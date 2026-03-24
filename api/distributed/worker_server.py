"""
Worker server for distributed hybrid inference.

Runs on the spot instance (H100/A100/2xA40) and executes 14B model
segments on demand. The coordinator sends checkpoint paths via HTTP,
the worker reads from shared volume, runs the denoising steps, and
writes the result back.

Endpoints:
    GET  /health           - Worker status and GPU info
    POST /run-segment      - Execute a segment of denoising steps
    GET  /segment-status/{task_id} - Poll segment progress
"""

import logging
import math
import os
import sys
import time
import threading
import uuid
from contextlib import asynccontextmanager, contextmanager
from typing import Optional

import torch
import torch.cuda.amp as amp
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add project root for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from api.distributed.checkpoint import (
    load_checkpoint,
    save_segment_result,
    restore_scheduler_state,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class SegmentRequest(BaseModel):
    checkpoint_path: str   # Path to input checkpoint on shared volume
    result_path: str       # Path to write output on shared volume


class SegmentStatus(BaseModel):
    task_id: str
    status: str            # "running", "completed", "failed"
    current_step: int = 0
    total_steps: int = 0
    elapsed_time: float = 0.0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Worker state
# ---------------------------------------------------------------------------

class WorkerState:
    """Tracks the current segment execution."""

    def __init__(self):
        self.model_manager = None
        self.current_task: Optional[SegmentStatus] = None
        self.lock = threading.Lock()
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))


_state = WorkerState()

# ---------------------------------------------------------------------------
# Segment execution (runs in a thread)
# ---------------------------------------------------------------------------

def _run_segment(task_id: str, checkpoint_path: str, result_path: str):
    """
    Load checkpoint, run denoising steps with 14B model, save result.

    This runs in a background thread so the HTTP endpoint returns immediately.
    """
    try:
        device = _state.model_manager.device
        model = _state.model_manager.get_model("14B")

        # Load checkpoint from shared volume
        checkpoint = load_checkpoint(checkpoint_path, device)

        latents = checkpoint["latents"]
        context = checkpoint["context"]
        context_null = checkpoint["context_null"]
        params = checkpoint["sampling_params"]
        guidance_scale = params["guidance_scale"]
        seq_len = params.get("seq_len", 0)

        # Recreate scheduler and restore state
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=model.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        # We need to call set_timesteps first to initialize the scheduler,
        # then overwrite with the checkpointed state
        total_steps = checkpoint["scheduler_state"]["num_inference_steps"]
        scheduler.set_timesteps(total_steps, device=device, shift=params.get("shift", 5.0))
        restore_scheduler_state(scheduler, checkpoint["scheduler_state"], device)

        # Determine which timesteps this segment should run
        global_step_start = checkpoint["global_step"]
        num_steps = params["num_steps"]  # steps for this segment
        timesteps = scheduler.timesteps
        segment_timesteps = timesteps[global_step_start:global_step_start + num_steps]

        # Model args
        arg_c = {"context": context, "seq_len": seq_len}
        arg_null = {"context": context_null, "seq_len": seq_len}

        # Update task status
        with _state.lock:
            if _state.current_task and _state.current_task.task_id == task_id:
                _state.current_task.total_steps = num_steps

        # Noop context manager for no_sync
        @contextmanager
        def noop_no_sync():
            yield

        # Ensure model is on device
        model.model.to(device)
        no_sync = getattr(model.model, "no_sync", noop_no_sync)

        cache_hits = 0
        fresh_computes = 0
        segment_start = time.time()

        with amp.autocast(dtype=model.param_dtype), torch.no_grad(), no_sync():
            for i, t in enumerate(segment_timesteps):
                # Update progress
                with _state.lock:
                    if _state.current_task and _state.current_task.task_id == task_id:
                        _state.current_task.current_step = i + 1
                        _state.current_task.elapsed_time = time.time() - segment_start

                latent_model_input = [latents]
                timestep = torch.stack([t])

                # Conditional prediction
                noise_pred_cond = model.model(latent_model_input, t=timestep, **arg_c)[0]
                # Unconditional prediction
                noise_pred_uncond = model.model(latent_model_input, t=timestep, **arg_null)[0]

                # Classifier-free guidance
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                fresh_computes += 1

                # Scheduler step
                temp_x0 = scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents.unsqueeze(0),
                    return_dict=False,
                )[0]
                latents = temp_x0.squeeze(0)

        segment_time = time.time() - segment_start

        # Save result to shared volume
        save_segment_result(
            path=result_path,
            latents=latents,
            scheduler=scheduler,
            global_step=global_step_start + num_steps,
            segment_idx=checkpoint["segment_idx"],
            cache_hits=cache_hits,
            fresh_computes=fresh_computes,
            segment_time=segment_time,
        )

        with _state.lock:
            if _state.current_task and _state.current_task.task_id == task_id:
                _state.current_task.status = "completed"
                _state.current_task.elapsed_time = segment_time

        logger.info(f"Segment completed: {num_steps} steps in {segment_time:.1f}s")

    except Exception as e:
        logger.error(f"Segment execution failed: {e}")
        import traceback
        traceback.print_exc()
        with _state.lock:
            if _state.current_task and _state.current_task.task_id == task_id:
                _state.current_task.status = "failed"
                _state.current_task.error = str(e)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load 14B model at startup."""
    from api.model_manager import initialize_models

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    logger.info(f"Worker starting (rank={rank}, world_size={world_size})")

    _state.model_manager = initialize_models(
        device_id=rank,
        mode="worker",
        world_size=world_size,
        rank=rank,
    )
    _state.rank = rank
    _state.world_size = world_size

    yield

    if _state.model_manager:
        _state.model_manager.cleanup()


app = FastAPI(title="Wan2.1 Worker (14B)", lifespan=lifespan)


@app.get("/health")
async def health():
    """Worker health check."""
    gpu_memory = _state.model_manager.get_gpu_memory() if _state.model_manager else {}
    is_busy = False
    with _state.lock:
        is_busy = _state.current_task is not None and _state.current_task.status == "running"

    return {
        "status": "ok",
        "model": "14B",
        "mode": "worker",
        "rank": _state.rank,
        "world_size": _state.world_size,
        "gpu_memory": gpu_memory,
        "is_busy": is_busy,
        "models_loaded": _state.model_manager.loaded_models if _state.model_manager else [],
    }


@app.post("/run-segment")
async def run_segment(request: SegmentRequest):
    """
    Execute a segment of denoising steps using the 14B model.

    Reads checkpoint from shared volume, runs steps, writes result back.
    Returns immediately with a task_id for polling.
    """
    # Only rank 0 receives HTTP requests
    with _state.lock:
        if _state.current_task and _state.current_task.status == "running":
            raise HTTPException(status_code=409, detail="Worker is busy with another segment")

    if not os.path.exists(request.checkpoint_path):
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {request.checkpoint_path}")

    task_id = str(uuid.uuid4())[:8]

    with _state.lock:
        _state.current_task = SegmentStatus(
            task_id=task_id,
            status="running",
        )

    # Run in background thread
    thread = threading.Thread(
        target=_run_segment,
        args=(task_id, request.checkpoint_path, request.result_path),
        daemon=True,
    )
    thread.start()

    return {"task_id": task_id, "status": "running"}


@app.get("/segment-status/{task_id}")
async def segment_status(task_id: str):
    """Poll the status of a running segment."""
    with _state.lock:
        if _state.current_task is None:
            raise HTTPException(status_code=404, detail="No active task")
        if _state.current_task.task_id != task_id:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        return _state.current_task.model_copy()
