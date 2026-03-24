"""
Checkpoint serialization for distributed hybrid inference.

Saves and restores the full diffusion state at segment boundaries,
including latents, text embeddings, and scheduler internal state
(model_outputs history required by multistep solvers like UniPC/DPMSolver).
"""

import logging
import os
import tempfile
from typing import Optional

import torch

logger = logging.getLogger(__name__)

CHECKPOINT_VERSION = 1
CHECKPOINT_MAGIC = "wan21_distributed_checkpoint"


def serialize_scheduler_state(scheduler) -> dict:
    """
    Extract mutable state from a FlowUniPCMultistepScheduler or
    FlowDPMSolverMultistepScheduler for serialization.

    The multistep solvers maintain a sliding window of previous model outputs
    (model_outputs) and timesteps (timestep_list) used for higher-order updates.
    These MUST be preserved for correct resumption.
    """
    state = {
        "model_outputs": [
            x.cpu() if isinstance(x, torch.Tensor) else x
            for x in scheduler.model_outputs
        ],
        "lower_order_nums": scheduler.lower_order_nums,
        "_step_index": scheduler._step_index,
        "_begin_index": scheduler._begin_index,
        "sigmas": scheduler.sigmas.cpu() if isinstance(scheduler.sigmas, torch.Tensor) else scheduler.sigmas,
        "timesteps": scheduler.timesteps.cpu(),
        "num_inference_steps": scheduler.num_inference_steps,
    }

    # UniPC-specific state
    if hasattr(scheduler, "timestep_list"):
        state["timestep_list"] = list(scheduler.timestep_list)
    if hasattr(scheduler, "last_sample"):
        state["last_sample"] = (
            scheduler.last_sample.cpu()
            if isinstance(scheduler.last_sample, torch.Tensor)
            else scheduler.last_sample
        )
    if hasattr(scheduler, "this_order"):
        state["this_order"] = scheduler.this_order

    return state


def restore_scheduler_state(scheduler, state: dict, device: torch.device):
    """
    Restore mutable state into a scheduler instance.

    The scheduler must already have been created with matching config
    (same num_train_timesteps, solver_order, etc.) and set_timesteps() called.
    This function overwrites the mutable fields with the checkpointed values.
    """
    scheduler.model_outputs = [
        x.to(device) if isinstance(x, torch.Tensor) else x
        for x in state["model_outputs"]
    ]
    scheduler.lower_order_nums = state["lower_order_nums"]
    scheduler._step_index = state["_step_index"]
    scheduler._begin_index = state["_begin_index"]
    scheduler.sigmas = state["sigmas"]  # kept on CPU per original code
    scheduler.timesteps = state["timesteps"].to(device)
    scheduler.num_inference_steps = state["num_inference_steps"]

    # UniPC-specific
    if "timestep_list" in state:
        scheduler.timestep_list = state["timestep_list"]
    if "last_sample" in state:
        scheduler.last_sample = (
            state["last_sample"].to(device)
            if isinstance(state["last_sample"], torch.Tensor)
            else state["last_sample"]
        )
    if "this_order" in state:
        scheduler.this_order = state["this_order"]


def save_checkpoint(
    path: str,
    job_id: str,
    segment_idx: int,
    global_step: int,
    latents: torch.Tensor,
    context: list,
    context_null: list,
    scheduler,
    sampling_params: dict,
    seed: int,
    atomic: bool = True,
):
    """
    Save a segment-boundary checkpoint to disk.

    Args:
        path: File path to write the checkpoint
        job_id: Job identifier
        segment_idx: Which segment we're about to start
        global_step: Absolute step index at segment boundary
        latents: Current latent tensor
        context: T5 prompt embeddings (list of tensors)
        context_null: T5 negative prompt embeddings
        scheduler: The scheduler instance (state will be serialized)
        sampling_params: Dict with guidance_scale, target_shape, etc.
        seed: The seed used for this generation
        atomic: If True, write to temp file then rename (crash-safe)
    """
    checkpoint = {
        "magic": CHECKPOINT_MAGIC,
        "version": CHECKPOINT_VERSION,
        "job_id": job_id,
        "segment_idx": segment_idx,
        "global_step": global_step,
        "latents": latents.cpu(),
        "context": [c.cpu() for c in context],
        "context_null": [c.cpu() for c in context_null],
        "scheduler_state": serialize_scheduler_state(scheduler),
        "sampling_params": sampling_params,
        "seed": seed,
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)

    if atomic:
        # Write to temp file in same directory, then atomic rename
        dir_name = os.path.dirname(path)
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        try:
            os.close(fd)
            torch.save(checkpoint, tmp_path)
            os.replace(tmp_path, path)
        except Exception:
            # Clean up temp file on failure
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
    else:
        torch.save(checkpoint, path)

    logger.info(f"Checkpoint saved: {path} (segment={segment_idx}, step={global_step})")


def load_checkpoint(path: str, device: torch.device) -> dict:
    """
    Load a checkpoint from disk and move tensors to the specified device.

    Returns a dict with all checkpoint fields, tensors on device.
    Validates the checkpoint magic and version.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    if checkpoint.get("magic") != CHECKPOINT_MAGIC:
        raise ValueError(f"Invalid checkpoint file: {path} (bad magic)")
    if checkpoint.get("version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"Checkpoint version mismatch: expected {CHECKPOINT_VERSION}, "
            f"got {checkpoint.get('version')}"
        )

    # Move tensors to device
    checkpoint["latents"] = checkpoint["latents"].to(device)
    checkpoint["context"] = [c.to(device) for c in checkpoint["context"]]
    checkpoint["context_null"] = [c.to(device) for c in checkpoint["context_null"]]

    logger.info(f"Checkpoint loaded: {path} (segment={checkpoint['segment_idx']}, step={checkpoint['global_step']})")
    return checkpoint


def save_segment_result(
    path: str,
    latents: torch.Tensor,
    scheduler,
    global_step: int,
    segment_idx: int,
    cache_hits: int = 0,
    fresh_computes: int = 0,
    segment_time: float = 0.0,
    atomic: bool = True,
):
    """
    Save the result of a completed segment (written by the worker).

    Contains the updated latents and scheduler state after running N steps.
    """
    result = {
        "magic": CHECKPOINT_MAGIC,
        "version": CHECKPOINT_VERSION,
        "type": "segment_result",
        "latents": latents.cpu(),
        "scheduler_state": serialize_scheduler_state(scheduler),
        "global_step": global_step,
        "segment_idx": segment_idx,
        "cache_hits": cache_hits,
        "fresh_computes": fresh_computes,
        "segment_time": segment_time,
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)

    if atomic:
        dir_name = os.path.dirname(path)
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        try:
            os.close(fd)
            torch.save(result, tmp_path)
            os.replace(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
    else:
        torch.save(result, path)

    logger.info(f"Segment result saved: {path}")


def load_segment_result(path: str, device: torch.device) -> dict:
    """
    Load a segment result from disk.

    Returns dict with latents on device and scheduler_state (CPU tensors).
    """
    result = torch.load(path, map_location="cpu", weights_only=False)

    if result.get("magic") != CHECKPOINT_MAGIC:
        raise ValueError(f"Invalid segment result file: {path}")
    if result.get("type") != "segment_result":
        raise ValueError(f"Expected segment_result, got: {result.get('type')}")

    result["latents"] = result["latents"].to(device)

    logger.info(f"Segment result loaded: {path}")
    return result


def get_checkpoint_paths(checkpoint_dir: str, job_id: str, segment_idx: int) -> tuple:
    """Return (input_path, output_path) for a given job and segment."""
    job_dir = os.path.join(checkpoint_dir, job_id)
    input_path = os.path.join(job_dir, f"segment_{segment_idx}_input.pt")
    output_path = os.path.join(job_dir, f"segment_{segment_idx}_output.pt")
    return input_path, output_path


def cleanup_job_checkpoints(checkpoint_dir: str, job_id: str):
    """Remove all checkpoint files for a completed job."""
    job_dir = os.path.join(checkpoint_dir, job_id)
    if os.path.exists(job_dir):
        import shutil
        shutil.rmtree(job_dir)
        logger.info(f"Cleaned up checkpoints for job {job_id}")
