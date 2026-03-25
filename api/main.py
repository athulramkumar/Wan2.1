"""
Wan2.1 Video Generation API - FastAPI Application

Production-ready REST API for video generation using Wan2.1 models.
"""

import logging
import os
from contextlib import asynccontextmanager
from dataclasses import asdict

import fastapi
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .config import (
    DEFAULTS,
    DISTRIBUTED,
    VALID_MODELS,
    VALID_FRAME_COUNTS,
    VALID_FPS,
    VALID_RESOLUTIONS,
    OUTPUT_DIR,
    Messages,
)
from .schemas import (
    GenerationRequest,
    JobSubmitResponse,
    JobStatusResponse,
    HealthResponse,
    ConfigResponse,
    ErrorResponse,
)
from .model_manager import get_model_manager, initialize_models
from .job_queue import get_job_queue, initialize_job_queue, shutdown_job_queue
from .utils.scheduling import validate_schedule

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Application Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    
    Startup: Load models and start job queue
    Shutdown: Cleanup resources
    """
    # Startup
    logger.info("=" * 60)
    logger.info("STARTING WAN2.1 VIDEO GENERATION API")
    logger.info("=" * 60)
    
    # Initialize models (mode from env, set by run_server.py --mode flag)
    mode = os.environ.get("MODEL_MODE", "both")
    logger.info(f"Loading models (mode={mode}, this may take a few minutes)...")
    model_manager = initialize_models(device_id=0, mode=mode)
    
    if not model_manager.is_loaded:
        logger.error("Failed to load models!")
        raise RuntimeError("Failed to load models")
    
    # Initialize job queue
    logger.info("Starting job queue...")
    job_queue = initialize_job_queue(model_manager)
    
    logger.info("=" * 60)
    logger.info("✓ SERVER READY")
    logger.info("=" * 60)
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    shutdown_job_queue()
    model_manager.cleanup()
    logger.info("✓ Shutdown complete")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Wan2.1 Video Generation API",
    description="""
    Generate videos using Wan2.1 text-to-video models.
    
    ## Features
    - **Baseline 14B**: Highest quality, slower generation
    - **Baseline 1.3B**: Fast generation, good quality
    - **Hybrid**: Best of both - 14B for structure, 1.3B for speed
    
    ## Workflow
    1. POST /generate - Submit a generation job
    2. GET /status/{job_id} - Poll for progress
    3. GET /video/{job_id} - Download the video when ready
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Health & Config Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Check server health and model status.
    
    Returns loaded models, GPU memory usage, and queue status.
    """
    model_manager = get_model_manager()
    job_queue = get_job_queue()
    
    if model_manager is None or not model_manager.is_loaded:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "models_loaded": [],
                "message": Messages.MODELS_LOADING
            }
        )
    
    gpu_mem = model_manager.get_gpu_memory()
    
    return HealthResponse(
        status=Messages.SERVER_HEALTHY,
        models_loaded=model_manager.loaded_models,
        gpu_memory_used=gpu_mem["allocated_gb"],
        queue_size=job_queue.get_queue_size() if job_queue else 0
    )


@app.get("/config", response_model=ConfigResponse, tags=["System"])
async def get_config():
    """
    Get default configuration and valid options.
    
    Useful for UI/client to know what values are acceptable.
    """
    return ConfigResponse(
        defaults={
            "model": DEFAULTS.model,
            "frame_count": DEFAULTS.frame_count,
            "fps": DEFAULTS.fps,
            "width": DEFAULTS.width,
            "height": DEFAULTS.height,
            "sampling_steps": DEFAULTS.sampling_steps,
            "guidance_scale": DEFAULTS.guidance_scale,
            "shift": DEFAULTS.shift,
            "enable_caching": DEFAULTS.enable_caching,
            "cache_start_step": DEFAULTS.cache_start_step,
            "cache_end_step": DEFAULTS.cache_end_step,
            "cache_interval": DEFAULTS.cache_interval,
            "seed": DEFAULTS.seed,
        },
        valid_models=VALID_MODELS,
        valid_frame_counts=VALID_FRAME_COUNTS,
        valid_fps=VALID_FPS,
        valid_resolutions=[list(r) for r in VALID_RESOLUTIONS],
    )


# =============================================================================
# Compute Endpoints (for local-orchestrator experiments)
# =============================================================================

@app.post("/run-steps-inline", tags=["Compute"])
async def run_steps_inline(request: fastapi.Request):
    """
    Run N denoising steps with the locally loaded model.
    Accepts checkpoint as binary POST body, returns result as binary.
    Used by the local MacBook orchestrator for the relay approach.
    """
    import io
    import torch
    import torch.cuda.amp as amp
    from contextlib import contextmanager
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    from api.distributed.checkpoint import restore_scheduler_state, serialize_scheduler_state

    body = await request.body()
    model_manager = get_model_manager()
    device = model_manager.device

    checkpoint = torch.load(io.BytesIO(body), map_location="cpu", weights_only=False)

    latents = checkpoint["latents"].to(device)
    context = [c.to(device) for c in checkpoint["context"]]
    context_null = [c.to(device) for c in checkpoint["context_null"]]
    params = checkpoint["sampling_params"]
    guidance_scale = params["guidance_scale"]
    seq_len = params.get("seq_len", 0)
    model_name = params.get("model_name", "1.3B")

    model = model_manager.get_model(model_name)

    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=model.num_train_timesteps, shift=1, use_dynamic_shifting=False,
    )
    total_steps = checkpoint["scheduler_state"]["num_inference_steps"]
    scheduler.set_timesteps(total_steps, device=device, shift=params.get("shift", 5.0))
    restore_scheduler_state(scheduler, checkpoint["scheduler_state"], device)

    global_step_start = checkpoint["global_step"]
    num_steps = params["num_steps"]
    segment_timesteps = scheduler.timesteps[global_step_start:global_step_start + num_steps]

    arg_c = {"context": context, "seq_len": seq_len}
    arg_null = {"context": context_null, "seq_len": seq_len}

    @contextmanager
    def noop():
        yield

    import time as _time
    model.model.to(device)
    no_sync = getattr(model.model, "no_sync", noop)
    start = _time.time()

    with amp.autocast(dtype=model.param_dtype), torch.no_grad(), no_sync():
        for i, t in enumerate(segment_timesteps):
            latent_model_input = [latents]
            timestep = torch.stack([t])
            noise_pred_cond = model.model(latent_model_input, t=timestep, **arg_c)[0]
            noise_pred_uncond = model.model(latent_model_input, t=timestep, **arg_null)[0]
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            temp_x0 = scheduler.step(noise_pred.unsqueeze(0), t, latents.unsqueeze(0), return_dict=False)[0]
            latents = temp_x0.squeeze(0)

    segment_time = _time.time() - start

    result = {
        "magic": "wan21_distributed_checkpoint",
        "version": 1,
        "type": "segment_result",
        "latents": latents.cpu(),
        "scheduler_state": serialize_scheduler_state(scheduler),
        "global_step": global_step_start + num_steps,
        "segment_idx": checkpoint.get("segment_idx", 0),
        "segment_time": segment_time,
        "fresh_computes": num_steps,
    }
    buf = io.BytesIO()
    torch.save(result, buf)
    return fastapi.responses.Response(content=buf.getvalue(), media_type="application/octet-stream")


@app.post("/encode-text", tags=["Compute"])
async def encode_text(request: dict):
    """Encode text with T5 and return embeddings as binary."""
    import io
    import torch

    model_manager = get_model_manager()
    device = model_manager.device
    model = model_manager.get_model(list(model_manager.get_models().keys())[0])

    prompt = request.get("prompt", "")
    negative_prompt = request.get("negative_prompt", "")

    model.text_encoder.model.to(device)
    context = model.text_encoder([prompt], device)
    context_null = model.text_encoder([negative_prompt], device)
    model.text_encoder.model.cpu()
    torch.cuda.empty_cache()

    result = {"context": context, "context_null": context_null}
    buf = io.BytesIO()
    torch.save(result, buf)
    return fastapi.responses.Response(content=buf.getvalue(), media_type="application/octet-stream")


@app.post("/decode-latents", tags=["Compute"])
async def decode_latents(request: fastapi.Request):
    """Decode latents to video with VAE, return MP4."""
    import io
    import torch
    import tempfile
    from wan.utils.utils import cache_video

    body = await request.body()
    data = torch.load(io.BytesIO(body), map_location="cpu", weights_only=False)

    model_manager = get_model_manager()
    device = model_manager.device
    model = model_manager.get_model(list(model_manager.get_models().keys())[0])

    latents = data["latents"].to(device)
    fps = data.get("fps", 16)

    with torch.no_grad():
        videos = model.vae.decode([latents])

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    cache_video(videos[0][None], save_file=tmp.name, fps=fps, nrow=1, normalize=True, value_range=(-1, 1))

    with open(tmp.name, "rb") as f:
        video_bytes = f.read()
    import os
    os.unlink(tmp.name)

    return fastapi.responses.Response(content=video_bytes, media_type="video/mp4")


# =============================================================================
# Admin Endpoints (for experiment orchestration)
# =============================================================================

@app.post("/admin/set-worker-url", tags=["Admin"])
async def set_worker_url(request: dict):
    """
    Dynamically set the remote worker URL for distributed mode.

    Called by the experiment orchestration script when a new worker pod starts.
    Resets the worker status tracker so the coordinator will try the new worker.
    """
    worker_url = request.get("worker_url", "")
    if not worker_url:
        raise HTTPException(status_code=400, detail="worker_url is required")

    # Update the distributed config
    DISTRIBUTED.enabled = True
    DISTRIBUTED.worker_url = worker_url

    # Reset the generate_video singletons so they pick up the new URL
    from .generator import generate_video
    if hasattr(generate_video, "_worker_client"):
        generate_video._worker_client.worker_url = worker_url.rstrip("/")
    if hasattr(generate_video, "_worker_status"):
        generate_video._worker_status.reset()

    logger.info(f"Worker URL set to: {worker_url}")
    return {"status": "ok", "worker_url": worker_url, "distributed_enabled": True}


@app.post("/admin/disable-distributed", tags=["Admin"])
async def disable_distributed():
    """Disable distributed mode (fall back to local-only generation)."""
    DISTRIBUTED.enabled = False
    DISTRIBUTED.worker_url = ""
    logger.info("Distributed mode disabled")
    return {"status": "ok", "distributed_enabled": False}


# =============================================================================
# Generation Endpoints
# =============================================================================

@app.post("/generate", response_model=JobSubmitResponse, tags=["Generation"])
async def submit_generation(request: GenerationRequest):
    """
    Submit a video generation job.
    
    Returns a job_id that can be used to track progress and retrieve the video.
    
    ## Model Options
    - `baseline_14B`: Full 14B model (highest quality, ~6-7 min)
    - `baseline_1.3B`: Full 1.3B model (fastest, ~1-2 min)
    - `hybrid`: Mix of both (balanced, ~3-4 min)
    
    ## Caching
    When `enable_caching=true`, the API will cache and reuse noise predictions
    to speed up generation. Configure with:
    - `cache_start_step`: When to start caching (default: 10)
    - `cache_end_step`: When to stop caching (default: 40)
    - `cache_interval`: Cache every N steps (default: 3)
    """
    job_queue = get_job_queue()
    
    if job_queue is None:
        raise HTTPException(
            status_code=503,
            detail="Server not ready. Models may still be loading."
        )
    
    # Validate hybrid schedule if provided
    if request.model == "hybrid" and request.schedule is not None:
        is_valid, error_msg = validate_schedule(request.schedule, request.sampling_steps)
        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid schedule: {error_msg}"
            )
    
    # Validate caching config
    if request.enable_caching:
        if request.cache_start_step >= request.sampling_steps:
            raise HTTPException(
                status_code=400,
                detail=f"cache_start_step ({request.cache_start_step}) must be < sampling_steps ({request.sampling_steps})"
            )
        if request.cache_end_step is not None:
            if request.cache_end_step < request.cache_start_step:
                raise HTTPException(
                    status_code=400,
                    detail=f"cache_end_step ({request.cache_end_step}) must be >= cache_start_step ({request.cache_start_step})"
                )
    
    # Submit job
    job = job_queue.submit_job(request)
    
    return JobSubmitResponse(
        job_id=job.job_id,
        status=job.status.value,
        message=Messages.JOB_SUBMITTED
    )


@app.get("/status/{job_id}", response_model=JobStatusResponse, tags=["Generation"])
async def get_job_status(job_id: str):
    """
    Get the status of a generation job.
    
    Poll this endpoint to track progress. When status is "completed",
    use `/video/{job_id}` to download the video.
    
    ## Status Values
    - `queued`: Waiting to start
    - `processing`: Currently generating
    - `completed`: Done - video ready for download
    - `failed`: Generation failed - check error field
    """
    job_queue = get_job_queue()
    
    if job_queue is None:
        raise HTTPException(status_code=503, detail="Server not ready")
    
    job = job_queue.get_job(job_id)
    
    if job is None:
        raise HTTPException(status_code=404, detail=Messages.JOB_NOT_FOUND)
    
    return JobStatusResponse(**job.to_status_dict())


@app.get("/video/{job_id}", tags=["Generation"])
async def get_video(job_id: str):
    """
    Download the generated video.
    
    Only available when job status is "completed".
    Returns an MP4 video file that can be played directly or used in `<video>` tags.
    """
    job_queue = get_job_queue()
    
    if job_queue is None:
        raise HTTPException(status_code=503, detail="Server not ready")
    
    job = job_queue.get_job(job_id)
    
    if job is None:
        raise HTTPException(status_code=404, detail=Messages.JOB_NOT_FOUND)
    
    if job.status.value != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Video not ready. Current status: {job.status.value}"
        )
    
    if job.video_path is None or not os.path.exists(job.video_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    return FileResponse(
        path=job.video_path,
        media_type="video/mp4",
        filename=f"wan2.1_{job_id}.mp4",
        headers={
            "Content-Disposition": f"inline; filename=wan2.1_{job_id}.mp4"
        }
    )


# =============================================================================
# Error Handlers
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "HTTPException", "message": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "InternalServerError", "message": str(exc)}
    )

