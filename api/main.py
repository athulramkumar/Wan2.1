"""
Wan2.1 Video Generation API - FastAPI Application

Production-ready REST API for video generation using Wan2.1 models.
"""

import logging
import os
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .config import (
    DEFAULTS,
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
    
    # Initialize models
    logger.info("Loading models (this may take a few minutes)...")
    model_manager = initialize_models(device_id=0)
    
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

