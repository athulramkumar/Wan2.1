"""
Pydantic schemas for API request/response validation.
"""

from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator

from .config import (
    VALID_MODELS,
    VALID_FRAME_COUNTS,
    VALID_FPS,
    DEFAULTS,
)


# =============================================================================
# Request Schemas
# =============================================================================

class GenerationRequest(BaseModel):
    """Request schema for video generation."""
    
    # Required
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Text prompt describing the video to generate"
    )
    
    # Model selection
    model: Literal["baseline_14B", "baseline_1.3B", "hybrid"] = Field(
        default=DEFAULTS.model,
        description="Which model(s) to use for generation"
    )
    
    # Hybrid schedule (only used when model="hybrid")
    schedule: Optional[list[list]] = Field(
        default=None,
        description="Hybrid schedule as [[model, steps], ...]. Must sum to sampling_steps."
    )
    
    # Video settings
    frame_count: int = Field(
        default=DEFAULTS.frame_count,
        description="Number of frames to generate (must be 4n+1)"
    )
    fps: int = Field(
        default=DEFAULTS.fps,
        description="Output video FPS"
    )
    width: int = Field(
        default=DEFAULTS.width,
        description="Video width in pixels"
    )
    height: int = Field(
        default=DEFAULTS.height,
        description="Video height in pixels"
    )
    
    # Sampling settings
    sampling_steps: int = Field(
        default=DEFAULTS.sampling_steps,
        ge=10,
        le=100,
        description="Number of diffusion sampling steps"
    )
    guidance_scale: float = Field(
        default=DEFAULTS.guidance_scale,
        ge=1.0,
        le=20.0,
        description="Classifier-free guidance scale"
    )
    shift: float = Field(
        default=DEFAULTS.shift,
        ge=1.0,
        le=20.0,
        description="Noise schedule shift parameter"
    )
    
    # Caching settings
    enable_caching: bool = Field(
        default=DEFAULTS.enable_caching,
        description="Enable simple activation caching for speedup"
    )
    cache_start_step: int = Field(
        default=DEFAULTS.cache_start_step,
        ge=0,
        description="Absolute step to start caching"
    )
    cache_end_step: Optional[int] = Field(
        default=DEFAULTS.cache_end_step,
        description="Absolute step to stop caching (None = until end)"
    )
    cache_interval: int = Field(
        default=DEFAULTS.cache_interval,
        ge=2,
        le=10,
        description="Cache interval (compute fresh every N steps)"
    )
    
    # Reproducibility
    seed: int = Field(
        default=DEFAULTS.seed,
        description="Random seed (-1 for random)"
    )
    negative_prompt: str = Field(
        default=DEFAULTS.negative_prompt,
        max_length=2000,
        description="Negative prompt for content exclusion"
    )
    
    @field_validator("frame_count")
    @classmethod
    def validate_frame_count(cls, v):
        if v not in VALID_FRAME_COUNTS:
            raise ValueError(f"frame_count must be one of {VALID_FRAME_COUNTS}")
        return v
    
    @field_validator("fps")
    @classmethod
    def validate_fps(cls, v):
        if v not in VALID_FPS:
            raise ValueError(f"fps must be one of {VALID_FPS}")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "A cat playing piano in a cozy living room",
                "model": "hybrid",
                "frame_count": 81,
                "fps": 16,
                "sampling_steps": 50,
                "enable_caching": True,
                "cache_start_step": 10,
                "cache_end_step": 40,
                "cache_interval": 3
            }
        }


# =============================================================================
# Response Schemas
# =============================================================================

class JobSubmitResponse(BaseModel):
    """Response after submitting a generation job."""
    
    job_id: str = Field(description="Unique job identifier")
    status: str = Field(description="Job status")
    message: str = Field(description="Status message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "abc123",
                "status": "queued",
                "message": "Job submitted successfully"
            }
        }


class CacheStatistics(BaseModel):
    """Cache statistics for a generation run."""
    
    cache_hits: int = Field(description="Number of steps using cache")
    fresh_computes: int = Field(description="Number of steps computed fresh")
    cache_hit_rate: float = Field(description="Cache hit rate (0.0-1.0)")


class JobMetadata(BaseModel):
    """Metadata about a completed job."""
    
    model: str
    schedule: Optional[list[list]] = None
    frame_count: int
    fps: int
    width: int
    height: int
    sampling_steps: int
    seed: int
    prompt: str


class JobStatusResponse(BaseModel):
    """Response for job status query."""
    
    job_id: str = Field(description="Unique job identifier")
    status: Literal["queued", "processing", "completed", "failed"] = Field(
        description="Current job status"
    )
    progress: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Progress percentage (0-100)"
    )
    
    # Progress details (only when processing)
    current_step: Optional[int] = Field(default=None, description="Current sampling step")
    total_steps: Optional[int] = Field(default=None, description="Total sampling steps")
    model_in_use: Optional[str] = Field(default=None, description="Currently active model")
    
    # Timing
    elapsed_time: Optional[float] = Field(default=None, description="Elapsed time in seconds")
    estimated_remaining: Optional[float] = Field(default=None, description="Estimated remaining time")
    
    # Results (only when completed)
    video_url: Optional[str] = Field(default=None, description="URL to download video")
    generation_time: Optional[float] = Field(default=None, description="Total generation time")
    
    # Cache statistics (if caching was enabled)
    cache_statistics: Optional[CacheStatistics] = Field(default=None)
    
    # Metadata (only when completed)
    metadata: Optional[JobMetadata] = Field(default=None)
    
    # Error info (only when failed)
    error: Optional[str] = Field(default=None, description="Error message if failed")
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "abc123",
                "status": "processing",
                "progress": 45,
                "current_step": 23,
                "total_steps": 50,
                "model_in_use": "1.3B",
                "elapsed_time": 120.5
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(description="Server status")
    models_loaded: list[str] = Field(description="List of loaded models")
    gpu_memory_used: Optional[float] = Field(default=None, description="GPU memory in GB")
    queue_size: int = Field(default=0, description="Number of jobs in queue")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "models_loaded": ["14B", "1.3B"],
                "gpu_memory_used": 63.89,
                "queue_size": 0
            }
        }


class ConfigResponse(BaseModel):
    """Configuration/defaults response."""
    
    defaults: dict = Field(description="Default generation settings")
    valid_models: list[str] = Field(description="Valid model choices")
    valid_frame_counts: list[int] = Field(description="Valid frame count values")
    valid_fps: list[int] = Field(description="Valid FPS values")
    valid_resolutions: list[list[int]] = Field(description="Valid resolution pairs")


class ErrorResponse(BaseModel):
    """Error response."""
    
    error: str = Field(description="Error type")
    message: str = Field(description="Error message")
    details: Optional[dict] = Field(default=None, description="Additional details")

