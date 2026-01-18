"""
Configuration for Wan2.1 Video Generation API.

Contains default values and server settings.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# Path Configuration
# =============================================================================

# Base directory (where the Wan2.1 code lives)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model checkpoint directories
CHECKPOINT_DIR_14B = os.path.join(BASE_DIR, "Wan2.1-T2V-14B")
CHECKPOINT_DIR_1_3B = os.path.join(BASE_DIR, "Wan2.1-T2V-1.3B")

# Output directory for generated videos
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "jobs")


# =============================================================================
# Server Configuration
# =============================================================================

@dataclass
class ServerConfig:
    """Server configuration settings."""
    host: str = "0.0.0.0"
    port: int = 30396
    workers: int = 1  # Single worker for GPU
    reload: bool = False  # Don't reload in production


# =============================================================================
# Model Configuration
# =============================================================================

# Valid model choices
VALID_MODELS = ["baseline_14B", "baseline_1.3B", "hybrid"]

# Valid frame counts (must be 4n+1)
VALID_FRAME_COUNTS = [17, 33, 49, 65, 81]

# Valid FPS values
VALID_FPS = [8, 16, 24]

# Valid resolutions (width, height)
VALID_RESOLUTIONS = [
    (832, 480),   # 480p
    (1280, 720),  # 720p
]


# =============================================================================
# Generation Defaults
# =============================================================================

@dataclass
class GenerationDefaults:
    """Default values for video generation."""
    
    # Model selection
    model: str = "baseline_14B"
    
    # Default hybrid schedule: 30% 14B, 70% 1.3B
    default_schedule: list = field(default_factory=lambda: [["14B", 15], ["1.3B", 35]])
    
    # Video settings
    frame_count: int = 81
    fps: int = 16
    width: int = 832
    height: int = 480
    
    # Sampling settings
    sampling_steps: int = 50
    guidance_scale: float = 5.0
    shift: float = 5.0
    
    # Caching settings (disabled by default)
    enable_caching: bool = False
    cache_start_step: int = 10
    cache_end_step: Optional[int] = 40
    cache_interval: int = 3
    
    # Reproducibility
    seed: int = -1  # -1 means random
    negative_prompt: str = ""


# =============================================================================
# Negative Prompt (used when none provided)
# =============================================================================

DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, "
    "works, paintings, images, static, overall gray, worst quality, "
    "low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn face, deformed, disfigured, "
    "misaligned limbs, extra limbs, missing arms, missing legs, "
    "extra arms, extra legs, fused fingers, too many fingers, "
    "long neck, username, watermark, signature"
)


# =============================================================================
# API Response Messages
# =============================================================================

class Messages:
    """Standard API response messages."""
    
    JOB_SUBMITTED = "Job submitted successfully"
    JOB_NOT_FOUND = "Job not found"
    SERVER_HEALTHY = "Server is healthy"
    MODELS_LOADING = "Models are still loading"
    INVALID_MODEL = "Invalid model. Must be one of: baseline_14B, baseline_1.3B, hybrid"
    INVALID_FRAME_COUNT = f"Invalid frame_count. Must be one of: {VALID_FRAME_COUNTS}"
    INVALID_SCHEDULE = "Invalid schedule. Must sum to sampling_steps"


# =============================================================================
# Instantiate defaults
# =============================================================================

DEFAULTS = GenerationDefaults()
SERVER = ServerConfig()

