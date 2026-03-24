"""
Video Generation Logic.

Implements baseline and hybrid video generation with optional caching.
Based on the hybrid_model_sampling.ipynb implementation.
"""

import logging
import math
import os
import random
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Callable

import torch
import torch.cuda.amp as amp
from tqdm import tqdm

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from wan.utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from wan.utils.utils import cache_video

from .model_manager import ModelManager
from .utils.caching import should_use_cache, get_cache_statistics
from .utils.scheduling import parse_schedule, get_model_for_step, get_schedule_summary
from .config import OUTPUT_DIR, DEFAULT_NEGATIVE_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class GenerationProgress:
    """Tracks generation progress."""
    current_step: int = 0
    total_steps: int = 0
    model_in_use: str = ""
    cache_hits: int = 0
    fresh_computes: int = 0
    start_time: float = 0.0
    
    @property
    def progress_percent(self) -> int:
        if self.total_steps == 0:
            return 0
        return int((self.current_step / self.total_steps) * 100)
    
    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time
    
    @property
    def estimated_remaining(self) -> float:
        if self.current_step == 0:
            return 0.0
        time_per_step = self.elapsed_time / self.current_step
        remaining_steps = self.total_steps - self.current_step
        return time_per_step * remaining_steps


@dataclass
class GenerationResult:
    """Result of video generation."""
    success: bool
    video_path: Optional[str] = None
    generation_time: float = 0.0
    cache_statistics: Optional[dict] = None
    error: Optional[str] = None
    seed_used: int = -1


def generate_video(
    model_manager: ModelManager,
    prompt: str,
    model_type: str,  # "baseline_14B", "baseline_1.3B", or "hybrid"
    schedule: Optional[list] = None,  # For hybrid: [("14B", 15), ("1.3B", 35)]
    # Video settings
    width: int = 832,
    height: int = 480,
    frame_count: int = 81,
    fps: int = 16,
    # Sampling settings
    sampling_steps: int = 50,
    guidance_scale: float = 5.0,
    shift: float = 5.0,
    # Caching settings
    enable_caching: bool = False,
    cache_start_step: int = 10,
    cache_end_step: Optional[int] = 40,
    cache_interval: int = 3,
    # Other
    seed: int = -1,
    negative_prompt: str = "",
    job_id: str = "",
    progress_callback: Optional[Callable[[GenerationProgress], None]] = None,
) -> GenerationResult:
    """
    Generate a video using the specified model configuration.

    In distributed mode (DISTRIBUTED.enabled), delegates to the coordinator
    which orchestrates across local 1.3B and remote 14B worker.
    Otherwise, runs the original single-GPU pipeline.
    """
    # Distributed mode dispatch
    from .config import DISTRIBUTED
    if DISTRIBUTED.enabled and DISTRIBUTED.worker_url:
        from .distributed.coordinator import run_distributed_generation
        from .distributed.worker_client import WorkerClient
        from .distributed.worker_status import WorkerStatus
        from .distributed.spot_manager import SpotManager
        from .config import SPOT

        # Use module-level singletons for client/status/spot across jobs
        if not hasattr(generate_video, "_worker_client"):
            generate_video._worker_client = WorkerClient(
                worker_url=DISTRIBUTED.worker_url,
                health_timeout=DISTRIBUTED.health_check_timeout,
            )
            generate_video._worker_status = WorkerStatus(
                max_consecutive_failures=DISTRIBUTED.max_consecutive_failures,
                backoff_duration=DISTRIBUTED.backoff_duration,
            )
            generate_video._spot_manager = SpotManager(
                api_key=SPOT.api_key,
                gpu_type=SPOT.gpu_type,
                gpu_count=SPOT.gpu_count,
                volume_id=SPOT.volume_id,
                template_id=SPOT.template_id,
                worker_port=SPOT.worker_port,
                idle_timeout=SPOT.idle_timeout,
                max_start_wait=SPOT.max_start_wait,
            ) if SPOT.api_key else None

        return run_distributed_generation(
            model_manager=model_manager,
            worker_client=generate_video._worker_client,
            worker_status=generate_video._worker_status,
            spot_manager=getattr(generate_video, "_spot_manager", None),
            prompt=prompt,
            model_type=model_type,
            schedule=schedule,
            width=width, height=height,
            frame_count=frame_count, fps=fps,
            sampling_steps=sampling_steps,
            guidance_scale=guidance_scale, shift=shift,
            enable_caching=enable_caching,
            cache_start_step=cache_start_step,
            cache_end_step=cache_end_step,
            cache_interval=cache_interval,
            seed=seed,
            negative_prompt=negative_prompt,
            job_id=job_id,
            progress_callback=progress_callback,
        )

    start_time = time.time()
    progress = GenerationProgress(
        total_steps=sampling_steps,
        start_time=start_time
    )
    
    try:
        # Get device from model manager
        device = model_manager.device
        
        # Determine which model(s) to use
        if model_type == "baseline_14B":
            models_to_use = {"14B": model_manager.get_model("14B")}
            parsed_schedule = [("14B", sampling_steps)]
        elif model_type == "baseline_1.3B":
            models_to_use = {"1.3B": model_manager.get_model("1.3B")}
            parsed_schedule = [("1.3B", sampling_steps)]
        elif model_type == "hybrid":
            models_to_use = model_manager.get_models()
            if schedule is None:
                # Default hybrid schedule
                steps_14B = int(sampling_steps * 0.3)
                steps_1_3B = sampling_steps - steps_14B
                parsed_schedule = [("14B", steps_14B), ("1.3B", steps_1_3B)]
            else:
                parsed_schedule = parse_schedule(schedule)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        logger.info(f"Generation config:")
        logger.info(f"  Model: {model_type}")
        logger.info(f"  Schedule: {get_schedule_summary(parsed_schedule)}")
        logger.info(f"  Resolution: {width}x{height}, {frame_count} frames")
        logger.info(f"  Sampling steps: {sampling_steps}")
        logger.info(f"  Caching: {enable_caching}")
        
        # Get first model for shared components (VAE, T5)
        first_model = models_to_use[parsed_schedule[0][0]]
        
        # Setup negative prompt
        n_prompt = negative_prompt if negative_prompt else DEFAULT_NEGATIVE_PROMPT
        
        # Setup seed
        actual_seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=device)
        seed_g.manual_seed(actual_seed)
        logger.info(f"  Seed: {actual_seed}")
        
        # Calculate target shape
        F = frame_count
        vae_stride = first_model.vae_stride
        patch_size = first_model.patch_size
        
        target_shape = (
            first_model.vae.model.z_dim,
            (F - 1) // vae_stride[0] + 1,
            height // vae_stride[1],
            width // vae_stride[2]
        )
        
        seq_len = math.ceil(
            (target_shape[2] * target_shape[3]) /
            (patch_size[1] * patch_size[2]) *
            target_shape[1]
        )
        
        logger.info(f"  Latent shape: {target_shape}")
        logger.info(f"  Sequence length: {seq_len}")
        
        # Encode text prompt
        logger.info("Encoding text prompt...")
        first_model.text_encoder.model.to(device)
        context = first_model.text_encoder([prompt], device)
        context_null = first_model.text_encoder([n_prompt], device)
        first_model.text_encoder.model.cpu()
        torch.cuda.empty_cache()
        
        # Initialize noise
        noise = torch.randn(
            target_shape[0],
            target_shape[1],
            target_shape[2],
            target_shape[3],
            dtype=torch.float32,
            device=device,
            generator=seed_g
        )
        
        # Setup scheduler
        sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=first_model.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False
        )
        sample_scheduler.set_timesteps(sampling_steps, device=device, shift=shift)
        timesteps = sample_scheduler.timesteps
        
        # Initialize latents
        latents = noise
        
        # Prepare model arguments
        arg_c = {'context': context, 'seq_len': seq_len}
        arg_null = {'context': context_null, 'seq_len': seq_len}
        
        # Caching state
        cached_noise_pred = None
        cache_hits = 0
        fresh_computes = 0
        
        # Noop context manager
        @contextmanager
        def noop_no_sync():
            yield
        
        # Run sampling
        logger.info("Starting sampling...")
        step_idx = 0
        
        for segment_idx, (model_name, num_steps) in enumerate(parsed_schedule):
            model = models_to_use[model_name]
            logger.info(f"Segment {segment_idx + 1}: {model_name} ({num_steps} steps)")
            
            # Ensure model is on device
            model.model.to(device)
            no_sync = getattr(model.model, 'no_sync', noop_no_sync)
            
            # Get timesteps for this segment
            segment_timesteps = timesteps[step_idx:step_idx + num_steps]
            
            with amp.autocast(dtype=model.param_dtype), torch.no_grad(), no_sync():
                for i, t in enumerate(segment_timesteps):
                    global_step = step_idx + i
                    
                    # Update progress
                    progress.current_step = global_step + 1
                    progress.model_in_use = model_name
                    progress.cache_hits = cache_hits
                    progress.fresh_computes = fresh_computes
                    
                    if progress_callback:
                        progress_callback(progress)
                    
                    latent_model_input = [latents]
                    timestep = torch.stack([t])
                    
                    # Check if we should use cache
                    use_cache = (
                        enable_caching and
                        cached_noise_pred is not None and
                        should_use_cache(global_step, cache_start_step, cache_end_step, cache_interval)
                    )
                    
                    if use_cache:
                        noise_pred = cached_noise_pred
                        cache_hits += 1
                    else:
                        # Conditional prediction
                        noise_pred_cond = model.model(latent_model_input, t=timestep, **arg_c)[0]
                        # Unconditional prediction
                        noise_pred_uncond = model.model(latent_model_input, t=timestep, **arg_null)[0]
                        
                        # Classifier-free guidance
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                        
                        # Store for potential cache reuse
                        if enable_caching:
                            cached_noise_pred = noise_pred.clone()
                        
                        fresh_computes += 1
                    
                    # Scheduler step
                    temp_x0 = sample_scheduler.step(
                        noise_pred.unsqueeze(0),
                        t,
                        latents.unsqueeze(0),
                        return_dict=False,
                        generator=seed_g
                    )[0]
                    latents = temp_x0.squeeze(0)
            
            step_idx += num_steps
        
        # Final progress update
        progress.current_step = sampling_steps
        progress.cache_hits = cache_hits
        progress.fresh_computes = fresh_computes
        if progress_callback:
            progress_callback(progress)
        
        # Decode latents
        logger.info("Decoding latents...")
        with torch.no_grad():
            videos = first_model.vae.decode([latents])
        
        # Save video
        output_filename = f"{job_id}.mp4" if job_id else f"video_{int(time.time())}.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        cache_video(
            videos[0][None],
            save_file=output_path,
            fps=fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1)
        )
        
        generation_time = time.time() - start_time
        logger.info(f"✓ Video saved to: {output_path}")
        logger.info(f"  Generation time: {generation_time:.2f}s")
        
        # Build cache statistics
        cache_stats = None
        if enable_caching:
            cache_stats = {
                "cache_hits": cache_hits,
                "fresh_computes": fresh_computes,
                "cache_hit_rate": round(cache_hits / sampling_steps, 3) if sampling_steps > 0 else 0
            }
            logger.info(f"  Cache stats: {cache_hits} hits, {fresh_computes} fresh")
        
        # Cleanup
        del noise, latents, sample_scheduler
        torch.cuda.empty_cache()
        
        return GenerationResult(
            success=True,
            video_path=output_path,
            generation_time=generation_time,
            cache_statistics=cache_stats,
            seed_used=actual_seed
        )
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()
        
        return GenerationResult(
            success=False,
            error=str(e),
            generation_time=time.time() - start_time
        )

