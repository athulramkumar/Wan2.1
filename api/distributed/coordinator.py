"""
Coordinator for distributed hybrid inference.

Orchestrates the diffusion loop across coordinator (1.3B, local) and
worker (14B, remote spot instance). Falls back to 1.3B if the worker
is unavailable, times out, or is preempted mid-segment.

The coordinator:
1. Encodes text with T5 (local)
2. Initializes noise and scheduler
3. For each segment in the schedule:
   - If "14B" and worker available: delegate via checkpoint on shared volume
   - Otherwise: run locally with 1.3B
4. Decodes latents with VAE (local)
"""

import logging
import math
import os
import random
import sys
import time
from contextlib import contextmanager
from typing import Optional, Callable

import torch
import torch.cuda.amp as amp

from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from wan.utils.utils import cache_video

from api.model_manager import ModelManager
from api.config import OUTPUT_DIR, DEFAULT_NEGATIVE_PROMPT, DISTRIBUTED
from api.utils.caching import should_use_cache
from api.utils.scheduling import parse_schedule, get_schedule_summary
from api.distributed.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    load_segment_result,
    serialize_scheduler_state,
    restore_scheduler_state,
    get_checkpoint_paths,
    cleanup_job_checkpoints,
)
from api.distributed.worker_client import WorkerClient
from api.distributed.worker_status import WorkerStatus
from api.distributed.spot_manager import SpotManager

logger = logging.getLogger(__name__)

# Re-use the same dataclasses from generator.py
from api.generator import GenerationProgress, GenerationResult


def _run_segment_locally(
    model,
    latents: torch.Tensor,
    scheduler,
    segment_timesteps: torch.Tensor,
    context: list,
    context_null: list,
    seq_len: int,
    guidance_scale: float,
    device: torch.device,
    enable_caching: bool = False,
    cached_noise_pred=None,
    cache_start_step: int = 10,
    cache_end_step: Optional[int] = 40,
    cache_interval: int = 3,
    global_step_offset: int = 0,
    progress_callback=None,
    model_name: str = "1.3B",
) -> tuple:
    """
    Run a segment of denoising steps locally.

    Returns (latents, cached_noise_pred, cache_hits, fresh_computes)
    """
    arg_c = {"context": context, "seq_len": seq_len}
    arg_null = {"context": context_null, "seq_len": seq_len}
    cache_hits = 0
    fresh_computes = 0

    @contextmanager
    def noop_no_sync():
        yield

    model.model.to(device)
    no_sync = getattr(model.model, "no_sync", noop_no_sync)

    with amp.autocast(dtype=model.param_dtype), torch.no_grad(), no_sync():
        for i, t in enumerate(segment_timesteps):
            global_step = global_step_offset + i

            if progress_callback:
                progress_callback(global_step + 1, model_name)

            latent_model_input = [latents]
            timestep = torch.stack([t])

            use_cache = (
                enable_caching
                and cached_noise_pred is not None
                and should_use_cache(global_step, cache_start_step, cache_end_step, cache_interval)
            )

            if use_cache:
                noise_pred = cached_noise_pred
                cache_hits += 1
            else:
                noise_pred_cond = model.model(latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_uncond = model.model(latent_model_input, t=timestep, **arg_null)[0]
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                if enable_caching:
                    cached_noise_pred = noise_pred.clone()
                fresh_computes += 1

            temp_x0 = scheduler.step(
                noise_pred.unsqueeze(0), t, latents.unsqueeze(0), return_dict=False
            )[0]
            latents = temp_x0.squeeze(0)

    return latents, cached_noise_pred, cache_hits, fresh_computes


def run_distributed_generation(
    model_manager: ModelManager,
    worker_client: WorkerClient,
    worker_status: WorkerStatus,
    prompt: str,
    spot_manager: Optional[SpotManager] = None,
    model_type: str = "hybrid",
    schedule: Optional[list] = None,
    width: int = 832,
    height: int = 480,
    frame_count: int = 81,
    fps: int = 16,
    sampling_steps: int = 50,
    guidance_scale: float = 5.0,
    shift: float = 5.0,
    enable_caching: bool = False,
    cache_start_step: int = 10,
    cache_end_step: Optional[int] = 40,
    cache_interval: int = 3,
    seed: int = -1,
    negative_prompt: str = "",
    job_id: str = "",
    progress_callback: Optional[Callable[[GenerationProgress], None]] = None,
) -> GenerationResult:
    """
    Generate a video using distributed coordinator/worker architecture.

    For "14B" segments in the schedule, attempts to delegate to the remote
    worker. Falls back to local 1.3B if worker is unavailable.
    """
    start_time = time.time()
    progress = GenerationProgress(total_steps=sampling_steps, start_time=start_time)

    try:
        device = model_manager.device

        # Build schedule
        if model_type == "hybrid":
            if schedule is None:
                steps_14B = int(sampling_steps * 0.3)
                steps_1_3B = sampling_steps - steps_14B
                parsed_schedule = [("14B", steps_14B), ("1.3B", steps_1_3B)]
            else:
                parsed_schedule = parse_schedule(schedule)
        elif model_type == "baseline_1.3B":
            parsed_schedule = [("1.3B", sampling_steps)]
        elif model_type == "baseline_14B":
            # In distributed mode, baseline_14B means try all on worker
            parsed_schedule = [("14B", sampling_steps)]
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # If schedule includes 14B segments, ensure worker is available
        has_14B_segments = any(m == "14B" for m, _ in parsed_schedule)
        if has_14B_segments and spot_manager and spot_manager.is_configured:
            spot_manager.cancel_idle_timer()
            worker_url = spot_manager.ensure_worker_available()
            if worker_url and worker_url != worker_client.worker_url:
                worker_client.worker_url = worker_url.rstrip("/")
                worker_status.reset()
                logger.info(f"Spot worker ready at {worker_url}")

        logger.info(f"Distributed generation:")
        logger.info(f"  Schedule: {get_schedule_summary(parsed_schedule)}")
        logger.info(f"  Worker: {worker_client.worker_url}")

        # Get the local 1.3B model (always available)
        local_model = model_manager.get_model("1.3B")

        # Setup
        n_prompt = negative_prompt if negative_prompt else DEFAULT_NEGATIVE_PROMPT
        actual_seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=device)
        seed_g.manual_seed(actual_seed)

        F = frame_count
        vae_stride = local_model.vae_stride
        patch_size = local_model.patch_size
        target_shape = (
            local_model.vae.model.z_dim,
            (F - 1) // vae_stride[0] + 1,
            height // vae_stride[1],
            width // vae_stride[2],
        )
        seq_len = math.ceil(
            (target_shape[2] * target_shape[3])
            / (patch_size[1] * patch_size[2])
            * target_shape[1]
        )

        # Encode text (on coordinator)
        logger.info("Encoding text prompt (coordinator)...")
        local_model.text_encoder.model.to(device)
        context = local_model.text_encoder([prompt], device)
        context_null = local_model.text_encoder([n_prompt], device)
        local_model.text_encoder.model.cpu()
        torch.cuda.empty_cache()

        # Initialize noise and scheduler
        noise = torch.randn(
            *target_shape, dtype=torch.float32, device=device, generator=seed_g
        )
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=local_model.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        scheduler.set_timesteps(sampling_steps, device=device, shift=shift)
        timesteps = scheduler.timesteps
        latents = noise

        # Caching state
        cached_noise_pred = None
        total_cache_hits = 0
        total_fresh_computes = 0
        segments_on_worker = 0
        segments_fallback = 0

        # Progress helper
        def update_progress(step, model_name):
            progress.current_step = step
            progress.model_in_use = model_name
            progress.cache_hits = total_cache_hits
            progress.fresh_computes = total_fresh_computes
            if progress_callback:
                progress_callback(progress)

        # Process each segment
        step_idx = 0
        checkpoint_dir = DISTRIBUTED.checkpoint_dir

        for segment_idx, (model_name, num_steps) in enumerate(parsed_schedule):
            segment_timesteps = timesteps[step_idx : step_idx + num_steps]

            if model_name == "14B" and worker_status.should_try_worker:
                # Try to delegate to worker via inline HTTP transfer
                success = False

                # Health check
                if worker_client.health_check():
                    import io as _io

                    # Serialize checkpoint to bytes
                    sampling_params = {
                        "guidance_scale": guidance_scale,
                        "target_shape": target_shape,
                        "num_steps": num_steps,
                        "seq_len": seq_len,
                        "shift": shift,
                    }
                    checkpoint_data = {
                        "magic": "wan21_distributed_checkpoint",
                        "version": 1,
                        "job_id": job_id,
                        "segment_idx": segment_idx,
                        "global_step": step_idx,
                        "latents": latents.cpu(),
                        "context": [c.cpu() for c in context],
                        "context_null": [c.cpu() for c in context_null],
                        "scheduler_state": serialize_scheduler_state(scheduler),
                        "sampling_params": sampling_params,
                        "seed": actual_seed,
                    }
                    buf = _io.BytesIO()
                    torch.save(checkpoint_data, buf)
                    checkpoint_bytes = buf.getvalue()
                    logger.info(f"Checkpoint serialized: {len(checkpoint_bytes)/1e6:.1f}MB")

                    try:
                        # Send checkpoint to worker, get result back inline
                        timeout = (
                            num_steps * DISTRIBUTED.expected_step_time_14b
                            + DISTRIBUTED.worker_timeout_margin
                            + 30  # extra for transfer time
                        )
                        update_progress(step_idx, "14B (remote)")

                        result_bytes = worker_client.run_segment_inline(
                            checkpoint_bytes, timeout=timeout
                        )

                        # Deserialize result
                        seg_result = torch.load(
                            _io.BytesIO(result_bytes), map_location="cpu", weights_only=False
                        )
                        latents = seg_result["latents"].to(device)
                        restore_scheduler_state(
                            scheduler, seg_result["scheduler_state"], device
                        )
                        total_cache_hits += seg_result.get("cache_hits", 0)
                        total_fresh_computes += seg_result.get("fresh_computes", 0)
                        worker_status.record_success()
                        segments_on_worker += 1
                        success = True
                        logger.info(
                            f"Segment {segment_idx} completed on worker "
                            f"({seg_result.get('segment_time', 0):.1f}s)"
                        )
                    except Exception as e:
                        worker_status.record_failure(
                            f"Segment {segment_idx}: {str(e)[:100]}"
                        )
                else:
                    worker_status.record_failure("Health check failed")

                if not success:
                    # Fallback: run on local 1.3B
                    logger.info(
                        f"Segment {segment_idx}: falling back to local 1.3B "
                        f"({num_steps} steps)"
                    )
                    segments_fallback += 1
                    latents, cached_noise_pred, hits, computes = _run_segment_locally(
                        model=local_model,
                        latents=latents,
                        scheduler=scheduler,
                        segment_timesteps=segment_timesteps,
                        context=context,
                        context_null=context_null,
                        seq_len=seq_len,
                        guidance_scale=guidance_scale,
                        device=device,
                        enable_caching=enable_caching,
                        cached_noise_pred=cached_noise_pred,
                        cache_start_step=cache_start_step,
                        cache_end_step=cache_end_step,
                        cache_interval=cache_interval,
                        global_step_offset=step_idx,
                        progress_callback=update_progress,
                        model_name="1.3B (fallback)",
                    )
                    total_cache_hits += hits
                    total_fresh_computes += computes

            else:
                # Run locally (1.3B segment or worker unavailable)
                model_label = model_name
                if model_name == "14B" and not worker_status.should_try_worker:
                    model_label = "1.3B (backoff)"
                    segments_fallback += 1

                latents, cached_noise_pred, hits, computes = _run_segment_locally(
                    model=local_model,
                    latents=latents,
                    scheduler=scheduler,
                    segment_timesteps=segment_timesteps,
                    context=context,
                    context_null=context_null,
                    seq_len=seq_len,
                    guidance_scale=guidance_scale,
                    device=device,
                    enable_caching=enable_caching,
                    cached_noise_pred=cached_noise_pred,
                    cache_start_step=cache_start_step,
                    cache_end_step=cache_end_step,
                    cache_interval=cache_interval,
                    global_step_offset=step_idx,
                    progress_callback=update_progress,
                    model_name=model_label,
                )
                total_cache_hits += hits
                total_fresh_computes += computes

            step_idx += num_steps

        # Final progress
        progress.current_step = sampling_steps
        progress.cache_hits = total_cache_hits
        progress.fresh_computes = total_fresh_computes
        if progress_callback:
            progress_callback(progress)

        # Decode latents (on coordinator)
        logger.info("Decoding latents (coordinator)...")
        with torch.no_grad():
            videos = local_model.vae.decode([latents])

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
            value_range=(-1, 1),
        )

        generation_time = time.time() - start_time

        # Clean up checkpoints
        try:
            cleanup_job_checkpoints(checkpoint_dir, job_id)
        except Exception as e:
            logger.warning(f"Failed to cleanup checkpoints: {e}")

        # Start idle timer for spot shutdown
        if spot_manager and spot_manager.is_configured:
            spot_manager.start_idle_timer()

        # Build stats
        cache_stats = None
        if enable_caching:
            cache_stats = {
                "cache_hits": total_cache_hits,
                "fresh_computes": total_fresh_computes,
                "cache_hit_rate": round(total_cache_hits / sampling_steps, 3) if sampling_steps > 0 else 0,
            }

        distributed_stats = {
            "segments_on_worker": segments_on_worker,
            "segments_fallback": segments_fallback,
            "worker_stats": worker_status.get_stats(),
        }

        logger.info(f"Generation complete: {generation_time:.1f}s")
        logger.info(f"  Worker segments: {segments_on_worker}, Fallback: {segments_fallback}")

        del noise, latents, scheduler
        torch.cuda.empty_cache()

        return GenerationResult(
            success=True,
            video_path=output_path,
            generation_time=generation_time,
            cache_statistics={**(cache_stats or {}), **distributed_stats},
            seed_used=actual_seed,
        )

    except Exception as e:
        logger.error(f"Distributed generation failed: {e}")
        import traceback
        traceback.print_exc()
        return GenerationResult(
            success=False,
            error=str(e),
            generation_time=time.time() - start_time,
        )
