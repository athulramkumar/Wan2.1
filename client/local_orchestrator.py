#!/usr/bin/env python3
"""
Local orchestrator for hybrid compute experiments.

Runs on your MacBook. Controls the diffusion loop by sending segments
to remote GPUs via HTTP. The MacBook holds the latent state and relays
between GPUs — no GPU-to-GPU networking needed.

Flow:
  1. 2xA40 encodes text (T5) → sends embeddings to MacBook
  2. MacBook initializes noise + scheduler locally (CPU)
  3. For 14B steps: MacBook sends latent → H100 runs steps → returns latent
  4. For 1.3B steps: MacBook sends latent → 2xA40 runs steps → returns latent
  5. 2xA40 decodes latents (VAE) → sends MP4 to MacBook

Usage:
    python client/local_orchestrator.py
"""

import io
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
import torch
from torch import Tensor

# URLs (via SSH tunnels on localhost)
H100_WORKER = os.environ.get("H100_URL", "http://localhost:18889")
A40_COORD = os.environ.get("A40_URL", "http://localhost:18890")

SEED = 42
PROMPT = "The Merced river is overflowing, birds flying in the sky, camera is zooming out to reveal an American Buffalo bathing in the river"
NEG_PROMPT = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, misaligned limbs, extra limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"

OUTDIR = Path("experiments/compute_comparison")


def serialize_scheduler_state(scheduler) -> dict:
    """Extract scheduler state for serialization."""
    state = {
        "model_outputs": [x.cpu() if isinstance(x, Tensor) else x for x in scheduler.model_outputs],
        "lower_order_nums": scheduler.lower_order_nums,
        "_step_index": scheduler._step_index,
        "_begin_index": scheduler._begin_index,
        "sigmas": scheduler.sigmas.cpu() if isinstance(scheduler.sigmas, Tensor) else scheduler.sigmas,
        "timesteps": scheduler.timesteps.cpu(),
        "num_inference_steps": scheduler.num_inference_steps,
    }
    if hasattr(scheduler, "timestep_list"):
        state["timestep_list"] = list(scheduler.timestep_list)
    if hasattr(scheduler, "last_sample"):
        state["last_sample"] = scheduler.last_sample.cpu() if isinstance(scheduler.last_sample, Tensor) else scheduler.last_sample
    if hasattr(scheduler, "this_order"):
        state["this_order"] = scheduler.this_order
    return state


def restore_scheduler_state(scheduler, state, device):
    """Restore scheduler from serialized state."""
    scheduler.model_outputs = [x.to(device) if isinstance(x, Tensor) else x for x in state["model_outputs"]]
    scheduler.lower_order_nums = state["lower_order_nums"]
    scheduler._step_index = state["_step_index"]
    scheduler._begin_index = state["_begin_index"]
    scheduler.sigmas = state["sigmas"]
    scheduler.timesteps = state["timesteps"].to(device)
    scheduler.num_inference_steps = state["num_inference_steps"]
    if "timestep_list" in state:
        scheduler.timestep_list = state["timestep_list"]
    if "last_sample" in state:
        scheduler.last_sample = state["last_sample"].to(device) if isinstance(state["last_sample"], Tensor) else state["last_sample"]
    if "this_order" in state:
        scheduler.this_order = state["this_order"]


def run_segment_remote(url: str, checkpoint_data: dict, timeout: float = 600) -> dict:
    """Send checkpoint to a remote GPU, get result back."""
    buf = io.BytesIO()
    torch.save(checkpoint_data, buf)
    payload = buf.getvalue()

    endpoint = "/run-segment-inline" if "/1888" in url and "9" in url[-1:] else "/run-steps-inline"
    print(f"    Sending {len(payload)/1e6:.1f}MB to {url}{endpoint}...", end="", flush=True)
    t0 = time.time()

    resp = requests.post(f"{url}{endpoint}", data=payload,
                         headers={"Content-Type": "application/octet-stream"},
                         timeout=timeout)
    resp.raise_for_status()

    result = torch.load(io.BytesIO(resp.content), map_location="cpu", weights_only=False)
    elapsed = time.time() - t0
    seg_time = result.get("segment_time", 0)
    print(f" done ({elapsed:.1f}s total, {seg_time:.1f}s compute)")
    return result


def encode_text(url: str, prompt: str, neg_prompt: str) -> tuple:
    """Get T5 embeddings from a remote GPU."""
    print(f"  Encoding text on {url}...", end="", flush=True)
    t0 = time.time()
    resp = requests.post(f"{url}/encode-text",
                         json={"prompt": prompt, "negative_prompt": neg_prompt},
                         timeout=300)
    resp.raise_for_status()
    data = torch.load(io.BytesIO(resp.content), map_location="cpu", weights_only=False)
    print(f" done ({time.time()-t0:.1f}s)")
    return data["context"], data["context_null"]


def decode_latents(url: str, latents: Tensor, fps: int = 16) -> bytes:
    """Send latents to remote GPU for VAE decode, get MP4 back."""
    print(f"  Decoding latents on {url}...", end="", flush=True)
    t0 = time.time()
    buf = io.BytesIO()
    torch.save({"latents": latents.cpu(), "fps": fps}, buf)
    resp = requests.post(f"{url}/decode-latents", data=buf.getvalue(),
                         headers={"Content-Type": "application/octet-stream"},
                         timeout=600)
    resp.raise_for_status()
    print(f" done ({time.time()-t0:.1f}s, {len(resp.content)/1e6:.1f}MB video)")
    return resp.content


def run_experiment(
    name: str,
    schedule: list,  # [("14B", 15, H100_URL), ("1.3B", 35, A40_URL)]
    encode_url: str,
    decode_url: str,
    vid_file: str,
    steps: int = 50,
    seed: int = SEED,
):
    """Run a single experiment with the local orchestrator."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  Schedule: {[(m,s) for m,s,_ in schedule]}")
    print(f"{'='*60}")

    t_start = time.time()

    # Step 1: Encode text
    context, context_null = encode_text(encode_url, PROMPT, NEG_PROMPT)

    # Step 2: Initialize noise + scheduler on CPU
    # We need model config values. Use defaults for Wan2.1.
    target_shape = (16, 21, 60, 104)  # z_dim=16, 81 frames, 480x832
    # Actually z_dim depends on the model. Let me check...
    # Wan2.1 VAE has z_dim=16 for T2V models
    # Wait, looking at code: first_model.vae.model.z_dim
    # From configs: t2v_14B and t2v_1_3B both use z_dim=16? Let me use 16.
    # Actually from our earlier experiments: target_shape = (4, 21, 60, 104)
    # z_dim = 4 based on the checkpoint data we saw earlier
    target_shape = (4, 21, 60, 104)

    seq_len = math.ceil((target_shape[2] * target_shape[3]) / (1 * 1) * target_shape[1])
    # patch_size for 14B is (1,2,2), for 1.3B is (1,2,2)
    # seq_len = ceil((60*104)/(2*2) * 21) = ceil(6240/4 * 21) = ceil(1560 * 21) = 32760
    seq_len = math.ceil((target_shape[2] * target_shape[3]) / (2 * 2) * target_shape[1])

    torch.manual_seed(seed)
    noise = torch.randn(*target_shape, dtype=torch.float32)
    latents = noise

    # Create scheduler on CPU
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000, shift=1, use_dynamic_shifting=False
    )
    scheduler.set_timesteps(steps, device="cpu", shift=5.0)

    # Step 3: Run segments
    step_idx = 0
    total_compute = 0
    total_transfer = 0

    for model_name, num_steps, gpu_url in schedule:
        print(f"\n  Segment: {model_name} x{num_steps} steps on {gpu_url}")

        checkpoint = {
            "magic": "wan21_distributed_checkpoint",
            "version": 1,
            "job_id": vid_file,
            "segment_idx": step_idx,
            "global_step": step_idx,
            "latents": latents.cpu(),
            "context": [c.cpu() for c in context],
            "context_null": [c.cpu() for c in context_null],
            "scheduler_state": serialize_scheduler_state(scheduler),
            "sampling_params": {
                "guidance_scale": 5.0,
                "target_shape": target_shape,
                "num_steps": num_steps,
                "seq_len": seq_len,
                "shift": 5.0,
                "model_name": model_name,
            },
            "seed": seed,
        }

        t_transfer = time.time()
        result = run_segment_remote(gpu_url, checkpoint, timeout=600)
        transfer_time = time.time() - t_transfer
        compute_time = result.get("segment_time", 0)
        total_compute += compute_time
        total_transfer += (transfer_time - compute_time)

        latents = result["latents"]
        restore_scheduler_state(scheduler, result["scheduler_state"], "cpu")
        step_idx += num_steps

    # Step 4: VAE decode
    video_bytes = decode_latents(decode_url, latents, fps=16)

    total_time = time.time() - t_start

    # Save
    OUTDIR.mkdir(parents=True, exist_ok=True)
    vid_path = OUTDIR / vid_file
    with open(vid_path, "wb") as f:
        f.write(video_bytes)

    status = {
        "name": name,
        "generation_time": total_time,
        "compute_time": total_compute,
        "transfer_overhead": total_transfer,
        "schedule": [(m, s) for m, s, _ in schedule],
        "status": "completed",
        "video_path": str(vid_path),
    }
    with open(OUTDIR / f"{vid_file.replace('.mp4','')}_status.json", "w") as f:
        json.dump(status, f, indent=2)

    print(f"\n  DONE: {total_time:.1f}s total ({total_compute:.1f}s compute, {total_transfer:.1f}s transfer)")
    print(f"  Video: {vid_path} ({len(video_bytes)/1e6:.1f}MB)")
    return status


def main():
    print("="*60)
    print("  LOCAL ORCHESTRATOR - HYBRID COMPUTE EXPERIMENTS")
    print(f"  H100 (14B): {H100_WORKER}")
    print(f"  2xA40 (1.3B): {A40_COORD}")
    print("="*60)

    # Verify both GPUs are up
    for name, url in [("H100", H100_WORKER), ("2xA40", A40_COORD)]:
        try:
            r = requests.get(f"{url}/health", timeout=5).json()
            print(f"  {name}: {r.get('status', '?')}")
        except Exception as e:
            print(f"  {name}: UNREACHABLE ({e})")
            sys.exit(1)

    results = []

    # Exp 1: All 50 steps on H100 (14B) — baseline
    r = run_experiment(
        "14B Baseline (H100)",
        [("14B", 50, H100_WORKER)],
        encode_url=A40_COORD, decode_url=A40_COORD,
        vid_file="local_exp1_14B_baseline.mp4",
    )
    results.append(r)

    # Exp 2: Hybrid 30/70 (H100 14B + 2xA40 1.3B)
    r = run_experiment(
        "Hybrid 30/70 (H100+2xA40)",
        [("14B", 15, H100_WORKER), ("1.3B", 35, A40_COORD)],
        encode_url=A40_COORD, decode_url=A40_COORD,
        vid_file="local_exp2_hybrid_30_70.mp4",
    )
    results.append(r)

    # Exp 3: Hybrid 20/80 (H100 14B + 2xA40 1.3B)
    r = run_experiment(
        "Hybrid 20/80 (H100+2xA40)",
        [("14B", 10, H100_WORKER), ("1.3B", 40, A40_COORD)],
        encode_url=A40_COORD, decode_url=A40_COORD,
        vid_file="local_exp3_hybrid_20_80.mp4",
    )
    results.append(r)

    # Exp 4: 1.3B only on 2xA40
    r = run_experiment(
        "1.3B Only (2xA40)",
        [("1.3B", 50, A40_COORD)],
        encode_url=A40_COORD, decode_url=A40_COORD,
        vid_file="local_exp4_1.3B_only.mp4",
    )
    results.append(r)

    # Summary
    print(f"\n{'='*60}")
    print("  RESULTS")
    print(f"{'='*60}")
    print(f"  {'Experiment':<35} {'Total':>7} {'Compute':>8} {'Transfer':>9}")
    print(f"  {'-'*60}")
    for r in results:
        print(f"  {r['name']:<35} {r['generation_time']:>6.1f}s {r['compute_time']:>7.1f}s {r['transfer_overhead']:>8.1f}s")

    with open(OUTDIR / "local_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results: {OUTDIR}/local_results.json")


if __name__ == "__main__":
    main()
