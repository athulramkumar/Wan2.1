#!/usr/bin/env python3
"""
Wan2.1 Distributed Worker - 14B Model Server

Runs on spot instances (H100/A100/2xA40) and executes 14B model
segments on demand from the coordinator.

For single GPU:
    python run_worker.py

For multi-GPU (2xA40):
    torchrun --nproc_per_node=2 run_worker.py

Environment variables:
    WAN_WORKER_PORT: Worker port (default: 8889)
    RUNPOD_POD_ID: RunPod pod ID for public URL
"""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def get_gpu_count() -> int:
    """Detect available GPUs."""
    try:
        import torch
        return torch.cuda.device_count()
    except ImportError:
        return 0


def get_public_url(port: int) -> str | None:
    """Generate RunPod public URL if running on RunPod."""
    pod_id = os.environ.get("RUNPOD_POD_ID")
    if pod_id:
        return f"https://{pod_id}-{port}.proxy.runpod.net"
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Wan2.1 Distributed Worker (14B model)",
    )
    parser.add_argument(
        "--host", type=str,
        default=os.environ.get("WAN_WORKER_HOST", "0.0.0.0"),
    )
    parser.add_argument(
        "--port", type=int,
        default=int(os.environ.get("WAN_WORKER_PORT", "8889")),
    )
    parser.add_argument(
        "--auto-multi-gpu", action="store_true",
        help="Auto-detect multiple GPUs and relaunch with torchrun",
    )
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # Auto-relaunch with torchrun for multi-GPU
    if args.auto_multi_gpu and world_size == 1:
        gpu_count = get_gpu_count()
        if gpu_count > 1:
            print(f"Detected {gpu_count} GPUs, relaunching with torchrun...")
            cmd = [
                sys.executable, "-m", "torch.distributed.run",
                f"--nproc_per_node={gpu_count}",
                __file__,
                "--host", args.host,
                "--port", str(args.port),
            ]
            os.execvp(cmd[0], cmd)

    # Only rank 0 starts the HTTP server
    if rank == 0:
        public_url = get_public_url(args.port)

        print()
        print("=" * 60)
        print("  WAN2.1 DISTRIBUTED WORKER (14B)")
        print("=" * 60)
        print(f"  Host: {args.host}")
        print(f"  Port: {args.port}")
        print(f"  Rank: {rank} / World Size: {world_size}")
        print(f"  GPUs: {world_size}")

        if public_url:
            print()
            print(f"  Public URL: {public_url}")
            print(f"  Health:     {public_url}/health")

        print()
        print("=" * 60)
        print()

        import uvicorn
        uvicorn.run(
            "api.distributed.worker_server:app",
            host=args.host,
            port=args.port,
            workers=1,
            log_level="info",
        )
    else:
        # Non-rank-0 processes: wait for work from rank 0
        # In the current design, the worker_server thread on rank 0
        # handles orchestration. Non-rank processes participate via
        # torch.distributed calls during model.forward().
        # For now, they just need to stay alive and have the model loaded.
        print(f"Worker rank {rank}: Loading model and waiting for distributed calls...")
        from api.model_manager import initialize_models
        import torch.distributed as dist

        model_manager = initialize_models(
            device_id=rank,
            mode="worker",
            world_size=world_size,
            rank=rank,
        )

        # Keep alive — rank 0's FastAPI calls will trigger distributed
        # operations via the shared model's forward pass
        print(f"Worker rank {rank}: Model loaded, waiting for distributed operations...")
        try:
            while True:
                import time
                time.sleep(60)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
