#!/usr/bin/env python3
"""
Experiments for distributed hybrid inference.

Compares quality, latency, and cost across:
1. Baseline: All 14B on single GPU
2. Baseline: All 1.3B on single GPU
3. Hybrid (local): 14B + 1.3B on single GPU
4. Distributed hybrid: 14B on spot worker, 1.3B on coordinator

Run this from the client machine (or coordinator pod).
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import requests

# Test prompts spanning different difficulty levels
TEST_PROMPTS = [
    "A golden retriever playing in a field of sunflowers on a sunny day",
    "A spaceship launching from a futuristic city at sunset with dramatic clouds",
    "A close-up of rain drops falling on a still pond creating ripples, slow motion",
    "A timelapse of a flower blooming in a garden with butterflies",
]


def submit_and_wait(api_url, prompt, model, schedule=None, caching=False,
                    steps=50, poll_interval=5, timeout=600):
    """Submit a job and wait for completion."""
    payload = {
        "prompt": prompt,
        "model": model,
        "sampling_steps": steps,
        "frame_count": 81,
        "width": 832,
        "height": 480,
    }
    if schedule:
        payload["schedule"] = schedule
    if caching:
        payload["enable_caching"] = True
        payload["cache_start_step"] = 10
        payload["cache_end_step"] = 40
        payload["cache_interval"] = 3

    # Submit
    resp = requests.post(f"{api_url}/generate", json=payload, timeout=30)
    resp.raise_for_status()
    job_id = resp.json()["job_id"]

    # Poll
    start = time.time()
    while time.time() - start < timeout:
        status_resp = requests.get(f"{api_url}/status/{job_id}", timeout=30)
        status = status_resp.json()

        if status["status"] == "completed":
            return {
                "job_id": job_id,
                "status": "completed",
                "generation_time": status.get("generation_time", 0),
                "cache_statistics": status.get("cache_statistics", {}),
                "metadata": status.get("metadata", {}),
            }
        elif status["status"] == "failed":
            return {
                "job_id": job_id,
                "status": "failed",
                "error": status.get("error", "unknown"),
            }

        time.sleep(poll_interval)

    return {"job_id": job_id, "status": "timeout"}


def run_experiments(api_url, output_dir, prompts=None, steps=50):
    """Run the full experiment suite."""
    prompts = prompts or TEST_PROMPTS[:2]  # Default to 2 prompts
    os.makedirs(output_dir, exist_ok=True)

    configurations = [
        {"name": "baseline_1.3B", "model": "baseline_1.3B"},
        {"name": "hybrid_local", "model": "hybrid", "schedule": [["14B", 15], ["1.3B", 35]]},
        {"name": "distributed_hybrid", "model": "hybrid", "schedule": [["14B", 15], ["1.3B", 35]]},
    ]

    # Check if server is in distributed mode
    health = requests.get(f"{api_url}/health", timeout=10).json()
    is_distributed = health.get("distributed_mode", False)

    results = []

    for config in configurations:
        # Skip distributed config if not in distributed mode
        if config["name"] == "distributed_hybrid" and not is_distributed:
            print(f"Skipping {config['name']} (server not in distributed mode)")
            continue
        # Skip hybrid_local if in distributed mode (they'd be the same)
        if config["name"] == "hybrid_local" and is_distributed:
            print(f"Skipping {config['name']} (server is in distributed mode)")
            continue

        for i, prompt in enumerate(prompts):
            print(f"\n{'='*60}")
            print(f"Config: {config['name']} | Prompt {i+1}/{len(prompts)}")
            print(f"  {prompt[:60]}...")
            print(f"{'='*60}")

            result = submit_and_wait(
                api_url, prompt,
                model=config["model"],
                schedule=config.get("schedule"),
                steps=steps,
            )
            result["config"] = config["name"]
            result["prompt"] = prompt
            result["prompt_idx"] = i
            results.append(result)

            status = result["status"]
            if status == "completed":
                gen_time = result["generation_time"]
                distributed_stats = result.get("cache_statistics", {})
                worker_segs = distributed_stats.get("segments_on_worker", "N/A")
                fallback_segs = distributed_stats.get("segments_fallback", "N/A")
                print(f"  Time: {gen_time:.1f}s | Worker: {worker_segs} | Fallback: {fallback_segs}")
            else:
                print(f"  Status: {status}")
                if "error" in result:
                    print(f"  Error: {result['error']}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(output_dir, f"distributed_experiment_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for config_name in set(r["config"] for r in results):
        config_results = [r for r in results if r["config"] == config_name and r["status"] == "completed"]
        if config_results:
            avg_time = sum(r["generation_time"] for r in config_results) / len(config_results)
            print(f"  {config_name}: avg {avg_time:.1f}s ({len(config_results)} runs)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Distributed hybrid inference experiments")
    parser.add_argument("--server", required=True, help="API server URL")
    parser.add_argument("--output", default="experiments/distributed", help="Output directory")
    parser.add_argument("--steps", type=int, default=50, help="Sampling steps")
    parser.add_argument("--prompts", type=int, default=2, help="Number of test prompts")
    args = parser.parse_args()

    run_experiments(args.server, args.output, TEST_PROMPTS[:args.prompts], args.steps)


if __name__ == "__main__":
    main()
