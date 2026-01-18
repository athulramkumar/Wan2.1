#!/usr/bin/env python3
"""
Wan2.1 Video Generation Client

A simple script to generate videos using the Wan2.1 API.
Downloads the video and saves metadata on completion.

Usage:
    python generate_video.py "A cat playing piano"
    python generate_video.py "A sunset over mountains" --model hybrid --seed 42
    python generate_video.py "Dancing robots" --model baseline_1.3B --frames 33

Requirements:
    pip install requests
"""

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime

try:
    import requests
except ImportError:
    print("Error: 'requests' package not found.")
    print("Install it with: pip install requests")
    sys.exit(1)


# =============================================================================
# Configuration - CHANGE THIS TO YOUR SERVER URL
# =============================================================================
API_BASE_URL = "https://jvkiftqnv4abuo-8888.proxy.runpod.net"
# =============================================================================


def sanitize_filename(text: str, max_length: int = 50) -> str:
    """Convert text to a safe filename."""
    # Remove special characters, keep alphanumeric and spaces
    safe = re.sub(r'[^\w\s-]', '', text)
    # Replace spaces with underscores
    safe = re.sub(r'\s+', '_', safe)
    # Truncate
    return safe[:max_length]


def submit_job(server_url: str, prompt: str, **kwargs) -> dict:
    """Submit a generation job to the API."""
    payload = {"prompt": prompt, **kwargs}
    
    response = requests.post(
        f"{server_url}/generate",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=30
    )
    response.raise_for_status()
    return response.json()


def get_status(server_url: str, job_id: str) -> dict:
    """Get job status from the API."""
    response = requests.get(
        f"{server_url}/status/{job_id}",
        timeout=30
    )
    response.raise_for_status()
    return response.json()


def download_video(server_url: str, job_id: str, output_path: str) -> bool:
    """Download the generated video."""
    response = requests.get(
        f"{server_url}/video/{job_id}",
        timeout=300,  # 5 min timeout for large videos
        stream=True
    )
    response.raise_for_status()
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    return True


def print_progress_bar(progress: int, width: int = 40):
    """Print a progress bar."""
    filled = int(width * progress / 100)
    bar = '█' * filled + '░' * (width - filled)
    print(f"\r  [{bar}] {progress}%", end='', flush=True)


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {mins}m {secs}s"


def open_video(video_path: str) -> bool:
    """Open video with the default system player."""
    try:
        system = platform.system()
        
        if system == "Darwin":  # macOS
            subprocess.run(["open", video_path], check=True)
        elif system == "Windows":
            os.startfile(video_path)
        else:  # Linux
            subprocess.run(["xdg-open", video_path], check=True)
        
        return True
    except Exception as e:
        print(f"  Could not open video: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate videos using Wan2.1 API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_video.py "A cat playing piano"
  python generate_video.py "A sunset" --model hybrid --caching
  python generate_video.py "Robots dancing" --model baseline_1.3B --frames 33 --steps 30
        """
    )
    
    # Required
    parser.add_argument("prompt", help="Text prompt for video generation")
    
    # Model settings
    parser.add_argument(
        "--model", "-m",
        choices=["baseline_14B", "baseline_1.3B", "hybrid"],
        default="hybrid",
        help="Model to use (default: hybrid)"
    )
    parser.add_argument(
        "--schedule",
        type=str,
        default=None,
        help='Hybrid schedule as JSON, e.g., \'[["14B", 15], ["1.3B", 35]]\''
    )
    
    # Video settings
    parser.add_argument("--frames", type=int, default=81, help="Frame count (default: 81)")
    parser.add_argument("--fps", type=int, default=16, help="Output FPS (default: 16)")
    parser.add_argument("--steps", type=int, default=50, help="Sampling steps (default: 50)")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed (-1 for random)")
    
    # Caching
    parser.add_argument("--caching", action="store_true", help="Enable caching for speedup")
    parser.add_argument("--cache-start", type=int, default=10, help="Cache start step")
    parser.add_argument("--cache-end", type=int, default=40, help="Cache end step")
    parser.add_argument("--cache-interval", type=int, default=3, help="Cache interval")
    
    # Output
    parser.add_argument(
        "--output-dir", "-o",
        default=".",
        help="Output directory (default: current directory)"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't open video after generation (default: opens video)"
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=5,
        help="Status poll interval in seconds (default: 5)"
    )
    
    # Server
    parser.add_argument(
        "--server",
        default=API_BASE_URL,
        help=f"API server URL (default: {API_BASE_URL})"
    )
    
    args = parser.parse_args()
    
    # Use server URL from args
    server_url = args.server
    
    # Build request payload
    payload = {
        "prompt": args.prompt,
        "model": args.model,
        "frame_count": args.frames,
        "fps": args.fps,
        "sampling_steps": args.steps,
        "seed": args.seed,
        "enable_caching": args.caching,
    }
    
    if args.caching:
        payload["cache_start_step"] = args.cache_start
        payload["cache_end_step"] = args.cache_end
        payload["cache_interval"] = args.cache_interval
    
    if args.schedule:
        payload["schedule"] = json.loads(args.schedule)
    
    # Print header
    print()
    print("=" * 60)
    print("  WAN2.1 VIDEO GENERATION")
    print("=" * 60)
    print(f"  Server:  {server_url}")
    print(f"  Prompt:  {args.prompt[:50]}{'...' if len(args.prompt) > 50 else ''}")
    print(f"  Model:   {args.model}")
    print(f"  Frames:  {args.frames}")
    print(f"  Steps:   {args.steps}")
    print(f"  Caching: {'Yes' if args.caching else 'No'}")
    print("=" * 60)
    print()
    
    # Submit job
    print("📤 Submitting job...")
    try:
        result = submit_job(server_url, **payload)
        job_id = result["job_id"]
        print(f"✓ Job submitted: {job_id}")
    except requests.exceptions.RequestException as e:
        print(f"✗ Failed to submit job: {e}")
        sys.exit(1)
    
    # Poll for status
    print()
    print("⏳ Generating video...")
    start_time = time.time()
    last_progress = -1
    
    try:
        while True:
            status = get_status(server_url, job_id)
            progress = status.get("progress", 0)
            state = status.get("status", "unknown")
            
            if progress != last_progress:
                print_progress_bar(progress)
                last_progress = progress
            
            if state == "completed":
                print()  # New line after progress bar
                break
            elif state == "failed":
                print()
                print(f"\n✗ Generation failed: {status.get('error', 'Unknown error')}")
                sys.exit(1)
            
            # Show additional info
            model_in_use = status.get("model_in_use", "")
            elapsed = time.time() - start_time
            if model_in_use:
                print(f" | {model_in_use} | {format_duration(elapsed)}", end='', flush=True)
            else:
                print(f" | {format_duration(elapsed)}", end='', flush=True)
            
            time.sleep(args.poll_interval)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted. Job may still be running on server.")
        print(f"   Check status: curl {server_url}/status/{job_id}")
        sys.exit(1)
    
    # Get final status
    final_status = get_status(server_url, job_id)
    generation_time = final_status.get("generation_time", 0)
    metadata = final_status.get("metadata", {})
    cache_stats = final_status.get("cache_statistics")
    
    print(f"✓ Generation complete in {format_duration(generation_time)}")
    
    if cache_stats:
        hit_rate = cache_stats.get("cache_hit_rate", 0) * 100
        print(f"  Cache hit rate: {hit_rate:.1f}%")
    
    # Create output filenames
    safe_prompt = sanitize_filename(args.prompt)
    base_name = f"{safe_prompt}_{job_id}"
    
    os.makedirs(args.output_dir, exist_ok=True)
    video_path = os.path.join(args.output_dir, f"{base_name}.mp4")
    json_path = os.path.join(args.output_dir, f"{base_name}.json")
    
    # Download video
    print()
    print("📥 Downloading video...")
    try:
        download_video(server_url, job_id, video_path)
        file_size = os.path.getsize(video_path) / (1024 * 1024)
        print(f"✓ Video saved: {video_path} ({file_size:.1f} MB)")
    except Exception as e:
        print(f"✗ Failed to download video: {e}")
        sys.exit(1)
    
    # Save metadata
    metadata_to_save = {
        "job_id": job_id,
        "prompt": args.prompt,
        "model": args.model,
        "frame_count": args.frames,
        "fps": args.fps,
        "sampling_steps": args.steps,
        "seed": metadata.get("seed", args.seed),
        "caching_enabled": args.caching,
        "generation_time_seconds": generation_time,
        "cache_statistics": cache_stats,
        "server_url": server_url,
        "generated_at": datetime.now().isoformat(),
        "video_file": os.path.basename(video_path),
    }
    
    if args.schedule:
        metadata_to_save["schedule"] = json.loads(args.schedule)
    
    with open(json_path, 'w') as f:
        json.dump(metadata_to_save, f, indent=2)
    print(f"✓ Metadata saved: {json_path}")
    
    # Final summary
    print()
    print("=" * 60)
    print("  ✅ DONE!")
    print("=" * 60)
    print(f"  Video:    {video_path}")
    print(f"  Metadata: {json_path}")
    print(f"  Time:     {format_duration(generation_time)} ({generation_time:.1f}s)")
    print("=" * 60)
    print()
    
    # Open video if not disabled
    if not args.no_display:
        print("🎬 Opening video...")
        open_video(video_path)


if __name__ == "__main__":
    main()

