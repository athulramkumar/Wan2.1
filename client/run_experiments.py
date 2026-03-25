#!/usr/bin/env python3
"""
Wan2.1 Hybrid Model Experiment Runner

Runs a series of experiments comparing baseline_14B vs hybrid model
with different configurations. Generates HTML report with latency
and quality (CLIP similarity) metrics.

Usage:
    python run_experiments.py
"""

import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
import numpy as np

# For CLIP similarity
try:
    import torch
    import cv2
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: CLIP not available. Quality metrics will be skipped.")
    print("Install with: pip install transformers torch opencv-python")

# =============================================================================
# Configuration
# =============================================================================

# Server URL (same as generate_video.py)
API_BASE_URL = "https://t6npgjjo8y0yf0-8888.proxy.runpod.net"

# Experiment prompt and seed
EXPERIMENT_PROMPT = "The Merced river is overflowing, birds flying in the sky, camera is zooming out to reveal an American Buffalo bathing in the river"
EXPERIMENT_SEED = 42

# Output directory
EXPERIMENT_DIR = Path(__file__).parent.parent / "experiments"

# =============================================================================
# Experiment Definitions
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    model: str  # baseline_14B, baseline_1.3B, hybrid
    steps: int
    frames: int
    caching: bool
    is_baseline: bool = False
    
    def get_filename_base(self) -> str:
        cache_str = "cache" if self.caching else "nocache"
        return f"{self.model}_{self.steps}steps_{self.frames}frames_{cache_str}"


@dataclass
class ExperimentResult:
    """Results from a single experiment."""
    config: dict
    job_id: str
    generation_time: float  # seconds
    cache_hit_rate: Optional[float]
    video_path: str
    clip_similarity: Optional[float] = None
    speedup_vs_baseline: Optional[float] = None
    completed_at: str = ""
    
    def to_dict(self) -> dict:
        return {
            "config": self.config,
            "job_id": self.job_id,
            "generation_time": self.generation_time,
            "cache_hit_rate": self.cache_hit_rate,
            "video_path": self.video_path,
            "clip_similarity": self.clip_similarity,
            "speedup_vs_baseline": self.speedup_vs_baseline,
            "completed_at": self.completed_at,
        }


# Define all experiments
EXPERIMENTS = [
    ExperimentConfig(
        name="Golden Baseline (14B, 50 steps)",
        model="baseline_14B",
        steps=50,
        frames=81,
        caching=False,
        is_baseline=True,
    ),
    ExperimentConfig(
        name="Hybrid 30 steps (no cache)",
        model="hybrid",
        steps=30,
        frames=81,
        caching=False,
    ),
    ExperimentConfig(
        name="Hybrid 30 steps (with cache)",
        model="hybrid",
        steps=30,
        frames=81,
        caching=True,
    ),
    ExperimentConfig(
        name="Hybrid 50 steps (no cache)",
        model="hybrid",
        steps=50,
        frames=81,
        caching=False,
    ),
    ExperimentConfig(
        name="Hybrid 50 steps (with cache)",
        model="hybrid",
        steps=50,
        frames=81,
        caching=True,
    ),
]


# =============================================================================
# API Client Functions
# =============================================================================

def submit_job(prompt: str, model: str, steps: int, frames: int, 
               caching: bool, seed: int) -> dict:
    """Submit a generation job to the API."""
    payload = {
        "prompt": prompt,
        "model": model,
        "frame_count": frames,
        "fps": 16,
        "sampling_steps": steps,
        "seed": seed,
        "enable_caching": caching,
    }
    
    if caching:
        payload["cache_start_step"] = 10
        payload["cache_end_step"] = min(40, steps - 5)
        payload["cache_interval"] = 3
    
    response = requests.post(
        f"{API_BASE_URL}/generate",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=30
    )
    response.raise_for_status()
    return response.json()


def get_status(job_id: str) -> dict:
    """Get job status from the API."""
    response = requests.get(f"{API_BASE_URL}/status/{job_id}", timeout=30)
    response.raise_for_status()
    return response.json()


def download_video(job_id: str, output_path: str) -> bool:
    """Download the generated video."""
    response = requests.get(
        f"{API_BASE_URL}/video/{job_id}",
        timeout=300,
        stream=True
    )
    response.raise_for_status()
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return True


def wait_for_completion(job_id: str, poll_interval: int = 5) -> dict:
    """Wait for job to complete and return final status."""
    print(f"  Waiting for job {job_id}...")
    
    while True:
        status = get_status(job_id)
        state = status.get("status", "unknown")
        progress = status.get("progress", 0)
        model_in_use = status.get("model_in_use", "")
        
        # Progress display
        bar_width = 30
        filled = int(bar_width * progress / 100)
        bar = '█' * filled + '░' * (bar_width - filled)
        model_str = f" [{model_in_use}]" if model_in_use else ""
        print(f"\r  [{bar}] {progress}%{model_str}    ", end='', flush=True)
        
        if state == "completed":
            print()  # New line
            return status
        elif state == "failed":
            print()
            raise RuntimeError(f"Job failed: {status.get('error', 'Unknown error')}")
        
        time.sleep(poll_interval)


# =============================================================================
# CLIP Similarity Functions
# =============================================================================

class CLIPEvaluator:
    """Compute CLIP-based similarity between videos."""
    
    def __init__(self):
        if not CLIP_AVAILABLE:
            raise RuntimeError("CLIP dependencies not available")
        
        print("Loading CLIP model...")
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.model.to(self.device)
        self.model.eval()
        print(f"CLIP loaded on {self.device}")
    
    def extract_frames(self, video_path: str, num_frames: int = 16) -> list:
        """Extract evenly spaced frames from video."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames evenly
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        return frames
    
    def get_video_embedding(self, video_path: str) -> np.ndarray:
        """Get average CLIP embedding for video frames."""
        frames = self.extract_frames(video_path)
        
        if not frames:
            raise ValueError(f"Could not extract frames from {video_path}")
        
        # Process frames in batches
        embeddings = []
        batch_size = 4
        
        with torch.no_grad():
            for i in range(0, len(frames), batch_size):
                batch = frames[i:i + batch_size]
                inputs = self.processor(images=batch, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                features = self.model.get_image_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)
                embeddings.append(features.cpu().numpy())
        
        # Average all frame embeddings
        all_embeddings = np.concatenate(embeddings, axis=0)
        avg_embedding = all_embeddings.mean(axis=0)
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
        
        return avg_embedding
    
    def compute_similarity(self, video_path1: str, video_path2: str) -> float:
        """Compute cosine similarity between two videos."""
        emb1 = self.get_video_embedding(video_path1)
        emb2 = self.get_video_embedding(video_path2)
        
        similarity = np.dot(emb1, emb2)
        return float(similarity)


# =============================================================================
# HTML Report Generator
# =============================================================================

def generate_html_report(results: list[ExperimentResult], output_path: str):
    """Generate an HTML report with experiment results."""
    
    baseline_time = None
    for r in results:
        if r.config.get("is_baseline"):
            baseline_time = r.generation_time
            break
    
    # Build results table rows
    table_rows = ""
    for r in results:
        speedup = f"{r.speedup_vs_baseline:.2f}x" if r.speedup_vs_baseline else "—"
        cache_rate = f"{r.cache_hit_rate*100:.1f}%" if r.cache_hit_rate else "—"
        clip_sim = f"{r.clip_similarity:.4f}" if r.clip_similarity else "—"
        is_baseline = "✓" if r.config.get("is_baseline") else ""
        
        # Color code speedup
        speedup_class = ""
        if r.speedup_vs_baseline:
            if r.speedup_vs_baseline >= 2:
                speedup_class = "speedup-great"
            elif r.speedup_vs_baseline >= 1.5:
                speedup_class = "speedup-good"
        
        # Color code similarity
        sim_class = ""
        if r.clip_similarity:
            if r.clip_similarity >= 0.95:
                sim_class = "sim-great"
            elif r.clip_similarity >= 0.90:
                sim_class = "sim-good"
            elif r.clip_similarity < 0.85:
                sim_class = "sim-poor"
        
        video_name = os.path.basename(r.video_path)
        
        table_rows += f"""
        <tr>
            <td>{r.config.get('name', '')}</td>
            <td>{r.config.get('model', '')}</td>
            <td>{r.config.get('steps', '')}</td>
            <td>{r.config.get('caching', False) and '✓' or '—'}</td>
            <td><strong>{r.generation_time:.1f}s</strong></td>
            <td class="{speedup_class}">{speedup}</td>
            <td>{cache_rate}</td>
            <td class="{sim_class}">{clip_sim}</td>
            <td>{is_baseline}</td>
            <td><a href="{video_name}" target="_blank">🎬 View</a></td>
        </tr>
        """
    
    # Build video previews
    video_previews = ""
    for r in results:
        video_name = os.path.basename(r.video_path)
        label = r.config.get('name', video_name)
        video_previews += f"""
        <div class="video-card">
            <h4>{label}</h4>
            <video controls width="400">
                <source src="{video_name}" type="video/mp4">
            </video>
            <p>Time: {r.generation_time:.1f}s</p>
        </div>
        """
    
    # Build latency chart data
    chart_labels = [r.config.get('name', '')[:20] for r in results]
    chart_times = [r.generation_time for r in results]
    chart_colors = ['#4CAF50' if r.config.get('is_baseline') else '#2196F3' for r in results]
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Wan2.1 Experiment Results</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --text-primary: #c9d1d9;
            --text-secondary: #8b949e;
            --accent-green: #3fb950;
            --accent-blue: #58a6ff;
            --accent-yellow: #d29922;
            --accent-red: #f85149;
            --border-color: #30363d;
        }}
        
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        
        body {{
            font-family: 'SF Mono', 'Fira Code', 'JetBrains Mono', monospace;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 2rem;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        h1 {{
            font-size: 2rem;
            margin-bottom: 0.5rem;
            color: var(--accent-blue);
        }}
        
        h2 {{
            font-size: 1.3rem;
            margin: 2rem 0 1rem;
            color: var(--accent-green);
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 0.5rem;
        }}
        
        .subtitle {{
            color: var(--text-secondary);
            margin-bottom: 2rem;
        }}
        
        .prompt-box {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 2rem;
        }}
        
        .prompt-box strong {{
            color: var(--accent-yellow);
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--bg-secondary);
            border-radius: 8px;
            overflow: hidden;
        }}
        
        th, td {{
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        
        th {{
            background: var(--bg-tertiary);
            color: var(--text-secondary);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.5px;
        }}
        
        tr:hover {{
            background: var(--bg-tertiary);
        }}
        
        .speedup-great {{ color: var(--accent-green); font-weight: bold; }}
        .speedup-good {{ color: var(--accent-yellow); }}
        .sim-great {{ color: var(--accent-green); font-weight: bold; }}
        .sim-good {{ color: var(--accent-yellow); }}
        .sim-poor {{ color: var(--accent-red); }}
        
        .charts {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            margin: 2rem 0;
        }}
        
        .chart-container {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1.5rem;
        }}
        
        .video-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
            gap: 1.5rem;
            margin: 2rem 0;
        }}
        
        .video-card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem;
        }}
        
        .video-card h4 {{
            color: var(--accent-blue);
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
        }}
        
        .video-card video {{
            width: 100%;
            border-radius: 4px;
        }}
        
        .video-card p {{
            color: var(--text-secondary);
            margin-top: 0.5rem;
            font-size: 0.85rem;
        }}
        
        a {{
            color: var(--accent-blue);
            text-decoration: none;
        }}
        
        a:hover {{
            text-decoration: underline;
        }}
        
        .timestamp {{
            color: var(--text-secondary);
            font-size: 0.8rem;
            margin-top: 2rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎬 Wan2.1 Experiment Results</h1>
        <p class="subtitle">Hybrid Model Benchmarking - Latency & Quality Analysis</p>
        
        <div class="prompt-box">
            <strong>Prompt:</strong> {EXPERIMENT_PROMPT}<br>
            <strong>Seed:</strong> {EXPERIMENT_SEED} | <strong>Frames:</strong> 81 | <strong>FPS:</strong> 16
        </div>
        
        <h2>📊 Results Summary</h2>
        <table>
            <thead>
                <tr>
                    <th>Experiment</th>
                    <th>Model</th>
                    <th>Steps</th>
                    <th>Cache</th>
                    <th>Latency</th>
                    <th>Speedup</th>
                    <th>Cache Hit</th>
                    <th>CLIP Sim</th>
                    <th>Baseline</th>
                    <th>Video</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
        
        <h2>📈 Latency Comparison</h2>
        <div class="charts">
            <div class="chart-container">
                <canvas id="latencyChart"></canvas>
            </div>
            <div class="chart-container">
                <canvas id="similarityChart"></canvas>
            </div>
        </div>
        
        <h2>🎥 Video Previews</h2>
        <div class="video-grid">
            {video_previews}
        </div>
        
        <p class="timestamp">Generated: {datetime.now().isoformat()}</p>
    </div>
    
    <script>
        // Latency Chart
        new Chart(document.getElementById('latencyChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(chart_labels)},
                datasets: [{{
                    label: 'Generation Time (seconds)',
                    data: {json.dumps(chart_times)},
                    backgroundColor: {json.dumps(chart_colors)},
                    borderRadius: 4,
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Generation Latency',
                        color: '#c9d1d9'
                    }},
                    legend: {{ display: false }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        grid: {{ color: '#30363d' }},
                        ticks: {{ color: '#8b949e' }}
                    }},
                    x: {{
                        grid: {{ display: false }},
                        ticks: {{ color: '#8b949e', maxRotation: 45 }}
                    }}
                }}
            }}
        }});
        
        // Similarity Chart
        const simData = {json.dumps([r.clip_similarity if r.clip_similarity else 0 for r in results])};
        new Chart(document.getElementById('similarityChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(chart_labels)},
                datasets: [{{
                    label: 'CLIP Similarity to Baseline',
                    data: simData,
                    backgroundColor: simData.map(v => v >= 0.95 ? '#3fb950' : v >= 0.90 ? '#d29922' : '#58a6ff'),
                    borderRadius: 4,
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Quality (CLIP Similarity)',
                        color: '#c9d1d9'
                    }},
                    legend: {{ display: false }}
                }},
                scales: {{
                    y: {{
                        min: 0.7,
                        max: 1.0,
                        grid: {{ color: '#30363d' }},
                        ticks: {{ color: '#8b949e' }}
                    }},
                    x: {{
                        grid: {{ display: false }},
                        ticks: {{ color: '#8b949e', maxRotation: 45 }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"  📄 Report saved: {output_path}")


# =============================================================================
# Main Experiment Runner
# =============================================================================

def run_experiment(config: ExperimentConfig, output_dir: Path) -> ExperimentResult:
    """Run a single experiment and return results."""
    print(f"\n{'='*60}")
    print(f"🧪 Running: {config.name}")
    print(f"{'='*60}")
    print(f"  Model: {config.model}")
    print(f"  Steps: {config.steps}")
    print(f"  Frames: {config.frames}")
    print(f"  Caching: {config.caching}")
    print()
    
    # Submit job
    print("  📤 Submitting job...")
    result = submit_job(
        prompt=EXPERIMENT_PROMPT,
        model=config.model,
        steps=config.steps,
        frames=config.frames,
        caching=config.caching,
        seed=EXPERIMENT_SEED,
    )
    job_id = result["job_id"]
    print(f"  ✓ Job ID: {job_id}")
    
    # Wait for completion
    start_time = time.time()
    final_status = wait_for_completion(job_id)
    
    generation_time = final_status.get("generation_time", time.time() - start_time)
    cache_stats = final_status.get("cache_statistics")
    cache_hit_rate = cache_stats.get("cache_hit_rate") if cache_stats else None
    
    print(f"  ✓ Completed in {generation_time:.1f}s")
    if cache_hit_rate:
        print(f"  ✓ Cache hit rate: {cache_hit_rate*100:.1f}%")
    
    # Download video
    video_filename = f"{config.get_filename_base()}.mp4"
    video_path = output_dir / video_filename
    
    print(f"  📥 Downloading video...")
    download_video(job_id, str(video_path))
    print(f"  ✓ Saved: {video_path}")
    
    # Create result
    result = ExperimentResult(
        config=asdict(config),
        job_id=job_id,
        generation_time=generation_time,
        cache_hit_rate=cache_hit_rate,
        video_path=str(video_path),
        completed_at=datetime.now().isoformat(),
    )
    
    return result


def main():
    """Run all experiments."""
    print("\n" + "="*60)
    print("  WAN2.1 HYBRID MODEL EXPERIMENTS")
    print("="*60)
    print(f"  Prompt: {EXPERIMENT_PROMPT[:50]}...")
    print(f"  Seed: {EXPERIMENT_SEED}")
    print(f"  Total experiments: {len(EXPERIMENTS)}")
    print("="*60 + "\n")
    
    # Create output directory
    output_dir = EXPERIMENT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Check server health
    print("\n🔍 Checking server health...")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        health = response.json()
        print(f"  ✓ Server healthy")
        print(f"  Models loaded: {health.get('models_loaded', [])}")
        print(f"  GPU memory: {health.get('gpu_memory_used', 'N/A')} GB")
    except Exception as e:
        print(f"  ✗ Server not reachable: {e}")
        print("  Please ensure the server is running and accessible.")
        sys.exit(1)
    
    # Load CLIP evaluator
    clip_evaluator = None
    if CLIP_AVAILABLE:
        try:
            clip_evaluator = CLIPEvaluator()
        except Exception as e:
            print(f"  Warning: Could not load CLIP: {e}")
    
    # Run experiments
    results: list[ExperimentResult] = []
    baseline_result: Optional[ExperimentResult] = None
    baseline_time: Optional[float] = None
    
    for i, config in enumerate(EXPERIMENTS):
        print(f"\n[{i+1}/{len(EXPERIMENTS)}]")
        
        try:
            result = run_experiment(config, output_dir)
            
            # Track baseline
            if config.is_baseline:
                baseline_result = result
                baseline_time = result.generation_time
                result.clip_similarity = 1.0  # Baseline is identical to itself
                result.speedup_vs_baseline = 1.0
            else:
                # Calculate speedup
                if baseline_time:
                    result.speedup_vs_baseline = baseline_time / result.generation_time
                
                # Calculate CLIP similarity
                if clip_evaluator and baseline_result:
                    print("  📊 Computing CLIP similarity...")
                    try:
                        similarity = clip_evaluator.compute_similarity(
                            baseline_result.video_path,
                            result.video_path
                        )
                        result.clip_similarity = similarity
                        print(f"  ✓ CLIP similarity: {similarity:.4f}")
                    except Exception as e:
                        print(f"  ⚠ Could not compute similarity: {e}")
            
            results.append(result)
            
            # Save intermediate results
            results_path = output_dir / "experiment_results.json"
            with open(results_path, 'w') as f:
                json.dump([r.to_dict() for r in results], f, indent=2)
            
            # Generate HTML report after each experiment
            report_path = output_dir / "report.html"
            generate_html_report(results, str(report_path))
            
            # Print summary
            print(f"\n  📋 EXPERIMENT SUMMARY")
            print(f"  {'─'*40}")
            print(f"  Latency:     {result.generation_time:.1f}s")
            if result.speedup_vs_baseline and not config.is_baseline:
                print(f"  Speedup:     {result.speedup_vs_baseline:.2f}x faster")
            if result.cache_hit_rate:
                print(f"  Cache Rate:  {result.cache_hit_rate*100:.1f}%")
            if result.clip_similarity and not config.is_baseline:
                print(f"  CLIP Sim:    {result.clip_similarity:.4f}")
            print(f"  {'─'*40}")
            
        except Exception as e:
            print(f"\n  ✗ Experiment failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final summary
    print("\n" + "="*60)
    print("  🎉 ALL EXPERIMENTS COMPLETED!")
    print("="*60)
    print(f"\n  Results saved to: {output_dir}")
    print(f"  HTML Report: {output_dir / 'report.html'}")
    print(f"\n  Summary:")
    print(f"  {'─'*50}")
    
    for r in results:
        name = r.config.get('name', '')[:30]
        time_str = f"{r.generation_time:.1f}s"
        speedup = f"{r.speedup_vs_baseline:.2f}x" if r.speedup_vs_baseline else "—"
        sim = f"{r.clip_similarity:.3f}" if r.clip_similarity else "—"
        print(f"  {name:<32} | {time_str:>8} | {speedup:>6} | {sim}")
    
    print(f"  {'─'*50}")
    print()


if __name__ == "__main__":
    main()

