#!/usr/bin/env python3
"""
Wan2.1 Caching Strategy Experiments

Tests different caching configurations and 40 steps.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import requests
import numpy as np
import torch
import cv2
from transformers import CLIPProcessor, CLIPModel

# =============================================================================
# Configuration
# =============================================================================

API_BASE_URL = "https://t6npgjjo8y0yf0-8888.proxy.runpod.net"
EXPERIMENT_PROMPT = "The Merced river is overflowing, birds flying in the sky, camera is zooming out to reveal an American Buffalo bathing in the river"
EXPERIMENT_SEED = 42
EXPERIMENT_DIR = Path(__file__).parent.parent / "experiments"

# =============================================================================
# Experiment Definitions
# =============================================================================

@dataclass
class CachingConfig:
    """Caching configuration."""
    enabled: bool
    start_step: int = 10
    end_step: int = 40
    interval: int = 3
    name: str = ""
    
    def __post_init__(self):
        if not self.name and self.enabled:
            self.name = f"cache_{self.start_step}-{self.end_step}_int{self.interval}"


@dataclass 
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    model: str
    steps: int
    frames: int
    caching: CachingConfig
    is_baseline: bool = False
    
    def get_filename_base(self) -> str:
        if self.caching.enabled:
            cache_str = self.caching.name
        else:
            cache_str = "nocache"
        return f"{self.model}_{self.steps}steps_{self.frames}frames_{cache_str}"


# New experiments to run
NEW_EXPERIMENTS = [
    # 40 steps experiments
    ExperimentConfig(
        name="Hybrid 40 steps (no cache)",
        model="hybrid",
        steps=40,
        frames=81,
        caching=CachingConfig(enabled=False),
    ),
    ExperimentConfig(
        name="Hybrid 40 steps (default cache)",
        model="hybrid",
        steps=40,
        frames=81,
        caching=CachingConfig(enabled=True, start_step=10, end_step=35, interval=3, name="default"),
    ),
    ExperimentConfig(
        name="Hybrid 40 steps (aggressive cache)",
        model="hybrid",
        steps=40,
        frames=81,
        caching=CachingConfig(enabled=True, start_step=5, end_step=38, interval=2, name="aggressive"),
    ),
    ExperimentConfig(
        name="Hybrid 40 steps (conservative cache)",
        model="hybrid",
        steps=40,
        frames=81,
        caching=CachingConfig(enabled=True, start_step=15, end_step=30, interval=4, name="conservative"),
    ),
    
    # 50 steps with different caching strategies
    ExperimentConfig(
        name="Hybrid 50 steps (aggressive cache)",
        model="hybrid",
        steps=50,
        frames=81,
        caching=CachingConfig(enabled=True, start_step=5, end_step=45, interval=2, name="aggressive"),
    ),
    ExperimentConfig(
        name="Hybrid 50 steps (conservative cache)",
        model="hybrid",
        steps=50,
        frames=81,
        caching=CachingConfig(enabled=True, start_step=15, end_step=35, interval=4, name="conservative"),
    ),
    
    # 30 steps with different caching strategies  
    ExperimentConfig(
        name="Hybrid 30 steps (aggressive cache)",
        model="hybrid",
        steps=30,
        frames=81,
        caching=CachingConfig(enabled=True, start_step=5, end_step=25, interval=2, name="aggressive"),
    ),
]


# =============================================================================
# API Functions
# =============================================================================

def submit_job(prompt: str, model: str, steps: int, frames: int, 
               caching: CachingConfig, seed: int) -> dict:
    """Submit a generation job to the API."""
    payload = {
        "prompt": prompt,
        "model": model,
        "frame_count": frames,
        "fps": 16,
        "sampling_steps": steps,
        "seed": seed,
        "enable_caching": caching.enabled,
    }
    
    if caching.enabled:
        payload["cache_start_step"] = caching.start_step
        payload["cache_end_step"] = caching.end_step
        payload["cache_interval"] = caching.interval
    
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
        try:
            status = get_status(job_id)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"\n  ⚠ Job {job_id} not found, waiting...")
                time.sleep(poll_interval)
                continue
            raise
        
        state = status.get("status", "unknown")
        progress = status.get("progress", 0)
        model_in_use = status.get("model_in_use", "")
        
        bar_width = 30
        filled = int(bar_width * progress / 100)
        bar = '█' * filled + '░' * (bar_width - filled)
        model_str = f" [{model_in_use}]" if model_in_use else ""
        print(f"\r  [{bar}] {progress}%{model_str}    ", end='', flush=True)
        
        if state == "completed":
            print()
            return status
        elif state == "failed":
            print()
            raise RuntimeError(f"Job failed: {status.get('error', 'Unknown error')}")
        
        time.sleep(poll_interval)


# =============================================================================
# CLIP Evaluator
# =============================================================================

class CLIPEvaluator:
    """Compute CLIP-based similarity between videos."""
    
    def __init__(self):
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
        
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        return frames
    
    def get_video_embedding(self, video_path: str) -> np.ndarray:
        """Get average CLIP embedding for video frames."""
        frames = self.extract_frames(video_path)
        
        if not frames:
            raise ValueError(f"Could not extract frames from {video_path}")
        
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

def generate_html_report(results: list, output_path: str):
    """Generate an HTML report with experiment results."""
    
    baseline_time = None
    for r in results:
        if r.get("config", {}).get("is_baseline"):
            baseline_time = r.get("generation_time")
            break
    
    # Group results by step count for better organization
    by_steps = {}
    for r in results:
        steps = r.get("config", {}).get("steps", 0)
        if steps not in by_steps:
            by_steps[steps] = []
        by_steps[steps].append(r)
    
    table_rows = ""
    for r in results:
        config = r.get("config", {})
        caching = config.get("caching", {})
        
        speedup = f"{r.get('speedup_vs_baseline', 0):.2f}x" if r.get('speedup_vs_baseline') else "—"
        cache_rate = f"{r.get('cache_hit_rate', 0)*100:.1f}%" if r.get('cache_hit_rate') else "—"
        clip_sim = f"{r.get('clip_similarity', 0):.4f}" if r.get('clip_similarity') else "—"
        is_baseline = "✓" if config.get("is_baseline") else ""
        
        # Cache config display
        if isinstance(caching, dict) and caching.get("enabled"):
            cache_config = f"{caching.get('start_step', '-')}-{caching.get('end_step', '-')} / {caching.get('interval', '-')}"
        else:
            cache_config = "—"
        
        speedup_class = ""
        if r.get('speedup_vs_baseline'):
            if r['speedup_vs_baseline'] >= 3:
                speedup_class = "speedup-great"
            elif r['speedup_vs_baseline'] >= 2:
                speedup_class = "speedup-good"
        
        sim_class = ""
        if r.get('clip_similarity'):
            if r['clip_similarity'] >= 0.97:
                sim_class = "sim-great"
            elif r['clip_similarity'] >= 0.93:
                sim_class = "sim-good"
            elif r['clip_similarity'] < 0.88:
                sim_class = "sim-poor"
        
        video_name = os.path.basename(r.get('video_path', ''))
        
        table_rows += f"""
        <tr>
            <td>{config.get('name', '')}</td>
            <td>{config.get('steps', '')}</td>
            <td>{cache_config}</td>
            <td><strong>{r.get('generation_time', 0):.1f}s</strong></td>
            <td class="{speedup_class}">{speedup}</td>
            <td>{cache_rate}</td>
            <td class="{sim_class}">{clip_sim}</td>
            <td>{is_baseline}</td>
            <td><a href="{video_name}" target="_blank">🎬</a></td>
        </tr>
        """
    
    video_previews = ""
    for r in results:
        video_name = os.path.basename(r.get('video_path', ''))
        label = r.get('config', {}).get('name', video_name)
        gen_time = r.get('generation_time', 0)
        clip_sim = r.get('clip_similarity')
        clip_str = f"{clip_sim:.4f}" if clip_sim else "N/A"
        speedup = r.get('speedup_vs_baseline')
        speedup_str = f"{speedup:.2f}x" if speedup else "—"
        video_previews += f"""
        <div class="video-card">
            <h4>{label}</h4>
            <video controls width="400">
                <source src="{video_name}" type="video/mp4">
            </video>
            <p>⏱️ {gen_time:.1f}s ({speedup_str}) | 📊 CLIP: {clip_str}</p>
        </div>
        """
    
    chart_labels = [r.get('config', {}).get('name', '')[:25] for r in results]
    chart_times = [r.get('generation_time', 0) for r in results]
    chart_sims = [r.get('clip_similarity', 0) or 0 for r in results]
    chart_colors = ['#4CAF50' if r.get('config', {}).get('is_baseline') else '#2196F3' for r in results]
    
    # Create speedup data
    speedups = [r.get('speedup_vs_baseline', 0) or 0 for r in results]
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Wan2.1 Experiment Results - Extended</title>
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
            --accent-purple: #a371f7;
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
        
        .container {{ max-width: 1600px; margin: 0 auto; }}
        
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
        
        .subtitle {{ color: var(--text-secondary); margin-bottom: 2rem; }}
        
        .prompt-box {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 2rem;
        }}
        
        .prompt-box strong {{ color: var(--accent-yellow); }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        
        .stat-card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1.5rem;
            text-align: center;
        }}
        
        .stat-card .value {{
            font-size: 2rem;
            font-weight: bold;
            color: var(--accent-green);
        }}
        
        .stat-card .label {{
            color: var(--text-secondary);
            font-size: 0.8rem;
            text-transform: uppercase;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--bg-secondary);
            border-radius: 8px;
            overflow: hidden;
            font-size: 0.9rem;
        }}
        
        th, td {{
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        
        th {{
            background: var(--bg-tertiary);
            color: var(--text-secondary);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.7rem;
            letter-spacing: 0.5px;
        }}
        
        tr:hover {{ background: var(--bg-tertiary); }}
        
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
            font-size: 0.85rem;
        }}
        
        .video-card video {{ width: 100%; border-radius: 4px; }}
        
        .video-card p {{
            color: var(--text-secondary);
            margin-top: 0.5rem;
            font-size: 0.8rem;
        }}
        
        a {{ color: var(--accent-blue); text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        
        .timestamp {{
            color: var(--text-secondary);
            font-size: 0.8rem;
            margin-top: 2rem;
        }}
        
        .findings {{
            background: var(--bg-secondary);
            border: 1px solid var(--accent-green);
            border-radius: 8px;
            padding: 1.5rem;
            margin: 2rem 0;
        }}
        
        .findings h3 {{
            color: var(--accent-green);
            margin-bottom: 1rem;
        }}
        
        .findings ul {{
            list-style: none;
            padding: 0;
        }}
        
        .findings li {{
            padding: 0.5rem 0;
            border-bottom: 1px solid var(--border-color);
        }}
        
        .findings li:last-child {{ border-bottom: none; }}
        
        .findings .highlight {{
            color: var(--accent-yellow);
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎬 Wan2.1 Experiment Results</h1>
        <p class="subtitle">Hybrid Model Benchmarking with Caching Strategy Analysis</p>
        
        <div class="prompt-box">
            <strong>Prompt:</strong> {EXPERIMENT_PROMPT}<br>
            <strong>Seed:</strong> {EXPERIMENT_SEED} | <strong>Frames:</strong> 81 | <strong>FPS:</strong> 16
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="value">{len(results)}</div>
                <div class="label">Total Experiments</div>
            </div>
            <div class="stat-card">
                <div class="value">{max(speedups):.1f}x</div>
                <div class="label">Max Speedup</div>
            </div>
            <div class="stat-card">
                <div class="value">{min(chart_times):.0f}s</div>
                <div class="label">Fastest Time</div>
            </div>
            <div class="stat-card">
                <div class="value">{max([s for s in chart_sims if s < 1.0] or [0]):.2f}</div>
                <div class="label">Best Non-Baseline CLIP</div>
            </div>
        </div>
        
        <h2>📊 Results Summary</h2>
        <table>
            <thead>
                <tr>
                    <th>Experiment</th>
                    <th>Steps</th>
                    <th>Cache Config</th>
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
        
        <h2>📈 Performance Analysis</h2>
        <div class="charts">
            <div class="chart-container">
                <canvas id="latencyChart"></canvas>
            </div>
            <div class="chart-container">
                <canvas id="speedupChart"></canvas>
            </div>
        </div>
        
        <div class="charts">
            <div class="chart-container">
                <canvas id="similarityChart"></canvas>
            </div>
            <div class="chart-container">
                <canvas id="tradeoffChart"></canvas>
            </div>
        </div>
        
        <h2>🎥 Video Previews</h2>
        <div class="video-grid">
            {video_previews}
        </div>
        
        <p class="timestamp">Generated: {datetime.now().isoformat()}</p>
    </div>
    
    <script>
        const labels = {json.dumps(chart_labels)};
        const times = {json.dumps(chart_times)};
        const sims = {json.dumps(chart_sims)};
        const speedups = {json.dumps(speedups)};
        const colors = {json.dumps(chart_colors)};
        
        // Latency Chart
        new Chart(document.getElementById('latencyChart'), {{
            type: 'bar',
            data: {{
                labels: labels,
                datasets: [{{
                    label: 'Generation Time (seconds)',
                    data: times,
                    backgroundColor: colors,
                    borderRadius: 4,
                }}]
            }},
            options: {{
                responsive: true,
                indexAxis: 'y',
                plugins: {{
                    title: {{ display: true, text: 'Generation Latency (lower is better)', color: '#c9d1d9' }},
                    legend: {{ display: false }}
                }},
                scales: {{
                    x: {{ beginAtZero: true, grid: {{ color: '#30363d' }}, ticks: {{ color: '#8b949e' }} }},
                    y: {{ grid: {{ display: false }}, ticks: {{ color: '#8b949e', font: {{ size: 10 }} }} }}
                }}
            }}
        }});
        
        // Speedup Chart
        new Chart(document.getElementById('speedupChart'), {{
            type: 'bar',
            data: {{
                labels: labels,
                datasets: [{{
                    label: 'Speedup vs Baseline',
                    data: speedups,
                    backgroundColor: speedups.map(v => v >= 3 ? '#3fb950' : v >= 2 ? '#d29922' : '#58a6ff'),
                    borderRadius: 4,
                }}]
            }},
            options: {{
                responsive: true,
                indexAxis: 'y',
                plugins: {{
                    title: {{ display: true, text: 'Speedup vs Baseline (higher is better)', color: '#c9d1d9' }},
                    legend: {{ display: false }}
                }},
                scales: {{
                    x: {{ beginAtZero: true, grid: {{ color: '#30363d' }}, ticks: {{ color: '#8b949e' }} }},
                    y: {{ grid: {{ display: false }}, ticks: {{ color: '#8b949e', font: {{ size: 10 }} }} }}
                }}
            }}
        }});
        
        // Similarity Chart
        new Chart(document.getElementById('similarityChart'), {{
            type: 'bar',
            data: {{
                labels: labels,
                datasets: [{{
                    label: 'CLIP Similarity',
                    data: sims,
                    backgroundColor: sims.map(v => v >= 0.97 ? '#3fb950' : v >= 0.93 ? '#d29922' : '#58a6ff'),
                    borderRadius: 4,
                }}]
            }},
            options: {{
                responsive: true,
                indexAxis: 'y',
                plugins: {{
                    title: {{ display: true, text: 'Quality - CLIP Similarity (higher is better)', color: '#c9d1d9' }},
                    legend: {{ display: false }}
                }},
                scales: {{
                    x: {{ min: 0.8, max: 1.0, grid: {{ color: '#30363d' }}, ticks: {{ color: '#8b949e' }} }},
                    y: {{ grid: {{ display: false }}, ticks: {{ color: '#8b949e', font: {{ size: 10 }} }} }}
                }}
            }}
        }});
        
        // Quality vs Speed Tradeoff
        const tradeoffData = labels.map((label, i) => ({{
            x: speedups[i],
            y: sims[i],
            label: label
        }})).filter(d => d.x > 0 && d.y > 0);
        
        new Chart(document.getElementById('tradeoffChart'), {{
            type: 'scatter',
            data: {{
                datasets: [{{
                    label: 'Experiments',
                    data: tradeoffData,
                    backgroundColor: tradeoffData.map(d => d.y >= 0.97 ? '#3fb950' : d.y >= 0.93 ? '#d29922' : '#58a6ff'),
                    pointRadius: 8,
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{ display: true, text: 'Quality vs Speed Tradeoff', color: '#c9d1d9' }},
                    tooltip: {{
                        callbacks: {{
                            label: (ctx) => tradeoffData[ctx.dataIndex].label + ': ' + ctx.parsed.x.toFixed(2) + 'x, CLIP ' + ctx.parsed.y.toFixed(4)
                        }}
                    }}
                }},
                scales: {{
                    x: {{ 
                        title: {{ display: true, text: 'Speedup (higher is faster)', color: '#8b949e' }},
                        grid: {{ color: '#30363d' }}, 
                        ticks: {{ color: '#8b949e' }} 
                    }},
                    y: {{ 
                        min: 0.85, 
                        max: 1.0,
                        title: {{ display: true, text: 'CLIP Similarity (higher is better)', color: '#8b949e' }},
                        grid: {{ color: '#30363d' }}, 
                        ticks: {{ color: '#8b949e' }} 
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
# Main
# =============================================================================

def run_experiment(config: ExperimentConfig, output_dir: Path, 
                   baseline_time: float, clip_eval, baseline_video: str) -> dict:
    """Run a single experiment and return results."""
    print(f"\n{'='*60}")
    print(f"🧪 Running: {config.name}")
    print(f"{'='*60}")
    print(f"  Model: {config.model}")
    print(f"  Steps: {config.steps}")
    print(f"  Caching: {config.caching.enabled}")
    if config.caching.enabled:
        print(f"    Start: {config.caching.start_step}, End: {config.caching.end_step}, Interval: {config.caching.interval}")
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
    final_status = wait_for_completion(job_id)
    
    generation_time = final_status.get("generation_time", 0)
    cache_stats = final_status.get("cache_statistics")
    cache_hit_rate = cache_stats.get("cache_hit_rate") if cache_stats else None
    
    print(f"  ✓ Completed in {generation_time:.1f}s")
    if cache_hit_rate:
        print(f"  ✓ Cache hit rate: {cache_hit_rate*100:.1f}%")
    
    # Download video
    video_path = output_dir / f"{config.get_filename_base()}.mp4"
    print(f"  📥 Downloading video...")
    download_video(job_id, str(video_path))
    print(f"  ✓ Saved: {video_path}")
    
    # Compute CLIP similarity
    clip_similarity = None
    if clip_eval and baseline_video:
        print("  📊 Computing CLIP similarity...")
        try:
            clip_similarity = clip_eval.compute_similarity(baseline_video, str(video_path))
            print(f"  ✓ CLIP similarity: {clip_similarity:.4f}")
        except Exception as e:
            print(f"  ⚠ Could not compute similarity: {e}")
    
    # Calculate speedup
    speedup = baseline_time / generation_time if baseline_time else None
    
    # Create result dict
    result = {
        "config": {
            "name": config.name,
            "model": config.model,
            "steps": config.steps,
            "frames": config.frames,
            "caching": asdict(config.caching),
            "is_baseline": config.is_baseline,
        },
        "job_id": job_id,
        "generation_time": generation_time,
        "cache_hit_rate": cache_hit_rate,
        "video_path": str(video_path),
        "clip_similarity": clip_similarity,
        "speedup_vs_baseline": speedup,
        "completed_at": datetime.now().isoformat(),
    }
    
    # Print summary
    print(f"\n  📋 EXPERIMENT SUMMARY")
    print(f"  {'─'*40}")
    print(f"  Latency:     {generation_time:.1f}s")
    if speedup:
        print(f"  Speedup:     {speedup:.2f}x faster")
    if cache_hit_rate:
        print(f"  Cache Rate:  {cache_hit_rate*100:.1f}%")
    if clip_similarity:
        print(f"  CLIP Sim:    {clip_similarity:.4f}")
    print(f"  {'─'*40}")
    
    return result


def main():
    print("\n" + "="*60)
    print("  CACHING STRATEGY EXPERIMENTS")
    print("="*60)
    print(f"  New experiments: {len(NEW_EXPERIMENTS)}")
    print("="*60 + "\n")
    
    output_dir = EXPERIMENT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / "experiment_results.json"
    
    # Load existing results
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing results")
    else:
        results = []
    
    # Find baseline
    baseline_time = None
    baseline_video = None
    for r in results:
        if r.get("config", {}).get("is_baseline"):
            baseline_time = r.get("generation_time")
            baseline_video = r.get("video_path")
            break
    
    if not baseline_time:
        print("⚠ No baseline found in existing results!")
        return
    
    print(f"Baseline time: {baseline_time:.1f}s")
    print(f"Baseline video: {baseline_video}")
    
    # Check server health
    print("\n🔍 Checking server health...")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        health = response.json()
        print(f"  ✓ Server healthy")
        print(f"  Models loaded: {health.get('models_loaded', [])}")
    except Exception as e:
        print(f"  ✗ Server not reachable: {e}")
        return
    
    # Load CLIP evaluator
    print("\n📊 Loading CLIP model...")
    try:
        clip_eval = CLIPEvaluator()
    except Exception as e:
        print(f"  ⚠ Could not load CLIP: {e}")
        clip_eval = None
    
    # Check which experiments are already done
    existing_names = {r.get("config", {}).get("name") for r in results}
    experiments_to_run = [e for e in NEW_EXPERIMENTS if e.name not in existing_names]
    
    print(f"\n📋 Experiments to run: {len(experiments_to_run)}")
    for e in experiments_to_run:
        print(f"  - {e.name}")
    
    # Run new experiments
    for i, config in enumerate(experiments_to_run):
        print(f"\n[{i+1}/{len(experiments_to_run)}]")
        
        try:
            result = run_experiment(
                config, output_dir, baseline_time, clip_eval, baseline_video
            )
            results.append(result)
            
            # Save after each experiment
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Regenerate report
            report_path = output_dir / "report.html"
            generate_html_report(results, str(report_path))
            
        except Exception as e:
            print(f"\n  ✗ Experiment failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Final summary
    print("\n" + "="*70)
    print("  🎉 ALL EXPERIMENTS COMPLETE!")
    print("="*70)
    print(f"\n  {'Experiment':<40} | {'Time':>7} | {'Speedup':>7} | {'CLIP':>7}")
    print(f"  {'-'*70}")
    
    for r in sorted(results, key=lambda x: x.get('generation_time', 999)):
        name = r.get('config', {}).get('name', '')[:40]
        time_str = f"{r.get('generation_time', 0):.1f}s"
        speedup = f"{r.get('speedup_vs_baseline', 0):.2f}x" if r.get('speedup_vs_baseline') else "—"
        sim = f"{r.get('clip_similarity', 0):.4f}" if r.get('clip_similarity') else "—"
        print(f"  {name:<40} | {time_str:>7} | {speedup:>7} | {sim:>7}")
    
    print(f"  {'-'*70}")
    print(f"\n  📄 HTML Report: {output_dir / 'report.html'}")
    print()


if __name__ == "__main__":
    main()

