#!/usr/bin/env python3
"""
Run the missing experiment and compute CLIP similarities for all videos.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

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
# API Functions
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
# HTML Report Generator
# =============================================================================

def generate_html_report(results: list, output_path: str):
    """Generate an HTML report with experiment results."""
    
    baseline_time = None
    for r in results:
        if r.get("config", {}).get("is_baseline"):
            baseline_time = r.get("generation_time")
            break
    
    table_rows = ""
    for r in results:
        config = r.get("config", {})
        speedup = f"{r.get('speedup_vs_baseline', 0):.2f}x" if r.get('speedup_vs_baseline') else "—"
        cache_rate = f"{r.get('cache_hit_rate', 0)*100:.1f}%" if r.get('cache_hit_rate') else "—"
        clip_sim = f"{r.get('clip_similarity', 0):.4f}" if r.get('clip_similarity') else "—"
        is_baseline = "✓" if config.get("is_baseline") else ""
        
        speedup_class = ""
        if r.get('speedup_vs_baseline'):
            if r['speedup_vs_baseline'] >= 2:
                speedup_class = "speedup-great"
            elif r['speedup_vs_baseline'] >= 1.5:
                speedup_class = "speedup-good"
        
        sim_class = ""
        if r.get('clip_similarity'):
            if r['clip_similarity'] >= 0.95:
                sim_class = "sim-great"
            elif r['clip_similarity'] >= 0.90:
                sim_class = "sim-good"
            elif r['clip_similarity'] < 0.85:
                sim_class = "sim-poor"
        
        video_name = os.path.basename(r.get('video_path', ''))
        
        table_rows += f"""
        <tr>
            <td>{config.get('name', '')}</td>
            <td>{config.get('model', '')}</td>
            <td>{config.get('steps', '')}</td>
            <td>{config.get('caching', False) and '✓' or '—'}</td>
            <td><strong>{r.get('generation_time', 0):.1f}s</strong></td>
            <td class="{speedup_class}">{speedup}</td>
            <td>{cache_rate}</td>
            <td class="{sim_class}">{clip_sim}</td>
            <td>{is_baseline}</td>
            <td><a href="{video_name}" target="_blank">🎬 View</a></td>
        </tr>
        """
    
    video_previews = ""
    for r in results:
        video_name = os.path.basename(r.get('video_path', ''))
        label = r.get('config', {}).get('name', video_name)
        gen_time = r.get('generation_time', 0)
        clip_sim = r.get('clip_similarity')
        clip_str = f"{clip_sim:.4f}" if clip_sim else "N/A"
        video_previews += f"""
        <div class="video-card">
            <h4>{label}</h4>
            <video controls width="400">
                <source src="{video_name}" type="video/mp4">
            </video>
            <p>Time: {gen_time:.1f}s | CLIP: {clip_str}</p>
        </div>
        """
    
    chart_labels = [r.get('config', {}).get('name', '')[:20] for r in results]
    chart_times = [r.get('generation_time', 0) for r in results]
    chart_colors = ['#4CAF50' if r.get('config', {}).get('is_baseline') else '#2196F3' for r in results]
    
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
                    title: {{ display: true, text: 'Generation Latency', color: '#c9d1d9' }},
                    legend: {{ display: false }}
                }},
                scales: {{
                    y: {{ beginAtZero: true, grid: {{ color: '#30363d' }}, ticks: {{ color: '#8b949e' }} }},
                    x: {{ grid: {{ display: false }}, ticks: {{ color: '#8b949e', maxRotation: 45 }} }}
                }}
            }}
        }});
        
        const simData = {json.dumps([r.get('clip_similarity', 0) or 0 for r in results])};
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
                    title: {{ display: true, text: 'Quality (CLIP Similarity)', color: '#c9d1d9' }},
                    legend: {{ display: false }}
                }},
                scales: {{
                    y: {{ min: 0.7, max: 1.0, grid: {{ color: '#30363d' }}, ticks: {{ color: '#8b949e' }} }},
                    x: {{ grid: {{ display: false }}, ticks: {{ color: '#8b949e', maxRotation: 45 }} }}
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

def main():
    print("\n" + "="*60)
    print("  COMPLETING EXPERIMENTS & COMPUTING CLIP SIMILARITY")
    print("="*60 + "\n")
    
    results_path = EXPERIMENT_DIR / "experiment_results.json"
    
    # Load existing results
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing results")
    else:
        print("No existing results found")
        results = []
    
    # Check if we need to run the missing experiment
    has_hybrid_50_cache = any(
        r.get("config", {}).get("name") == "Hybrid 50 steps (with cache)"
        for r in results
    )
    
    baseline_time = None
    baseline_video = None
    for r in results:
        if r.get("config", {}).get("is_baseline"):
            baseline_time = r.get("generation_time")
            baseline_video = r.get("video_path")
            break
    
    if not has_hybrid_50_cache:
        print("\n🧪 Running missing experiment: Hybrid 50 steps (with cache)")
        print("="*60)
        
        try:
            result = submit_job(
                prompt=EXPERIMENT_PROMPT,
                model="hybrid",
                steps=50,
                frames=81,
                caching=True,
                seed=EXPERIMENT_SEED,
            )
            job_id = result["job_id"]
            print(f"  ✓ Job ID: {job_id}")
            
            final_status = wait_for_completion(job_id)
            
            generation_time = final_status.get("generation_time", 0)
            cache_stats = final_status.get("cache_statistics")
            cache_hit_rate = cache_stats.get("cache_hit_rate") if cache_stats else None
            
            print(f"  ✓ Completed in {generation_time:.1f}s")
            if cache_hit_rate:
                print(f"  ✓ Cache hit rate: {cache_hit_rate*100:.1f}%")
            
            video_path = EXPERIMENT_DIR / "hybrid_50steps_81frames_cache.mp4"
            print(f"  📥 Downloading video...")
            download_video(job_id, str(video_path))
            print(f"  ✓ Saved: {video_path}")
            
            new_result = {
                "config": {
                    "name": "Hybrid 50 steps (with cache)",
                    "model": "hybrid",
                    "steps": 50,
                    "frames": 81,
                    "caching": True,
                    "is_baseline": False,
                },
                "job_id": job_id,
                "generation_time": generation_time,
                "cache_hit_rate": cache_hit_rate,
                "video_path": str(video_path),
                "speedup_vs_baseline": baseline_time / generation_time if baseline_time else None,
                "completed_at": datetime.now().isoformat(),
            }
            
            results.append(new_result)
            
            print(f"\n  📋 EXPERIMENT SUMMARY")
            print(f"  {'─'*40}")
            print(f"  Latency:     {generation_time:.1f}s")
            print(f"  Speedup:     {new_result['speedup_vs_baseline']:.2f}x faster")
            if cache_hit_rate:
                print(f"  Cache Rate:  {cache_hit_rate*100:.1f}%")
            print(f"  {'─'*40}")
            
        except Exception as e:
            print(f"  ✗ Failed to run missing experiment: {e}")
            import traceback
            traceback.print_exc()
    
    # Load CLIP evaluator and compute similarities
    print("\n📊 Computing CLIP similarities...")
    print("="*60)
    
    try:
        clip_eval = CLIPEvaluator()
        
        if baseline_video and os.path.exists(baseline_video):
            print(f"\nBaseline video: {baseline_video}")
            
            for r in results:
                config = r.get("config", {})
                video_path = r.get("video_path")
                
                if not video_path or not os.path.exists(video_path):
                    print(f"  ⚠ Video not found: {video_path}")
                    continue
                
                if config.get("is_baseline"):
                    r["clip_similarity"] = 1.0
                    print(f"  ✓ {config.get('name')}: 1.0000 (baseline)")
                else:
                    print(f"  Computing similarity for {config.get('name')}...")
                    try:
                        similarity = clip_eval.compute_similarity(baseline_video, video_path)
                        r["clip_similarity"] = similarity
                        print(f"  ✓ {config.get('name')}: {similarity:.4f}")
                    except Exception as e:
                        print(f"  ⚠ Error: {e}")
        else:
            print("  ⚠ Baseline video not found, skipping CLIP similarity")
            
    except Exception as e:
        print(f"  ✗ CLIP evaluation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Save results
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved: {results_path}")
    
    # Generate HTML report
    report_path = EXPERIMENT_DIR / "report.html"
    generate_html_report(results, str(report_path))
    
    # Final summary
    print("\n" + "="*60)
    print("  🎉 COMPLETE!")
    print("="*60)
    print(f"\n  Summary:")
    print(f"  {'─'*60}")
    print(f"  {'Experiment':<35} | {'Time':>8} | {'Speedup':>7} | {'CLIP':>6}")
    print(f"  {'─'*60}")
    
    for r in results:
        name = r.get('config', {}).get('name', '')[:35]
        time_str = f"{r.get('generation_time', 0):.1f}s"
        speedup = f"{r.get('speedup_vs_baseline', 0):.2f}x" if r.get('speedup_vs_baseline') else "—"
        sim = f"{r.get('clip_similarity', 0):.4f}" if r.get('clip_similarity') else "—"
        print(f"  {name:<35} | {time_str:>8} | {speedup:>7} | {sim:>6}")
    
    print(f"  {'─'*60}")
    print(f"\n  📄 HTML Report: {report_path}")
    print()


if __name__ == "__main__":
    main()

