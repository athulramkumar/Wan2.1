#!/usr/bin/env python3
"""
Hybrid Compute Strategy Experiments for Wan2.1

Comprehensive comparison of different hybrid inference strategies,
measuring latency, estimated cost, and quality (CLIP similarity).

Strategies tested:
  1. Baselines: pure 14B, pure 1.3B
  2. Split ratios: 10/90, 20/80, 30/70, 40/60, 50/50 (14B/1.3B)
  3. Ordering: 14B-first vs 14B-last vs 14B-sandwich
  4. Caching combinations: best splits with/without caching
  5. Step counts: 30, 40, 50 steps with best split

All experiments use the same seed for fair comparison.
Generates an HTML report with charts for latency, cost, and quality.

Usage:
    python run_hybrid_strategy_experiments.py --server https://pod-id-8888.proxy.runpod.net
    python run_hybrid_strategy_experiments.py --server http://localhost:8888 --quick
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
import numpy as np

# CLIP (optional, for quality comparison)
try:
    import torch
    import cv2
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Note: CLIP not available. Quality metrics will be skipped.")
    print("Install with: pip install transformers torch opencv-python")


# =============================================================================
# Configuration
# =============================================================================

EXPERIMENT_SEED = 42
EXPERIMENT_DIR = Path(__file__).parent.parent / "experiments" / "hybrid_strategies"

# Multiple prompts for statistical robustness
PROMPTS = [
    {
        "id": "nature",
        "text": "The Merced river is overflowing, birds flying in the sky, camera is zooming out to reveal an American Buffalo bathing in the river",
    },
    {
        "id": "scifi",
        "text": "A spaceship launching from a futuristic city at sunset with dramatic clouds and neon lights reflecting off glass buildings",
    },
    {
        "id": "closeup",
        "text": "A close-up of rain drops falling on a still pond creating ripples in slow motion, with cherry blossom petals floating on the surface",
    },
]

# GPU cost rates ($/hr) for cost estimation
COST_RATES = {
    "H100": 3.50,
    "H100_spot": 1.75,
    "A100": 2.50,
    "A100_spot": 1.50,
    "A40x2_spot": 1.40,
    "RTX4090": 0.44,
    "A4000": 0.30,
}


# =============================================================================
# Experiment Definitions
# =============================================================================

@dataclass
class StrategyConfig:
    """Configuration for a single experiment strategy."""
    name: str
    description: str
    model: str                      # baseline_14B, baseline_1.3B, hybrid
    steps: int = 50
    schedule: Optional[list] = None # [[model, steps], ...]
    caching: bool = False
    cache_start: int = 10
    cache_end: int = 40
    cache_interval: int = 3
    is_baseline: bool = False
    category: str = "other"         # baseline, split, ordering, caching, steps

    def schedule_json(self) -> Optional[list]:
        if self.schedule:
            return [[s[0], s[1]] for s in self.schedule]
        return None


def build_experiment_configs(total_steps: int = 50, quick: bool = False) -> list:
    """Build the full list of experiment configurations."""
    configs = []

    # ── Category 1: Baselines ──
    configs.append(StrategyConfig(
        name="Baseline 14B",
        description="All steps on 14B (quality reference)",
        model="baseline_14B",
        steps=total_steps,
        is_baseline=True,
        category="baseline",
    ))
    configs.append(StrategyConfig(
        name="Baseline 1.3B",
        description="All steps on 1.3B (speed/cost reference)",
        model="baseline_1.3B",
        steps=total_steps,
        category="baseline",
    ))

    # ── Category 2: Split ratios (14B first) ──
    splits = [(10, 90), (20, 80), (30, 70), (40, 60), (50, 50)]
    if quick:
        splits = [(20, 80), (30, 70), (50, 50)]

    for pct_14b, pct_1_3b in splits:
        steps_14b = int(total_steps * pct_14b / 100)
        steps_1_3b = total_steps - steps_14b
        configs.append(StrategyConfig(
            name=f"Split {pct_14b}/{pct_1_3b}",
            description=f"{steps_14b} steps 14B then {steps_1_3b} steps 1.3B",
            model="hybrid",
            steps=total_steps,
            schedule=[("14B", steps_14b), ("1.3B", steps_1_3b)],
            category="split",
        ))

    # ── Category 3: Ordering variations (with 30/70 split) ──
    steps_14b_30 = int(total_steps * 0.3)
    steps_1_3b_70 = total_steps - steps_14b_30

    configs.append(StrategyConfig(
        name="14B-First (30/70)",
        description="14B runs early steps (structure), 1.3B refines",
        model="hybrid",
        steps=total_steps,
        schedule=[("14B", steps_14b_30), ("1.3B", steps_1_3b_70)],
        category="ordering",
    ))
    configs.append(StrategyConfig(
        name="14B-Last (30/70)",
        description="1.3B runs early, 14B corrects final steps",
        model="hybrid",
        steps=total_steps,
        schedule=[("1.3B", steps_1_3b_70), ("14B", steps_14b_30)],
        category="ordering",
    ))

    if not quick:
        # Sandwich: 14B-1.3B-14B
        steps_14b_half = steps_14b_30 // 2
        steps_14b_remainder = steps_14b_30 - steps_14b_half
        configs.append(StrategyConfig(
            name="14B-Sandwich (30/70)",
            description="14B first + last, 1.3B in middle",
            model="hybrid",
            steps=total_steps,
            schedule=[
                ("14B", steps_14b_half),
                ("1.3B", steps_1_3b_70),
                ("14B", steps_14b_remainder),
            ],
            category="ordering",
        ))

    # ── Category 4: Caching combinations ──
    for pct_14b, pct_1_3b in [(30, 70)]:
        steps_14b = int(total_steps * pct_14b / 100)
        steps_1_3b = total_steps - steps_14b
        configs.append(StrategyConfig(
            name=f"Split {pct_14b}/{pct_1_3b} + Cache",
            description=f"Hybrid {pct_14b}/{pct_1_3b} with activation caching",
            model="hybrid",
            steps=total_steps,
            schedule=[("14B", steps_14b), ("1.3B", steps_1_3b)],
            caching=True,
            cache_start=10,
            cache_end=min(40, total_steps - 5),
            cache_interval=3,
            category="caching",
        ))

    if not quick:
        # 1.3B with caching (speed baseline)
        configs.append(StrategyConfig(
            name="1.3B + Cache",
            description="All 1.3B with activation caching",
            model="baseline_1.3B",
            steps=total_steps,
            caching=True,
            cache_start=10,
            cache_end=min(40, total_steps - 5),
            cache_interval=3,
            category="caching",
        ))

    # ── Category 5: Step count variations (with 30/70 split) ──
    if not quick:
        for step_count in [30, 40]:
            s14 = int(step_count * 0.3)
            s13 = step_count - s14
            configs.append(StrategyConfig(
                name=f"Split 30/70 @ {step_count} steps",
                description=f"30/70 split at {step_count} total steps",
                model="hybrid",
                steps=step_count,
                schedule=[("14B", s14), ("1.3B", s13)],
                category="steps",
            ))

    return configs


# =============================================================================
# API Client
# =============================================================================

class APIClient:
    """Client for the Wan2.1 generation API."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def health(self) -> dict:
        resp = self.session.get(f"{self.base_url}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def submit(self, prompt: str, config: StrategyConfig, seed: int) -> str:
        payload = {
            "prompt": prompt,
            "model": config.model,
            "sampling_steps": config.steps,
            "frame_count": 81,
            "fps": 16,
            "width": 832,
            "height": 480,
            "seed": seed,
            "enable_caching": config.caching,
        }
        if config.schedule:
            payload["schedule"] = config.schedule_json()
        if config.caching:
            payload["cache_start_step"] = config.cache_start
            payload["cache_end_step"] = config.cache_end
            payload["cache_interval"] = config.cache_interval

        resp = self.session.post(
            f"{self.base_url}/generate", json=payload, timeout=30
        )
        resp.raise_for_status()
        return resp.json()["job_id"]

    def status(self, job_id: str) -> dict:
        resp = self.session.get(f"{self.base_url}/status/{job_id}", timeout=30)
        resp.raise_for_status()
        return resp.json()

    def download(self, job_id: str, path: str):
        resp = self.session.get(
            f"{self.base_url}/video/{job_id}", timeout=300, stream=True
        )
        resp.raise_for_status()
        with open(path, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)

    def wait(self, job_id: str, poll_interval: int = 5, timeout: int = 600) -> dict:
        start = time.time()
        while time.time() - start < timeout:
            st = self.status(job_id)
            state = st.get("status")
            progress = st.get("progress", 0)
            model = st.get("model_in_use", "")

            bar_w = 30
            filled = int(bar_w * progress / 100)
            bar = "\u2588" * filled + "\u2591" * (bar_w - filled)
            model_str = f" [{model}]" if model else ""
            print(f"\r    [{bar}] {progress}%{model_str}    ", end="", flush=True)

            if state == "completed":
                print()
                return st
            elif state == "failed":
                print()
                raise RuntimeError(f"Job failed: {st.get('error', 'unknown')}")
            time.sleep(poll_interval)
        raise TimeoutError(f"Job {job_id} timed out after {timeout}s")


# =============================================================================
# CLIP Evaluator (reused from existing experiments)
# =============================================================================

class CLIPEvaluator:
    def __init__(self):
        if not CLIP_AVAILABLE:
            raise RuntimeError("CLIP not available")
        print("  Loading CLIP model...")
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.model.to(self.device).eval()
        print(f"  CLIP loaded on {self.device}")

    def get_video_embedding(self, video_path: str, num_frames: int = 16):
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            raise ValueError(f"No frames from {video_path}")

        embeddings = []
        with torch.no_grad():
            for i in range(0, len(frames), 4):
                batch = frames[i : i + 4]
                inputs = self.processor(images=batch, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                feat = self.model.get_image_features(**inputs)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                embeddings.append(feat.cpu().numpy())

        all_emb = np.concatenate(embeddings, axis=0)
        avg = all_emb.mean(axis=0)
        return avg / np.linalg.norm(avg)

    def similarity(self, path1: str, path2: str) -> float:
        return float(np.dot(self.get_video_embedding(path1), self.get_video_embedding(path2)))


# =============================================================================
# Cost Estimation
# =============================================================================

def estimate_cost(generation_time: float, config: StrategyConfig, mode: str = "single_h100") -> dict:
    """
    Estimate GPU cost for a generation run.

    Modes:
      - single_h100: everything on one H100 on-demand
      - single_h100_spot: everything on one H100 spot
      - distributed_spot: 14B on H100 spot, 1.3B on RTX4090 stable
      - distributed_a40: 14B on 2xA40 spot, 1.3B on RTX4090 stable
    """
    hr = generation_time / 3600.0

    if mode == "single_h100":
        return {"cost_usd": hr * COST_RATES["H100"], "gpu": "H100 on-demand"}
    elif mode == "single_h100_spot":
        return {"cost_usd": hr * COST_RATES["H100_spot"], "gpu": "H100 spot"}
    elif mode == "distributed_spot":
        # Coordinator always on, worker only during 14B segments
        if config.schedule:
            total = config.steps
            steps_14b = sum(s for m, s in config.schedule if m == "14B")
            frac_14b = steps_14b / total if total > 0 else 0
        elif config.model == "baseline_14B":
            frac_14b = 1.0
        else:
            frac_14b = 0.0

        coord_cost = hr * COST_RATES["RTX4090"]
        worker_cost = hr * frac_14b * COST_RATES["H100_spot"]
        return {
            "cost_usd": coord_cost + worker_cost,
            "gpu": f"RTX4090 + H100 spot ({frac_14b*100:.0f}%)",
            "coordinator_cost": coord_cost,
            "worker_cost": worker_cost,
        }
    elif mode == "distributed_a40":
        if config.schedule:
            total = config.steps
            steps_14b = sum(s for m, s in config.schedule if m == "14B")
            frac_14b = steps_14b / total if total > 0 else 0
        elif config.model == "baseline_14B":
            frac_14b = 1.0
        else:
            frac_14b = 0.0

        coord_cost = hr * COST_RATES["A4000"]
        worker_cost = hr * frac_14b * COST_RATES["A40x2_spot"]
        return {
            "cost_usd": coord_cost + worker_cost,
            "gpu": f"A4000 + 2xA40 spot ({frac_14b*100:.0f}%)",
            "coordinator_cost": coord_cost,
            "worker_cost": worker_cost,
        }

    return {"cost_usd": 0, "gpu": "unknown"}


# =============================================================================
# HTML Report
# =============================================================================

def generate_report(results: list, prompts_used: list, output_path: str):
    """Generate comprehensive HTML report with latency/cost/quality charts."""

    # Find baseline
    baseline_time = None
    for r in results:
        if r.get("is_baseline") and r.get("status") == "completed":
            baseline_time = r["generation_time"]
            break

    # Build table rows
    table_rows = ""
    for r in results:
        if r.get("status") != "completed":
            continue
        name = r["name"]
        gen_t = r["generation_time"]
        speedup = f'{baseline_time / gen_t:.2f}x' if baseline_time else "-"
        clip_s = f'{r["clip_similarity"]:.4f}' if r.get("clip_similarity") else "-"
        cost_h100 = f'${r["cost_h100"]["cost_usd"]*100:.2f}c' if r.get("cost_h100") else "-"
        cost_dist = f'${r["cost_distributed"]["cost_usd"]*100:.2f}c' if r.get("cost_distributed") else "-"
        savings = ""
        if r.get("cost_h100") and r.get("cost_distributed"):
            s = (1 - r["cost_distributed"]["cost_usd"] / r["cost_h100"]["cost_usd"]) * 100
            savings = f'{s:.0f}%' if s > 0 else "-"
        cache_str = f'{r.get("cache_hit_rate", 0)*100:.0f}%' if r.get("cache_hit_rate") else "-"
        cat = r.get("category", "")
        baseline_mark = "ref" if r.get("is_baseline") else ""

        speedup_cls = ""
        if baseline_time and gen_t < baseline_time:
            ratio = baseline_time / gen_t
            speedup_cls = "speedup-great" if ratio >= 2 else ("speedup-good" if ratio >= 1.5 else "")

        sim_cls = ""
        if r.get("clip_similarity"):
            v = r["clip_similarity"]
            sim_cls = "sim-great" if v >= 0.95 else ("sim-good" if v >= 0.90 else ("sim-poor" if v < 0.85 else ""))

        table_rows += f"""
        <tr>
            <td>{name}</td>
            <td>{cat}</td>
            <td>{r.get('steps', 50)}</td>
            <td><strong>{gen_t:.1f}s</strong></td>
            <td class="{speedup_cls}">{speedup}</td>
            <td>{cache_str}</td>
            <td class="{sim_cls}">{clip_s}</td>
            <td>{cost_h100}</td>
            <td>{cost_dist}</td>
            <td>{savings}</td>
            <td>{baseline_mark}</td>
        </tr>"""

    # Chart data
    completed = [r for r in results if r.get("status") == "completed"]
    chart_names = [r["name"][:25] for r in completed]
    chart_times = [r["generation_time"] for r in completed]
    chart_clips = [r.get("clip_similarity", 0) or 0 for r in completed]
    chart_cost_h100 = [r.get("cost_h100", {}).get("cost_usd", 0) * 100 for r in completed]
    chart_cost_dist = [r.get("cost_distributed", {}).get("cost_usd", 0) * 100 for r in completed]
    chart_colors = ['#4CAF50' if r.get("is_baseline") else '#58a6ff' for r in completed]

    # Pareto frontier data (time vs quality)
    pareto_data = json.dumps([
        {"x": r["generation_time"], "y": r.get("clip_similarity", 0) or 0, "label": r["name"][:20]}
        for r in completed
    ])

    prompts_html = "<br>".join(f'{p["id"]}: "{p["text"][:80]}..."' for p in prompts_used)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Hybrid Compute Strategy Experiments</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg: #0d1117; --bg2: #161b22; --bg3: #21262d;
            --txt: #c9d1d9; --txt2: #8b949e;
            --green: #3fb950; --blue: #58a6ff; --yellow: #d29922; --red: #f85149;
            --border: #30363d;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: 'SF Mono', monospace; background: var(--bg); color: var(--txt); padding: 2rem; line-height: 1.6; }}
        .container {{ max-width: 1500px; margin: 0 auto; }}
        h1 {{ font-size: 1.8rem; color: var(--blue); margin-bottom: .3rem; }}
        h2 {{ font-size: 1.2rem; color: var(--green); margin: 2rem 0 1rem; border-bottom: 1px solid var(--border); padding-bottom: .5rem; }}
        .sub {{ color: var(--txt2); margin-bottom: 1.5rem; font-size: .9rem; }}
        .info-box {{ background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; padding: 1rem; margin-bottom: 1.5rem; font-size: .85rem; }}
        .info-box strong {{ color: var(--yellow); }}
        table {{ width: 100%; border-collapse: collapse; background: var(--bg2); border-radius: 8px; overflow: hidden; font-size: .85rem; }}
        th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid var(--border); }}
        th {{ background: var(--bg3); color: var(--txt2); font-weight: 600; text-transform: uppercase; font-size: .7rem; letter-spacing: .5px; }}
        tr:hover {{ background: var(--bg3); }}
        .speedup-great {{ color: var(--green); font-weight: bold; }}
        .speedup-good {{ color: var(--yellow); }}
        .sim-great {{ color: var(--green); font-weight: bold; }}
        .sim-good {{ color: var(--yellow); }}
        .sim-poor {{ color: var(--red); }}
        .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin: 1.5rem 0; }}
        .chart-box {{ background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; padding: 1.2rem; }}
        .chart-box.full {{ grid-column: 1 / -1; }}
        .ts {{ color: var(--txt2); font-size: .75rem; margin-top: 2rem; }}
        .key-findings {{ background: var(--bg2); border-left: 3px solid var(--green); padding: 1rem 1.5rem; margin: 1.5rem 0; border-radius: 0 8px 8px 0; }}
        .key-findings li {{ margin: .3rem 0; }}
    </style>
</head>
<body>
<div class="container">
    <h1>Hybrid Compute Strategy Experiments</h1>
    <p class="sub">Comparing latency, cost, and quality across {len(completed)} configurations</p>

    <div class="info-box">
        <strong>Prompts:</strong><br>{prompts_html}<br>
        <strong>Seed:</strong> {EXPERIMENT_SEED} | <strong>Frames:</strong> 81 | <strong>Resolution:</strong> 832x480
    </div>

    <h2>Results Table</h2>
    <table>
        <thead><tr>
            <th>Strategy</th><th>Category</th><th>Steps</th><th>Latency</th><th>Speedup</th>
            <th>Cache</th><th>CLIP Sim</th><th>Cost (H100)</th><th>Cost (Dist)</th><th>Savings</th><th>Ref</th>
        </tr></thead>
        <tbody>{table_rows}</tbody>
    </table>

    <h2>Charts</h2>
    <div class="charts">
        <div class="chart-box"><canvas id="latencyChart"></canvas></div>
        <div class="chart-box"><canvas id="qualityChart"></canvas></div>
        <div class="chart-box"><canvas id="costChart"></canvas></div>
        <div class="chart-box"><canvas id="paretoChart"></canvas></div>
    </div>

    <p class="ts">Generated: {datetime.now().isoformat()}</p>
</div>
<script>
    const names = {json.dumps(chart_names)};
    const colors = {json.dumps(chart_colors)};

    new Chart(document.getElementById('latencyChart'), {{
        type: 'bar',
        data: {{ labels: names, datasets: [{{ label: 'Seconds', data: {json.dumps(chart_times)}, backgroundColor: colors, borderRadius: 4 }}] }},
        options: {{ responsive: true, plugins: {{ title: {{ display: true, text: 'Generation Latency (seconds)', color: '#c9d1d9' }}, legend: {{ display: false }} }},
            scales: {{ y: {{ beginAtZero: true, grid: {{ color: '#30363d' }}, ticks: {{ color: '#8b949e' }} }}, x: {{ grid: {{ display: false }}, ticks: {{ color: '#8b949e', maxRotation: 60, font: {{ size: 10 }} }} }} }} }}
    }});

    new Chart(document.getElementById('qualityChart'), {{
        type: 'bar',
        data: {{ labels: names, datasets: [{{ label: 'CLIP Similarity', data: {json.dumps(chart_clips)},
            backgroundColor: {json.dumps(chart_clips)}.map(v => v >= 0.95 ? '#3fb950' : v >= 0.90 ? '#d29922' : '#58a6ff'), borderRadius: 4 }}] }},
        options: {{ responsive: true, plugins: {{ title: {{ display: true, text: 'Quality (CLIP Similarity to 14B Baseline)', color: '#c9d1d9' }}, legend: {{ display: false }} }},
            scales: {{ y: {{ min: 0.7, max: 1.0, grid: {{ color: '#30363d' }}, ticks: {{ color: '#8b949e' }} }}, x: {{ grid: {{ display: false }}, ticks: {{ color: '#8b949e', maxRotation: 60, font: {{ size: 10 }} }} }} }} }}
    }});

    new Chart(document.getElementById('costChart'), {{
        type: 'bar',
        data: {{ labels: names, datasets: [
            {{ label: 'H100 on-demand', data: {json.dumps(chart_cost_h100)}, backgroundColor: '#f85149', borderRadius: 4 }},
            {{ label: 'Distributed (spot)', data: {json.dumps(chart_cost_dist)}, backgroundColor: '#3fb950', borderRadius: 4 }}
        ] }},
        options: {{ responsive: true, plugins: {{ title: {{ display: true, text: 'Cost per Video (cents)', color: '#c9d1d9' }}, legend: {{ labels: {{ color: '#8b949e' }} }} }},
            scales: {{ y: {{ beginAtZero: true, grid: {{ color: '#30363d' }}, ticks: {{ color: '#8b949e' }} }}, x: {{ grid: {{ display: false }}, ticks: {{ color: '#8b949e', maxRotation: 60, font: {{ size: 10 }} }} }} }} }}
    }});

    const paretoData = {pareto_data};
    new Chart(document.getElementById('paretoChart'), {{
        type: 'scatter',
        data: {{ datasets: [{{ label: 'Strategies', data: paretoData.map(d => ({{ x: d.x, y: d.y }})),
            backgroundColor: paretoData.map(d => d.y >= 0.95 ? '#3fb950' : '#58a6ff'), pointRadius: 8 }}] }},
        options: {{ responsive: true, plugins: {{
            title: {{ display: true, text: 'Pareto: Latency vs Quality', color: '#c9d1d9' }},
            tooltip: {{ callbacks: {{ label: (ctx) => paretoData[ctx.dataIndex].label + ': ' + ctx.parsed.x.toFixed(1) + 's, CLIP=' + ctx.parsed.y.toFixed(4) }} }}
        }}, scales: {{
            x: {{ title: {{ display: true, text: 'Latency (s)', color: '#8b949e' }}, grid: {{ color: '#30363d' }}, ticks: {{ color: '#8b949e' }} }},
            y: {{ title: {{ display: true, text: 'CLIP Similarity', color: '#8b949e' }}, min: 0.7, max: 1.0, grid: {{ color: '#30363d' }}, ticks: {{ color: '#8b949e' }} }}
        }} }}
    }});
</script>
</body></html>"""

    with open(output_path, "w") as f:
        f.write(html)
    print(f"  Report saved: {output_path}")


# =============================================================================
# Main Runner
# =============================================================================

def run_single(client: APIClient, config: StrategyConfig, prompt: dict,
               seed: int, output_dir: Path) -> dict:
    """Run a single experiment, download video, return result dict."""
    job_id = client.submit(prompt["text"], config, seed)
    print(f"    Job: {job_id}")

    final = client.wait(job_id)

    gen_time = final.get("generation_time", 0)
    cache_stats = final.get("cache_statistics", {})
    cache_hit_rate = cache_stats.get("cache_hit_rate") if cache_stats else None

    # Download video
    sched_str = ""
    if config.schedule:
        sched_str = "_".join(f"{m}{s}" for m, s in config.schedule)
    else:
        sched_str = config.model
    safe_name = f"{prompt['id']}_{sched_str}_{config.steps}s"
    if config.caching:
        safe_name += "_cache"
    video_path = output_dir / f"{safe_name}.mp4"
    client.download(job_id, str(video_path))

    return {
        "name": config.name,
        "description": config.description,
        "category": config.category,
        "model": config.model,
        "steps": config.steps,
        "schedule": config.schedule,
        "caching": config.caching,
        "is_baseline": config.is_baseline,
        "prompt_id": prompt["id"],
        "job_id": job_id,
        "generation_time": gen_time,
        "cache_hit_rate": cache_hit_rate,
        "video_path": str(video_path),
        "status": "completed",
        "distributed_stats": cache_stats.get("segments_on_worker") if cache_stats else None,
        "cost_h100": estimate_cost(gen_time, config, "single_h100"),
        "cost_distributed": estimate_cost(gen_time, config, "distributed_spot"),
        "completed_at": datetime.now().isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid compute strategy experiments for Wan2.1"
    )
    parser.add_argument("--server", required=True, help="API server URL")
    parser.add_argument("--output", default=str(EXPERIMENT_DIR), help="Output dir")
    parser.add_argument("--steps", type=int, default=50, help="Total sampling steps")
    parser.add_argument("--quick", action="store_true", help="Run reduced set (fewer splits)")
    parser.add_argument("--prompts", type=int, default=1, help="Number of prompts (1-3)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing results")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"

    # Build configs
    configs = build_experiment_configs(args.steps, args.quick)
    prompts_to_use = PROMPTS[: args.prompts]

    print()
    print("=" * 60)
    print("  HYBRID COMPUTE STRATEGY EXPERIMENTS")
    print("=" * 60)
    print(f"  Server:      {args.server}")
    print(f"  Strategies:  {len(configs)}")
    print(f"  Prompts:     {len(prompts_to_use)}")
    print(f"  Total runs:  {len(configs) * len(prompts_to_use)}")
    print(f"  Steps:       {args.steps}")
    print(f"  Output:      {output_dir}")
    print("=" * 60)
    print()

    client = APIClient(args.server)

    # Health check
    print("Checking server...")
    try:
        health = client.health()
        print(f"  Server OK. Models: {health.get('models_loaded', [])}")
    except Exception as e:
        print(f"  Server unreachable: {e}")
        sys.exit(1)

    # Resume or start fresh
    results = []
    completed_keys = set()
    if args.resume and results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        completed_keys = {
            (r["name"], r.get("prompt_id", "")) for r in results if r.get("status") == "completed"
        }
        print(f"  Resuming: {len(completed_keys)} experiments already done")

    # Load CLIP evaluator
    clip_eval = None
    if CLIP_AVAILABLE:
        try:
            clip_eval = CLIPEvaluator()
        except Exception as e:
            print(f"  CLIP unavailable: {e}")

    # Run experiments
    total = len(configs) * len(prompts_to_use)
    run_idx = 0

    for prompt in prompts_to_use:
        for config in configs:
            run_idx += 1
            key = (config.name, prompt["id"])

            if key in completed_keys:
                print(f"[{run_idx}/{total}] SKIP (done): {config.name} / {prompt['id']}")
                continue

            print(f"\n[{run_idx}/{total}] {config.name} / {prompt['id']}")
            print(f"  {config.description}")

            try:
                result = run_single(client, config, prompt, EXPERIMENT_SEED, output_dir)
                results.append(result)

                gen_t = result["generation_time"]
                cost_h = result["cost_h100"]["cost_usd"] * 100
                cost_d = result["cost_distributed"]["cost_usd"] * 100
                print(f"    Time: {gen_t:.1f}s | H100: {cost_h:.2f}c | Dist: {cost_d:.2f}c")

            except Exception as e:
                print(f"    FAILED: {e}")
                results.append({
                    "name": config.name,
                    "category": config.category,
                    "prompt_id": prompt["id"],
                    "status": "failed",
                    "error": str(e),
                })

            # Save intermediate
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)

    # Compute CLIP similarities (per prompt, relative to 14B baseline)
    print("\nComputing CLIP similarities...")
    for prompt in prompts_to_use:
        prompt_results = [
            r for r in results
            if r.get("prompt_id") == prompt["id"] and r.get("status") == "completed"
        ]
        baseline_vid = None
        for r in prompt_results:
            if r.get("is_baseline") and os.path.exists(r.get("video_path", "")):
                baseline_vid = r["video_path"]
                r["clip_similarity"] = 1.0
                break

        if baseline_vid and clip_eval:
            for r in prompt_results:
                if r.get("is_baseline") or r.get("clip_similarity"):
                    continue
                vid = r.get("video_path", "")
                if os.path.exists(vid):
                    try:
                        sim = clip_eval.similarity(baseline_vid, vid)
                        r["clip_similarity"] = sim
                        print(f"  {r['name']} / {prompt['id']}: {sim:.4f}")
                    except Exception as e:
                        print(f"  {r['name']}: CLIP error: {e}")

    # If multiple prompts, average results per strategy
    if len(prompts_to_use) > 1:
        print("\nAveraging across prompts...")
        by_name = {}
        for r in results:
            if r.get("status") != "completed":
                continue
            name = r["name"]
            if name not in by_name:
                by_name[name] = []
            by_name[name].append(r)

        averaged = []
        for name, runs in by_name.items():
            avg = dict(runs[0])  # Copy first run as template
            avg["generation_time"] = np.mean([r["generation_time"] for r in runs])
            clips = [r["clip_similarity"] for r in runs if r.get("clip_similarity")]
            avg["clip_similarity"] = np.mean(clips) if clips else None
            avg["prompt_id"] = "averaged"
            avg["num_prompts"] = len(runs)
            # Recompute costs with averaged time
            cfg = StrategyConfig(name=name, description="", model=avg["model"],
                                 steps=avg["steps"], schedule=avg.get("schedule"))
            avg["cost_h100"] = estimate_cost(avg["generation_time"], cfg, "single_h100")
            avg["cost_distributed"] = estimate_cost(avg["generation_time"], cfg, "distributed_spot")
            averaged.append(avg)
        results.extend(averaged)

    # Save final results
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Generate report (use first prompt or averaged)
    report_results = [r for r in results if r.get("prompt_id") == prompts_to_use[0]["id"] and r.get("status") == "completed"]
    if not report_results:
        report_results = [r for r in results if r.get("status") == "completed"]

    report_path = output_dir / "report.html"
    generate_report(report_results, prompts_to_use, str(report_path))

    # Print summary
    print("\n" + "=" * 80)
    print("  RESULTS SUMMARY")
    print("=" * 80)
    print(f"  {'Strategy':<30} | {'Time':>7} | {'CLIP':>6} | {'H100':>8} | {'Dist':>8} | {'Save':>5}")
    print(f"  {'-'*75}")

    for r in sorted(report_results, key=lambda x: x.get("generation_time", 999)):
        name = r["name"][:30]
        t = f'{r["generation_time"]:.1f}s'
        clip_s = f'{r["clip_similarity"]:.4f}' if r.get("clip_similarity") else "-"
        ch = f'{r["cost_h100"]["cost_usd"]*100:.2f}c' if r.get("cost_h100") else "-"
        cd = f'{r["cost_distributed"]["cost_usd"]*100:.2f}c' if r.get("cost_distributed") else "-"
        sav = ""
        if r.get("cost_h100") and r.get("cost_distributed") and r["cost_h100"]["cost_usd"] > 0:
            sav = f'{(1 - r["cost_distributed"]["cost_usd"]/r["cost_h100"]["cost_usd"])*100:.0f}%'
        print(f"  {name:<30} | {t:>7} | {clip_s:>6} | {ch:>8} | {cd:>8} | {sav:>5}")

    print(f"  {'-'*75}")
    print(f"\n  Report: {report_path}")
    print(f"  Data:   {results_path}")
    print()


if __name__ == "__main__":
    main()
