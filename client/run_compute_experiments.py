#!/usr/bin/env python3
"""
Hybrid Compute Experiment Orchestrator

Spins up RunPod instances, runs experiments comparing different GPU
configurations for hybrid inference, downloads results, generates report.

Experiment matrix:
  1. H100 single-GPU baseline (14B only)
  2-3. H100 local hybrid (30/70, 20/80)
  4-5. Distributed: RTX4090 + H100 spot (30/70, 20/80)
  6-7. Distributed: RTX4090 + A100 spot (30/70, 20/80)
  8-9. Distributed: RTX4090 + 2xA40 spot (30/70, 20/80)
  10.  RTX4090 1.3B only (quality floor)

Usage:
    export RUNPOD_API_KEY=rpa_xxx
    python client/run_compute_experiments.py

    # Dry run (no pods created):
    python client/run_compute_experiments.py --dry-run

    # Skip phases you've already done:
    python client/run_compute_experiments.py --resume
"""

import argparse
import atexit
import json
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
import numpy as np

# CLIP for quality comparison
try:
    import torch
    import cv2
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Note: CLIP not available locally. Quality metrics skipped.")

# =============================================================================
# Configuration
# =============================================================================

RUNPOD_API_URL = "https://api.runpod.io/graphql"
GIT_BRANCH = "distributed-inference"
VOLUME_ID = os.environ.get("RUNPOD_VOLUME_ID", "")  # Set to "sy8ts7nyu2" for athul-dev, or empty for any datacenter
DOCKER_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"

EXPERIMENT_SEED = 42
EXPERIMENT_PROMPT = (
    "The Merced river is overflowing, birds flying in the sky, "
    "camera is zooming out to reveal an American Buffalo bathing in the river"
)
OUTPUT_DIR = Path(__file__).parent.parent / "experiments" / "compute_comparison"

# GPU cost rates $/hr
COST_RATES = {
    "NVIDIA H100 80GB HBM3": {"on_demand": 3.50, "spot": 1.75},
    "NVIDIA A100 80GB PCIe": {"on_demand": 2.50, "spot": 1.50},
    "NVIDIA A100-SXM4-80GB": {"on_demand": 2.70, "spot": 1.60},
    "NVIDIA A40": {"on_demand": 0.79, "spot": 0.70},
    "NVIDIA GeForce RTX 4090": {"on_demand": 0.44, "spot": 0.39},
}

SERVER_PORT = 8888
WORKER_PORT = 8889

# =============================================================================
# RunPod API Client
# =============================================================================

class RunPodClient:
    """Manages RunPod pods via GraphQL API."""

    def __init__(self, api_key: str, volume_id: str = ""):
        self.api_key = api_key
        self.volume_id = volume_id
        self._active_pods = []  # Track for cleanup

    def _gql(self, query: str, variables: dict = None) -> dict:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"query": query}
        if variables:
            payload["variables"] = variables
        resp = requests.post(RUNPOD_API_URL, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if "errors" in data:
            raise RuntimeError(f"RunPod API error: {data['errors']}")
        return data.get("data", {})

    def create_pod(
        self,
        name: str,
        gpu_type: str,
        gpu_count: int = 1,
        cloud_type: str = "SPOT",
        startup_cmd: str = "",
    ) -> str:
        """Create a pod, return pod_id."""
        query = """
        mutation($input: PodFindAndDeployOnDemandInput!) {
            podFindAndDeployOnDemand(input: $input) { id desiredStatus machineId }
        }"""
        inp = {
            "name": name,
            "gpuTypeId": gpu_type,
            "gpuCount": gpu_count,
            "cloudType": cloud_type,
            "dockerArgs": startup_cmd,
            "volumeInGb": 150,  # Local disk for model weights
            "containerDiskInGb": 20,
            "minVcpuCount": 4,
            "minMemoryInGb": 32,
            "imageName": DOCKER_IMAGE,
        }
        if self.volume_id:
            inp["networkVolumeId"] = self.volume_id
            inp["volumeInGb"] = 0  # Use network volume instead

        data = self._gql(query, {"input": inp})
        pod = data.get("podFindAndDeployOnDemand", {})
        pod_id = pod.get("id")
        if not pod_id:
            raise RuntimeError(f"Failed to create pod: {data}")
        self._active_pods.append(pod_id)
        print(f"    Pod created: {pod_id} ({gpu_type} x{gpu_count}, {cloud_type})")
        return pod_id

    def terminate_pod(self, pod_id: str):
        """Terminate (delete) a pod."""
        query = """mutation($podId: String!) { podTerminate(input: {podId: $podId}) }"""
        try:
            self._gql(query, {"podId": pod_id})
            if pod_id in self._active_pods:
                self._active_pods.remove(pod_id)
            print(f"    Pod terminated: {pod_id}")
        except Exception as e:
            print(f"    Warning: failed to terminate {pod_id}: {e}")

    def terminate_all(self):
        """Terminate all tracked pods (for cleanup)."""
        for pod_id in list(self._active_pods):
            self.terminate_pod(pod_id)

    def get_pod_url(self, pod_id: str, port: int) -> str:
        return f"https://{pod_id}-{port}.proxy.runpod.net"

    def wait_for_ready(self, pod_id: str, port: int, timeout: int = 600) -> str:
        """Wait for pod to be ready and return its URL."""
        url = self.get_pod_url(pod_id, port)
        health_url = f"{url}/health"
        start = time.time()
        print(f"    Waiting for {url} ...", end="", flush=True)

        while time.time() - start < timeout:
            try:
                resp = requests.get(health_url, timeout=10)
                if resp.status_code == 200:
                    print(f" ready ({time.time()-start:.0f}s)")
                    return url
            except Exception:
                pass
            print(".", end="", flush=True)
            time.sleep(15)

        print(f" TIMEOUT after {timeout}s")
        raise TimeoutError(f"Pod {pod_id} not ready after {timeout}s")


# =============================================================================
# Generation API Client
# =============================================================================

class GenerationClient:
    """Client for the Wan2.1 generation API."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def health(self) -> dict:
        return self.session.get(f"{self.base_url}/health", timeout=10).json()

    def set_worker_url(self, worker_url: str):
        resp = self.session.post(
            f"{self.base_url}/admin/set-worker-url",
            json={"worker_url": worker_url},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def disable_distributed(self):
        resp = self.session.post(
            f"{self.base_url}/admin/disable-distributed", timeout=10
        )
        resp.raise_for_status()
        return resp.json()

    def submit(self, prompt, model, steps, schedule=None, seed=42) -> str:
        payload = {
            "prompt": prompt,
            "model": model,
            "sampling_steps": steps,
            "frame_count": 81,
            "fps": 16,
            "width": 832,
            "height": 480,
            "seed": seed,
        }
        if schedule:
            payload["schedule"] = schedule
        resp = self.session.post(
            f"{self.base_url}/generate", json=payload, timeout=30
        )
        resp.raise_for_status()
        return resp.json()["job_id"]

    def wait(self, job_id: str, timeout: int = 900) -> dict:
        start = time.time()
        while time.time() - start < timeout:
            resp = self.session.get(
                f"{self.base_url}/status/{job_id}", timeout=30
            )
            st = resp.json()
            progress = st.get("progress", 0)
            model = st.get("model_in_use", "")
            bar_w = 25
            filled = int(bar_w * progress / 100)
            bar = "\u2588" * filled + "\u2591" * (bar_w - filled)
            print(f"\r      [{bar}] {progress}% {model:20s}", end="", flush=True)

            if st.get("status") == "completed":
                print()
                return st
            elif st.get("status") == "failed":
                print()
                raise RuntimeError(f"Job failed: {st.get('error')}")
            time.sleep(5)
        raise TimeoutError(f"Job {job_id} timed out")

    def download(self, job_id: str, path: str):
        resp = self.session.get(
            f"{self.base_url}/video/{job_id}", timeout=300, stream=True
        )
        resp.raise_for_status()
        with open(path, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)


# =============================================================================
# CLIP Evaluator
# =============================================================================

class CLIPEvaluator:
    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.model.to(self.device).eval()

    def video_embedding(self, path: str, n_frames=16):
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total - 1, n_frames, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            raise ValueError(f"No frames: {path}")
        embs = []
        with torch.no_grad():
            for i in range(0, len(frames), 4):
                inp = self.processor(images=frames[i:i+4], return_tensors="pt")
                inp = {k: v.to(self.device) for k, v in inp.items()}
                f = self.model.get_image_features(**inp)
                embs.append((f / f.norm(dim=-1, keepdim=True)).cpu().numpy())
        avg = np.concatenate(embs).mean(axis=0)
        return avg / np.linalg.norm(avg)

    def similarity(self, p1, p2):
        return float(np.dot(self.video_embedding(p1), self.video_embedding(p2)))


# =============================================================================
# Startup command builder
# =============================================================================

def _startup_cmd(role: str, extra_env: str = "", use_volume: bool = False) -> str:
    """Build pod startup command wrapped in bash -c for RunPod dockerArgs."""
    # Step 1: Clone repo if needed, checkout branch
    setup = (
        "if [ ! -d /workspace/wan2.1/Wan2.1/.git ]; then "
        "  mkdir -p /workspace/wan2.1 && cd /workspace/wan2.1 && "
        "  git clone https://github.com/athulramkumar/Wan2.1.git 2>&1; "
        "fi && "
        "cd /workspace/wan2.1/Wan2.1 && "
        f"git fetch origin 2>/dev/null; git checkout {GIT_BRANCH} 2>/dev/null; git pull origin {GIT_BRANCH} 2>/dev/null; "
        "pip install fastapi uvicorn requests pydantic huggingface_hub -q 2>/dev/null; "
    )

    # Step 2: Download model weights if not already present
    if not use_volume:
        # Determine which models this role needs
        if role in ("server_both",):
            # Need both 14B and 1.3B
            setup += (
                "if [ ! -d Wan2.1-T2V-1.3B ]; then "
                "  echo 'Downloading 1.3B model...' && "
                "  huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir Wan2.1-T2V-1.3B 2>&1; "
                "fi && "
                "if [ ! -d Wan2.1-T2V-14B ]; then "
                "  echo 'Downloading 14B model...' && "
                "  huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir Wan2.1-T2V-14B 2>&1; "
                "fi && "
            )
        elif role in ("coordinator",):
            # Only need 1.3B
            setup += (
                "if [ ! -d Wan2.1-T2V-1.3B ]; then "
                "  echo 'Downloading 1.3B model...' && "
                "  huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir Wan2.1-T2V-1.3B 2>&1; "
                "fi && "
            )
        elif role in ("worker", "worker_multigpu"):
            # Only need 14B
            setup += (
                "if [ ! -d Wan2.1-T2V-14B ]; then "
                "  echo 'Downloading 14B model...' && "
                "  huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir Wan2.1-T2V-14B 2>&1; "
                "fi && "
            )

    if role == "server_both":
        cmd = setup + "python run_server.py --port 8888"
    elif role == "coordinator":
        cmd = setup + f"{extra_env} python run_server.py --port 8888 --distributed --mode coordinator"
    elif role == "worker":
        cmd = setup + "python run_worker.py --port 8889"
    elif role == "worker_multigpu":
        cmd = setup + "python run_worker.py --port 8889 --auto-multi-gpu"
    else:
        raise ValueError(f"Unknown role: {role}")
    return f'bash -c "{cmd}"'


# =============================================================================
# Experiment definitions
# =============================================================================

@dataclass
class Experiment:
    id: int
    name: str
    phase: str
    model: str
    schedule: Optional[list] = None
    compute_desc: str = ""
    gpu_type: str = ""
    cloud_type: str = ""
    distributed: bool = False

EXPERIMENTS = [
    # Phase A: H100 single-GPU
    Experiment(1, "Baseline 14B (H100)", "A", "baseline_14B",
              compute_desc="Single H100", gpu_type="NVIDIA H100 80GB HBM3"),
    Experiment(2, "Hybrid 30/70 local (H100)", "A", "hybrid",
              schedule=[["14B", 15], ["1.3B", 35]],
              compute_desc="Single H100", gpu_type="NVIDIA H100 80GB HBM3"),
    Experiment(3, "Hybrid 20/80 local (H100)", "A", "hybrid",
              schedule=[["14B", 10], ["1.3B", 40]],
              compute_desc="Single H100", gpu_type="NVIDIA H100 80GB HBM3"),
    # Phase B: Distributed with H100 spot
    Experiment(4, "Dist 30/70 (RTX4090+H100)", "B", "hybrid",
              schedule=[["14B", 15], ["1.3B", 35]],
              compute_desc="RTX4090 + H100 spot", distributed=True),
    Experiment(5, "Dist 20/80 (RTX4090+H100)", "B", "hybrid",
              schedule=[["14B", 10], ["1.3B", 40]],
              compute_desc="RTX4090 + H100 spot", distributed=True),
    # Phase C: Distributed with A100 spot
    Experiment(6, "Dist 30/70 (RTX4090+A100)", "C", "hybrid",
              schedule=[["14B", 15], ["1.3B", 35]],
              compute_desc="RTX4090 + A100 spot", distributed=True),
    Experiment(7, "Dist 20/80 (RTX4090+A100)", "C", "hybrid",
              schedule=[["14B", 10], ["1.3B", 40]],
              compute_desc="RTX4090 + A100 spot", distributed=True),
    # Phase D: Distributed with 2xA40 spot
    Experiment(8, "Dist 30/70 (RTX4090+2xA40)", "D", "hybrid",
              schedule=[["14B", 15], ["1.3B", 35]],
              compute_desc="RTX4090 + 2xA40 spot", distributed=True),
    Experiment(9, "Dist 20/80 (RTX4090+2xA40)", "D", "hybrid",
              schedule=[["14B", 10], ["1.3B", 40]],
              compute_desc="RTX4090 + 2xA40 spot", distributed=True),
    # Phase E: 1.3B floor
    Experiment(10, "1.3B only (RTX4090)", "E", "baseline_1.3B",
              compute_desc="Single RTX4090", gpu_type="NVIDIA GeForce RTX 4090"),
]


# =============================================================================
# HTML Report
# =============================================================================

def generate_report(results: list, output_path: str):
    baseline_time = next(
        (r["generation_time"] for r in results if r["exp_id"] == 1 and r["status"] == "completed"), None
    )
    completed = [r for r in results if r["status"] == "completed"]

    rows = ""
    for r in completed:
        t = r["generation_time"]
        speedup = f'{baseline_time/t:.2f}x' if baseline_time and t > 0 else "-"
        clip = f'{r["clip_similarity"]:.4f}' if r.get("clip_similarity") else "-"
        cost = f'${r.get("cost_usd",0)*100:.1f}c' if r.get("cost_usd") else "-"
        savings = ""
        baseline_cost = next((x.get("cost_usd",0) for x in completed if x["exp_id"]==1), 0)
        if r.get("cost_usd") and baseline_cost:
            s = (1 - r["cost_usd"]/baseline_cost)*100
            savings = f'{s:.0f}%' if s > 0 else "-"
        sp_cls = "color:#3fb950;font-weight:bold" if baseline_time and t < baseline_time else ""
        cl_cls = "color:#3fb950" if r.get("clip_similarity",0) >= 0.95 else ("color:#d29922" if r.get("clip_similarity",0) >= 0.90 else "")
        rows += f'<tr><td>{r["name"]}</td><td>{r["compute_desc"]}</td><td>{t:.1f}s</td><td style="{sp_cls}">{speedup}</td><td style="{cl_cls}">{clip}</td><td>{cost}</td><td>{savings}</td></tr>\n'

    names_j = json.dumps([r["name"][:22] for r in completed])
    times_j = json.dumps([r["generation_time"] for r in completed])
    clips_j = json.dumps([r.get("clip_similarity",0) or 0 for r in completed])
    costs_j = json.dumps([r.get("cost_usd",0)*100 for r in completed])
    colors_j = json.dumps(["#4CAF50" if r["exp_id"]==1 else ("#f85149" if not r.get("distributed") else "#58a6ff") for r in completed])

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Hybrid Compute Experiments</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
:root{{--bg:#0d1117;--bg2:#161b22;--bg3:#21262d;--t:#c9d1d9;--t2:#8b949e;--g:#3fb950;--b:#58a6ff;--y:#d29922;--r:#f85149;--bd:#30363d}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'SF Mono',monospace;background:var(--bg);color:var(--t);padding:2rem;line-height:1.6}}
.c{{max-width:1400px;margin:0 auto}}
h1{{font-size:1.8rem;color:var(--b);margin-bottom:.3rem}}
h2{{font-size:1.2rem;color:var(--g);margin:2rem 0 1rem;border-bottom:1px solid var(--bd);padding-bottom:.5rem}}
.s{{color:var(--t2);margin-bottom:1.5rem}}
.info{{background:var(--bg2);border:1px solid var(--bd);border-radius:8px;padding:1rem;margin-bottom:1.5rem;font-size:.85rem}}
.info strong{{color:var(--y)}}
table{{width:100%;border-collapse:collapse;background:var(--bg2);border-radius:8px;overflow:hidden;font-size:.85rem}}
th,td{{padding:10px 14px;text-align:left;border-bottom:1px solid var(--bd)}}
th{{background:var(--bg3);color:var(--t2);font-weight:600;text-transform:uppercase;font-size:.7rem}}
tr:hover{{background:var(--bg3)}}
.g{{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;margin:1.5rem 0}}
.ch{{background:var(--bg2);border:1px solid var(--bd);border-radius:8px;padding:1.2rem}}
.ts{{color:var(--t2);font-size:.75rem;margin-top:2rem}}
</style></head><body><div class="c">
<h1>Hybrid Compute Experiments</h1>
<p class="s">Comparing GPU configurations for distributed hybrid inference</p>
<div class="info">
<strong>Prompt:</strong> {EXPERIMENT_PROMPT[:80]}...<br>
<strong>Seed:</strong> {EXPERIMENT_SEED} | <strong>Frames:</strong> 81 | <strong>Steps:</strong> 50 | <strong>Schedules:</strong> 30/70 and 20/80
</div>
<h2>Results</h2>
<table><thead><tr><th>Experiment</th><th>Compute</th><th>Latency</th><th>Speedup</th><th>CLIP Sim</th><th>Cost</th><th>Savings</th></tr></thead>
<tbody>{rows}</tbody></table>
<h2>Charts</h2>
<div class="g">
<div class="ch"><canvas id="c1"></canvas></div>
<div class="ch"><canvas id="c2"></canvas></div>
<div class="ch"><canvas id="c3"></canvas></div>
<div class="ch"><canvas id="c4"></canvas></div>
</div>
<p class="ts">Generated: {datetime.now().isoformat()}</p>
</div><script>
const N={names_j},T={times_j},CL={clips_j},CO={costs_j},COL={colors_j};
const o={{responsive:true,plugins:{{legend:{{display:false}}}},scales:{{y:{{beginAtZero:true,grid:{{color:'#30363d'}},ticks:{{color:'#8b949e'}}}},x:{{grid:{{display:false}},ticks:{{color:'#8b949e',maxRotation:60,font:{{size:9}}}}}}}}}};
new Chart(document.getElementById('c1'),{{type:'bar',data:{{labels:N,datasets:[{{label:'Seconds',data:T,backgroundColor:COL,borderRadius:4}}]}},options:{{...o,plugins:{{...o.plugins,title:{{display:true,text:'Generation Latency (s)',color:'#c9d1d9'}}}}}}}});
new Chart(document.getElementById('c2'),{{type:'bar',data:{{labels:N,datasets:[{{label:'CLIP',data:CL,backgroundColor:CL.map(v=>v>=.95?'#3fb950':v>=.9?'#d29922':'#58a6ff'),borderRadius:4}}]}},options:{{...o,scales:{{...o.scales,y:{{min:.7,max:1,grid:{{color:'#30363d'}},ticks:{{color:'#8b949e'}}}}}},plugins:{{...o.plugins,title:{{display:true,text:'Quality (CLIP Similarity)',color:'#c9d1d9'}}}}}}}});
new Chart(document.getElementById('c3'),{{type:'bar',data:{{labels:N,datasets:[{{label:'Cents',data:CO,backgroundColor:COL,borderRadius:4}}]}},options:{{...o,plugins:{{...o.plugins,title:{{display:true,text:'Cost per Video (cents)',color:'#c9d1d9'}}}}}}}});
const PD=N.map((n,i)=>({{x:T[i],y:CL[i],l:n}}));
new Chart(document.getElementById('c4'),{{type:'scatter',data:{{datasets:[{{data:PD.map(d=>({{x:d.x,y:d.y}})),backgroundColor:COL,pointRadius:8}}]}},options:{{responsive:true,plugins:{{title:{{display:true,text:'Pareto: Latency vs Quality',color:'#c9d1d9'}},tooltip:{{callbacks:{{label:c=>PD[c.dataIndex].l+': '+c.parsed.x.toFixed(1)+'s, CLIP='+c.parsed.y.toFixed(4)}}}}}},scales:{{x:{{title:{{display:true,text:'Latency (s)',color:'#8b949e'}},grid:{{color:'#30363d'}},ticks:{{color:'#8b949e'}}}},y:{{title:{{display:true,text:'CLIP Similarity',color:'#8b949e'}},min:.7,max:1,grid:{{color:'#30363d'}},ticks:{{color:'#8b949e'}}}}}}}}}});
</script></body></html>"""
    with open(output_path, "w") as f:
        f.write(html)
    print(f"  Report: {output_path}")


# =============================================================================
# Main Orchestrator
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Hybrid compute experiments")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without creating pods")
    parser.add_argument("--resume", action="store_true", help="Skip completed experiments")
    parser.add_argument("--volume-id", default=os.environ.get("RUNPOD_VOLUME_ID", ""),
                        help="RunPod network volume ID (athul-dev)")
    args = parser.parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY", "")
    if not api_key and not args.dry_run:
        print("ERROR: Set RUNPOD_API_KEY environment variable")
        sys.exit(1)

    volume_id = args.volume_id
    if not volume_id:
        print("  No volume ID set — pods will download models from HuggingFace (any datacenter)")

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"

    # Load existing results for resume
    results = []
    completed_ids = set()
    if args.resume and results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        completed_ids = {r["exp_id"] for r in results if r.get("status") == "completed"}
        print(f"Resuming: {len(completed_ids)} experiments already done")

    print()
    print("=" * 65)
    print("  HYBRID COMPUTE EXPERIMENTS")
    print("=" * 65)
    print(f"  Experiments: {len(EXPERIMENTS)}")
    print(f"  Prompt: {EXPERIMENT_PROMPT[:50]}...")
    print(f"  Frames: 81 | Steps: 50 | Seed: {EXPERIMENT_SEED}")
    print(f"  Output: {output_dir}")
    print("=" * 65)

    if args.dry_run:
        print("\n  DRY RUN — experiment plan:")
        for exp in EXPERIMENTS:
            skip = "(SKIP)" if exp.id in completed_ids else ""
            sched = f" {exp.schedule}" if exp.schedule else ""
            print(f"  [{exp.phase}] #{exp.id}: {exp.name}{sched} {skip}")
        return

    rp = RunPodClient(api_key, volume_id)

    # Cleanup on exit
    def cleanup(*_):
        print("\n  Cleaning up pods...")
        rp.terminate_all()
    atexit.register(cleanup)
    signal.signal(signal.SIGINT, lambda *_: (cleanup(), sys.exit(0)))

    def run_experiment(exp: Experiment, client: GenerationClient) -> dict:
        """Run one experiment and return result dict."""
        if exp.id in completed_ids:
            print(f"    SKIP (already done)")
            return next(r for r in results if r["exp_id"] == exp.id)

        print(f"    Submitting: model={exp.model}, schedule={exp.schedule}")
        start = time.time()
        job_id = client.submit(
            EXPERIMENT_PROMPT, exp.model, 50, exp.schedule, EXPERIMENT_SEED
        )
        print(f"    Job: {job_id}")
        status = client.wait(job_id)
        gen_time = status.get("generation_time", time.time() - start)

        # Download video
        vid_name = f"exp{exp.id}_{exp.name.replace(' ','_').replace('/','-')[:30]}.mp4"
        vid_path = str(output_dir / vid_name)
        client.download(job_id, vid_path)

        dist_stats = status.get("cache_statistics", {})

        return {
            "exp_id": exp.id,
            "name": exp.name,
            "phase": exp.phase,
            "compute_desc": exp.compute_desc,
            "model": exp.model,
            "schedule": exp.schedule,
            "distributed": exp.distributed,
            "generation_time": gen_time,
            "video_path": vid_path,
            "job_id": job_id,
            "status": "completed",
            "segments_on_worker": dist_stats.get("segments_on_worker"),
            "segments_fallback": dist_stats.get("segments_fallback"),
            "completed_at": datetime.now().isoformat(),
        }

    def save_results():
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

    # ── Phase A: H100 single-GPU baselines ──
    print(f"\n{'='*65}")
    print("  PHASE A: H100 Single-GPU Baselines")
    print(f"{'='*65}")

    phase_a_exps = [e for e in EXPERIMENTS if e.phase == "A"]
    if all(e.id in completed_ids for e in phase_a_exps):
        print("  All Phase A experiments already done, skipping pod creation")
    else:
        h100_pod = rp.create_pod(
            "wan21-h100-baseline",
            "NVIDIA H100 80GB HBM3", 1, "SECURE",
            _startup_cmd("server_both"),
        )
        h100_url = rp.wait_for_ready(h100_pod, SERVER_PORT, timeout=1800)
        h100_client = GenerationClient(h100_url)

        h100_start = time.time()
        for exp in phase_a_exps:
            print(f"\n  [{exp.phase}] #{exp.id}: {exp.name}")
            r = run_experiment(exp, h100_client)
            # Cost: H100 on-demand for the generation time
            r["cost_usd"] = r["generation_time"] / 3600 * COST_RATES["NVIDIA H100 80GB HBM3"]["on_demand"]
            results.append(r)
            save_results()

        h100_wall = time.time() - h100_start
        rp.terminate_pod(h100_pod)
        print(f"\n  Phase A done. H100 wall time: {h100_wall:.0f}s, cost: ${h100_wall/3600*3.50:.2f}")

    # ── Phase B-D: Distributed experiments ──
    # Create coordinator (RTX4090, stays up for B-E)
    distributed_phases = {
        "B": ("NVIDIA H100 80GB HBM3", 1, "SPOT", "worker"),
        "C": ("NVIDIA A100-SXM4-80GB", 1, "SECURE", "worker"),
        "D": ("NVIDIA A40", 2, "SPOT", "worker_multigpu"),
    }

    needs_coordinator = any(
        e.id not in completed_ids
        for e in EXPERIMENTS if e.phase in ("B", "C", "D", "E")
    )

    coord_pod = None
    coord_url = None
    coord_client = None
    coord_start = None

    if needs_coordinator:
        print(f"\n{'='*65}")
        print("  Creating RTX4090 Coordinator (stays up for Phase B-E)")
        print(f"{'='*65}")
        coord_pod = rp.create_pod(
            "wan21-coordinator",
            "NVIDIA GeForce RTX 4090", 1, "SECURE",
            _startup_cmd("coordinator"),
        )
        coord_url = rp.wait_for_ready(coord_pod, SERVER_PORT, timeout=1800)
        coord_client = GenerationClient(coord_url)
        coord_start = time.time()

    for phase_key, (gpu_type, gpu_count, cloud_type, role) in distributed_phases.items():
        phase_exps = [e for e in EXPERIMENTS if e.phase == phase_key]
        if all(e.id in completed_ids for e in phase_exps):
            print(f"\n  Phase {phase_key}: all done, skipping")
            continue

        print(f"\n{'='*65}")
        print(f"  PHASE {phase_key}: Distributed with {gpu_type} x{gpu_count}")
        print(f"{'='*65}")

        worker_pod = rp.create_pod(
            f"wan21-worker-{phase_key.lower()}",
            gpu_type, gpu_count, cloud_type,
            _startup_cmd(role),
        )
        worker_url = rp.wait_for_ready(worker_pod, WORKER_PORT, timeout=1800)

        # Tell coordinator about this worker
        coord_client.set_worker_url(worker_url)
        print(f"  Coordinator pointed to worker: {worker_url}")

        worker_start = time.time()
        for exp in phase_exps:
            print(f"\n  [{exp.phase}] #{exp.id}: {exp.name}")
            r = run_experiment(exp, coord_client)
            # Cost: coordinator time + worker time (proportional to 14B steps)
            gen_t = r["generation_time"]
            coord_rate = COST_RATES.get("NVIDIA GeForce RTX 4090", {}).get("on_demand", 0.44)
            worker_rate = COST_RATES.get(gpu_type, {}).get("spot", 1.50)
            total_steps = 50
            steps_14b = sum(s for m, s in (exp.schedule or []) if m == "14B")
            frac_14b = steps_14b / total_steps if total_steps else 0
            r["cost_usd"] = (gen_t / 3600) * (coord_rate + worker_rate * frac_14b)
            results.append(r)
            save_results()

        worker_wall = time.time() - worker_start
        rp.terminate_pod(worker_pod)
        spot_cost = worker_wall / 3600 * COST_RATES.get(gpu_type, {}).get("spot", 1.50)
        print(f"\n  Phase {phase_key} done. Worker wall: {worker_wall:.0f}s, spot cost: ${spot_cost:.2f}")

    # ── Phase E: 1.3B only on coordinator ──
    phase_e_exps = [e for e in EXPERIMENTS if e.phase == "E"]
    if coord_client and not all(e.id in completed_ids for e in phase_e_exps):
        print(f"\n{'='*65}")
        print("  PHASE E: 1.3B Only (RTX4090)")
        print(f"{'='*65}")

        coord_client.disable_distributed()
        for exp in phase_e_exps:
            print(f"\n  [{exp.phase}] #{exp.id}: {exp.name}")
            r = run_experiment(exp, coord_client)
            r["cost_usd"] = r["generation_time"] / 3600 * 0.44
            results.append(r)
            save_results()

    # Terminate coordinator
    if coord_pod:
        coord_wall = time.time() - coord_start if coord_start else 0
        rp.terminate_pod(coord_pod)
        print(f"\n  Coordinator total wall time: {coord_wall:.0f}s, cost: ${coord_wall/3600*0.44:.2f}")

    # ── Phase F: CLIP analysis ──
    print(f"\n{'='*65}")
    print("  PHASE F: Quality Analysis")
    print(f"{'='*65}")

    baseline_vid = next(
        (r["video_path"] for r in results if r["exp_id"] == 1 and r["status"] == "completed"), None
    )

    if CLIP_AVAILABLE and baseline_vid and os.path.exists(baseline_vid):
        print("  Loading CLIP...")
        clip = CLIPEvaluator()
        for r in results:
            if r.get("status") != "completed":
                continue
            if r["exp_id"] == 1:
                r["clip_similarity"] = 1.0
                continue
            vid = r.get("video_path", "")
            if os.path.exists(vid):
                try:
                    sim = clip.similarity(baseline_vid, vid)
                    r["clip_similarity"] = sim
                    print(f"    #{r['exp_id']} {r['name'][:30]}: {sim:.4f}")
                except Exception as e:
                    print(f"    #{r['exp_id']}: CLIP error: {e}")
        save_results()
    else:
        print("  CLIP not available or baseline video missing. Skipping quality metrics.")

    # ── Generate report ──
    report_path = str(output_dir / "report.html")
    generate_report(results, report_path)

    # ── Summary ──
    total_cost = sum(r.get("cost_usd", 0) for r in results if r.get("status") == "completed")
    print(f"\n{'='*65}")
    print("  COMPLETE")
    print(f"{'='*65}")
    print(f"  {'Experiment':<35} | {'Time':>7} | {'CLIP':>6} | {'Cost':>7}")
    print(f"  {'-'*65}")
    for r in sorted(results, key=lambda x: x.get("exp_id", 99)):
        if r.get("status") != "completed":
            continue
        n = r["name"][:35]
        t = f'{r["generation_time"]:.1f}s'
        c = f'{r.get("clip_similarity",0):.4f}' if r.get("clip_similarity") else "-"
        co = f'${r.get("cost_usd",0)*100:.1f}c'
        print(f"  {n:<35} | {t:>7} | {c:>6} | {co:>7}")
    print(f"  {'-'*65}")
    print(f"  Total experiment GPU cost: ${total_cost:.2f}")
    print(f"  Report: {report_path}")
    print(f"  Data:   {results_path}")
    print()


if __name__ == "__main__":
    main()
