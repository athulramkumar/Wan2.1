# Wan2.1 Video Generation API Server

A production-ready FastAPI server for generating videos using Wan2.1 models.

## Features

- **Multiple Models**: Baseline 14B, Baseline 1.3B, and Hybrid mode
- **Hybrid Sampling**: Combine 14B quality with 1.3B speed using configurable schedules
- **Caching (Cache-DiT)**: Activation caching for faster inference
- **Job Queue**: Asynchronous processing with progress tracking
- **Auto Model Loading**: Both models loaded at startup, kept in GPU memory

## Quick Start

```bash
# From the Wan2.1 directory
cd /workspace/wan2.1/Wan2.1

# Activate virtual environment
source .venv/bin/activate

# Start server (default port 8888)
python run_server.py

# Or with specific port
python run_server.py --port 8000
```

## Server Options

```bash
python run_server.py --help

Options:
  --host HOST           Server host (default: 0.0.0.0)
  --port PORT           Server port (default: 8888)
  --kill-existing, -k   Kill existing process on port
  --yes, -y             Auto-confirm kill (no prompt)
  --reload              Enable auto-reload (dev only)
  --workers N           Number of workers (keep at 1 for GPU)
```

### Examples

```bash
# Start on default port
python run_server.py

# Start on port 8000
python run_server.py --port 8000

# Kill existing process and restart
python run_server.py --port 8888 -k -y
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Server health check |
| GET | `/config` | Server configuration |
| GET | `/docs` | Interactive API documentation |
| POST | `/generate` | Submit video generation job |
| GET | `/status/{job_id}` | Get job status and progress |
| GET | `/video/{job_id}` | Download generated video |

## Generate Request

```bash
curl -X POST http://localhost:8888/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cat playing piano",
    "model": "hybrid",
    "frame_count": 81,
    "fps": 16,
    "sampling_steps": 50,
    "seed": 42,
    "enable_caching": true,
    "cache_start_step": 10,
    "cache_end_step": 40,
    "cache_interval": 3
  }'
```

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | required | Text description of the video |
| `model` | string | `"hybrid"` | `"baseline_14B"`, `"baseline_1.3B"`, or `"hybrid"` |
| `schedule` | array | auto | Hybrid schedule, e.g., `[["14B", 15], ["1.3B", 35]]` |
| `frame_count` | int | 81 | Number of frames (17, 33, 49, 65, 81) |
| `fps` | int | 16 | Output video FPS |
| `sampling_steps` | int | 50 | Denoising steps |
| `seed` | int | -1 | Random seed (-1 for random) |
| `enable_caching` | bool | false | Enable Cache-DiT |
| `cache_start_step` | int | 10 | Start caching at this step |
| `cache_end_step` | int | 40 | Stop caching at this step |
| `cache_interval` | int | 3 | Recompute every N steps |

## Check Status

```bash
curl http://localhost:8888/status/{job_id}
```

Response:
```json
{
  "job_id": "abc12345",
  "status": "processing",
  "progress": 45,
  "model_in_use": "14B",
  "current_step": 22,
  "total_steps": 50
}
```

## Download Video

```bash
curl http://localhost:8888/video/{job_id} -o video.mp4
```

---

# RunPod Deployment

## The Problem

RunPod exposes services differently:
- **SSH Port** (e.g., `38.80.152.146:30396`): For SSH connections only, NOT HTTP
- **HTTP Proxy**: Uses `https://{pod_id}-{port}.proxy.runpod.net/` format

If you try to `curl` the SSH port, you'll get:
```
curl: (1) Received HTTP/0.9 when not allowed
```

## The Solution

### Option 1: Use an Already-Exposed HTTP Port (Recommended)

RunPod pre-configures certain ports for HTTP (like 8888 for Jupyter). Use one of these:

```bash
# Stop Jupyter (if running)
pkill -f jupyter

# Start API on port 8888
python run_server.py --port 8888 -k -y
```

Your public URL will be:
```
https://{pod_id}-8888.proxy.runpod.net/
```

### Option 2: Add Custom HTTP Port in RunPod Dashboard

1. Go to RunPod Dashboard → Your Pod → Settings
2. Find "Expose HTTP Ports" or "HTTP Services"
3. Add port `8000` (or your preferred port)
4. Save and restart pod

### Option 3: SSH Port Forwarding (For Personal Use)

If you only need access from your own machine:

```bash
# On your local machine
ssh root@38.80.152.146 -p 30396 -i ~/.ssh/id_ed25519 -L 8000:localhost:8000

# Then access via
curl http://localhost:8000/health
```

## Finding Your Pod ID

Your pod ID is in the SSH command RunPod provides:
```
ssh jvkiftqnv4abuo-64411ee4@ssh.runpod.io
    ^^^^^^^^^^^^^^
    This is your pod ID
```

Or from environment variable inside the pod:
```bash
echo $RUNPOD_POD_ID
```

## Public URL Format

```
https://{RUNPOD_POD_ID}-{PORT}.proxy.runpod.net/
```

Example:
```
https://jvkiftqnv4abuo-8888.proxy.runpod.net/health
https://jvkiftqnv4abuo-8888.proxy.runpod.net/docs
https://jvkiftqnv4abuo-8888.proxy.runpod.net/generate
```

## Checking What's Running

```bash
# See what ports nginx is proxying
ss -tlnp | grep nginx

# Check if your server is running
curl http://localhost:8888/health

# Test public URL from inside the pod
curl https://{pod_id}-8888.proxy.runpod.net/health
```

## Common Issues

### "Connection refused"
- Server not running or wrong port
- Check: `ps aux | grep run_server`

### "404 Not Found" from Cloudflare
- Port not configured for HTTP in RunPod
- Use port 8888 or add your port in RunPod dashboard

### "HTTP/0.9 when not allowed"
- You're hitting the SSH port, not HTTP proxy
- Use the `proxy.runpod.net` URL instead

### Models not loading (GPU stays at 0 MiB)
- Process might be loading to CPU first
- Wait 5-10 minutes for full model loading

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Server                          │
├─────────────────────────────────────────────────────────────┤
│  main.py          - API endpoints, startup/shutdown         │
│  model_manager.py - Load/manage 14B and 1.3B models         │
│  generator.py     - Video generation logic                  │
│  job_queue.py     - Async job processing                    │
│  schemas.py       - Pydantic request/response models        │
│  config.py        - Server configuration                    │
├─────────────────────────────────────────────────────────────┤
│  utils/                                                     │
│    scheduling.py  - Hybrid model scheduling                 │
│    caching.py     - Cache-DiT logic                         │
└─────────────────────────────────────────────────────────────┘
```

## GPU Memory Requirements

| Configuration | VRAM |
|--------------|------|
| 14B only | ~55 GB |
| 1.3B only | ~8 GB |
| Both models (hybrid) | ~64 GB |

Recommended: NVIDIA A100 80GB or H100

