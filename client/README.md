# Wan2.1 Video Generation Client

A simple Python script to generate videos using the Wan2.1 API from your local machine.

## Setup

### 1. Install Dependencies

```bash
pip install requests
```

### 2. Configure Server URL

Edit `generate_video.py` and set your server URL:

```python
# Line 36
API_BASE_URL = "https://jvkiftqnv4abuo-8888.proxy.runpod.net"
```

Or pass it as a command-line argument:
```bash
python generate_video.py "Your prompt" --server https://your-server-url.com
```

## Usage

### Basic Usage

```bash
python generate_video.py "A cat playing piano"
```

### With Options

```bash
# Hybrid model (default) with caching for speed
python generate_video.py "A magical forest at sunset" --caching --seed 42

# Fast generation with 1.3B model
python generate_video.py "Dancing robots" --model baseline_1.3B --frames 33 --steps 30

# High quality with 14B model
python generate_video.py "Ocean waves crashing" --model baseline_14B

# Custom output directory
python generate_video.py "Flying birds" --output-dir ~/Videos/wan
```

## Command-Line Options

```
positional arguments:
  prompt                Text prompt for video generation

options:
  -h, --help            Show help message
  --model, -m           Model: baseline_14B, baseline_1.3B, hybrid (default: hybrid)
  --schedule            Hybrid schedule JSON, e.g., '[["14B", 15], ["1.3B", 35]]'
  --frames              Frame count: 17, 33, 49, 65, 81 (default: 81)
  --fps                 Output FPS (default: 16)
  --steps               Sampling steps (default: 50)
  --seed                Random seed, -1 for random (default: -1)
  --caching             Enable Cache-DiT for faster generation
  --cache-start         Cache start step (default: 10)
  --cache-end           Cache end step (default: 40)
  --cache-interval      Cache interval (default: 3)
  --output-dir, -o      Output directory (default: current directory)
  --no-display          Don't auto-open video after generation
  --poll-interval       Status poll interval in seconds (default: 5)
  --server              API server URL
```

## Output Files

The script saves two files:

```
./A_cat_playing_piano_abc12345.mp4    # The generated video
./A_cat_playing_piano_abc12345.json   # Metadata with all parameters
```

### Metadata JSON Example

```json
{
  "job_id": "abc12345",
  "prompt": "A cat playing piano",
  "model": "hybrid",
  "frame_count": 81,
  "fps": 16,
  "sampling_steps": 50,
  "seed": 42,
  "caching_enabled": true,
  "generation_time_seconds": 127.5,
  "cache_statistics": {
    "cache_hits": 120,
    "cache_misses": 30,
    "cache_hit_rate": 0.8
  },
  "server_url": "https://jvkiftqnv4abuo-8888.proxy.runpod.net",
  "generated_at": "2026-01-18T18:30:00.000000",
  "video_file": "A_cat_playing_piano_abc12345.mp4"
}
```

## Examples

### Quick Test (Fast)

```bash
python generate_video.py "A bouncing ball" \
  --model baseline_1.3B \
  --frames 17 \
  --steps 20
```

### Standard Quality

```bash
python generate_video.py "A sunset over mountains" \
  --model hybrid \
  --seed 123
```

### High Quality with Caching

```bash
python generate_video.py "A detailed cityscape at night" \
  --model hybrid \
  --caching \
  --frames 81 \
  --steps 50 \
  --seed 42
```

### Maximum Quality (Slow)

```bash
python generate_video.py "Photorealistic ocean waves" \
  --model baseline_14B \
  --frames 81 \
  --steps 50
```

### Custom Hybrid Schedule

```bash
python generate_video.py "Abstract art" \
  --model hybrid \
  --schedule '[["14B", 20], ["1.3B", 30]]'
```

## Progress Display

```
============================================================
  WAN2.1 VIDEO GENERATION
============================================================
  Server:  https://jvkiftqnv4abuo-8888.proxy.runpod.net
  Prompt:  A cat playing piano
  Model:   hybrid
  Frames:  81
  Caching: Yes
============================================================

📤 Submitting job...
✓ Job submitted: abc12345

⏳ Generating video...
  [████████████████████░░░░░░░░░░░░░░░░░░░░] 52% | 14B | 65s
```

## Troubleshooting

### "Connection refused"

Server is not running. Start it on RunPod:
```bash
python run_server.py --port 8888
```

### "Failed to submit job: 404"

Wrong server URL. Check the URL in the script or use `--server`:
```bash
python generate_video.py "test" --server https://correct-url.proxy.runpod.net
```

### "timeout"

- Server might be overloaded
- Try increasing `--poll-interval 10`
- Check server health: `curl {server_url}/health`

### Interrupted Generation

If you Ctrl+C during generation, the job continues on the server. Check status:
```bash
curl https://your-server/status/{job_id}
```

## Model Comparison

| Model | Quality | Speed | VRAM |
|-------|---------|-------|------|
| `baseline_14B` | Highest | Slowest | ~55 GB |
| `baseline_1.3B` | Good | Fastest | ~8 GB |
| `hybrid` | High | Medium | ~64 GB |
| `hybrid` + caching | High | Fast | ~64 GB |

## Tips

1. **Use caching** (`--caching`) for 20-30% speedup with minimal quality loss
2. **Start with fewer frames** (`--frames 33`) for quick iterations
3. **Use 1.3B for drafts** then switch to hybrid/14B for final
4. **Set a seed** (`--seed 42`) for reproducible results
5. **Organize outputs** with `--output-dir ~/Videos/project_name`

