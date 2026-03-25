#!/bin/bash
# Download models on a remote instance using Python directly
# Usage: bash scripts/download_models.sh <host> <port> <models>
# models: 1.3B | 14B | both
HOST="$1"; PORT="$2"; MODELS="$3"
SSH_KEY="$HOME/.ssh/id_ed25519"

echo "=== Downloading $MODELS models on $HOST:$PORT ==="
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" -p "$PORT" "root@$HOST" bash -s -- "$MODELS" <<'REMOTE'
MODELS="$1"
cd /workspace/wan2.1/Wan2.1

# Use Python directly for downloads
python3 << 'PYEOF'
import os, sys
models = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MODELS", "both")

try:
    from huggingface_hub import snapshot_download
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub", "-q"])
    from huggingface_hub import snapshot_download

if models in ("1.3B", "both"):
    target = "Wan2.1-T2V-1.3B"
    if not os.path.exists(os.path.join(target, "models_t5_umt5-xxl-enc-bf16.pth")):
        print(f"Downloading {target}...")
        snapshot_download("Wan-AI/Wan2.1-T2V-1.3B", local_dir=target)
        print(f"{target} done")
    else:
        print(f"{target} already present")

if models in ("14B", "both"):
    target = "Wan2.1-T2V-14B"
    if not os.path.exists(os.path.join(target, "models_t5_umt5-xxl-enc-bf16.pth")):
        print(f"Downloading {target}...")
        snapshot_download("Wan-AI/Wan2.1-T2V-14B", local_dir=target)
        print(f"{target} done")
    else:
        print(f"{target} already present")
PYEOF

echo "=== Model dirs ==="
ls -d Wan2.1-T2V-*/ 2>/dev/null || echo "No model dirs found"
du -sh Wan2.1-T2V-*/ 2>/dev/null
REMOTE
