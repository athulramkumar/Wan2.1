#!/bin/bash
# Setup a RunPod instance with the Wan2.1 repo and models
# Usage: bash scripts/setup_instance.sh <host> <port> <role> [model_size]
# role: coordinator (downloads 1.3B) or worker (downloads 14B) or both
# Example: bash scripts/setup_instance.sh 157.157.221.29 21010 coordinator

HOST="$1"
PORT="$2"
ROLE="$3"
SSH_KEY="$HOME/.ssh/id_ed25519"

if [ -z "$HOST" ] || [ -z "$PORT" ] || [ -z "$ROLE" ]; then
    echo "Usage: $0 <host> <port> <role>"
    echo "  role: coordinator, worker, both"
    exit 1
fi

echo "=== Setting up $ROLE on $HOST:$PORT ==="

ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" -p "$PORT" "root@$HOST" bash <<REMOTE
set -e

echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU"

echo ""
echo "=== Setting up repo ==="
if [ ! -d /workspace/wan2.1/Wan2.1/.git ]; then
    mkdir -p /workspace/wan2.1
    cd /workspace/wan2.1
    git clone https://github.com/athulramkumar/Wan2.1.git 2>&1
fi

cd /workspace/wan2.1/Wan2.1
git remote set-url origin https://github.com/athulramkumar/Wan2.1.git 2>/dev/null
git fetch origin 2>&1
git checkout distributed-inference 2>&1 || git checkout -b distributed-inference origin/distributed-inference 2>&1
git pull origin distributed-inference 2>&1

echo ""
echo "=== Installing dependencies ==="
pip install -r requirements.txt -q 2>&1 | tail -5
pip install fastapi uvicorn requests pydantic huggingface_hub -q 2>&1 | tail -3

echo ""
echo "=== Downloading model weights ==="
if [ "$ROLE" = "coordinator" ] || [ "$ROLE" = "both" ]; then
    if [ ! -d Wan2.1-T2V-1.3B ] || [ ! -f Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth ]; then
        echo "Downloading 1.3B model..."
        huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir Wan2.1-T2V-1.3B 2>&1 | tail -5
    else
        echo "1.3B model already present"
    fi
fi

if [ "$ROLE" = "worker" ] || [ "$ROLE" = "both" ]; then
    if [ ! -d Wan2.1-T2V-14B ] || [ ! -f Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth ]; then
        echo "Downloading 14B model..."
        huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir Wan2.1-T2V-14B 2>&1 | tail -5
    else
        echo "14B model already present"
    fi
fi

echo ""
echo "=== Setup complete ==="
ls -la /workspace/wan2.1/Wan2.1/Wan2.1-T2V-*/
REMOTE

echo ""
echo "Done setting up $ROLE on $HOST:$PORT"
