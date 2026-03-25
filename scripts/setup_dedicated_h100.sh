#!/bin/bash
# Setup dedicated H100 (port 31090) - both models for baseline experiments
SSH_KEY="$HOME/.ssh/id_ed25519"
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" -p 31090 root@38.80.152.148 bash <<'REMOTE'
set -e
echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

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

echo "=== Installing deps ==="
pip install -r requirements.txt -q 2>&1 | tail -5
pip install fastapi uvicorn requests pydantic -q

echo "=== Checking models ==="
ls -d Wan2.1-T2V-14B Wan2.1-T2V-1.3B 2>/dev/null || echo "Models not found - downloading..."
if [ ! -d Wan2.1-T2V-14B ]; then
    pip install huggingface_hub -q
    echo "Downloading 14B..."
    huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir Wan2.1-T2V-14B 2>&1 | tail -3
fi
if [ ! -d Wan2.1-T2V-1.3B ]; then
    echo "Downloading 1.3B..."
    huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir Wan2.1-T2V-1.3B 2>&1 | tail -3
fi

echo "=== Starting server (mode=both, port 8890) ==="
pkill -f "run_server.*8890" 2>/dev/null; sleep 2
nohup python run_server.py --port 8890 > /tmp/wan_server.log 2>&1 &
echo "PID: $!"

echo "Waiting for health..."
for i in $(seq 1 90); do
    sleep 10
    if curl -s http://localhost:8890/health 2>/dev/null | grep -q "healthy"; then
        echo "READY ($((i*10))s)"
        curl -s http://localhost:8890/health
        exit 0
    fi
    echo "  Loading... ($((i*10))s)"
done
echo "TIMEOUT - last log:"
tail -30 /tmp/wan_server.log
exit 1
REMOTE
