#!/bin/bash
# Fix H100 - kill old processes, clear GPU mem, restart fresh
SSH_KEY="$HOME/.ssh/id_ed25519"
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" -p 30646 root@38.80.152.148 bash <<'REMOTE'
echo "=== Killing old server ==="
pkill -9 -f "run_server" 2>/dev/null
pkill -9 -f "uvicorn" 2>/dev/null
sleep 3

echo "=== Clearing GPU ==="
python3 -c "import torch; torch.cuda.empty_cache(); print(f'GPU mem: {torch.cuda.memory_allocated()/1e9:.1f}GB')" 2>/dev/null

echo "=== GPU status ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

echo "=== Updating code ==="
cd /workspace/wan2.1/Wan2.1
git pull origin distributed-inference 2>&1 | tail -3

echo "=== Starting server ==="
nohup python run_server.py --port 8890 > /tmp/wan_server.log 2>&1 &
echo "PID: $!"
echo "Waiting for health..."
for i in $(seq 1 90); do
    sleep 10
    if curl -s http://localhost:8890/health 2>/dev/null | grep -q "healthy"; then
        echo "READY ($((i*10))s)"
        curl -s http://localhost:8890/health | python3 -m json.tool
        exit 0
    fi
    echo "  Loading... ($((i*10))s)"
done
echo "TIMEOUT"
tail -30 /tmp/wan_server.log
exit 1
REMOTE
