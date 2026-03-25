#!/bin/bash
# Force update code on H100 and restart worker
SSH_KEY="$HOME/.ssh/id_ed25519"
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" -p 30646 root@38.80.152.148 bash <<'REMOTE'
cd /workspace/wan2.1/Wan2.1
echo "=== Force reset to remote ==="
git stash 2>&1
git checkout distributed-inference 2>&1
git fetch origin 2>&1
git reset --hard origin/distributed-inference 2>&1
echo "=== Git log ==="
git log --oneline -3

echo "=== Restarting worker ==="
pkill -f "run_worker" 2>/dev/null; sleep 2
nohup python3 run_worker.py --port 8889 > /tmp/wan_worker.log 2>&1 &
echo "Worker PID: $!"
REMOTE

echo "Waiting for worker..."
for i in $(seq 1 60); do
    sleep 10
    if curl -s http://localhost:18889/health 2>/dev/null | grep -q "ok"; then
        echo "H100 worker READY ($((i*10))s)"
        curl -s http://localhost:18889/health | python3 -m json.tool
        exit 0
    fi
    echo "  Loading... ($((i*10))s)"
done
