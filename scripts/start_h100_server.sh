#!/bin/bash
# Just start the server on the dedicated H100 (models already there)
SSH_KEY="$HOME/.ssh/id_ed25519"
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" -p 31090 root@38.80.152.148 bash <<'REMOTE'
cd /workspace/wan2.1/Wan2.1
pkill -f "run_server.*8890" 2>/dev/null; sleep 2
nohup python3 run_server.py --port 8890 > /tmp/wan_server.log 2>&1 &
echo "Started PID $!"
REMOTE
echo "H100 server starting. Polling health..."
# Poll from local via SSH tunnel
pkill -f "ssh.*18890.*31090" 2>/dev/null; sleep 1
ssh -o StrictHostKeyChecking=no -i "$HOME/.ssh/id_ed25519" -f -N -L 18890:localhost:8890 -p 31090 root@38.80.152.148
for i in $(seq 1 90); do
    sleep 10
    if curl -s http://localhost:18890/health 2>/dev/null | grep -q "healthy"; then
        echo "H100 READY ($((i*10))s)"
        curl -s http://localhost:18890/health | python3 -m json.tool
        exit 0
    fi
    echo "  Loading... ($((i*10))s)"
done
echo "TIMEOUT"
