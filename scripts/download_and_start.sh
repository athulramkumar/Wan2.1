#!/bin/bash
# Download models and start server/worker on a remote instance
# Usage: bash scripts/download_and_start.sh <host> <port> <role> <server_port>
# role: server_both | coordinator | worker

HOST="$1"; PORT="$2"; ROLE="$3"; SRVPORT="${4:-8890}"
SSH_KEY="$HOME/.ssh/id_ed25519"

echo "=== $ROLE on $HOST:$PORT (server port $SRVPORT) ==="

ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" -p "$PORT" "root@$HOST" bash -s -- "$ROLE" "$SRVPORT" <<'REMOTE'
ROLE="$1"; SRVPORT="$2"
set -e
cd /workspace/wan2.1/Wan2.1
export PATH="$PATH:/usr/local/bin:$HOME/.local/bin"

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

echo "=== Download models with python -m ==="
if [ "$ROLE" = "coordinator" ] || [ "$ROLE" = "server_both" ]; then
    if [ ! -f Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth ]; then
        echo "Downloading 1.3B..."
        python3 -m huggingface_hub.cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir Wan2.1-T2V-1.3B 2>&1 | tail -3
    else
        echo "1.3B already present"
    fi
fi

if [ "$ROLE" = "worker" ] || [ "$ROLE" = "server_both" ]; then
    if [ ! -f Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth ]; then
        echo "Downloading 14B..."
        python3 -m huggingface_hub.cli download Wan-AI/Wan2.1-T2V-14B --local-dir Wan2.1-T2V-14B 2>&1 | tail -3
    else
        echo "14B already present"
    fi
fi

echo "=== Model dirs ==="
ls -d Wan2.1-T2V-*/ 2>/dev/null

echo "=== Starting service ==="
pkill -f "run_server.*$SRVPORT" 2>/dev/null
pkill -f "run_worker.*$SRVPORT" 2>/dev/null
sleep 2

if [ "$ROLE" = "server_both" ]; then
    nohup python3 run_server.py --port $SRVPORT > /tmp/wan_server.log 2>&1 &
elif [ "$ROLE" = "coordinator" ]; then
    nohup python3 run_server.py --port $SRVPORT --distributed --mode coordinator > /tmp/wan_server.log 2>&1 &
elif [ "$ROLE" = "worker" ]; then
    nohup python3 run_worker.py --port $SRVPORT > /tmp/wan_worker.log 2>&1 &
fi
echo "PID: $!"

echo "Waiting for health..."
for i in $(seq 1 120); do
    sleep 10
    if curl -s "http://localhost:$SRVPORT/health" 2>/dev/null | grep -qE "healthy|ok"; then
        echo "READY ($((i*10))s)"
        curl -s "http://localhost:$SRVPORT/health"
        exit 0
    fi
    echo "  Loading... ($((i*10))s)"
done
echo "TIMEOUT"
tail -30 /tmp/wan_server.log /tmp/wan_worker.log 2>/dev/null
exit 1
REMOTE
