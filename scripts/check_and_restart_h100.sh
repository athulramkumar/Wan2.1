#!/bin/bash
# Check H100 server status and restart if needed
H100="root@38.80.152.148"
PORT=30646

echo "=== Checking H100 ==="
ssh -o StrictHostKeyChecking=no -p $PORT $H100 bash <<'REMOTE'
echo "GPU status:"
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader

echo ""
echo "Server process:"
ps aux | grep run_server | grep -v grep || echo "Server NOT running"

echo ""
echo "Last 20 lines of log:"
tail -20 /tmp/wan_server.log 2>/dev/null

echo ""
echo "Restarting server..."
pkill -f "run_server.py" 2>/dev/null
sleep 3

cd /workspace/wan2.1/Wan2.1
nohup python run_server.py --port 8890 > /tmp/wan_server.log 2>&1 &
echo "Server restarted (PID: $!)"

echo "Waiting for health..."
for i in $(seq 1 60); do
    sleep 10
    if curl -s http://localhost:8890/health | grep -q "healthy"; then
        echo "Server ready!"
        exit 0
    fi
    echo "  Loading... ($((i*10))s)"
done
echo "FAILED to start"
exit 1
REMOTE
