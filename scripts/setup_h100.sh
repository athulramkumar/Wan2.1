#!/bin/bash
# Setup and start server on the existing H100 instance
set -e

cd /workspace/wan2.1/Wan2.1

echo "=== Installing dependencies ==="
pip install -r requirements.txt -q 2>&1 | tail -5
pip install fastapi uvicorn requests pydantic -q

echo "=== Checking for running server ==="
pkill -f "run_server.py" 2>/dev/null || true
sleep 2

echo "=== Starting server on port 8890 ==="
nohup python run_server.py --port 8890 > /tmp/wan_server.log 2>&1 &
echo "Server PID: $!"
echo "Logs: /tmp/wan_server.log"
echo "Waiting for model load..."

# Wait up to 10 minutes for health endpoint
for i in $(seq 1 60); do
    sleep 10
    if curl -s http://localhost:8890/health | grep -q "healthy"; then
        echo "Server ready!"
        exit 0
    fi
    echo "  Still loading... ($((i*10))s)"
done

echo "Server failed to start. Last 30 lines of log:"
tail -30 /tmp/wan_server.log
exit 1
