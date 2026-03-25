#!/bin/bash
# Start H100 as worker (14B) and RTX4000 as coordinator (1.3B)
SSH_KEY="$HOME/.ssh/id_ed25519"

echo "=== Starting H100 as 14B worker (port 8889) ==="
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" -p 30646 root@38.80.152.148 bash <<'REMOTE'
cd /workspace/wan2.1/Wan2.1
pkill -f "run_worker" 2>/dev/null; sleep 2
nohup python3 run_worker.py --port 8889 > /tmp/wan_worker.log 2>&1 &
echo "Worker PID: $!"
REMOTE

echo "=== Starting RTX4000 as 1.3B coordinator (port 8890) ==="
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" -p 21010 root@157.157.221.29 bash <<'REMOTE'
cd /workspace/wan2.1/Wan2.1
pkill -f "run_server" 2>/dev/null; sleep 2
nohup python3 run_server.py --port 8890 --distributed --mode coordinator > /tmp/wan_server.log 2>&1 &
echo "Coordinator PID: $!"
REMOTE

echo ""
echo "=== Setting up SSH tunnels ==="
pkill -f "ssh.*-N.*-L" 2>/dev/null; sleep 1

# Tunnel to RTX4000 coordinator
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" -f -N -L 18890:localhost:8890 -p 21010 root@157.157.221.29 2>&1
echo "Coordinator tunnel: localhost:18890 -> RTX4000:8890"

# Tunnel to H100 worker
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" -f -N -L 18889:localhost:8889 -p 30646 root@38.80.152.148 2>&1
echo "Worker tunnel: localhost:18889 -> H100:8889"

echo ""
echo "=== Waiting for services ==="
echo "Waiting for H100 worker..."
for i in $(seq 1 90); do
    sleep 10
    if curl -s http://localhost:18889/health 2>/dev/null | grep -q "ok"; then
        echo "H100 worker READY ($((i*10))s)"
        break
    fi
    echo "  Worker loading... ($((i*10))s)"
done

echo "Waiting for RTX4000 coordinator..."
for i in $(seq 1 90); do
    sleep 10
    if curl -s http://localhost:18890/health 2>/dev/null | grep -q "healthy"; then
        echo "RTX4000 coordinator READY ($((i*10))s)"
        break
    fi
    echo "  Coordinator loading... ($((i*10))s)"
done

echo ""
echo "=== Status ==="
echo "H100 worker:"
curl -s http://localhost:18889/health 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "  Not ready"
echo ""
echo "RTX4000 coordinator:"
curl -s http://localhost:18890/health 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "  Not ready"
