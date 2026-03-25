#!/bin/bash
# Pull latest code, restart services, run experiments
SSH_KEY="$HOME/.ssh/id_ed25519"
A40_HOST="69.30.85.135"; A40_PORT="22182"
H100_HOST="38.80.152.148"; H100_PORT="30646"

echo "=== Pulling code on H100 ==="
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" -p $H100_PORT root@$H100_HOST \
    "cd /workspace/wan2.1/Wan2.1 && git pull origin distributed-inference 2>&1 | tail -3" 2>&1

echo "=== Pulling code on 2xA40 ==="
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" -p $A40_PORT root@$A40_HOST \
    "cd /workspace/wan2.1/Wan2.1 && git remote set-url origin https://github.com/athulramkumar/Wan2.1.git 2>/dev/null; git pull origin distributed-inference 2>&1 | tail -3" 2>&1

echo "=== Restarting H100 worker ==="
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" -p $H100_PORT root@$H100_HOST bash <<'REMOTE'
cd /workspace/wan2.1/Wan2.1
pkill -f "run_worker" 2>/dev/null; sleep 2
nohup python3 run_worker.py --port 8889 > /tmp/wan_worker.log 2>&1 &
echo "Worker PID: $!"
REMOTE

echo "=== Restarting 2xA40 coordinator ==="
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" -p $A40_PORT root@$A40_HOST bash <<'REMOTE'
cd /workspace/wan2.1/Wan2.1
pkill -f "run_server" 2>/dev/null; sleep 2
nohup python3 run_server.py --port 8890 --distributed --mode coordinator > /tmp/wan_server.log 2>&1 &
echo "Coordinator PID: $!"
REMOTE

echo "=== Re-establishing tunnels ==="
pkill -f "ssh.*-N.*-L" 2>/dev/null; sleep 1
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" -f -N -L 18890:localhost:8890 -p $A40_PORT root@$A40_HOST
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" -f -N -L 18889:localhost:8889 -p $H100_PORT root@$H100_HOST
# Coordinator -> Worker tunnel
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" -p $A40_PORT root@$A40_HOST \
    "pkill -f 'ssh.*-L.*8889' 2>/dev/null; sleep 1; ssh -o StrictHostKeyChecking=no -i /root/.ssh/id_ed25519 -f -N -L 8889:localhost:8889 -p $H100_PORT root@$H100_HOST" 2>&1

echo "=== Waiting for services ==="
for i in $(seq 1 90); do
    sleep 10
    w=$(curl -s http://localhost:18889/health 2>/dev/null | grep -c "ok")
    c=$(curl -s http://localhost:18890/health 2>/dev/null | grep -c "healthy")
    echo "  ($((i*10))s) Worker=$w Coordinator=$c"
    [ "$w" = "1" ] && [ "$c" = "1" ] && break
done

echo ""
echo "=== Service status ==="
echo "Worker:"
curl -s http://localhost:18889/health 2>/dev/null | python3 -c "import json,sys;d=json.load(sys.stdin);print(f'  {d}')" 2>/dev/null || echo "  NOT READY"
echo "Coordinator:"
curl -s http://localhost:18890/health 2>/dev/null | python3 -c "import json,sys;d=json.load(sys.stdin);print(f'  {d}')" 2>/dev/null || echo "  NOT READY"

# Connect coordinator to worker
echo ""
echo "=== Connecting coordinator to worker ==="
curl -s -X POST "http://localhost:18890/admin/set-worker-url" \
    -H "Content-Type: application/json" \
    -d '{"worker_url":"http://localhost:8889"}' 2>/dev/null | python3 -m json.tool

echo ""
echo "=== Running experiments ==="
bash scripts/run_final.sh
