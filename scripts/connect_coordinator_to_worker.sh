#!/bin/bash
# Create SSH tunnel from RTX4000 coordinator to H100 worker
# Then set the worker URL on the coordinator
SSH_KEY="$HOME/.ssh/id_ed25519"

echo "=== Step 1: Copy SSH key to coordinator (RTX4000) ==="
scp -o StrictHostKeyChecking=no -i "$SSH_KEY" -P 21010 \
    "$SSH_KEY" root@157.157.221.29:/root/.ssh/id_ed25519 2>&1
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" -p 21010 root@157.157.221.29 \
    "chmod 600 /root/.ssh/id_ed25519" 2>&1

echo "=== Step 2: Create tunnel from coordinator to H100 worker ==="
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" -p 21010 root@157.157.221.29 bash <<'REMOTE'
# Kill old tunnels
pkill -f "ssh.*-L.*8889" 2>/dev/null; sleep 1

# Create tunnel: coordinator:8889 -> H100:8889
ssh -o StrictHostKeyChecking=no -i /root/.ssh/id_ed25519 \
    -f -N -L 8889:localhost:8889 -p 30646 root@38.80.152.148 2>&1
echo "Tunnel created"

# Test it
sleep 3
curl -s http://localhost:8889/health 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "Worker not reachable yet"
REMOTE

echo "=== Step 3: Set worker URL on coordinator ==="
# Use the local tunnel to reach the coordinator
curl -s -X POST "http://localhost:18890/admin/set-worker-url" \
    -H "Content-Type: application/json" \
    -d '{"worker_url":"http://localhost:8889"}' 2>/dev/null | python3 -m json.tool

echo "=== Done ==="
