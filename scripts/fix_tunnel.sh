#!/bin/bash
# Fix the tunnel from 2xA40 to H100
SSH_KEY="$HOME/.ssh/id_ed25519"

echo "=== Step 1: Verify SSH key on 2xA40 ==="
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" -p 22182 root@69.30.85.135 \
    "ls -la /root/.ssh/id_ed25519 && echo 'Key exists'" 2>&1

echo ""
echo "=== Step 2: Test SSH from 2xA40 to H100 ==="
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" -p 22182 root@69.30.85.135 \
    "ssh -o StrictHostKeyChecking=no -i /root/.ssh/id_ed25519 -p 30646 root@38.80.152.148 'echo SSH_OK; hostname' 2>&1" 2>&1

echo ""
echo "=== Step 3: Create tunnel ==="
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" -p 22182 root@69.30.85.135 bash <<'REMOTE'
pkill -f "ssh.*-L.*8889" 2>/dev/null
sleep 1
ssh -o StrictHostKeyChecking=no -i /root/.ssh/id_ed25519 -f -N \
    -L 8889:localhost:8889 -p 30646 root@38.80.152.148 2>&1
echo "Tunnel command exit: $?"
sleep 3
echo "Testing tunnel..."
curl -s http://localhost:8889/health 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "FAILED: cannot reach worker"
REMOTE
