#!/bin/bash
# Set up shared checkpoint directory between 2xA40 (coordinator) and H100 (worker)
# using SSHFS to mount H100's /workspace/checkpoints on 2xA40
SSH_KEY="$HOME/.ssh/id_ed25519"

echo "=== Creating checkpoint dir on H100 ==="
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" -p 30646 root@38.80.152.148 \
    "mkdir -p /workspace/checkpoints && echo 'Created'" 2>&1

echo ""
echo "=== Mounting H100 checkpoints dir on 2xA40 via SSHFS ==="
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" -p 22182 root@69.30.85.135 bash <<'REMOTE'
# Install sshfs if needed
which sshfs > /dev/null 2>&1 || apt-get install -y sshfs 2>&1 | tail -3

# Create mount point
mkdir -p /workspace/checkpoints

# Unmount if already mounted
fusermount -u /workspace/checkpoints 2>/dev/null

# Mount H100's checkpoint dir
sshfs -o StrictHostKeyChecking=no,IdentityFile=/root/.ssh/id_ed25519 \
    -p 30646 root@38.80.152.148:/workspace/checkpoints /workspace/checkpoints 2>&1

# Test
echo "test_$(date +%s)" > /workspace/checkpoints/test_write
echo "Write test: $(cat /workspace/checkpoints/test_write)"
rm /workspace/checkpoints/test_write
echo "SSHFS mounted successfully"
REMOTE
