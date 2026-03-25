#!/bin/bash
# Copy SSH key to 2xA40 so it can tunnel to H100
SSH_KEY="$HOME/.ssh/id_ed25519"
scp -o StrictHostKeyChecking=no -i "$SSH_KEY" -P 22182 \
    "$SSH_KEY" root@69.30.85.135:/root/.ssh/id_ed25519
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" -p 22182 root@69.30.85.135 \
    "chmod 600 /root/.ssh/id_ed25519 && echo 'SSH key copied'"
