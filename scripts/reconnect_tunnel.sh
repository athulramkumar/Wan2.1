#!/bin/bash
# Re-establish SSH tunnel to H100
pkill -f "ssh.*18890.*30646" 2>/dev/null
sleep 1
ssh -o StrictHostKeyChecking=no -f -N -L 18890:localhost:8890 -p 30646 root@38.80.152.148
echo "Tunnel re-established"
curl -s http://localhost:18890/health | python3 -m json.tool
