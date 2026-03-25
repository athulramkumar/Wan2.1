#!/bin/bash
# Create a coordinator pod on RunPod (1.3B model only)
# Uses RTX A4000 (16GB) or similar cheap GPU

API_KEY="${RUNPOD_API_KEY}"
if [ -z "$API_KEY" ]; then
    echo "ERROR: Set RUNPOD_API_KEY"
    exit 1
fi

# Try RTX A4000 first (cheapest that fits 1.3B)
for GPU in "NVIDIA RTX A4000" "NVIDIA A40" "NVIDIA GeForce RTX 4090" "NVIDIA L4"; do
    echo "Trying $GPU..."
    RESULT=$(curl -s -H "Authorization: Bearer $API_KEY" \
        -H "Content-Type: application/json" \
        -d "{
            \"query\": \"mutation(\$input: PodFindAndDeployOnDemandInput!) { podFindAndDeployOnDemand(input: \$input) { id } }\",
            \"variables\": {
                \"input\": {
                    \"name\": \"wan21-coordinator\",
                    \"gpuTypeId\": \"$GPU\",
                    \"gpuCount\": 1,
                    \"cloudType\": \"SECURE\",
                    \"dockerArgs\": \"bash -c 'if [ ! -d /workspace/wan2.1/Wan2.1/.git ]; then mkdir -p /workspace/wan2.1 && cd /workspace/wan2.1 && git clone https://github.com/athulramkumar/Wan2.1.git 2>&1; fi && cd /workspace/wan2.1/Wan2.1 && git fetch origin 2>/dev/null; git checkout distributed-inference 2>/dev/null; git pull origin distributed-inference 2>/dev/null; pip install -r requirements.txt -q 2>&1 | tail -3; pip install fastapi uvicorn requests pydantic -q; echo READY && python run_server.py --port 8888 --distributed --mode coordinator'\",
                    \"volumeInGb\": 80,
                    \"containerDiskInGb\": 20,
                    \"minVcpuCount\": 4,
                    \"minMemoryInGb\": 16,
                    \"imageName\": \"runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04\"
                }
            }
        }" \
        https://api.runpod.io/graphql 2>&1)

    POD_ID=$(echo "$RESULT" | python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('data',{}).get('podFindAndDeployOnDemand',{}).get('id',''))" 2>/dev/null)

    if [ -n "$POD_ID" ] && [ "$POD_ID" != "" ] && [ "$POD_ID" != "None" ]; then
        echo "Created coordinator pod: $POD_ID ($GPU)"
        echo "URL: https://${POD_ID}-8888.proxy.runpod.net"
        echo "$POD_ID" > /tmp/coordinator_pod_id
        exit 0
    else
        echo "  Not available"
    fi
done

echo "ERROR: No GPUs available for coordinator"
exit 1
