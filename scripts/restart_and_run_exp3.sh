#!/bin/bash
# Restart H100 server and run experiment 3, then 10
H100="root@38.80.152.148"
PORT=30646
API="http://localhost:18890"
SEED=42
PROMPT="The Merced river is overflowing, birds flying in the sky, camera is zooming out to reveal an American Buffalo bathing in the river"
OUTDIR="experiments/compute_comparison"

echo "=== Restarting H100 server ==="
ssh -o StrictHostKeyChecking=no -p $PORT $H100 bash <<'REMOTE'
pkill -f "run_server.py" 2>/dev/null
sleep 5
# Clear GPU memory
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null
cd /workspace/wan2.1/Wan2.1
nohup python run_server.py --port 8890 > /tmp/wan_server.log 2>&1 &
echo "Server PID: $!"
REMOTE

echo "Waiting for server..."
for i in $(seq 1 90); do
    sleep 10
    if curl -s "$API/health" 2>/dev/null | grep -q "healthy"; then
        echo "Server ready! ($((i*10))s)"
        break
    fi
    echo "  Loading... ($((i*10))s)"
done

# Run Experiment 3
echo ""
echo "=========================================="
echo "  Exp 3: Hybrid 20/80 local (H100)"
echo "=========================================="
RESULT=$(curl -s -X POST "$API/generate" -H "Content-Type: application/json" \
    -d "{\"prompt\":\"$PROMPT\",\"model\":\"hybrid\",\"sampling_steps\":50,\"frame_count\":81,\"fps\":16,\"seed\":$SEED,\"schedule\":[[\"14B\",10],[\"1.3B\",40]]}")
JOB_ID=$(echo "$RESULT" | python3 -c "import json,sys;print(json.load(sys.stdin).get('job_id',''))" 2>/dev/null)
echo "  Job: $JOB_ID"

while true; do
    STATUS=$(curl -s "$API/status/$JOB_ID" 2>/dev/null)
    STATE=$(echo "$STATUS" | python3 -c "import json,sys;print(json.load(sys.stdin).get('status','unknown'))" 2>/dev/null)
    PROGRESS=$(echo "$STATUS" | python3 -c "import json,sys;print(json.load(sys.stdin).get('progress',0))" 2>/dev/null)
    MODEL=$(echo "$STATUS" | python3 -c "import json,sys;print(json.load(sys.stdin).get('model_in_use',''))" 2>/dev/null)

    if [ "$STATE" = "completed" ]; then
        GEN_TIME=$(echo "$STATUS" | python3 -c "import json,sys;print(json.load(sys.stdin).get('generation_time',0))")
        echo ""
        echo "  Completed in ${GEN_TIME}s"
        curl -s "$API/video/$JOB_ID" -o "$OUTDIR/exp3_hybrid_20_80.mp4"
        echo "$STATUS" > "$OUTDIR/exp3_hybrid_20_80_status.json"
        break
    elif [ "$STATE" = "failed" ]; then
        echo "  FAILED"
        echo "$STATUS"
        break
    fi
    printf "\r  [%3s%%] %-15s" "$PROGRESS" "$MODEL"
    sleep 5
done

# Wait a bit, then run Experiment 10
echo ""
echo "Waiting 10s between experiments..."
sleep 10

echo "=========================================="
echo "  Exp 10: 1.3B Only (H100)"
echo "=========================================="
RESULT=$(curl -s -X POST "$API/generate" -H "Content-Type: application/json" \
    -d "{\"prompt\":\"$PROMPT\",\"model\":\"baseline_1.3B\",\"sampling_steps\":50,\"frame_count\":81,\"fps\":16,\"seed\":$SEED}")
JOB_ID=$(echo "$RESULT" | python3 -c "import json,sys;print(json.load(sys.stdin).get('job_id',''))" 2>/dev/null)
echo "  Job: $JOB_ID"

while true; do
    STATUS=$(curl -s "$API/status/$JOB_ID" 2>/dev/null)
    STATE=$(echo "$STATUS" | python3 -c "import json,sys;print(json.load(sys.stdin).get('status','unknown'))" 2>/dev/null)
    PROGRESS=$(echo "$STATUS" | python3 -c "import json,sys;print(json.load(sys.stdin).get('progress',0))" 2>/dev/null)
    MODEL=$(echo "$STATUS" | python3 -c "import json,sys;print(json.load(sys.stdin).get('model_in_use',''))" 2>/dev/null)

    if [ "$STATE" = "completed" ]; then
        GEN_TIME=$(echo "$STATUS" | python3 -c "import json,sys;print(json.load(sys.stdin).get('generation_time',0))")
        echo ""
        echo "  Completed in ${GEN_TIME}s"
        curl -s "$API/video/$JOB_ID" -o "$OUTDIR/exp10_1.3B_only.mp4"
        echo "$STATUS" > "$OUTDIR/exp10_1.3B_only_status.json"
        break
    elif [ "$STATE" = "failed" ]; then
        echo "  FAILED"
        echo "$STATUS"
        break
    fi
    printf "\r  [%3s%%] %-15s" "$PROGRESS" "$MODEL"
    sleep 5
done

echo ""
echo "=========================================="
echo "  Phase A Results:"
echo "=========================================="
for f in "$OUTDIR"/exp*_status.json; do
    NAME=$(basename "$f" _status.json)
    TIME=$(python3 -c "import json;d=json.load(open('$f'));print(f'{d.get(\"generation_time\",0):.1f}s')" 2>/dev/null)
    echo "  $NAME: $TIME"
done
