#!/bin/bash
# Quick update code on both machines (no restart of worker) and run experiments
SSH_KEY="$HOME/.ssh/id_ed25519"
SEED=42
PROMPT="The Merced river is overflowing, birds flying in the sky, camera is zooming out to reveal an American Buffalo bathing in the river"
OUTDIR="experiments/compute_comparison"
mkdir -p "$OUTDIR"
API="http://localhost:18890"

echo "=== Updating code ==="
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" -p 22182 root@69.30.85.135 \
    "cd /workspace/wan2.1/Wan2.1 && git pull origin distributed-inference 2>&1 | tail -3" 2>&1
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" -p 30646 root@38.80.152.148 \
    "cd /workspace/wan2.1/Wan2.1 && git reset --hard origin/distributed-inference 2>&1 | tail -1 && git pull origin distributed-inference 2>&1 | tail -3" 2>&1

echo "=== Restarting coordinator (schema fix) ==="
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" -p 22182 root@69.30.85.135 bash <<'REMOTE'
cd /workspace/wan2.1/Wan2.1
pkill -f "run_server" 2>/dev/null; sleep 2
nohup python3 run_server.py --port 8890 --distributed --mode coordinator > /tmp/wan_server.log 2>&1 &
echo "PID: $!"
REMOTE

echo "=== Waiting for coordinator ==="
for i in $(seq 1 30); do
    sleep 10
    if curl -s "$API/health" 2>/dev/null | grep -q "healthy"; then
        echo "READY ($((i*10))s)"
        break
    fi
    echo "  Loading... ($((i*10))s)"
done

echo "=== Setting worker URL ==="
curl -s -X POST "$API/admin/set-worker-url" -H "Content-Type: application/json" \
    -d '{"worker_url":"http://localhost:8889"}' 2>/dev/null | python3 -m json.tool

# Verify worker reachable
echo "=== Verifying worker reachable from coordinator ==="
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" -p 22182 root@69.30.85.135 \
    "curl -s http://localhost:8889/health 2>/dev/null | python3 -c 'import json,sys;d=json.load(sys.stdin);print(f\"Worker: {d[\\\"status\\\"]}\")'  2>/dev/null || echo 'CANNOT REACH WORKER'" 2>&1

run_exp() {
    local name="$1"; local payload="$2"; local vidfile="$3"
    echo ""
    echo "=========================================="
    echo "  $name"
    echo "=========================================="
    RESULT=$(curl -s -X POST "$API/generate" -H "Content-Type: application/json" -d "$payload")
    JOB_ID=$(echo "$RESULT" | python3 -c "import json,sys;print(json.load(sys.stdin).get('job_id',''))" 2>/dev/null)
    if [ -z "$JOB_ID" ] || [ "$JOB_ID" = "" ]; then echo "  ERROR: $RESULT"; return 1; fi
    echo "  Job: $JOB_ID"
    while true; do
        STATUS=$(curl -s "$API/status/$JOB_ID" 2>/dev/null)
        STATE=$(echo "$STATUS" | python3 -c "import json,sys;print(json.load(sys.stdin).get('status','unknown'))" 2>/dev/null)
        PROGRESS=$(echo "$STATUS" | python3 -c "import json,sys;print(json.load(sys.stdin).get('progress',0))" 2>/dev/null)
        MODEL=$(echo "$STATUS" | python3 -c "import json,sys;print(json.load(sys.stdin).get('model_in_use',''))" 2>/dev/null)
        if [ "$STATE" = "completed" ]; then
            GEN_TIME=$(echo "$STATUS" | python3 -c "import json,sys;print(f'{json.load(sys.stdin).get(\"generation_time\",0):.1f}')")
            echo ""; echo "  DONE: ${GEN_TIME}s"
            curl -s "$API/video/$JOB_ID" -o "$OUTDIR/$vidfile" 2>/dev/null
            echo "$STATUS" > "$OUTDIR/${vidfile%.mp4}_status.json"
            return 0
        elif [ "$STATE" = "failed" ]; then
            echo ""; echo "  FAILED: $(echo "$STATUS" | python3 -c "import json,sys;print(json.load(sys.stdin).get('error','?'))" 2>/dev/null)"
            return 1
        fi
        printf "\r  [%3s%%] %-20s" "$PROGRESS" "$MODEL"
        sleep 5
    done
}

echo ""
echo "=== RUNNING EXPERIMENTS ==="

# Exp 1: Hybrid 30/70 distributed
run_exp "Exp 1: Hybrid 30/70 (H100 14B + 2xA40 1.3B)" \
    "{\"prompt\":\"$PROMPT\",\"model\":\"hybrid\",\"sampling_steps\":50,\"frame_count\":81,\"fps\":16,\"seed\":$SEED,\"schedule\":[[\"14B\",15],[\"1.3B\",35]]}" \
    "exp1_hybrid_30_70.mp4"

# Exp 2: Hybrid 20/80 distributed
run_exp "Exp 2: Hybrid 20/80 (H100 14B + 2xA40 1.3B)" \
    "{\"prompt\":\"$PROMPT\",\"model\":\"hybrid\",\"sampling_steps\":50,\"frame_count\":81,\"fps\":16,\"seed\":$SEED,\"schedule\":[[\"14B\",10],[\"1.3B\",40]]}" \
    "exp2_hybrid_20_80.mp4"

# Exp 3: 1.3B only
curl -s -X POST "$API/admin/disable-distributed" > /dev/null
run_exp "Exp 3: 1.3B Only (2xA40)" \
    "{\"prompt\":\"$PROMPT\",\"model\":\"baseline_1.3B\",\"sampling_steps\":50,\"frame_count\":81,\"fps\":16,\"seed\":$SEED}" \
    "exp3_1.3B_only.mp4"

echo ""
echo "=========================================="
echo "  RESULTS"
echo "=========================================="
for f in "$OUTDIR"/exp*_status.json; do
    NAME=$(basename "$f" _status.json)
    TIME=$(python3 -c "import json;d=json.load(open('$f'));print(f'{d.get(\"generation_time\",0):.1f}s')" 2>/dev/null)
    echo "  $NAME: $TIME"
done
