#!/bin/bash
# Final experiment run:
#   2xA40 = coordinator (1.3B only, mode=coordinator)
#   H100 = worker (14B only)
SSH_KEY="$HOME/.ssh/id_ed25519"
A40_HOST="69.30.85.135"; A40_PORT="22182"
H100_HOST="38.80.152.148"; H100_PORT="30646"
SEED=42
PROMPT="The Merced river is overflowing, birds flying in the sky, camera is zooming out to reveal an American Buffalo bathing in the river"
OUTDIR="experiments/compute_comparison"
mkdir -p "$OUTDIR"

echo "============================================"
echo "  EXPERIMENT SUITE"
echo "  2xA40 coordinator (1.3B) + H100 worker (14B)"
echo "============================================"

# ─── Start services ───
echo "=== Starting 2xA40 as coordinator (1.3B only) ==="
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" -p $A40_PORT root@$A40_HOST bash <<'REMOTE'
cd /workspace/wan2.1/Wan2.1
pkill -f "run_server" 2>/dev/null; sleep 2
nohup python3 run_server.py --port 8890 --distributed --mode coordinator > /tmp/wan_server.log 2>&1 &
echo "PID: $!"
REMOTE

echo "=== Starting H100 as 14B worker ==="
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" -p $H100_PORT root@$H100_HOST bash <<'REMOTE'
cd /workspace/wan2.1/Wan2.1
pkill -f "run_worker" 2>/dev/null; sleep 2
nohup python3 run_worker.py --port 8889 > /tmp/wan_worker.log 2>&1 &
echo "PID: $!"
REMOTE

# ─── Tunnels ───
echo "=== Tunnels ==="
pkill -f "ssh.*-N.*-L" 2>/dev/null; sleep 1
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" -f -N -L 18890:localhost:8890 -p $A40_PORT root@$A40_HOST
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" -f -N -L 18889:localhost:8889 -p $H100_PORT root@$H100_HOST
# Coordinator -> Worker tunnel
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" -p $A40_PORT root@$A40_HOST \
    "pkill -f 'ssh.*-L.*8889' 2>/dev/null; sleep 1; ssh -o StrictHostKeyChecking=no -i /root/.ssh/id_ed25519 -f -N -L 8889:localhost:8889 -p $H100_PORT root@$H100_HOST" 2>&1
echo "  Tunnels ready"

# ─── Wait for services ───
echo "=== Waiting for H100 worker ==="
for i in $(seq 1 90); do
    sleep 10
    if curl -s http://localhost:18889/health 2>/dev/null | grep -q "ok"; then
        echo "  H100 worker READY ($((i*10))s)"
        break
    fi
    echo "  Loading... ($((i*10))s)"
done

echo "=== Waiting for 2xA40 coordinator ==="
for i in $(seq 1 90); do
    sleep 10
    if curl -s http://localhost:18890/health 2>/dev/null | grep -q "healthy"; then
        echo "  2xA40 coordinator READY ($((i*10))s)"
        break
    fi
    echo "  Loading... ($((i*10))s)"
done

API="http://localhost:18890"

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

# ─── Exp 1: 14B all via H100 (distributed, all 50 steps on worker) ───
echo ""
echo "=== Setting up distributed mode ==="
curl -s -X POST "$API/admin/set-worker-url" -H "Content-Type: application/json" \
    -d '{"worker_url":"http://localhost:8889"}' | python3 -c "import json,sys;print(json.load(sys.stdin))" 2>/dev/null

run_exp "Exp 1: 14B Baseline via H100 (all 50 steps on H100)" \
    "{\"prompt\":\"$PROMPT\",\"model\":\"baseline_14B\",\"sampling_steps\":50,\"frame_count\":81,\"fps\":16,\"seed\":$SEED}" \
    "exp1_14B_H100.mp4"

# ─── Exp 2: Hybrid 30/70 (H100 14B + 2xA40 1.3B) ───
run_exp "Exp 2: Hybrid 30/70 (H100+2xA40)" \
    "{\"prompt\":\"$PROMPT\",\"model\":\"hybrid\",\"sampling_steps\":50,\"frame_count\":81,\"fps\":16,\"seed\":$SEED,\"schedule\":[[\"14B\",15],[\"1.3B\",35]]}" \
    "exp2_hybrid_30_70.mp4"

# ─── Exp 3: Hybrid 20/80 (H100 14B + 2xA40 1.3B) ───
run_exp "Exp 3: Hybrid 20/80 (H100+2xA40)" \
    "{\"prompt\":\"$PROMPT\",\"model\":\"hybrid\",\"sampling_steps\":50,\"frame_count\":81,\"fps\":16,\"seed\":$SEED,\"schedule\":[[\"14B\",10],[\"1.3B\",40]]}" \
    "exp3_hybrid_20_80.mp4"

# ─── Exp 4: 1.3B only on 2xA40 ───
curl -s -X POST "$API/admin/disable-distributed" > /dev/null
run_exp "Exp 4: 1.3B Only (2xA40)" \
    "{\"prompt\":\"$PROMPT\",\"model\":\"baseline_1.3B\",\"sampling_steps\":50,\"frame_count\":81,\"fps\":16,\"seed\":$SEED}" \
    "exp4_1.3B_only.mp4"

# ─── Results ───
echo ""
echo "============================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "============================================"
for f in "$OUTDIR"/exp*_status.json; do
    NAME=$(basename "$f" _status.json)
    TIME=$(python3 -c "import json;d=json.load(open('$f'));print(f'{d.get(\"generation_time\",0):.1f}s')" 2>/dev/null)
    echo "  $NAME: $TIME"
done
echo ""
ls -lh "$OUTDIR"/*.mp4 2>/dev/null
