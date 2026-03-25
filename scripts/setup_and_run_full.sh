#!/bin/bash
# Full experiment suite using:
#   2xA40 (69.30.85.135:22182) - runs as server with both models (mode=both)
#   H100 (38.80.152.148:30646) - runs as 14B worker
#
# Experiments:
#   1. 14B baseline on 2xA40 (local, all steps)
#   2. 14B baseline via H100 worker (distributed, all steps on H100)
#   3. Hybrid 30/70 distributed (14B on H100, 1.3B on 2xA40)
#   4. Hybrid 20/80 distributed (14B on H100, 1.3B on 2xA40)
#   5. 1.3B only on 2xA40

SSH_KEY="$HOME/.ssh/id_ed25519"
A40_HOST="69.30.85.135"; A40_PORT="22182"
H100_HOST="38.80.152.148"; H100_PORT="30646"
SRV_PORT=8890
WRK_PORT=8889
SEED=42
PROMPT="The Merced river is overflowing, birds flying in the sky, camera is zooming out to reveal an American Buffalo bathing in the river"
OUTDIR="experiments/compute_comparison"
mkdir -p "$OUTDIR"

echo "============================================"
echo "  HYBRID COMPUTE EXPERIMENTS"
echo "============================================"
echo "  2xA40: $A40_HOST:$A40_PORT (server)"
echo "  H100:  $H100_HOST:$H100_PORT (worker)"
echo "============================================"

# ─── Step 1: Start server on 2xA40 (mode=both) ───
echo ""
echo "=== Starting 2xA40 server (mode=both) ==="
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" -p $A40_PORT root@$A40_HOST bash <<'REMOTE'
cd /workspace/wan2.1/Wan2.1
pkill -f "run_server" 2>/dev/null; sleep 2
nohup python3 run_server.py --port 8890 > /tmp/wan_server.log 2>&1 &
echo "Server PID: $!"
REMOTE

# ─── Step 2: Start worker on H100 ───
echo ""
echo "=== Starting H100 worker ==="
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" -p $H100_PORT root@$H100_HOST bash <<'REMOTE'
cd /workspace/wan2.1/Wan2.1
pkill -f "run_worker" 2>/dev/null; sleep 2
nohup python3 run_worker.py --port 8889 > /tmp/wan_worker.log 2>&1 &
echo "Worker PID: $!"
REMOTE

# ─── Step 3: SSH tunnels ───
echo ""
echo "=== Setting up SSH tunnels ==="
pkill -f "ssh.*-N.*-L" 2>/dev/null; sleep 1

# Local -> 2xA40 server
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" -f -N -L 18890:localhost:$SRV_PORT -p $A40_PORT root@$A40_HOST
echo "  localhost:18890 -> 2xA40:$SRV_PORT"

# Local -> H100 worker
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" -f -N -L 18889:localhost:$WRK_PORT -p $H100_PORT root@$H100_HOST
echo "  localhost:18889 -> H100:$WRK_PORT"

# 2xA40 -> H100 worker (so coordinator can reach worker)
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" -p $A40_PORT root@$A40_HOST bash <<REMOTE2
pkill -f "ssh.*-L.*8889" 2>/dev/null; sleep 1
ssh -o StrictHostKeyChecking=no -i /root/.ssh/id_ed25519 -f -N -L $WRK_PORT:localhost:$WRK_PORT -p $H100_PORT root@$H100_HOST 2>&1
echo "  2xA40:$WRK_PORT -> H100:$WRK_PORT"
REMOTE2

# ─── Step 4: Wait for services ───
echo ""
echo "=== Waiting for 2xA40 server ==="
for i in $(seq 1 90); do
    sleep 10
    if curl -s http://localhost:18890/health 2>/dev/null | grep -q "healthy"; then
        echo "  2xA40 READY ($((i*10))s)"
        curl -s http://localhost:18890/health | python3 -c "import json,sys;d=json.load(sys.stdin);print(f'  Models: {d[\"models_loaded\"]}  GPU: {d[\"gpu_memory_used\"]:.1f}GB')"
        break
    fi
    echo "  Loading... ($((i*10))s)"
done

echo ""
echo "=== Waiting for H100 worker ==="
for i in $(seq 1 90); do
    sleep 10
    if curl -s http://localhost:18889/health 2>/dev/null | grep -q "ok"; then
        echo "  H100 worker READY ($((i*10))s)"
        break
    fi
    echo "  Loading... ($((i*10))s)"
done

# ─── Experiment helper ───
API="http://localhost:18890"

run_exp() {
    local name="$1"; local payload="$2"; local vidfile="$3"
    echo ""
    echo "=========================================="
    echo "  $name"
    echo "=========================================="

    RESULT=$(curl -s -X POST "$API/generate" -H "Content-Type: application/json" -d "$payload")
    JOB_ID=$(echo "$RESULT" | python3 -c "import json,sys;print(json.load(sys.stdin).get('job_id',''))" 2>/dev/null)
    if [ -z "$JOB_ID" ] || [ "$JOB_ID" = "" ]; then
        echo "  ERROR: $RESULT"
        return 1
    fi
    echo "  Job: $JOB_ID"

    while true; do
        STATUS=$(curl -s "$API/status/$JOB_ID" 2>/dev/null)
        STATE=$(echo "$STATUS" | python3 -c "import json,sys;print(json.load(sys.stdin).get('status','unknown'))" 2>/dev/null)
        PROGRESS=$(echo "$STATUS" | python3 -c "import json,sys;print(json.load(sys.stdin).get('progress',0))" 2>/dev/null)
        MODEL=$(echo "$STATUS" | python3 -c "import json,sys;print(json.load(sys.stdin).get('model_in_use',''))" 2>/dev/null)

        if [ "$STATE" = "completed" ]; then
            GEN_TIME=$(echo "$STATUS" | python3 -c "import json,sys;print(f'{json.load(sys.stdin).get(\"generation_time\",0):.1f}')")
            echo ""
            echo "  DONE: ${GEN_TIME}s"
            curl -s "$API/video/$JOB_ID" -o "$OUTDIR/$vidfile" 2>/dev/null
            echo "$STATUS" > "$OUTDIR/${vidfile%.mp4}_status.json"
            return 0
        elif [ "$STATE" = "failed" ]; then
            ERROR=$(echo "$STATUS" | python3 -c "import json,sys;print(json.load(sys.stdin).get('error','unknown'))" 2>/dev/null)
            echo ""
            echo "  FAILED: $ERROR"
            return 1
        fi
        printf "\r  [%3s%%] %-20s" "$PROGRESS" "$MODEL"
        sleep 5
    done
}

# ─── Phase 1: Local baselines on 2xA40 (no distributed) ───
echo ""
echo "=========================================="
echo "  PHASE 1: LOCAL BASELINES (2xA40)"
echo "=========================================="
curl -s -X POST "$API/admin/disable-distributed" > /dev/null

run_exp "Exp 1: 14B Baseline (2xA40 local)" \
    "{\"prompt\":\"$PROMPT\",\"model\":\"baseline_14B\",\"sampling_steps\":50,\"frame_count\":81,\"fps\":16,\"seed\":$SEED}" \
    "exp1_14B_2xA40_local.mp4"

run_exp "Exp 5: 1.3B Only (2xA40 local)" \
    "{\"prompt\":\"$PROMPT\",\"model\":\"baseline_1.3B\",\"sampling_steps\":50,\"frame_count\":81,\"fps\":16,\"seed\":$SEED}" \
    "exp5_1.3B_2xA40_local.mp4"

# ─── Phase 2: Distributed (14B on H100, 1.3B on 2xA40) ───
echo ""
echo "=========================================="
echo "  PHASE 2: DISTRIBUTED (2xA40 + H100)"
echo "=========================================="
curl -s -X POST "$API/admin/set-worker-url" -H "Content-Type: application/json" \
    -d '{"worker_url":"http://localhost:8889"}' > /dev/null

run_exp "Exp 2: 14B via H100 worker (distributed)" \
    "{\"prompt\":\"$PROMPT\",\"model\":\"baseline_14B\",\"sampling_steps\":50,\"frame_count\":81,\"fps\":16,\"seed\":$SEED}" \
    "exp2_14B_H100_distributed.mp4"

run_exp "Exp 3: Hybrid 30/70 (H100 14B + 2xA40 1.3B)" \
    "{\"prompt\":\"$PROMPT\",\"model\":\"hybrid\",\"sampling_steps\":50,\"frame_count\":81,\"fps\":16,\"seed\":$SEED,\"schedule\":[[\"14B\",15],[\"1.3B\",35]]}" \
    "exp3_hybrid_30_70.mp4"

run_exp "Exp 4: Hybrid 20/80 (H100 14B + 2xA40 1.3B)" \
    "{\"prompt\":\"$PROMPT\",\"model\":\"hybrid\",\"sampling_steps\":50,\"frame_count\":81,\"fps\":16,\"seed\":$SEED,\"schedule\":[[\"14B\",10],[\"1.3B\",40]]}" \
    "exp4_hybrid_20_80.mp4"

# ─── Results ───
echo ""
echo "============================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "============================================"
echo ""
for f in "$OUTDIR"/exp*_status.json; do
    NAME=$(basename "$f" _status.json)
    TIME=$(python3 -c "import json;d=json.load(open('$f'));print(f'{d.get(\"generation_time\",0):.1f}s')" 2>/dev/null)
    echo "  $NAME: $TIME"
done
echo ""
ls -lh "$OUTDIR"/*.mp4 2>/dev/null
