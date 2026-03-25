#!/bin/bash
# Run all hybrid compute experiments
# H100 worker at localhost:18889, RTX4000 coordinator at localhost:18890
set -e

COORD="http://localhost:18890"
# Worker URL as seen BY THE COORDINATOR (via SSH tunnel on coordinator machine)
WORKER_ON_COORD="http://localhost:8889"
SEED=42
PROMPT="The Merced river is overflowing, birds flying in the sky, camera is zooming out to reveal an American Buffalo bathing in the river"
OUTDIR="experiments/compute_comparison"
mkdir -p "$OUTDIR"

submit_and_wait() {
    local name="$1"; local payload="$2"; local vidfile="$3"; local api="$4"

    echo ""
    echo "=========================================="
    echo "  $name"
    echo "=========================================="
    echo "  API: $api"

    # Wait for queue to be empty
    for attempt in $(seq 1 60); do
        QUEUE=$(curl -s "$api/health" 2>/dev/null | python3 -c "import json,sys;print(json.load(sys.stdin).get('queue_size',0))" 2>/dev/null)
        [ "$QUEUE" = "0" ] && break
        echo "  Queue not empty ($QUEUE), waiting..."
        sleep 10
    done

    RESULT=$(curl -s -X POST "$api/generate" -H "Content-Type: application/json" -d "$payload" 2>/dev/null)
    JOB_ID=$(echo "$RESULT" | python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('job_id',''))" 2>/dev/null)

    if [ -z "$JOB_ID" ] || [ "$JOB_ID" = "" ]; then
        echo "  ERROR submitting: $RESULT"
        return 1
    fi
    echo "  Job: $JOB_ID"

    local start_time=$(date +%s)
    while true; do
        STATUS=$(curl -s "$api/status/$JOB_ID" 2>/dev/null)
        STATE=$(echo "$STATUS" | python3 -c "import json,sys;print(json.load(sys.stdin).get('status','unknown'))" 2>/dev/null)
        PROGRESS=$(echo "$STATUS" | python3 -c "import json,sys;print(json.load(sys.stdin).get('progress',0))" 2>/dev/null)
        MODEL=$(echo "$STATUS" | python3 -c "import json,sys;print(json.load(sys.stdin).get('model_in_use',''))" 2>/dev/null)

        if [ "$STATE" = "completed" ]; then
            GEN_TIME=$(echo "$STATUS" | python3 -c "import json,sys;print(f'{json.load(sys.stdin).get(\"generation_time\",0):.1f}')")
            echo ""
            echo "  DONE: ${GEN_TIME}s"
            curl -s "$api/video/$JOB_ID" -o "$OUTDIR/$vidfile" 2>/dev/null
            echo "$STATUS" > "$OUTDIR/${vidfile%.mp4}_status.json"
            echo "  Video: $OUTDIR/$vidfile"
            return 0
        elif [ "$STATE" = "failed" ]; then
            ERROR=$(echo "$STATUS" | python3 -c "import json,sys;print(json.load(sys.stdin).get('error','unknown'))" 2>/dev/null)
            echo ""
            echo "  FAILED: $ERROR"
            return 1
        fi

        local elapsed=$(( $(date +%s) - start_time ))
        printf "\r  [%3s%%] %-20s (%ds)" "$PROGRESS" "$MODEL" "$elapsed"
        sleep 5
    done
}

echo "============================================"
echo "  HYBRID COMPUTE EXPERIMENTS"
echo "  Coordinator: $COORD (RTX4000, 1.3B)"
echo "  Worker: $WORKER (H100, 14B)"
echo "============================================"

# ─── Phase A: Baselines on H100 (need to use H100 directly for pure 14B) ───
# For pure 14B baseline we need a server with both models.
# Since H100 is running as worker-only, and RTX4000 has only 1.3B,
# we'll run the baseline via distributed mode: all 50 steps on 14B via worker.

# First, tell coordinator about the worker
echo ""
echo "=== Connecting coordinator to worker ==="
curl -s -X POST "$COORD/admin/set-worker-url" -H "Content-Type: application/json" \
    -d "{\"worker_url\":\"$WORKER_ON_COORD\"}" 2>/dev/null | python3 -m json.tool

# Exp 1: Baseline 14B (all steps on 14B via distributed — worker does everything)
submit_and_wait "Exp 1: Baseline 14B (H100 via distributed)" \
    "{\"prompt\":\"$PROMPT\",\"model\":\"baseline_14B\",\"sampling_steps\":50,\"frame_count\":81,\"fps\":16,\"seed\":$SEED}" \
    "exp1_baseline_14B.mp4" "$COORD"

# Exp 2: Hybrid 30/70 distributed (15 steps on H100 14B, 35 on RTX4000 1.3B)
submit_and_wait "Exp 2: Hybrid 30/70 distributed (RTX4000+H100)" \
    "{\"prompt\":\"$PROMPT\",\"model\":\"hybrid\",\"sampling_steps\":50,\"frame_count\":81,\"fps\":16,\"seed\":$SEED,\"schedule\":[[\"14B\",15],[\"1.3B\",35]]}" \
    "exp2_hybrid_30_70_distributed.mp4" "$COORD"

# Exp 3: Hybrid 20/80 distributed (10 steps on H100 14B, 40 on RTX4000 1.3B)
submit_and_wait "Exp 3: Hybrid 20/80 distributed (RTX4000+H100)" \
    "{\"prompt\":\"$PROMPT\",\"model\":\"hybrid\",\"sampling_steps\":50,\"frame_count\":81,\"fps\":16,\"seed\":$SEED,\"schedule\":[[\"14B\",10],[\"1.3B\",40]]}" \
    "exp3_hybrid_20_80_distributed.mp4" "$COORD"

# Exp 4: 1.3B only on RTX4000 (disable distributed for this)
echo ""
echo "=== Disabling distributed for 1.3B-only test ==="
curl -s -X POST "$COORD/admin/disable-distributed" 2>/dev/null | python3 -m json.tool

submit_and_wait "Exp 4: 1.3B Only (RTX4000)" \
    "{\"prompt\":\"$PROMPT\",\"model\":\"baseline_1.3B\",\"sampling_steps\":50,\"frame_count\":81,\"fps\":16,\"seed\":$SEED}" \
    "exp4_1.3B_only.mp4" "$COORD"

# Re-enable distributed for summary
curl -s -X POST "$COORD/admin/set-worker-url" -H "Content-Type: application/json" \
    -d "{\"worker_url\":\"$WORKER_ON_COORD\"}" 2>/dev/null > /dev/null

echo ""
echo "============================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "============================================"
echo ""
echo "Results:"
for f in "$OUTDIR"/exp*_status.json; do
    NAME=$(basename "$f" _status.json)
    TIME=$(python3 -c "import json;d=json.load(open('$f'));print(f'{d.get(\"generation_time\",0):.1f}s')" 2>/dev/null)
    echo "  $NAME: $TIME"
done
echo ""
echo "Videos:"
ls -lh "$OUTDIR"/*.mp4 2>/dev/null
