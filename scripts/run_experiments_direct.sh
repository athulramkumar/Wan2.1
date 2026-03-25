#!/bin/bash
# Run Phase A experiments directly against H100 via SSH tunnel
# H100 server at localhost:18890 (tunneled)

API="http://localhost:18890"
SEED=42
PROMPT="The Merced river is overflowing, birds flying in the sky, camera is zooming out to reveal an American Buffalo bathing in the river"
OUTDIR="experiments/compute_comparison"
mkdir -p "$OUTDIR"

submit_and_wait() {
    local name="$1"
    local payload="$2"
    local vidfile="$3"

    echo ""
    echo "=========================================="
    echo "  $name"
    echo "=========================================="

    # Submit
    RESULT=$(curl -s -X POST "$API/generate" -H "Content-Type: application/json" -d "$payload")
    JOB_ID=$(echo "$RESULT" | python3 -c "import json,sys;print(json.load(sys.stdin)['job_id'])")
    echo "  Job: $JOB_ID"

    # Poll
    while true; do
        STATUS=$(curl -s "$API/status/$JOB_ID")
        STATE=$(echo "$STATUS" | python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('status','unknown'))")
        PROGRESS=$(echo "$STATUS" | python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('progress',0))")
        MODEL=$(echo "$STATUS" | python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('model_in_use',''))")

        if [ "$STATE" = "completed" ]; then
            GEN_TIME=$(echo "$STATUS" | python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('generation_time',0))")
            echo "  Completed in ${GEN_TIME}s"

            # Download video
            curl -s "$API/video/$JOB_ID" -o "$OUTDIR/$vidfile"
            echo "  Video: $OUTDIR/$vidfile"

            # Save status
            echo "$STATUS" | python3 -m json.tool > "$OUTDIR/${vidfile%.mp4}_status.json"
            break
        elif [ "$STATE" = "failed" ]; then
            echo "  FAILED!"
            echo "$STATUS" | python3 -m json.tool
            break
        fi

        printf "\r  [%3d%%] %s      " "$PROGRESS" "$MODEL"
        sleep 5
    done
}

echo "============================================"
echo "  PHASE A: H100 Single-GPU Baselines"
echo "  Server: $API"
echo "============================================"

# Experiment 1: Baseline 14B
submit_and_wait "Exp 1: Baseline 14B (50 steps)" \
    "{\"prompt\":\"$PROMPT\",\"model\":\"baseline_14B\",\"sampling_steps\":50,\"frame_count\":81,\"fps\":16,\"seed\":$SEED}" \
    "exp1_baseline_14B.mp4"

# Experiment 2: Hybrid 30/70 local
submit_and_wait "Exp 2: Hybrid 30/70 local (H100)" \
    "{\"prompt\":\"$PROMPT\",\"model\":\"hybrid\",\"sampling_steps\":50,\"frame_count\":81,\"fps\":16,\"seed\":$SEED,\"schedule\":[[\"14B\",15],[\"1.3B\",35]]}" \
    "exp2_hybrid_30_70.mp4"

# Experiment 3: Hybrid 20/80 local
submit_and_wait "Exp 3: Hybrid 20/80 local (H100)" \
    "{\"prompt\":\"$PROMPT\",\"model\":\"hybrid\",\"sampling_steps\":50,\"frame_count\":81,\"fps\":16,\"seed\":$SEED,\"schedule\":[[\"14B\",10],[\"1.3B\",40]]}" \
    "exp3_hybrid_20_80.mp4"

# Experiment 10: 1.3B only (on same H100 for now)
submit_and_wait "Exp 10: 1.3B Only" \
    "{\"prompt\":\"$PROMPT\",\"model\":\"baseline_1.3B\",\"sampling_steps\":50,\"frame_count\":81,\"fps\":16,\"seed\":$SEED}" \
    "exp10_1.3B_only.mp4"

echo ""
echo "============================================"
echo "  PHASE A COMPLETE"
echo "============================================"
echo "  Results in: $OUTDIR/"
ls -la "$OUTDIR/"*.mp4 2>/dev/null
