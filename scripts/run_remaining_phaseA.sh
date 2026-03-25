#!/bin/bash
# Run remaining Phase A experiments
API="http://localhost:18890"
SEED=42
PROMPT="The Merced river is overflowing, birds flying in the sky, camera is zooming out to reveal an American Buffalo bathing in the river"
OUTDIR="experiments/compute_comparison"

submit_and_wait() {
    local name="$1"
    local payload="$2"
    local vidfile="$3"

    echo ""
    echo "=========================================="
    echo "  $name"
    echo "=========================================="

    # Check server is healthy first
    HEALTH=$(curl -s "$API/health" 2>/dev/null)
    if ! echo "$HEALTH" | grep -q "healthy"; then
        echo "  ERROR: Server not healthy!"
        echo "  $HEALTH"
        return 1
    fi
    QUEUE=$(echo "$HEALTH" | python3 -c "import json,sys;print(json.load(sys.stdin).get('queue_size',0))" 2>/dev/null)
    if [ "$QUEUE" != "0" ]; then
        echo "  Waiting for queue to clear (size=$QUEUE)..."
        while true; do
            sleep 10
            QUEUE=$(curl -s "$API/health" | python3 -c "import json,sys;print(json.load(sys.stdin).get('queue_size',0))" 2>/dev/null)
            [ "$QUEUE" = "0" ] && break
        done
    fi

    RESULT=$(curl -s -X POST "$API/generate" -H "Content-Type: application/json" -d "$payload" 2>/dev/null)
    JOB_ID=$(echo "$RESULT" | python3 -c "import json,sys;print(json.load(sys.stdin).get('job_id',''))" 2>/dev/null)

    if [ -z "$JOB_ID" ] || [ "$JOB_ID" = "" ]; then
        echo "  ERROR: Failed to submit job"
        echo "  $RESULT"
        return 1
    fi
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
            curl -s "$API/video/$JOB_ID" -o "$OUTDIR/$vidfile"
            echo "  Video: $OUTDIR/$vidfile ($(du -h "$OUTDIR/$vidfile" | cut -f1))"
            echo "$STATUS" > "$OUTDIR/${vidfile%.mp4}_status.json"
            return 0
        elif [ "$STATE" = "failed" ]; then
            echo ""
            echo "  FAILED: $(echo "$STATUS" | python3 -c "import json,sys;print(json.load(sys.stdin).get('error','unknown'))")"
            return 1
        fi

        printf "\r  [%3s%%] %-15s" "$PROGRESS" "$MODEL"
        sleep 5
    done
}

echo "Running remaining Phase A experiments..."

# Experiment 3: Hybrid 20/80 local
if [ ! -f "$OUTDIR/exp3_hybrid_20_80.mp4" ]; then
    submit_and_wait "Exp 3: Hybrid 20/80 local (H100)" \
        "{\"prompt\":\"$PROMPT\",\"model\":\"hybrid\",\"sampling_steps\":50,\"frame_count\":81,\"fps\":16,\"seed\":$SEED,\"schedule\":[[\"14B\",10],[\"1.3B\",40]]}" \
        "exp3_hybrid_20_80.mp4"
else
    echo "Exp 3 already done, skipping"
fi

# Experiment 10: 1.3B only
if [ ! -f "$OUTDIR/exp10_1.3B_only.mp4" ]; then
    submit_and_wait "Exp 10: 1.3B Only (H100)" \
        "{\"prompt\":\"$PROMPT\",\"model\":\"baseline_1.3B\",\"sampling_steps\":50,\"frame_count\":81,\"fps\":16,\"seed\":$SEED}" \
        "exp10_1.3B_only.mp4"
else
    echo "Exp 10 already done, skipping"
fi

echo ""
echo "=========================================="
echo "  Phase A Results:"
echo "=========================================="
for f in "$OUTDIR"/exp*_status.json; do
    NAME=$(basename "$f" _status.json)
    TIME=$(python3 -c "import json;d=json.load(open('$f'));print(f'{d.get(\"generation_time\",0):.1f}s')" 2>/dev/null)
    echo "  $NAME: $TIME"
done
