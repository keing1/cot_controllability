#!/usr/bin/env bash
# Remaining overnight eval runs - pipelined job queue (max 3 concurrent).
#
# Jobs:
#   CoTControl 100 samples: qwen3.5-4b, qwen3-32b, qwen3-8b, gpt-oss-20b
#
# Keeps 3 Tinker slots active at all times by starting the next job
# as soon as one finishes.

cd "$(dirname "$0")/.."

# Job definitions (one per line)
JOBS=(
  "uv run python scripts/run_eval.py --model qwen/qwen3.5-4b --dataset cotcontrol --modes all --split all --n-samples 100 --backend tinker --max-concurrency 30"
  "uv run python scripts/run_eval.py --model qwen/qwen3-32b --dataset cotcontrol --modes all --split all --n-samples 100 --backend tinker --max-concurrency 30"
  "uv run python scripts/run_eval.py --model qwen/qwen3-8b --dataset cotcontrol --modes all --split all --n-samples 100 --backend tinker --max-concurrency 30"
  "uv run python scripts/run_eval.py --model openai/gpt-oss-20b --dataset cotcontrol --modes all --split all --n-samples 100 --backend tinker --max-concurrency 30"
)

MAX_CONCURRENT=3
PIDS=()
JOB_IDX=0
TOTAL=${#JOBS[@]}

echo "=== Pipelined overnight runs - $(date) ==="
echo "Total jobs: $TOTAL, max concurrent: $MAX_CONCURRENT"

start_job() {
    local idx=$1
    echo ""
    echo "[Job $((idx+1))/$TOTAL] Starting: ${JOBS[$idx]}"
    echo "  Time: $(date)"
    ${JOBS[$idx]} &
    PIDS+=($!)
}

# Start initial batch
while [ $JOB_IDX -lt $MAX_CONCURRENT ] && [ $JOB_IDX -lt $TOTAL ]; do
    start_job $JOB_IDX
    JOB_IDX=$((JOB_IDX + 1))
done

# Wait for jobs to finish, start new ones as slots open
while [ ${#PIDS[@]} -gt 0 ]; do
    # Wait for any one PID to finish
    wait -n "${PIDS[@]}" 2>/dev/null
    EXIT_CODE=$?

    # Find which PID(s) finished
    NEW_PIDS=()
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            NEW_PIDS+=("$pid")
        else
            echo "[$(date)] Process $pid finished (exit $EXIT_CODE)"
        fi
    done
    PIDS=("${NEW_PIDS[@]}")

    # Start next job if available
    if [ $JOB_IDX -lt $TOTAL ]; then
        start_job $JOB_IDX
        JOB_IDX=$((JOB_IDX + 1))
    fi
done

echo ""
echo "=== All remaining runs complete: $(date) ==="
