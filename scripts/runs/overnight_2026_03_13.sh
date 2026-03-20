#!/usr/bin/env bash
# Overnight eval runs - 2026-03-13
# Max 3 models per eval concurrently to avoid Tinker overload.
#
# Plan:
# Batch 1: ReasonIF - 4 Qwen3.5 models (3 concurrent, then 1)
# Batch 2: CoTControl - qwen3.5-397b-a17b, qwen3.5-35b-a3b, qwen3.5-27b (100 samples each)
# Batch 3: CoTControl - qwen3.5-4b, qwen3-32b (100 samples each)
# Batch 4: CoTControl top-up - qwen3-8b, gpt-oss-20b (extra 50 to reach 100 each)

set -e
cd "$(dirname "$0")/.."

echo "=== Overnight Eval Runs - $(date) ==="

# ---------------------------------------------------------------------------
# Batch 1: ReasonIF - 4 Qwen3.5 models (3 + 1)
# ---------------------------------------------------------------------------
echo ""
echo "=== BATCH 1a: ReasonIF - 3 Qwen3.5 models ==="
echo "Started: $(date)"

uv run python scripts/run_eval.py \
  --model qwen/qwen3.5-397b-a17b \
  --dataset reasonif \
  --modes all \
  --split all \
  --backend tinker \
  --max-concurrency 30 &
PID1=$!

uv run python scripts/run_eval.py \
  --model qwen/qwen3.5-35b-a3b \
  --dataset reasonif \
  --modes all \
  --split all \
  --backend tinker \
  --max-concurrency 30 &
PID2=$!

uv run python scripts/run_eval.py \
  --model qwen/qwen3.5-27b \
  --dataset reasonif \
  --modes all \
  --split all \
  --backend tinker \
  --max-concurrency 30 &
PID3=$!

wait $PID1 $PID2 $PID3
echo "Batch 1a done: $(date)"

echo ""
echo "=== BATCH 1b: ReasonIF - qwen3.5-4b ==="
echo "Started: $(date)"

uv run python scripts/run_eval.py \
  --model qwen/qwen3.5-4b \
  --dataset reasonif \
  --modes all \
  --split all \
  --backend tinker \
  --max-concurrency 30

echo "Batch 1b done: $(date)"

# ---------------------------------------------------------------------------
# Batch 2: CoTControl - 3 Qwen3.5 models (100 samples each)
# ---------------------------------------------------------------------------
echo ""
echo "=== BATCH 2: CoTControl - 3 Qwen3.5 models (100 samples) ==="
echo "Started: $(date)"

uv run python scripts/run_eval.py \
  --model qwen/qwen3.5-397b-a17b \
  --dataset cotcontrol \
  --modes all \
  --split all \
  --n-samples 100 \
  --backend tinker \
  --max-concurrency 30 &
PID1=$!

uv run python scripts/run_eval.py \
  --model qwen/qwen3.5-35b-a3b \
  --dataset cotcontrol \
  --modes all \
  --split all \
  --n-samples 100 \
  --backend tinker \
  --max-concurrency 30 &
PID2=$!

uv run python scripts/run_eval.py \
  --model qwen/qwen3.5-27b \
  --dataset cotcontrol \
  --modes all \
  --split all \
  --n-samples 100 \
  --backend tinker \
  --max-concurrency 30 &
PID3=$!

wait $PID1 $PID2 $PID3
echo "Batch 2 done: $(date)"

# ---------------------------------------------------------------------------
# Batch 3: CoTControl - qwen3.5-4b + qwen3-32b (100 samples each)
# ---------------------------------------------------------------------------
echo ""
echo "=== BATCH 3: CoTControl - qwen3.5-4b + qwen3-32b (100 samples) ==="
echo "Started: $(date)"

uv run python scripts/run_eval.py \
  --model qwen/qwen3.5-4b \
  --dataset cotcontrol \
  --modes all \
  --split all \
  --n-samples 100 \
  --backend tinker \
  --max-concurrency 30 &
PID1=$!

uv run python scripts/run_eval.py \
  --model qwen/qwen3-32b \
  --dataset cotcontrol \
  --modes all \
  --split all \
  --n-samples 100 \
  --backend tinker \
  --max-concurrency 30 &
PID2=$!

wait $PID1 $PID2
echo "Batch 3 done: $(date)"

# ---------------------------------------------------------------------------
# Batch 4: CoTControl - qwen3-8b + gpt-oss-20b (fresh 100-sample runs, replaces old 50-sample data)
# ---------------------------------------------------------------------------
echo ""
echo "=== BATCH 4: CoTControl - qwen3-8b + gpt-oss-20b (fresh 100 samples) ==="
echo "Started: $(date)"

uv run python scripts/run_eval.py \
  --model qwen/qwen3-8b \
  --dataset cotcontrol \
  --modes all \
  --split all \
  --n-samples 100 \
  --backend tinker \
  --max-concurrency 30 &
PID1=$!

uv run python scripts/run_eval.py \
  --model openai/gpt-oss-20b \
  --dataset cotcontrol \
  --modes all \
  --split all \
  --n-samples 100 \
  --backend tinker \
  --max-concurrency 30 &
PID2=$!

wait $PID1 $PID2
echo "Batch 4 done: $(date)"

echo ""
echo "=== All overnight runs complete: $(date) ==="
