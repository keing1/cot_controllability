#!/bin/bash
# Run calibration for all base models.
# GPT-OSS models: one run per reasoning effort (low, medium, high)
# OpenRouter models: single run (default high effort)
#
# Usage: bash scripts/runs/run_all_calibrations.sh

set -e
cd "$(dirname "$0")/../.."

SCRIPT=".venv/bin/python scripts/runs/calibrate_word_count.py"
COMMON="--n-runs 3 --max-concurrency 100 --max-tokens 28000 --temperature 1.0"

echo "=== GPT-OSS models (tinker, per-effort) ==="

echo "--- gpt-oss-20b @ low ---"
$SCRIPT --model openai/gpt-oss-20b --backend tinker --reasoning-effort low $COMMON

echo "--- gpt-oss-20b @ medium ---"
$SCRIPT --model openai/gpt-oss-20b --backend tinker --reasoning-effort medium $COMMON

echo "--- gpt-oss-20b @ high ---"
$SCRIPT --model openai/gpt-oss-20b --backend tinker --reasoning-effort high $COMMON

echo "--- gpt-oss-120b @ low ---"
$SCRIPT --model openai/gpt-oss-120b --backend tinker --reasoning-effort low $COMMON

echo "--- gpt-oss-120b @ medium ---"
$SCRIPT --model openai/gpt-oss-120b --backend tinker --reasoning-effort medium $COMMON

echo "--- gpt-oss-120b @ high ---"
$SCRIPT --model openai/gpt-oss-120b --backend tinker --reasoning-effort high $COMMON

echo "=== Qwen models (openrouter) ==="

echo "--- qwen3-8b ---"
$SCRIPT --model qwen/qwen3-8b --backend openrouter $COMMON

echo "--- qwen3-32b ---"
$SCRIPT --model qwen/qwen3-32b --backend openrouter $COMMON

echo "--- qwen3-235b-a22b ---"
$SCRIPT --model qwen/qwen3-235b-a22b --backend openrouter $COMMON

echo "--- qwen3.5-4b ---"
$SCRIPT --model qwen/qwen3.5-4b --backend openrouter $COMMON

echo "--- qwen3.5-27b ---"
$SCRIPT --model qwen/qwen3.5-27b --backend openrouter $COMMON

echo "--- qwen3.5-35b-a3b ---"
$SCRIPT --model qwen/qwen3.5-35b-a3b --backend openrouter $COMMON

echo "--- qwen3.5-397b-a17b ---"
$SCRIPT --model qwen/qwen3.5-397b-a17b --backend openrouter $COMMON

echo "=== Moonshot models (openrouter) ==="

echo "--- kimi-k2-thinking ---"
$SCRIPT --model moonshotai/kimi-k2-thinking --backend openrouter $COMMON

echo "--- kimi-k2.5 ---"
$SCRIPT --model moonshotai/kimi-k2.5 --backend openrouter $COMMON

echo "=== ALL CALIBRATIONS COMPLETE ==="
