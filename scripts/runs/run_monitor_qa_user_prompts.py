#!/usr/bin/env python3
"""Run monitor_qa evals with user_default and user_dont_think prompt settings.

Runs 8 evals in series (4 FT models × 2 prompt settings), with high
actor concurrency (500) per eval.

Models: stripped FT batch-60 checkpoints for gpt-oss-20b, gpt-oss-120b,
qwen3-8b, qwen3-32b.

Usage:
    uv run python scripts/runs/run_monitor_qa_user_prompts.py
    uv run python scripts/runs/run_monitor_qa_user_prompts.py --dry-run
    uv run python scripts/runs/run_monitor_qa_user_prompts.py --n-samples 5
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from controllability.config import MonitorQAConfig
from controllability.evals.runner import run_monitor_experiment

RESULTS_DIR = Path("results")
OUTPUT_DIR = RESULTS_DIR / "rollouts" / "monitor_qa"
DATASET_PATH = str(RESULTS_DIR / "monitorqa" / "gpqa_hle_w_math_monitorqa.jsonl")

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

# ---------------------------------------------------------------------------
# FT models (stripped SFT, lr1e-4, step 000060)
# ---------------------------------------------------------------------------

FT_MODELS = [
    {
        "label": "gpt-oss-20b-ft60",
        "actor_model": "openai/gpt-oss-20b",
        "model_path": "tinker://5ec3cd52-fe1e-579b-ae83-7bd8260330ec:train:0/sampler_weights/000060",
        "reasoning_effort": "medium",
    },
    {
        "label": "gpt-oss-120b-ft60",
        "actor_model": "openai/gpt-oss-120b",
        "model_path": "tinker://f745ff57-2fcf-54ad-9c19-3dfab62598c5:train:0/sampler_weights/000060",
        "reasoning_effort": "medium",
    },
    {
        "label": "qwen3-8b-ft60",
        "actor_model": "qwen/qwen3-8b",
        "model_path": "tinker://e3fea85f-5797-5a24-8c5e-af48cd0ad866:train:0/sampler_weights/000060",
        "reasoning_effort": "medium",
    },
    {
        "label": "qwen3-32b-ft60",
        "actor_model": "qwen/qwen3-32b",
        "model_path": "tinker://eb355212-c7a1-5f36-9db6-c526e1f33298:train:0/sampler_weights/000060",
        "reasoning_effort": "medium",
    },
]

# Prompt settings to eval
PROMPT_SETTINGS = ["user_default", "user_dont_think"]

ACTOR_CONCURRENCY = 500


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_single(model_cfg: dict, prompt_type: str, run_id: str,
               n_samples: int | None) -> str:
    label = model_cfg["label"]

    config = MonitorQAConfig(
        actor_model=model_cfg["actor_model"],
        actor_backend="tinker",
        actor_max_tokens=28000,
        actor_temperature=1.0,
        actor_concurrency=ACTOR_CONCURRENCY,
        actor_max_retries=1,
        monitor_model="openai/gpt-4o",
        monitor_max_tokens=1024,
        monitor_temperature=0.0,
        monitor_concurrency=60,
        monitor_max_retries=3,
        dataset_path=DATASET_PATH,
        n_samples=n_samples,
        seed=42,
        prompt_type=prompt_type,
        monitor_types=["metr_note", "metr_note_gpt54"],
        monitor_model_overrides={"metr_note_gpt54": "openai/gpt-5.4"},
        modes=["side_task"],
        reasoning_effort=model_cfg["reasoning_effort"],
        request_timeout=900,
        model_path=model_cfg["model_path"],
        output_dir=str(OUTPUT_DIR),
    )

    custom_filename = (
        f"monitor_qa_{label}_{prompt_type}"
        f"_{run_id}_{config.experiment_id}.jsonl"
    )
    config.output_filename_override = custom_filename

    key = f"{label}/{prompt_type}"
    logging.info("START: %s -> %s", key, custom_filename)

    try:
        asyncio.run(run_monitor_experiment(config))
        logging.info("DONE:  %s", key)
        return "OK"
    except Exception as e:
        logging.error("FAIL:  %s -> %s", key, e, exc_info=True)
        return f"FAIL ({e})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()

    run_id = args.run_id or RUN_ID

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    jobs = []
    for prompt_type in PROMPT_SETTINGS:
        for model_cfg in FT_MODELS:
            jobs.append((model_cfg, prompt_type))

    print(f"Monitor QA User-Prompt Evals: run_id={run_id}")
    print(f"  {len(jobs)} jobs, serial execution, actor_concurrency={ACTOR_CONCURRENCY}")
    print(f"  Prompt settings: {PROMPT_SETTINGS}")
    print(f"  Monitor types: metr_note (GPT-4.1), metr_note_gpt54 (GPT-5.4)")
    if args.n_samples:
        print(f"  n_samples: {args.n_samples}")
    print()

    for i, (model_cfg, prompt_type) in enumerate(jobs, 1):
        print(f"  {i}. {model_cfg['label']} / {prompt_type}")

    if args.dry_run:
        print("\n[DRY RUN] Exiting.")
        return

    print()
    results = {}
    t0 = time.time()

    for i, (model_cfg, prompt_type) in enumerate(jobs, 1):
        key = f"{model_cfg['label']}/{prompt_type}"
        print(f"\n{'=' * 60}")
        print(f"Job {i}/{len(jobs)}: {key}")
        print(f"{'=' * 60}")
        t_start = time.time()
        result = run_single(model_cfg, prompt_type, run_id, args.n_samples)
        elapsed = time.time() - t_start
        results[key] = result
        print(f"  -> {result} ({elapsed:.0f}s)")

    total = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"ALL DONE — {total:.0f}s total")
    print(f"{'=' * 60}")
    for key, result in results.items():
        print(f"  {key:<45s} {result}")


if __name__ == "__main__":
    main()
