#!/usr/bin/env python3
"""Rerun 2 contaminated/incomplete dont_think_v2 rollouts from run 20260414_023958.

1. gpt-oss-20b-ft60 / user_dont_think_v2 — contaminated, deleted, redo from scratch
2. qwen3-32b / user_dont_think_v2 — 900/1000, resume fills missing 100

Hardcodes the original experiment_ids and filenames so resume works correctly.

Usage:
    uv run python scripts/runs/rerun_contaminated.py --dry-run
    uv run python scripts/runs/rerun_contaminated.py
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from controllability.config import MonitorQAConfig
from controllability.evals.runner import run_monitor_experiment

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "rollouts" / "monitor_qa"
DATASET_PATH = str(RESULTS_DIR / "monitorqa" / "monitorqa_val.jsonl")

RUN_ID = "20260414_023958"

# Two jobs with their original filenames (hardcoded experiment_ids from the original run)
JOBS = [
    {
        "label": "gpt-oss-20b-ft60",
        "actor_model": "openai/gpt-oss-20b",
        "model_path": "tinker://5ec3cd52-fe1e-579b-ae83-7bd8260330ec:train:0/sampler_weights/000060",
        "reasoning_effort": "medium",
        "prompt_type": "user_dont_think_v2",
        "filename": "monitor_qa_gpt-oss-20b-ft60_user_dont_think_v2_val_split_20260414_023958_04e85624ecfc.jsonl",
    },
    {
        "label": "qwen3-32b",
        "actor_model": "qwen/qwen3-32b",
        "model_path": None,
        "reasoning_effort": "medium",
        "prompt_type": "user_dont_think_v2",
        "filename": "monitor_qa_qwen3-32b_user_dont_think_v2_val_split_20260414_023958_1f4801954de5.jsonl",
    },
]


def make_config(job: dict) -> MonitorQAConfig:
    return MonitorQAConfig(
        actor_model=job["actor_model"],
        actor_backend="tinker",
        actor_max_tokens=28000,
        actor_temperature=1.0,
        actor_concurrency=500,
        actor_max_retries=1,
        monitor_model="openai/gpt-4o",
        monitor_max_tokens=1024,
        monitor_temperature=0.0,
        monitor_concurrency=60,
        monitor_max_retries=3,
        dataset_path=DATASET_PATH,
        dataset_split="val",
        n_samples=None,
        seed=42,
        prompt_type=job["prompt_type"],
        monitor_types=[
            "metr_note", "metr_note_gpt54", "main_task_grading",
            "reasoning_audit",
        ],
        monitor_model_overrides={
            "metr_note_gpt54": "openai/gpt-5.4",
            "main_task_grading": "openai/gpt-5.4-mini",
            "reasoning_audit": "openai/gpt-5.4",
        },
        modes=["side_task"],
        reasoning_effort=job["reasoning_effort"],
        request_timeout=900,
        model_path=job["model_path"],
        output_dir=str(OUTPUT_DIR),
        output_filename_override=job["filename"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)

    from transformers import AutoTokenizer  # noqa: F401

    for job in JOBS:
        config = make_config(job)
        filepath = OUTPUT_DIR / job["filename"]

        print(f"\n{'='*60}")
        print(f"Job: {job['label']} / {job['prompt_type']}")
        print(f"  File: {job['filename']}")
        print(f"  Exists: {filepath.exists()}")
        if filepath.exists():
            lines = sum(1 for _ in open(filepath))
            print(f"  Current lines: {lines}")

        if args.dry_run:
            print("  [DRY RUN] Skipping.")
            continue

        t0 = time.time()
        asyncio.run(run_monitor_experiment(config))
        elapsed = time.time() - t0

        if filepath.exists():
            lines = sum(1 for _ in open(filepath))
            print(f"  Done in {elapsed:.0f}s — {lines} lines")
        else:
            print(f"  WARNING: no output file after {elapsed:.0f}s")

    print(f"\n{'='*60}")
    print("All done.")


if __name__ == "__main__":
    main()
