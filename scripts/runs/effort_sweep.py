#!/usr/bin/env python3
"""Reasoning effort sweep: eval gpt-oss-20b and gpt-oss-120b at different reasoning efforts.

Runs both reasonif (300 samples, 6 modes) and cotcontrol (100 stratified, 10 modes)
for each model × effort combination.

Usage:
    python scripts/runs/effort_sweep.py
    python scripts/runs/effort_sweep.py --dry-run
    python scripts/runs/effort_sweep.py --models gpt-oss-20b
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from controllability.config import ExperimentConfig
from controllability.evals.runner import run_experiment

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

EFFORT_SWEEP = [
    {"model": "openai/gpt-oss-20b", "efforts": ["low", "medium", "high"]},
    {"model": "openai/gpt-oss-120b", "efforts": ["low", "high"]},
]

DATASET_MODES = {
    "cotcontrol": [
        "baseline", "word_suppression", "multiple_word_suppression",
        "uppercase_thinking", "lowercase_thinking", "alternating_case",
        "repeat_sentences", "end_of_sentence", "meow_between_words",
        "ignore_question",
    ],
    "reasonif": [
        "reasoning_language", "number_words", "english_capital",
        "end_checker", "json_format", "no_comma",
    ],
}

EVAL_DATASETS = [
    {"dataset": "reasonif", "modes": "all", "n_samples": None},
    {"dataset": "cotcontrol", "modes": "all", "n_samples": 100},
]


def _resolve_modes(modes_str: str, dataset: str) -> list[str]:
    if modes_str == "all":
        family = dataset.split("/")[0]
        return DATASET_MODES[family]
    return [m.strip() for m in modes_str.split(",")]


def main():
    parser = argparse.ArgumentParser(description="Reasoning effort sweep evals")
    parser.add_argument("--dry-run", action="store_true", help="Print plan only")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model short names (e.g., gpt-oss-20b)")
    parser.add_argument("--max-concurrency", type=int, default=100)
    args = parser.parse_args()

    # Filter models
    sweeps = list(EFFORT_SWEEP)
    if args.models:
        requested = {m.strip() for m in args.models.split(",")}
        sweeps = [s for s in sweeps if s["model"].split("/")[-1] in requested]
        if not sweeps:
            print(f"ERROR: No matching models for --models={args.models}")
            sys.exit(1)

    # Build job list
    jobs = []
    for sweep in sweeps:
        model = sweep["model"]
        for effort in sweep["efforts"]:
            for eval_cfg in EVAL_DATASETS:
                dataset = eval_cfg["dataset"]
                modes = _resolve_modes(eval_cfg["modes"], dataset)
                n_samples = eval_cfg["n_samples"]
                jobs.append({
                    "model": model,
                    "effort": effort,
                    "dataset": dataset,
                    "modes": modes,
                    "n_samples": n_samples,
                })

    # Dry run
    if args.dry_run:
        print(f"\nEffort Sweep: {len(jobs)} eval runs\n")
        for i, job in enumerate(jobs, 1):
            n = job["n_samples"] or "all"
            print(f"  {i:2d}. {job['model']} effort={job['effort']} "
                  f"dataset={job['dataset']} samples={n} modes={len(job['modes'])}")
        return

    # Run evals
    start = datetime.now(timezone.utc)
    logging.info("Starting effort sweep: %d eval runs", len(jobs))

    results = {}
    for i, job in enumerate(jobs, 1):
        label = f"{job['model']} effort={job['effort']} {job['dataset']}"
        logging.info("[%d/%d] %s", i, len(jobs), label)

        config = ExperimentConfig(
            model=job["model"],
            dataset=job["dataset"],
            modes=job["modes"],
            split="all",
            fraction=1.0,
            seed=42,
            backend="tinker",
            max_concurrency=args.max_concurrency,
            max_retries=3,
            max_tokens=28000,
            temperature=1.0,
            output_dir="results/rollouts",
            n_samples=job["n_samples"],
            reasoning_effort=job["effort"],
            request_timeout=420,
        )

        logging.info("  experiment_id=%s  output=%s", config.experiment_id, config.output_filename)

        try:
            asyncio.run(run_experiment(config))
            results[label] = "OK"
            logging.info("  DONE: %s", label)
        except Exception as e:
            results[label] = f"FAIL: {e}"
            logging.error("  FAIL: %s — %s", label, e, exc_info=True)

    # Summary
    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    logging.info("=" * 60)
    logging.info("Effort sweep complete — %.0fs total", elapsed)
    for label, status in results.items():
        logging.info("  %-50s %s", label, status)

    n_ok = sum(1 for s in results.values() if s == "OK")
    n_fail = sum(1 for s in results.values() if s.startswith("FAIL"))
    logging.info("  %d succeeded, %d failed", n_ok, n_fail)


if __name__ == "__main__":
    main()
