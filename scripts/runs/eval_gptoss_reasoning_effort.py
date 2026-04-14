#!/usr/bin/env python3
"""Eval gpt-oss base models on ReasonIF with analysis_channel across reasoning efforts.

Runs gpt-oss-20b and gpt-oss-120b on the reasonif dataset with analysis_channel=True
under three reasoning efforts: medium (first), then low, then high.
One eval at a time (sequential), with high concurrency within each eval.

Usage:
    python scripts/runs/eval_gptoss_reasoning_effort.py --dry-run
    python scripts/runs/eval_gptoss_reasoning_effort.py
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from controllability.config import ExperimentConfig
from controllability.evals.runner import run_experiment

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
ROLLOUTS_DIR = RESULTS_DIR / "rollouts"

MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]
EFFORT_ORDER = ["medium", "low", "high"]
DATASET = "reasonif"

REASONIF_MODES = [
    "reasoning_language", "number_words", "english_capital",
    "end_checker", "json_format", "no_comma",
]


def run_eval(model: str, effort: str, run_id: str) -> None:
    """Run a single eval: one model, one reasoning effort, reasonif dataset."""
    model_short = model.replace("/", "_")

    config = ExperimentConfig(
        model=model,
        dataset=DATASET,
        modes=REASONIF_MODES,
        split="all",
        fraction=1.0,
        seed=42,
        backend="tinker",
        max_concurrency=100,
        max_retries=3,
        max_tokens=28000,
        temperature=1.0,
        output_dir=str(ROLLOUTS_DIR),
        n_samples=None,
        reasoning_effort=effort,
        request_timeout=420,
        analysis_channel=True,
    )

    custom_filename = (
        f"{model_short}_reasonif_ac_{effort}"
        f"_{run_id}_{config.experiment_id}.jsonl"
    )

    config.output_filename_override = custom_filename

    output_path = ROLLOUTS_DIR / custom_filename
    logging.info("Eval: %s effort=%s -> %s", model, effort, output_path.name)
    asyncio.run(run_experiment(config))
    logging.info("Eval complete: %s effort=%s", model, effort)


def print_dry_run(run_id: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"DRY RUN — Eval Plan (run_id={run_id})")
    print(f"{'=' * 70}")
    print(f"\nDataset:          {DATASET}")
    print(f"analysis_channel: True")
    print(f"Concurrency:      sequential (1 eval at a time, max_concurrency=100 within)")
    print(f"Modes:            {', '.join(REASONIF_MODES)}")
    print(f"\nEval order (effort: {' -> '.join(EFFORT_ORDER)}):\n")

    for i, effort in enumerate(EFFORT_ORDER):
        for model in MODELS:
            model_short = model.replace("/", "_")
            config = ExperimentConfig(
                model=model, dataset=DATASET, modes=REASONIF_MODES,
                split="all", fraction=1.0, seed=42, backend="tinker",
                max_concurrency=100, max_retries=3, max_tokens=28000,
                temperature=1.0, output_dir=str(ROLLOUTS_DIR),
                reasoning_effort=effort, request_timeout=420,
                analysis_channel=True,
            )
            fname = (
                f"{model_short}_reasonif_ac_{effort}"
                f"_{run_id}_{config.experiment_id}.jsonl"
            )
            print(f"  {i * len(MODELS) + MODELS.index(model) + 1}. {model}  effort={effort}")
            print(f"     -> {fname}")
    print(f"\nTotal evals: {len(MODELS) * len(EFFORT_ORDER)}")
    print()


def _setup_logging() -> None:
    log_file = RESULTS_DIR / "eval_gptoss_reasoning_effort.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)


def main():
    parser = argparse.ArgumentParser(
        description="Eval gpt-oss models on ReasonIF across reasoning efforts"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if args.dry_run:
        print_dry_run(run_id)
        return

    _setup_logging()
    start_time = datetime.now(timezone.utc)

    logging.info(
        "Eval gpt-oss reasoning effort: %d models x %d efforts, run_id=%s",
        len(MODELS), len(EFFORT_ORDER), run_id,
    )

    results: dict[str, str] = {}
    for effort in EFFORT_ORDER:
        for model in MODELS:
            key = f"{model.split('/')[-1]}_{effort}"
            try:
                run_eval(model, effort, run_id)
                results[key] = "OK"
            except Exception as e:
                logging.error("FAILED: %s -> %s", key, e)
                results[key] = f"FAIL ({e})"

    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
    logging.info("")
    logging.info("=" * 70)
    logging.info("ALL EVALS COMPLETE — %.0fs total", elapsed)
    logging.info("=" * 70)
    for key, status in results.items():
        logging.info("  %-40s %s", key, status)
    n_ok = sum(1 for s in results.values() if s == "OK")
    n_fail = len(results) - n_ok
    logging.info("  %d succeeded, %d failed", n_ok, n_fail)


if __name__ == "__main__":
    main()
