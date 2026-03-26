#!/usr/bin/env python3
"""Evaluate gpt-oss FT checkpoints (lr=1e-4, step 60) at low and high reasoning effort.

Runs 8 evals total: 2 models x 2 efforts x 2 datasets, 3 concurrent at a time.
Order: gpt-oss-120b first, then gpt-oss-20b.

Usage:
    python scripts/runs/effort_eval_ft_ckpts.py
    python scripts/runs/effort_eval_ft_ckpts.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from controllability.config import ExperimentConfig
from controllability.evals.runner import run_experiment

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

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
    {"dataset": "reasonif", "modes": DATASET_MODES["reasonif"], "n_samples": None},
    {"dataset": "cotcontrol", "modes": DATASET_MODES["cotcontrol"], "n_samples": 100},
]


@dataclass
class EvalJob:
    label: str
    model: str
    model_path: str
    dataset: str
    modes: list[str]
    n_samples: int | None
    reasoning_effort: str


# Step-60 checkpoints from the lr=1e-4 run (20260319_093421)
JOBS: list[EvalJob] = []

for effort in ["low", "high"]:
    for eval_cfg in EVAL_DATASETS:
        JOBS.append(EvalJob(
            label=f"gpt-oss-120b-rif-step60 effort={effort} {eval_cfg['dataset']}",
            model="openai/gpt-oss-120b",
            model_path="tinker://1cbfc28d-caa9-5008-89c3-d7cf8ce7f349:train:0/sampler_weights/000060",
            dataset=eval_cfg["dataset"],
            modes=eval_cfg["modes"],
            n_samples=eval_cfg["n_samples"],
            reasoning_effort=effort,
        ))

for effort in ["low", "high"]:
    for eval_cfg in EVAL_DATASETS:
        JOBS.append(EvalJob(
            label=f"gpt-oss-20b-rif-step60 effort={effort} {eval_cfg['dataset']}",
            model="openai/gpt-oss-20b",
            model_path="tinker://6adcf41e-e791-573d-baa1-57d773d8ebef:train:0/sampler_weights/000060",
            dataset=eval_cfg["dataset"],
            modes=eval_cfg["modes"],
            n_samples=eval_cfg["n_samples"],
            reasoning_effort=effort,
        ))


async def run_one(job: EvalJob) -> tuple[str, str]:
    """Run a single eval job, return (label, status)."""
    config = ExperimentConfig(
        model=job.model,
        dataset=job.dataset,
        modes=job.modes,
        split="all",
        fraction=1.0,
        seed=42,
        backend="tinker",
        max_concurrency=100,
        max_retries=3,
        max_tokens=28000,
        temperature=1.0,
        output_dir="results/rollouts",
        n_samples=job.n_samples,
        model_path=job.model_path,
        reasoning_effort=job.reasoning_effort,
        request_timeout=420,
    )

    logging.info("START: %s  (experiment_id=%s, output=%s)",
                 job.label, config.experiment_id, config.output_filename)
    try:
        await run_experiment(config)
        logging.info("DONE:  %s", job.label)
        return job.label, "OK"
    except Exception as e:
        logging.error("FAIL:  %s — %s", job.label, e, exc_info=True)
        return job.label, f"FAIL: {e}"


async def run_with_concurrency(jobs: list[EvalJob], max_concurrent: int = 3):
    """Run jobs with a concurrency limit, starting new ones as old ones finish."""
    semaphore = asyncio.Semaphore(max_concurrent)
    results: dict[str, str] = {}

    async def limited_run(job: EvalJob):
        async with semaphore:
            label, status = await run_one(job)
            results[label] = status

    tasks = [asyncio.create_task(limited_run(job)) for job in jobs]
    await asyncio.gather(*tasks)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Eval FT checkpoints at different reasoning efforts"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-concurrent", type=int, default=3)
    args = parser.parse_args()

    if args.dry_run:
        print(f"\n{len(JOBS)} eval jobs (max {args.max_concurrent} concurrent):\n")
        for i, job in enumerate(JOBS, 1):
            n = job.n_samples or "all"
            print(f"  {i}. {job.label}  samples={n}  modes={len(job.modes)}")
        return

    start = datetime.now(timezone.utc)
    logging.info("Starting %d evals (max %d concurrent)", len(JOBS), args.max_concurrent)

    results = asyncio.run(run_with_concurrency(JOBS, args.max_concurrent))

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    logging.info("=" * 60)
    logging.info("Complete — %.0fs total", elapsed)
    for label, status in results.items():
        logging.info("  %-55s %s", label, status)
    n_ok = sum(1 for s in results.values() if s == "OK")
    n_fail = sum(1 for s in results.values() if s.startswith("FAIL"))
    logging.info("  %d succeeded, %d failed", n_ok, n_fail)


if __name__ == "__main__":
    main()
