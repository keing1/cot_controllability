#!/usr/bin/env python3
"""Overnight batch evaluation orchestrator.

Runs 11 models × 2 datasets (ReasonIF + CoTControl) with:
  - 32K max tokens, 7-min timeout, temperature=1.0, seed=42
  - Max 3 models running concurrently via asyncio.Semaphore
  - Resume-safe: re-running skips completed (sample_id, mode) pairs

Usage:
    python scripts/overnight_batch.py
    python scripts/overnight_batch.py --max-concurrent 2
    python scripts/overnight_batch.py --dry-run
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from controllability.evals.batch_eval import ModelEvalSpec, run_model_eval

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

ALL_MODELS: list[ModelEvalSpec] = [
    # Already done: qwen3-8b, qwen3-32b, qwen3.5-4b (will resume-skip)
    ModelEvalSpec(model="qwen/qwen3-8b", backend="tinker", max_concurrency=30),
    ModelEvalSpec(model="qwen/qwen3-32b", backend="tinker", max_concurrency=30),
    ModelEvalSpec(model="qwen/qwen3.5-4b", backend="tinker", max_concurrency=30),
    # Prioritized: gpt-oss-120b at position 4
    ModelEvalSpec(model="openai/gpt-oss-120b", backend="tinker", max_concurrency=30, reasoning_effort="medium"),
    # Resuming in-progress
    ModelEvalSpec(model="qwen/qwen3.5-27b", backend="tinker", max_concurrency=30),
    ModelEvalSpec(model="qwen/qwen3.5-35b-a3b", backend="tinker", max_concurrency=30),
    ModelEvalSpec(model="qwen/qwen3.5-397b-a17b", backend="tinker", max_concurrency=30),
    # Remaining
    ModelEvalSpec(model="qwen/qwen3-235b-a22b", backend="openrouter", max_concurrency=60),
    # gpt-oss-20b skipped — already has full base runs
    ModelEvalSpec(model="moonshotai/kimi-k2.5", backend="openrouter", max_concurrency=60),
    ModelEvalSpec(model="moonshotai/kimi-k2-thinking", backend="openrouter", max_concurrency=60),
]

# Shared defaults applied to all specs
for spec in ALL_MODELS:
    spec.max_tokens = 28000 if spec.backend == "tinker" else 32768
    spec.request_timeout = 420
    spec.temperature = 1.0
    spec.seed = 42
    spec.cotcontrol_n_samples = 100


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def run_with_semaphore(
    sem: asyncio.Semaphore,
    spec: ModelEvalSpec,
    results: dict[str, str],
) -> None:
    """Run a single model eval, gated by the semaphore."""
    async with sem:
        model = spec.model
        start = datetime.now(timezone.utc)
        logging.info("START  %s (%s)", model, spec.backend)
        try:
            paths = await run_model_eval(spec)
            elapsed = (datetime.now(timezone.utc) - start).total_seconds()
            results[model] = f"OK ({len(paths)} files, {elapsed:.0f}s)"
            logging.info("DONE   %s — %s", model, results[model])
        except Exception as e:
            elapsed = (datetime.now(timezone.utc) - start).total_seconds()
            results[model] = f"FAIL ({type(e).__name__}: {e}, {elapsed:.0f}s)"
            logging.error("FAIL   %s — %s", model, results[model])


async def main(max_concurrent: int = 3, dry_run: bool = False) -> None:
    start_time = datetime.now(timezone.utc)
    logging.info(
        "Overnight batch: %d models, max %d concurrent",
        len(ALL_MODELS),
        max_concurrent,
    )

    if dry_run:
        print("\nDry-run mode — would evaluate these models:\n")
        for i, spec in enumerate(ALL_MODELS, 1):
            datasets = spec.datasets or ["reasonif", "cotcontrol"]
            print(
                f"  {i:2d}. {spec.model:40s} backend={spec.backend:12s} "
                f"effort={spec.reasoning_effort:6s} concurrency={spec.max_concurrency} "
                f"datasets={datasets}"
            )
        print(f"\nTotal: {len(ALL_MODELS)} models × 2 datasets = {len(ALL_MODELS) * 2} eval runs")
        return

    sem = asyncio.Semaphore(max_concurrent)
    results: dict[str, str] = {}

    tasks = [
        asyncio.create_task(run_with_semaphore(sem, spec, results))
        for spec in ALL_MODELS
    ]
    await asyncio.gather(*tasks, return_exceptions=True)

    # Print summary
    elapsed_total = (datetime.now(timezone.utc) - start_time).total_seconds()
    print(f"\n{'='*70}")
    print(f"OVERNIGHT BATCH COMPLETE — {elapsed_total:.0f}s total")
    print(f"{'='*70}")
    for model, status in results.items():
        print(f"  {model:45s} {status}")

    n_ok = sum(1 for s in results.values() if s.startswith("OK"))
    n_fail = sum(1 for s in results.values() if s.startswith("FAIL"))
    print(f"\n  {n_ok} succeeded, {n_fail} failed out of {len(ALL_MODELS)} models")


def _setup_logging(log_file: Path) -> None:
    """Configure logging to both stdout and a log file."""
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overnight batch evaluation")
    parser.add_argument(
        "--max-concurrent", type=int, default=3,
        help="Max models running concurrently (default: 3)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print model list without running evals",
    )
    args = parser.parse_args()

    _setup_logging(Path("results/overnight_batch.log"))
    asyncio.run(main(max_concurrent=args.max_concurrent, dry_run=args.dry_run))
