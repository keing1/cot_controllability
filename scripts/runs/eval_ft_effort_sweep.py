#!/usr/bin/env python3
"""Eval stripped FT step-60 gpt-oss models at low/high effort on both datasets.

2 models x 2 efforts x 2 evals = 8 jobs, 3 concurrent at a time.

Usage:
    python scripts/runs/eval_ft_effort_sweep.py --dry-run
    python scripts/runs/eval_ft_effort_sweep.py
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from controllability.config import ExperimentConfig
from controllability.evals.runner import run_experiment

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
ROLLOUTS_DIR = RESULTS_DIR / "rollouts"

MAX_CONCURRENT = 3

MODELS = [
    {
        "model": "openai/gpt-oss-20b",
        "short": "gpt-oss-20b",
        "sampler_path": "tinker://5ec3cd52-fe1e-579b-ae83-7bd8260330ec:train:0/sampler_weights/000060",
    },
    {
        "model": "openai/gpt-oss-120b",
        "short": "gpt-oss-120b",
        "sampler_path": "tinker://f745ff57-2fcf-54ad-9c19-3dfab62598c5:train:0/sampler_weights/000060",
    },
]

EFFORTS = ["low", "high"]

DATASETS = {
    "reasonif": {
        "modes": [
            "reasoning_language", "number_words", "english_capital",
            "end_checker", "json_format", "no_comma",
        ],
        "n_samples": None,
    },
    "cotcontrol": {
        "modes": [
            "baseline", "word_suppression", "multiple_word_suppression",
            "uppercase_thinking", "lowercase_thinking", "alternating_case",
            "repeat_sentences", "end_of_sentence", "meow_between_words",
            "ignore_question",
        ],
        "n_samples": 100,
    },
}


def run_single_eval(model_cfg: dict, effort: str, dataset: str, run_id: str) -> str:
    """Run one eval in a worker process. Returns status string."""
    ds = DATASETS[dataset]
    use_ac = dataset == "reasonif"  # gpt-oss always uses analysis_channel for reasonif

    model_label = f"{model_cfg['model']}-rif-stripped-lr1e-4-000060"

    config = ExperimentConfig(
        model=model_cfg["model"],
        dataset=dataset,
        modes=ds["modes"],
        split="all",
        fraction=1.0,
        seed=42,
        backend="tinker",
        max_concurrency=100,
        max_retries=3,
        max_tokens=28000,
        temperature=1.0,
        output_dir=str(ROLLOUTS_DIR),
        n_samples=ds["n_samples"],
        model_path=model_cfg["sampler_path"],
        reasoning_effort=effort,
        request_timeout=420,
        analysis_channel=use_ac,
    )

    model_short = model_label.replace("/", "_")
    ac_suffix = "_ac" if use_ac else ""
    custom_filename = (
        f"{model_short}_{dataset}{ac_suffix}_{effort}"
        f"_{run_id}_{config.experiment_id}.jsonl"
    )

    config.output_filename_override = custom_filename

    key = f"{model_cfg['short']}_{dataset}_{effort}"
    logging.info("START: %s -> %s", key, custom_filename)
    try:
        asyncio.run(run_experiment(config))
        logging.info("DONE:  %s -> OK", key)
        return f"{key}: OK"
    except Exception as e:
        logging.error("FAIL:  %s -> %s", key, e)
        return f"{key}: FAIL ({e})"


def build_jobs(run_id: str) -> list[dict]:
    """Build list of eval jobs."""
    jobs = []
    for effort in EFFORTS:
        for m in MODELS:
            for dataset in DATASETS:
                jobs.append({
                    "model_cfg": m,
                    "effort": effort,
                    "dataset": dataset,
                    "run_id": run_id,
                })
    return jobs


def print_dry_run(run_id: str) -> None:
    jobs = build_jobs(run_id)
    print(f"\n{'=' * 70}")
    print(f"DRY RUN — FT Effort Sweep (run_id={run_id})")
    print(f"{'=' * 70}")
    print(f"\nMax concurrent: {MAX_CONCURRENT}")
    print(f"Jobs: {len(jobs)}\n")
    for i, j in enumerate(jobs, 1):
        m = j["model_cfg"]
        ac = " (AC)" if j["dataset"] == "reasonif" else ""
        print(f"  {i}. {m['short']}  effort={j['effort']}  dataset={j['dataset']}{ac}")
        print(f"     checkpoint: {m['sampler_path']}")
    print()


def _setup_logging() -> None:
    log_file = RESULTS_DIR / "eval_ft_effort_sweep.log"
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
        description="Eval FT gpt-oss step-60 at low/high effort on both datasets"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if args.dry_run:
        print_dry_run(run_id)
        return

    _setup_logging()
    start = datetime.now(timezone.utc)
    jobs = build_jobs(run_id)

    logging.info("FT effort sweep: %d jobs, max_concurrent=%d, run_id=%s",
                 len(jobs), MAX_CONCURRENT, run_id)

    results = []
    with ProcessPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
        futures = {
            pool.submit(
                run_single_eval,
                j["model_cfg"], j["effort"], j["dataset"], j["run_id"],
            ): j
            for j in jobs
        }
        for fut in as_completed(futures):
            result = fut.result()
            results.append(result)
            logging.info("  Finished: %s  (%d/%d)", result, len(results), len(jobs))

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    logging.info("")
    logging.info("=" * 70)
    logging.info("FT EFFORT SWEEP COMPLETE — %.0fs total", elapsed)
    logging.info("=" * 70)
    for r in results:
        logging.info("  %s", r)
    n_ok = sum(1 for r in results if "OK" in r)
    logging.info("  %d succeeded, %d failed", n_ok, len(results) - n_ok)


if __name__ == "__main__":
    main()
