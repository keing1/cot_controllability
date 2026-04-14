#!/usr/bin/env python3
"""Overnight monitor_qa eval sweep — BASE MODELS ONLY.

12 jobs: 4 base models × 3 prompt settings.
  - 4 models: gpt-oss-20b, gpt-oss-120b, qwen3-8b, qwen3-32b
  - 3 prompt settings: dont_think, baseline, default
  - Order: dont_think × 4, baseline × 4, default × 4
  - Concurrency: 3 jobs at a time
  - Backend: tinker

Usage:
    uv run python scripts/runs/overnight_monitor_qa_base_only.py
    uv run python scripts/runs/overnight_monitor_qa_base_only.py --dry-run
    uv run python scripts/runs/overnight_monitor_qa_base_only.py --n-samples 50
"""

import argparse
import asyncio
import logging
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

multiprocessing.set_start_method("spawn", force=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from controllability.config import MonitorQAConfig
from controllability.evals.runner import run_monitor_experiment

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
SFT_DIR = RESULTS_DIR / "sft"
OUTPUT_DIR = RESULTS_DIR / "rollouts" / "monitor_qa"
DATASET_PATH = str(RESULTS_DIR / "monitorqa" / "gpqa_hle_w_math_monitorqa.jsonl")

MAX_CONCURRENT = 3

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

# Base models (Tinker)
BASE_MODELS = [
    {
        "label": "gpt-oss-20b",
        "actor_model": "openai/gpt-oss-20b",
        "actor_backend": "tinker",
        "model_path": None,
        "reasoning_effort": "medium",
    },
    {
        "label": "gpt-oss-120b",
        "actor_model": "openai/gpt-oss-120b",
        "actor_backend": "tinker",
        "model_path": None,
        "reasoning_effort": "medium",
    },
    {
        "label": "qwen3-8b",
        "actor_model": "qwen/qwen3-8b",
        "actor_backend": "tinker",
        "model_path": None,
        "reasoning_effort": "medium",
    },
    {
        "label": "qwen3-32b",
        "actor_model": "qwen/qwen3-32b",
        "actor_backend": "tinker",
        "model_path": None,
        "reasoning_effort": "medium",
    },
]

# Prompt settings: (prompt_type, modes)
# "baseline" mode means the actor gets no side task — uses default prompt builder
PROMPT_SETTINGS = [
    ("dont_think", ["side_task"]),
    ("default",    ["baseline"]),
    ("default",    ["side_task"]),
]

PROMPT_SETTING_LABELS = {
    ("dont_think", "side_task"): "dont_think",
    ("default", "baseline"):    "baseline",
    ("default", "side_task"):   "default",
}


# ---------------------------------------------------------------------------
# Single job runner (runs in subprocess)
# ---------------------------------------------------------------------------

def run_single_eval(
    model_cfg: dict,
    prompt_type: str,
    modes: list[str],
    run_id: str,
    n_samples: int | None,
) -> str:
    """Run one monitor_qa eval in a worker process."""
    label = model_cfg["label"]
    mode_str = modes[0]
    setting_label = PROMPT_SETTING_LABELS.get((prompt_type, mode_str), prompt_type)

    config = MonitorQAConfig(
        actor_model=model_cfg["actor_model"],
        actor_backend=model_cfg["actor_backend"],
        actor_max_tokens=28000,
        actor_temperature=1.0,
        actor_concurrency=40,
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
        modes=modes,
        reasoning_effort=model_cfg["reasoning_effort"],
        request_timeout=900,
        model_path=model_cfg["model_path"],
        output_dir=str(OUTPUT_DIR),
    )

    # Custom filename for clarity
    custom_filename = (
        f"monitor_qa_{label}_{setting_label}"
        f"_{run_id}_{config.experiment_id}.jsonl"
    )
    config.output_filename_override = custom_filename

    key = f"{label}/{setting_label}"
    logging.info("START: %s -> %s", key, custom_filename)

    try:
        asyncio.run(run_monitor_experiment(config))
        logging.info("DONE:  %s -> OK", key)
        return f"{key}: OK"
    except Exception as e:
        logging.error("FAIL:  %s -> %s", key, e, exc_info=True)
        return f"{key}: FAIL ({e})"


# ---------------------------------------------------------------------------
# Job list
# ---------------------------------------------------------------------------

def build_jobs(run_id: str, n_samples: int | None) -> list[dict]:
    """Build job list: 3 prompt settings × 4 base models = 12 jobs.

    Ordered: dont_think × 4, baseline × 4, default × 4.
    """
    jobs = []
    for prompt_type, modes in PROMPT_SETTINGS:
        for model_cfg in BASE_MODELS:
            jobs.append({
                "model_cfg": model_cfg,
                "prompt_type": prompt_type,
                "modes": modes,
                "run_id": run_id,
                "n_samples": n_samples,
            })
    return jobs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def print_dry_run(run_id: str, n_samples: int | None) -> None:
    jobs = build_jobs(run_id, n_samples)
    print(f"\n{'=' * 70}")
    print(f"DRY RUN — Monitor QA Base-Only (run_id={run_id})")
    print(f"{'=' * 70}")
    print(f"\nDataset: {DATASET_PATH}")
    print(f"N samples: {n_samples or 'all'}")
    print(f"Max concurrent: {MAX_CONCURRENT}")
    print(f"Jobs: {len(jobs)}\n")

    for i, j in enumerate(jobs, 1):
        m = j["model_cfg"]
        mode_str = j["modes"][0]
        setting = PROMPT_SETTING_LABELS.get((j["prompt_type"], mode_str), j["prompt_type"])
        print(f"  {i:>2}. {m['label']:<25} setting={setting:<12} backend={m['actor_backend']}")
    print()


def _setup_logging(run_id: str) -> None:
    log_file = RESULTS_DIR / f"overnight_monitor_qa_base_only_{run_id}.log"
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
    from controllability.process_guard import ProcessGuard

    parser = argparse.ArgumentParser(description="Overnight monitor_qa eval sweep (base models only)")
    parser.add_argument("--dry-run", action="store_true", help="Print job list without running")
    parser.add_argument("--run-id", type=str, default=None, help="Custom run ID")
    parser.add_argument("--n-samples", type=int, default=None, help="Subsample N samples per eval")
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT)
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if args.dry_run:
        print_dry_run(run_id, args.n_samples)
        return

    _setup_logging(run_id)

    # Kill any surviving workers from a previous run.
    guard = ProcessGuard(RESULTS_DIR / "logs" / "overnight_monitor_qa_base_only.pids")
    killed = guard.kill_previous()
    if killed:
        logging.info("Killed %d orphaned worker(s) from previous run", killed)

    guard.install_signal_handlers()

    start = datetime.now(timezone.utc)
    jobs = build_jobs(run_id, args.n_samples)

    logging.info("Monitor QA base-only: %d jobs, max_concurrent=%d, run_id=%s",
                 len(jobs), args.max_concurrent, run_id)

    results = []
    with ProcessPoolExecutor(max_workers=args.max_concurrent) as pool:
        futures = {
            pool.submit(
                run_single_eval,
                j["model_cfg"], j["prompt_type"], j["modes"],
                j["run_id"], j["n_samples"],
            ): j
            for j in jobs
        }
        # Save worker PIDs so the next invocation can kill orphans.
        guard.save_pids(pool)

        for fut in as_completed(futures):
            result = fut.result()
            results.append(result)
            logging.info("  Finished: %s  (%d/%d)", result, len(results), len(jobs))

    guard.cleanup()
    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    logging.info("")
    logging.info("=" * 70)
    logging.info("MONITOR QA BASE-ONLY COMPLETE — %.0fs total (%.1f hours)", elapsed, elapsed / 3600)
    logging.info("=" * 70)
    for r in sorted(results):
        logging.info("  %s", r)
    n_ok = sum(1 for r in results if "OK" in r)
    logging.info("  %d succeeded, %d failed", n_ok, len(results) - n_ok)


if __name__ == "__main__":
    main()
