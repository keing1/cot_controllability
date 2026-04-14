#!/usr/bin/env python3
"""Rerun user_dont_think (v1) for all 8 models on val split.

Purpose: sanity check that the current codebase reproduces the same
evasion rates as the original overnight_val_prompts.py run (20260409).
If the numbers differ, there's been a code change affecting results.

Config is copied from overnight_new_prompts.py (the script that ran
the 3 new variants) so that everything except the prompt variant is
identical.

8 jobs: 8 models × 1 prompt variant (user_dont_think), side_task mode.
  - 4 base models + 4 FT models (same checkpoints)
  - 500 actor concurrency per eval (same as overnight_new_prompts.py)
  - If >20% of rollouts error, strips errors and reruns at 200 concurrency
  - Monitors: metr_note (GPT-4.1), metr_note_gpt54 (GPT-5.4), main_task_grading (GPT-5.4-mini),
    reasoning_audit (GPT-5.4)

Usage:
    uv run python scripts/runs/rerun_dont_think_v1.py --dry-run
    uv run python scripts/runs/rerun_dont_think_v1.py
    uv run python scripts/runs/rerun_dont_think_v1.py --n-samples 5
    uv run python scripts/runs/rerun_dont_think_v1.py --parallelism 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from controllability.config import MonitorQAConfig
from controllability.evals.runner import run_monitor_experiment

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "rollouts" / "monitor_qa"
DATASET_PATH = str(RESULTS_DIR / "monitorqa" / "monitorqa_val.jsonl")

# ---------------------------------------------------------------------------
# Models (identical to overnight_new_prompts.py / overnight_val_prompts.py)
# ---------------------------------------------------------------------------

BASE_MODELS = [
    {
        "label": "gpt-oss-20b",
        "actor_model": "openai/gpt-oss-20b",
        "model_path": None,
        "reasoning_effort": "medium",
    },
    {
        "label": "gpt-oss-120b",
        "actor_model": "openai/gpt-oss-120b",
        "model_path": None,
        "reasoning_effort": "medium",
    },
    {
        "label": "qwen3-8b",
        "actor_model": "qwen/qwen3-8b",
        "model_path": None,
        "reasoning_effort": "medium",
    },
    {
        "label": "qwen3-32b",
        "actor_model": "qwen/qwen3-32b",
        "model_path": None,
        "reasoning_effort": "medium",
    },
]

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

ALL_MODELS = BASE_MODELS + FT_MODELS

PROMPT_VARIANTS = [
    "user_dont_think",
]

HIGH_CONCURRENCY = 500
LOW_CONCURRENCY = 200
ERROR_THRESHOLD = 0.20  # 20%
DEFAULT_PARALLELISM = 3


# ---------------------------------------------------------------------------
# Helpers (same as overnight_new_prompts.py)
# ---------------------------------------------------------------------------

def count_errors(filepath: Path) -> tuple[int, int]:
    """Count (errors, total) in a rollout JSONL file."""
    errors = 0
    total = 0
    with open(filepath) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                r = json.loads(s)
            except json.JSONDecodeError:
                continue
            total += 1
            if r.get("actor_error"):
                errors += 1
    return errors, total


def remove_errored_lines(filepath: Path) -> int:
    """Remove rollout lines with actor_error from a JSONL file. Returns count removed."""
    header_lines: list[str] = []
    keep_lines: list[str] = []
    removed = 0
    with open(filepath) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                header_lines.append(line)
                continue
            try:
                r = json.loads(s)
            except json.JSONDecodeError:
                continue
            if r.get("actor_error"):
                removed += 1
            else:
                keep_lines.append(json.dumps(r) + "\n")
    with open(filepath, "w") as f:
        for line in header_lines:
            f.write(line if line.endswith("\n") else line + "\n")
        for line in keep_lines:
            f.write(line)
    return removed


def make_config(
    model_cfg: dict,
    prompt_type: str,
    run_id: str,
    n_samples: int | None,
    concurrency: int,
) -> MonitorQAConfig:
    return MonitorQAConfig(
        actor_model=model_cfg["actor_model"],
        actor_backend="tinker",
        actor_max_tokens=28000,
        actor_temperature=1.0,
        actor_concurrency=concurrency,
        actor_max_retries=1,
        monitor_model="openai/gpt-4o",
        monitor_max_tokens=1024,
        monitor_temperature=0.0,
        monitor_concurrency=60,
        monitor_max_retries=3,
        dataset_path=DATASET_PATH,
        dataset_split="val",
        n_samples=n_samples,
        seed=42,
        prompt_type=prompt_type,
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
        reasoning_effort=model_cfg["reasoning_effort"],
        request_timeout=900,
        model_path=model_cfg["model_path"],
        output_dir=str(OUTPUT_DIR),
    )


def run_single(
    model_cfg: dict,
    prompt_type: str,
    run_id: str,
    n_samples: int | None,
) -> str:
    """Run one eval. If >20% errors at HIGH concurrency, strip errors and rerun at LOW."""
    label = model_cfg["label"]
    key = f"{label}/{prompt_type}"

    for attempt, concurrency in enumerate([HIGH_CONCURRENCY, LOW_CONCURRENCY], 1):
        config = make_config(model_cfg, prompt_type, run_id, n_samples, concurrency)

        custom_filename = (
            f"monitor_qa_{label}_{prompt_type}_val_split"
            f"_{run_id}_{config.experiment_id}.jsonl"
        )
        filepath = OUTPUT_DIR / custom_filename

        # Override output_filename
        config.output_filename_override = custom_filename

        logging.info(
            "START: %s (attempt %d, concurrency=%d) -> %s",
            key, attempt, concurrency, custom_filename,
        )

        try:
            asyncio.run(run_monitor_experiment(config))
        except Exception as e:
            logging.error("FAIL: %s -> %s", key, e, exc_info=True)
            return f"FAIL ({e})"

        # Check error rate
        if filepath.exists():
            errors, total = count_errors(filepath)
            error_rate = errors / total if total > 0 else 0
            logging.info(
                "  %s: %d/%d errors (%.1f%%)",
                key, errors, total, error_rate * 100,
            )

            if error_rate > ERROR_THRESHOLD and attempt == 1:
                logging.warning(
                    "  %s: %.1f%% errors > %.0f%% threshold — removing %d errored lines and retrying with concurrency=%d",
                    key, error_rate * 100, ERROR_THRESHOLD * 100, errors, LOW_CONCURRENCY,
                )
                removed = remove_errored_lines(filepath)
                logging.info("  %s: removed %d errored lines, %d good lines remain", key, removed, total - removed)
                continue
            else:
                return f"OK ({errors}/{total} errors, concurrency={concurrency})"
        else:
            logging.warning("  %s: output file not found after run", key)
            return "FAIL (no output file)"

    return "FAIL (exhausted retries)"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Rerun user_dont_think v1 for sanity check")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--parallelism", type=int, default=DEFAULT_PARALLELISM,
                        help=f"Number of evals to run simultaneously (default: {DEFAULT_PARALLELISM})")
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Logging
    log_file = RESULTS_DIR / f"rerun_dont_think_v1_{run_id}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)

    # Build jobs
    jobs = []
    for prompt_type in PROMPT_VARIANTS:
        for model_cfg in ALL_MODELS:
            jobs.append((model_cfg, prompt_type))

    print(f"Rerun user_dont_think (v1) Sanity Check: run_id={run_id}")
    print(f"  {len(jobs)} jobs, {args.parallelism} running simultaneously")
    print(f"  Dataset: {DATASET_PATH} (val split)")
    print(f"  Concurrency per eval: {HIGH_CONCURRENCY} (retry at {LOW_CONCURRENCY} if >{ERROR_THRESHOLD*100:.0f}% errors)")
    print(f"  Monitors: metr_note (GPT-4.1), metr_note_gpt54 (GPT-5.4), main_task_grading (GPT-5.4-mini),")
    print(f"            reasoning_audit (GPT-5.4)")
    print(f"  Prompt variant: {PROMPT_VARIANTS}")
    if args.n_samples:
        print(f"  n_samples: {args.n_samples}")
    print()

    for i, (model_cfg, prompt_type) in enumerate(jobs, 1):
        ft = " (FT)" if model_cfg["model_path"] else ""
        print(f"  {i:>2}. {model_cfg['label']:<25} {prompt_type}{ft}")

    if args.dry_run:
        print(f"\n[DRY RUN] {len(jobs)} jobs would be submitted. Exiting.")
        return

    print()
    results = {}
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.parallelism) as pool:
        futures = {
            pool.submit(run_single, model_cfg, prompt_type, run_id, args.n_samples): f"{model_cfg['label']}/{prompt_type}"
            for model_cfg, prompt_type in jobs
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                result = future.result()
            except Exception as e:
                result = f"FAIL ({e})"
            results[key] = result
            done = len(results)
            print(f"  [{done}/{len(jobs)}] {key:<45s} -> {result}")

    total = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"ALL DONE — {total:.0f}s total ({total/3600:.1f} hours)")
    print(f"{'=' * 60}")
    for key, result in sorted(results.items()):
        print(f"  {key:<45s} {result}")
    n_ok = sum(1 for r in results.values() if r.startswith("OK"))
    print(f"\n  {n_ok} succeeded, {len(results) - n_ok} failed")


if __name__ == "__main__":
    main()
