#!/usr/bin/env python3
"""Re-run qwen3.5 FT evals for error rollouts only.

1. Filters error rollouts from existing JSONL files (keeps clean ones)
2. Re-runs evals so resume logic retries only failed samples
3. Uses increased request_timeout (480s vs previous 300s)
4. Max 3 models concurrent, 1 eval per model at a time

Usage:
    python scripts/runs/rerun_error_evals.py --dry-run
    python scripts/runs/rerun_error_evals.py --max-workers 3
    python scripts/runs/rerun_error_evals.py --single-model qwen3.5-27b --run-id 20260322_082246
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
SFT_DIR = RESULTS_DIR / "sft"
ROLLOUTS_DIR = RESULTS_DIR / "rollouts"

REQUEST_TIMEOUT = 480  # 8 min (up from 300s / 5 min)

# Same eval config as overnight_ft_eval_v3.py
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

EVAL_N_SAMPLES = {
    "cotcontrol": 100,
    "reasonif": None,
}

# Model specs (same as overnight_ft_eval_v3.py)
MODEL_SPECS = {
    "qwen3.5-4b": {"model": "qwen/qwen3.5-4b", "reasoning_effort": "none"},
    "qwen3.5-27b": {"model": "qwen/qwen3.5-27b", "reasoning_effort": "none"},
    "qwen3.5-35b-a3b": {"model": "qwen/qwen3.5-35b-a3b", "reasoning_effort": "none"},
    "qwen3.5-397b-a17b": {"model": "qwen/qwen3.5-397b-a17b", "reasoning_effort": "none"},
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RolloutFile:
    path: Path
    model_short: str       # e.g. "qwen3.5-27b"
    lr: str                # e.g. "lr1e-4"
    checkpoint: str        # e.g. "000060", "final"
    dataset: str           # e.g. "cotcontrol", "reasonif"
    run_id: str            # e.g. "20260322_082246"
    experiment_id: str     # e.g. "85f56e1991eb"
    total: int = 0
    clean: int = 0
    errors: int = 0


def parse_rollout_filename(path: Path) -> RolloutFile | None:
    """Parse a qwen3.5 FT rollout filename into components."""
    fname = path.name
    # Pattern: qwen_qwen3.5-{model}-rif-{lr}-{ckpt}_{dataset}_all_{run_id}_{exp_id}.jsonl
    m = re.match(
        r"qwen_qwen3\.5-(\S+?)-rif-"
        r"(?:lr([\d.]+e-\d+)-)?"
        r"(\d{3,}|final)_"
        r"(\w+)_all_"
        r"(\d{8}_\d{6})_"
        r"([0-9a-f]+)\.jsonl$",
        fname,
    )
    if not m:
        return None

    model_suffix = m.group(1)
    lr_value = m.group(2) or "1e-4"
    checkpoint = m.group(3)
    dataset = m.group(4)
    run_id = m.group(5)
    experiment_id = m.group(6)

    return RolloutFile(
        path=path,
        model_short=f"qwen3.5-{model_suffix}",
        lr=f"lr{lr_value}",
        checkpoint=checkpoint,
        dataset=dataset,
        run_id=run_id,
        experiment_id=experiment_id,
    )


def count_errors(path: Path) -> tuple[int, int, int]:
    """Count (total, clean, errors) in a rollout JSONL file."""
    total = 0
    clean = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            total += 1
            r = json.loads(line)
            reasoning = (r.get("reasoning") or "").strip()
            response = (r.get("response") or "").strip()
            if reasoning and response:
                clean += 1
    return total, clean, total - clean


# ---------------------------------------------------------------------------
# Filter phase
# ---------------------------------------------------------------------------

def filter_errors(path: Path, dry_run: bool = False) -> tuple[int, int]:
    """Remove error rollouts from a JSONL file. Returns (kept, removed)."""
    header_lines = []
    clean_lines = []
    error_count = 0

    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                header_lines.append(line)
                continue
            r = json.loads(stripped)
            reasoning = (r.get("reasoning") or "").strip()
            response = (r.get("response") or "").strip()
            if reasoning and response:
                clean_lines.append(line)
            else:
                error_count += 1

    if error_count == 0:
        return len(clean_lines), 0

    if not dry_run:
        with open(path, "w") as f:
            for line in header_lines:
                f.write(line)
            for line in clean_lines:
                f.write(line)

    return len(clean_lines), error_count


# ---------------------------------------------------------------------------
# Eval phase
# ---------------------------------------------------------------------------

def load_sampler_path(model_short: str, lr: str, run_id: str, checkpoint: str) -> str | None:
    """Load sampler_path for a checkpoint from the SFT log directory."""
    # lr is like "lr1e-4" -> strip "lr" prefix for path
    lr_label = lr
    log_dir = SFT_DIR / f"{model_short}-reasonif-{lr_label}-{run_id}"
    ckpt_file = log_dir / "checkpoints.jsonl"

    if not ckpt_file.exists():
        return None

    with open(ckpt_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            if c.get("name") == checkpoint:
                return c.get("sampler_path")

    return None


def run_eval_for_file(rf: RolloutFile, request_timeout: int) -> None:
    """Re-run eval for a single rollout file."""
    from controllability.config import ExperimentConfig
    from controllability.evals.runner import run_experiment

    spec = MODEL_SPECS.get(rf.model_short)
    if not spec:
        logging.error("Unknown model: %s", rf.model_short)
        return

    sampler_path = load_sampler_path(rf.model_short, rf.lr, rf.run_id, rf.checkpoint)
    if not sampler_path:
        logging.error("No sampler_path found for %s %s %s %s",
                       rf.model_short, rf.lr, rf.run_id, rf.checkpoint)
        return

    model = spec["model"]
    dataset = rf.dataset
    modes = DATASET_MODES.get(dataset, [])
    n_samples = EVAL_N_SAMPLES.get(dataset)

    model_label = f"{model}-rif-{rf.lr}-{rf.checkpoint}"
    logging.info("Eval: %s on %s (%d errors to retry)", model_label, dataset, rf.errors)

    config = ExperimentConfig(
        model=model,
        dataset=dataset,
        modes=modes,
        split="all",
        fraction=1.0,
        seed=42,
        backend="tinker",
        max_concurrency=70,
        max_retries=3,
        max_tokens=28000,
        temperature=1.0,
        output_dir=str(ROLLOUTS_DIR),
        n_samples=n_samples,
        model_path=sampler_path,
        reasoning_effort=spec["reasoning_effort"],
        request_timeout=request_timeout,
    )

    # Override output_filename to point to existing file
    custom_filename = rf.path.name
    config.__class__.output_filename = property(
        lambda self, fn=custom_filename: fn
    )

    logging.info("  Output: %s", rf.path)
    asyncio.run(run_experiment(config))


def run_evals_for_model(model_short: str, files: list[RolloutFile],
                        request_timeout: int) -> None:
    """Run all evals for a single model sequentially."""
    logging.info("=" * 60)
    logging.info("Model: %s (%d evals to re-run)", model_short, len(files))
    logging.info("=" * 60)

    for i, rf in enumerate(files, 1):
        logging.info("[%d/%d] %s %s %s %s (errors=%d)",
                     i, len(files), rf.model_short, rf.lr, rf.checkpoint,
                     rf.dataset, rf.errors)
        try:
            run_eval_for_file(rf, request_timeout)
        except Exception as e:
            logging.error("Failed: %s — %s: %s", rf.path.name,
                          type(e).__name__, e, exc_info=True)

    logging.info("All evals complete for %s", model_short)


# ---------------------------------------------------------------------------
# Pool manager
# ---------------------------------------------------------------------------

def run_pool(model_files: dict[str, list[RolloutFile]], max_workers: int,
             request_timeout: int) -> dict[str, str]:
    """Run model evals with a subprocess pool. Max 1 eval per model, max N models."""
    jobs = list(model_files.keys())
    if not jobs:
        print("No evals to re-run.")
        return {}

    # Get the run_id from the first file (all should be the same)
    first_file = next(iter(model_files.values()))[0]
    run_id = first_file.run_id

    print(f"\n{'=' * 70}")
    print(f"RE-RUN EVALS ({len(jobs)} models, max {max_workers} concurrent)")
    print(f"{'=' * 70}")
    for model in jobs:
        n_files = len(model_files[model])
        n_errors = sum(rf.errors for rf in model_files[model])
        print(f"  {model:25s}  {n_files} evals, {n_errors} errors to retry")

    active: dict[str, tuple[subprocess.Popen, Path]] = {}
    pending = list(jobs)
    results: dict[str, str] = {}

    def launch_next():
        if not pending:
            return
        model = pending.pop(0)
        log_file = RESULTS_DIR / f"rerun_{model}.log"

        cmd = [
            sys.executable, str(SCRIPT_PATH),
            "--single-model", model,
            "--run-id", run_id,
            "--request-timeout", str(request_timeout),
        ]

        print(f"\n  Launching: {model}  (log: {log_file})")
        with open(log_file, "w") as lf:
            proc = subprocess.Popen(
                cmd,
                stdout=lf,
                stderr=subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
            )
        active[model] = (proc, log_file)

    # Fill initial slots
    for _ in range(min(max_workers, len(pending))):
        launch_next()

    # Poll for completions
    while active:
        time.sleep(5)
        done_keys = []
        for key, (proc, log_file) in active.items():
            ret = proc.poll()
            if ret is not None:
                done_keys.append(key)
                status = "OK" if ret == 0 else f"FAIL (exit {ret})"
                results[key] = status
                print(f"  Finished: {key} -> {status}")

        for key in done_keys:
            del active[key]

        while pending and len(active) < max_workers:
            launch_next()

    return results


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging(log_file: Path | None = None) -> None:
    target = log_file or RESULTS_DIR / "rerun_error_evals.log"
    target.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(target, mode="a")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Re-run qwen3.5 FT evals for error rollouts"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without making changes")
    parser.add_argument("--filter-only", action="store_true",
                        help="Only filter error rollouts, don't re-run evals")
    parser.add_argument("--max-workers", type=int, default=3,
                        help="Max concurrent model evals (default: 3)")
    parser.add_argument("--request-timeout", type=int, default=REQUEST_TIMEOUT,
                        help=f"Per-request timeout in seconds (default: {REQUEST_TIMEOUT})")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Run ID for checkpoint lookup (auto-detected if not set)")
    parser.add_argument("--single-model", type=str, default=None,
                        help="Internal: run evals for a single model (used by pool)")

    args = parser.parse_args()

    # --- Discover rollout files ---
    all_files = sorted(glob.glob(str(ROLLOUTS_DIR / "*qwen3.5*rif*.jsonl")))
    rollout_files: list[RolloutFile] = []

    for path in all_files:
        rf = parse_rollout_filename(Path(path))
        if rf is None:
            continue
        total, clean, errors = count_errors(rf.path)
        rf.total = total
        rf.clean = clean
        rf.errors = errors
        rollout_files.append(rf)

    # Expected total rollouts per file
    EXPECTED_TOTAL = {"cotcontrol": 1000, "reasonif": 300}

    # Filter to files that are incomplete (have errors OR missing rollouts)
    error_files = []
    for rf in rollout_files:
        expected = EXPECTED_TOTAL.get(rf.dataset, 0)
        missing = expected - rf.clean
        if rf.errors > 0 or missing > 0:
            # Use missing count as errors if no error rollouts present
            # (they were already filtered out in a previous run)
            if rf.errors == 0 and missing > 0:
                rf.errors = missing
            error_files.append(rf)

    # Filter by run_id if specified
    if args.run_id:
        error_files = [rf for rf in error_files if rf.run_id == args.run_id]

    # --- Single-model mode ---
    if args.single_model:
        model_files = [rf for rf in error_files if rf.model_short == args.single_model]
        if not model_files:
            print(f"No error files found for {args.single_model}")
            return

        log_file = RESULTS_DIR / f"rerun_{args.single_model}.log"
        _setup_logging(log_file)

        # Filter phase
        for rf in model_files:
            kept, removed = filter_errors(rf.path, dry_run=False)
            logging.info("Filtered %s: kept=%d removed=%d", rf.path.name, kept, removed)

        # Eval phase
        run_evals_for_model(args.single_model, model_files, args.request_timeout)
        return

    # --- Normal mode ---

    # Print summary
    total_errors = sum(rf.errors for rf in error_files)
    total_rollouts = sum(rf.total for rf in rollout_files)
    print(f"\nFound {len(rollout_files)} qwen3.5 FT rollout files")
    print(f"  {len(error_files)} files have errors ({total_errors} error rollouts / {total_rollouts} total)")

    if not error_files:
        print("Nothing to re-run!")
        return

    # Group by model
    model_files: dict[str, list[RolloutFile]] = {}
    for rf in error_files:
        model_files.setdefault(rf.model_short, []).append(rf)

    # Print per-file details
    print(f"\n{'File':<80s} {'Total':>5s} {'Clean':>5s} {'Errors':>6s} {'Err%':>6s}")
    print("-" * 105)
    for rf in sorted(error_files, key=lambda r: (r.model_short, r.lr, r.checkpoint, r.dataset)):
        err_pct = rf.errors / rf.total * 100 if rf.total > 0 else 0
        print(f"{rf.path.name:<80s} {rf.total:5d} {rf.clean:5d} {rf.errors:6d} {err_pct:5.1f}%")

    print(f"\nModels: {', '.join(sorted(model_files.keys()))}")
    print(f"Request timeout: {args.request_timeout}s")

    if args.dry_run:
        print("\n[DRY RUN] Would filter error rollouts and re-run evals.")
        return

    if args.filter_only:
        print("\nFiltering error rollouts...")
        for rf in error_files:
            kept, removed = filter_errors(rf.path)
            print(f"  {rf.path.name}: kept={kept} removed={removed}")
        print("Done. Run without --filter-only to also re-run evals.")
        return

    # Run pool
    _setup_logging()
    results = run_pool(model_files, args.max_workers, args.request_timeout)

    # Summary
    n_ok = sum(1 for s in results.values() if s == "OK")
    n_fail = sum(1 for s in results.values() if s != "OK")
    print(f"\n{'=' * 70}")
    print(f"RE-RUN COMPLETE")
    print(f"{'=' * 70}")
    for key, status in results.items():
        print(f"  {key:25s}  {status}")
    print(f"\n  {n_ok} succeeded, {n_fail} failed")


if __name__ == "__main__":
    main()
