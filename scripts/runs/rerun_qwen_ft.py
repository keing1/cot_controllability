#!/usr/bin/env python3
"""Re-run 6 Qwen FT evals with updated timeout settings (660s/720s/1r).

6 jobs: 2 models × 3 prompt settings, 3 concurrent.

Usage:
    uv run python scripts/runs/rerun_qwen_ft.py
    uv run python scripts/runs/rerun_qwen_ft.py --dry-run
"""

import argparse
import asyncio
import logging
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

# Use 'spawn' so workers import fresh code instead of inheriting
# the parent's in-memory modules via fork.
multiprocessing.set_start_method("spawn", force=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "rollouts" / "monitor_qa"
DATASET_PATH = str(RESULTS_DIR / "monitorqa" / "gpqa_hle_w_math_monitorqa.jsonl")

MAX_CONCURRENT = 2

# Qwen FT models (Tinker, stripped lr1e-4 step 000060)
MODELS = [
    {
        "label": "qwen3-8b-ft60",
        "actor_model": "qwen/qwen3-8b",
        "actor_backend": "tinker",
        "model_path": "tinker://e3fea85f-5797-5a24-8c5e-af48cd0ad866:train:0/sampler_weights/000060",
        "reasoning_effort": "medium",
        "actor_concurrency": 50,
    },
    {
        "label": "qwen3-32b-ft60",
        "actor_model": "qwen/qwen3-32b",
        "actor_backend": "tinker",
        "model_path": "tinker://eb355212-c7a1-5f36-9db6-c526e1f33298:train:0/sampler_weights/000060",
        "reasoning_effort": "medium",
        "actor_concurrency": 50,
    },
]

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


def run_single_eval(
    model_cfg: dict,
    prompt_type: str,
    modes: list[str],
    run_id: str,
    n_samples: int | None,
) -> str:
    """Run one monitor_qa eval in a worker process."""
    # Import inside worker so forked processes get the latest code
    from controllability.config import MonitorQAConfig
    from controllability.evals.runner import run_monitor_experiment

    label = model_cfg["label"]
    mode_str = modes[0]
    setting_label = PROMPT_SETTING_LABELS.get((prompt_type, mode_str), prompt_type)

    config = MonitorQAConfig(
        actor_model=model_cfg["actor_model"],
        actor_backend=model_cfg["actor_backend"],
        actor_max_tokens=28000,
        actor_temperature=1.0,
        actor_concurrency=model_cfg.get("actor_concurrency", 50),
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

    custom_filename = (
        f"monitor_qa_{label}_{setting_label}"
        f"_{run_id}_{config.experiment_id}.jsonl"
    )
    config.output_filename_override = custom_filename

    key = f"{label}/{setting_label}"

    # Per-job log file: redirect stdout/stderr so workers don't interleave
    job_log = RESULTS_DIR / "logs" / f"rerun_{label}_{setting_label}_{run_id}.log"
    job_log.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(job_log, "w", buffering=1)  # line-buffered
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = log_fh
    sys.stderr = log_fh

    logging.info("START: %s -> %s (log: %s)", key, custom_filename, job_log.name)

    try:
        asyncio.run(run_monitor_experiment(config))
        sys.stdout, sys.stderr = old_stdout, old_stderr
        log_fh.close()
        logging.info("DONE:  %s -> OK", key)
        return f"{key}: OK"
    except Exception as e:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        log_fh.close()
        logging.error("FAIL:  %s -> %s", key, e, exc_info=True)
        return f"{key}: FAIL ({e})"


def build_jobs(run_id: str, n_samples: int | None) -> list[dict]:
    jobs = []
    for prompt_type, modes in PROMPT_SETTINGS:
        for model_cfg in MODELS:
            jobs.append({
                "model_cfg": model_cfg,
                "prompt_type": prompt_type,
                "modes": modes,
                "run_id": run_id,
                "n_samples": n_samples,
            })
    return jobs


def main():
    from controllability.process_guard import ProcessGuard

    parser = argparse.ArgumentParser(description="Re-run Qwen FT evals with updated timeout")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT)
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if args.dry_run:
        jobs = build_jobs(run_id, args.n_samples)
        print(f"\nDRY RUN — Qwen FT re-run (660s/720s/1r)")
        print(f"Jobs: {len(jobs)}, Max concurrent: {args.max_concurrent}\n")
        for i, j in enumerate(jobs, 1):
            m = j["model_cfg"]
            mode_str = j["modes"][0]
            setting = PROMPT_SETTING_LABELS.get((j["prompt_type"], mode_str), j["prompt_type"])
            print(f"  {i}. {m['label']:<20} setting={setting:<12}")
        return

    # Setup logging
    log_file = RESULTS_DIR / "logs" / f"rerun_qwen_ft_{run_id}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)

    # Kill any surviving workers from a previous run.
    guard = ProcessGuard(RESULTS_DIR / "logs" / "rerun_qwen_ft.pids")
    killed = guard.kill_previous()
    if killed:
        logging.info("Killed %d orphaned worker(s) from previous run", killed)

    guard.install_signal_handlers()

    start = datetime.now(timezone.utc)
    jobs = build_jobs(run_id, args.n_samples)
    logging.info("Qwen FT re-run: %d jobs, max_concurrent=%d, run_id=%s", len(jobs), args.max_concurrent, run_id)

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
    logging.info("=" * 60)
    logging.info("QWEN FT RE-RUN COMPLETE — %.0fs total (%.1f hours)", elapsed, elapsed / 3600)
    logging.info("=" * 60)
    for r in sorted(results):
        logging.info("  %s", r)
    n_ok = sum(1 for r in results if "OK" in r)
    logging.info("  %d succeeded, %d failed", n_ok, len(results) - n_ok)


if __name__ == "__main__":
    main()
