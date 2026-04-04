#!/usr/bin/env python3
"""Test different timeout settings for Tinker FT models.

Runs small-scale experiments with qwen3-8b-ft60 and qwen3-32b-ft60
to find timeout settings that reduce error rates without hugely
slowing throughput.

Usage:
    uv run python scripts/runs/test_timeout_settings.py
    uv run python scripts/runs/test_timeout_settings.py --dry-run
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from controllability.config import MonitorQAConfig
from controllability.evals.runner import run_monitor_experiment
from controllability.inference.batch import _single_with_retry

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = str(REPO_ROOT / "results" / "monitorqa" / "gpqa_hle_w_math_monitorqa.jsonl")

# Models to test
MODELS = [
    {
        "label": "qwen3-8b-ft60",
        "actor_model": "qwen/qwen3-8b",
        "actor_backend": "tinker",
        "model_path": "tinker://e3fea85f-5797-5a24-8c5e-af48cd0ad866:train:0/sampler_weights/000060",
    },
    {
        "label": "qwen3-32b-ft60",
        "actor_model": "qwen/qwen3-32b",
        "actor_backend": "tinker",
        "model_path": "tinker://eb355212-c7a1-5f36-9db6-c526e1f33298:train:0/sampler_weights/000060",
    },
]

# Timeout configs to test: (label, request_timeout, sample_timeout, max_retries)
TIMEOUT_CONFIGS = [
    # Current settings (baseline)
    ("current_300s_720s_3r", 300, 720, 3),
    # Longer request timeout, fewer retries — give each attempt more time
    ("600s_720s_1r", 600, 720, 1),
    # Longer request timeout, no retries — single long attempt
    ("660s_720s_0r", 660, 720, 0),
    # Even longer single attempt
    ("900s_960s_0r", 900, 960, 0),
]

N_SAMPLES = 50
SEED = 123  # Different seed from overnight to avoid overlap


async def run_one_test(
    model_cfg: dict,
    timeout_label: str,
    request_timeout: int,
    sample_timeout: int,
    max_retries: int,
    n_samples: int = 50,
) -> dict:
    """Run a single test configuration."""
    config = MonitorQAConfig(
        actor_model=model_cfg["actor_model"],
        actor_backend=model_cfg["actor_backend"],
        actor_max_tokens=28000,
        actor_temperature=1.0,
        actor_concurrency=50,
        actor_max_retries=max_retries,
        monitor_model="openai/gpt-4o",
        monitor_max_tokens=1024,
        monitor_temperature=0.0,
        monitor_concurrency=60,
        monitor_max_retries=3,
        dataset_path=DATASET_PATH,
        n_samples=n_samples,
        seed=SEED,
        prompt_type="default",
        modes=["side_task"],
        reasoning_effort="medium",
        request_timeout=request_timeout,
        model_path=model_cfg["model_path"],
        output_dir=str(REPO_ROOT / "results" / "rollouts" / "monitor_qa" / "timeout_tests"),
    )

    # Monkey-patch sample_timeout for this test
    import controllability.inference.batch as batch_mod
    original_fn = batch_mod._single_with_retry

    async def patched_single_with_retry(client, request, semaphore, max_retries_arg, sample_timeout_arg=None):
        return await original_fn(client, request, semaphore, max_retries_arg, sample_timeout=sample_timeout)

    batch_mod._single_with_retry = patched_single_with_retry

    # Custom filename
    custom_filename = (
        f"timeout_test_{model_cfg['label']}_{timeout_label}"
        f"_{config.experiment_id}.jsonl"
    )
    config.__class__.output_filename = property(
        lambda self, fn=custom_filename: fn
    )

    label = f"{model_cfg['label']}/{timeout_label}"
    print(f"\nSTART: {label}")
    print(f"  request_timeout={request_timeout}s  sample_timeout={sample_timeout}s  max_retries={max_retries}")

    start = time.time()
    try:
        await run_monitor_experiment(config)
        elapsed = time.time() - start
        print(f"DONE:  {label}  elapsed={elapsed:.0f}s")
    except Exception as e:
        elapsed = time.time() - start
        print(f"FAIL:  {label}  elapsed={elapsed:.0f}s  error={e}")
    finally:
        # Restore original function
        batch_mod._single_with_retry = original_fn

    # Read results and compute stats
    output_path = Path(config.output_dir) / custom_filename
    results = {"label": label, "elapsed": elapsed}
    if output_path.exists():
        rollouts = []
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rollouts.append(json.loads(line))
                    except:
                        pass

        n = len(rollouts)
        errors = sum(1 for r in rollouts if r.get("actor_error"))
        timeouts = sum(1 for r in rollouts if "Timeout" in str(r.get("actor_error", "") or "")
                       or "sample_timeout" in str(r.get("actor_error", "") or ""))
        valid = [r for r in rollouts if not r.get("actor_error")]
        latencies = sorted(r["actor_latency_ms"] / 1000 for r in valid if r.get("actor_latency_ms"))

        results["n"] = n
        results["errors"] = errors
        results["timeouts"] = timeouts
        results["error_rate"] = f"{100*errors/n:.1f}%" if n else "N/A"
        results["timeout_rate"] = f"{100*timeouts/n:.1f}%" if n else "N/A"
        if latencies:
            results["p50_latency"] = f"{latencies[len(latencies)//2]:.0f}s"
            results["p90_latency"] = f"{latencies[int(len(latencies)*0.9)]:.0f}s"
            results["max_latency"] = f"{latencies[-1]:.0f}s"

    return results


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--models", default="both", choices=["8b", "32b", "both"])
    args = parser.parse_args()

    n_samples = args.n_samples

    models = MODELS
    if args.models == "8b":
        models = [m for m in MODELS if "8b" in m["label"]]
    elif args.models == "32b":
        models = [m for m in MODELS if "32b" in m["label"]]

    if args.dry_run:
        print(f"Timeout test: {len(models)} models × {len(TIMEOUT_CONFIGS)} configs = {len(models)*len(TIMEOUT_CONFIGS)} runs")
        print(f"N samples: {N_SAMPLES}\n")
        for m in models:
            for label, rt, st, mr in TIMEOUT_CONFIGS:
                print(f"  {m['label']:<20} {label:<25} req_t={rt}s  sample_t={st}s  retries={mr}")
        return

    all_results = []

    for model_cfg in models:
        for timeout_label, request_timeout, sample_timeout, max_retries in TIMEOUT_CONFIGS:
            result = await run_one_test(
                model_cfg, timeout_label, request_timeout, sample_timeout, max_retries,
                n_samples=n_samples,
            )
            all_results.append(result)

    # Print summary
    print("\n" + "=" * 90)
    print("TIMEOUT TEST RESULTS")
    print("=" * 90)
    print(f"{'Config':<45} {'N':>4} {'Err%':>6} {'Tmo%':>6} {'p50':>6} {'p90':>6} {'max':>6} {'Wall':>6}")
    print("-" * 90)
    for r in all_results:
        print(
            f"{r['label']:<45} "
            f"{r.get('n','?'):>4} "
            f"{r.get('error_rate','?'):>6} "
            f"{r.get('timeout_rate','?'):>6} "
            f"{r.get('p50_latency','?'):>6} "
            f"{r.get('p90_latency','?'):>6} "
            f"{r.get('max_latency','?'):>6} "
            f"{r.get('elapsed',0):>5.0f}s"
        )


if __name__ == "__main__":
    asyncio.run(main())
