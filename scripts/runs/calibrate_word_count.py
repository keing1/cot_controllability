#!/usr/bin/env python3
"""Calibrate number_words thresholds per model.

Runs each model on all 300 ReasonIF questions (without constraints) n_runs times,
measures reasoning word counts, and computes the 20th percentile per source dataset
to produce model-specific thresholds for the number_words instruction type.

Usage:
    # Single model
    python scripts/runs/calibrate_word_count.py \
        --model openai/gpt-oss-20b --backend tinker --reasoning-effort medium

    # Multiple runs
    python scripts/runs/calibrate_word_count.py \
        --model openai/gpt-oss-120b --backend tinker --reasoning-effort low --n-runs 5

    # OpenRouter model
    python scripts/runs/calibrate_word_count.py \
        --model qwen/qwen3-235b-a22b --backend openrouter
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import nltk.tokenize
import numpy as np

# Ensure project is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from controllability.config import ExperimentConfig, Settings
from controllability.datasets.loader import load_dataset
from controllability.inference.base import InferenceClient
from controllability.inference.batch import run_batch
from controllability.inference.openrouter import OpenRouterClient
from controllability.types import InferenceRequest

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
CALIBRATION_DIR = RESULTS_DIR / "calibration"
REFERENCE_FILE = CALIBRATION_DIR / "number_words_reference.json"

# Prompt template: same as ReasonIF but without the instruction line
BASELINE_PROMPT = (
    "Think step-by-step, and place only your final answer inside the tags "
    "`<answer>` and `</answer>`.\n\n"
    "Here is the question:\n\n{question}"
)

PERCENTILE = 20  # 20th percentile, matching ReasonIF paper


def count_words(text: str) -> int:
    """Count words using the same tokenizer as ReasonIF's checker."""
    tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
    return len(tokenizer.tokenize(text))


def _create_client(
    backend: str, model: str, settings: Settings,
    model_path: str | None = None, reasoning_effort: str = "none",
    request_timeout: int = 420,
) -> InferenceClient:
    """Create inference client."""
    if backend == "openrouter":
        if not settings.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY not set")
        return OpenRouterClient(
            api_key=settings.openrouter_api_key,
            request_timeout=request_timeout,
        )
    elif backend == "tinker":
        from controllability.inference.tinker_client import TinkerClient
        if not settings.tinker_api_key:
            raise ValueError("TINKER_API_KEY not set")
        return TinkerClient(
            model=model,
            model_path=model_path,
            reasoning_effort=reasoning_effort,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")


ROLLOUT_DIR = RESULTS_DIR / "calibration" / "rollouts"


async def calibrate_model(
    model: str,
    backend: str = "tinker",
    model_path: str | None = None,
    reasoning_effort: str = "none",
    n_runs: int = 3,
    max_concurrency: int = 100,
    max_tokens: int = 28000,
    temperature: float = 1.0,
    request_timeout: int = 420,
) -> tuple[dict[str, int], list[dict]]:
    """Run model on all ReasonIF questions n_runs times, return 20th percentile
    word counts per source dataset.

    Returns:
        Tuple of (calibration dict, rollout records list).
        calibration: {"aime": 1200, "amc": 300, ...}
        rollouts: list of per-sample dicts with reasoning, word_count, etc.
    """
    settings = Settings()

    # Load all 300 ReasonIF questions
    samples = load_dataset("reasonif")
    logging.info("Loaded %d ReasonIF samples", len(samples))

    # Build baseline prompts (no instruction constraints)
    requests: list[InferenceRequest] = []
    request_meta: list[dict] = []  # track source + run index

    for run_idx in range(n_runs):
        for sample in samples:
            question = sample.question
            user_prompt = BASELINE_PROMPT.format(question=question)

            messages = [{"role": "user", "content": user_prompt}]
            req = InferenceRequest(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            requests.append(req)
            request_meta.append({
                "source": sample.metadata.get("source", "unknown"),
                "sample_id": sample.id,
                "run_idx": run_idx,
            })

    total = len(requests)
    logging.info(
        "Calibrating %s: %d samples × %d runs = %d requests (concurrency=%d)",
        model, len(samples), n_runs, total, max_concurrency,
    )

    # Run inference
    client = _create_client(
        backend, model, settings,
        model_path=model_path,
        reasoning_effort=reasoning_effort,
        request_timeout=request_timeout,
    )
    try:
        responses = await run_batch(
            client, requests,
            max_concurrency=max_concurrency,
            max_retries=3,
            desc=f"  calibrate({model})",
        )
    finally:
        await client.close()

    # Collect word counts per source and build rollout records
    source_word_counts: dict[str, list[int]] = {}
    rollout_records: list[dict] = []
    errors = 0
    no_reasoning = 0

    for meta, resp in zip(request_meta, responses):
        source = meta["source"]
        record = {
            "sample_id": meta["sample_id"],
            "source": source,
            "run_idx": meta["run_idx"],
            "error": resp.error,
            "reasoning": resp.reasoning,
            "content": resp.content,
            "word_count": None,
            "latency_ms": resp.latency_ms,
        }
        if resp.error:
            errors += 1
        elif not resp.reasoning:
            no_reasoning += 1
        else:
            wc = count_words(resp.reasoning)
            record["word_count"] = wc
            source_word_counts.setdefault(source, []).append(wc)
        rollout_records.append(record)

    if errors:
        logging.warning("  %d/%d requests had errors", errors, total)
    if no_reasoning:
        logging.warning("  %d/%d responses had no reasoning trace", no_reasoning, total)

    # Compute 20th percentile per source
    calibration: dict[str, int] = {}
    for source in sorted(source_word_counts.keys()):
        counts = source_word_counts[source]
        p20 = int(np.percentile(counts, PERCENTILE))
        median = int(np.median(counts))
        mean = int(np.mean(counts))
        logging.info(
            "  %s: n=%d, p20=%d, median=%d, mean=%d, min=%d, max=%d",
            source, len(counts), p20, median, mean, min(counts), max(counts),
        )
        calibration[source] = p20

    return calibration, rollout_records


def _model_key(model: str, reasoning_effort: str | None = None) -> str:
    """Create a key for the reference file. Include effort for gpt-oss models."""
    # e.g. "openai/gpt-oss-20b" -> "gpt-oss-20b"
    short = model.split("/")[-1]
    if reasoning_effort and "gpt-oss" in model:
        return f"{short}@{reasoning_effort}"
    return short


def save_calibration(
    model: str,
    calibration: dict[str, int],
    reasoning_effort: str | None = None,
    raw_counts: dict | None = None,
) -> None:
    """Save calibration to the reference file, merging with existing entries."""
    import fcntl

    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)

    key = _model_key(model, reasoning_effort)

    # Use file locking to prevent concurrent read-modify-write races
    lock_file = CALIBRATION_DIR / ".reference.lock"
    with open(lock_file, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            reference: dict = {}
            if REFERENCE_FILE.exists():
                with open(REFERENCE_FILE) as f:
                    reference = json.load(f)

            reference[key] = calibration

            with open(REFERENCE_FILE, "w") as f:
                json.dump(reference, f, indent=2)
                f.write("\n")
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)

    logging.info("Saved calibration for '%s' to %s", key, REFERENCE_FILE)

    # Also save raw stats for analysis
    if raw_counts:
        stats_file = CALIBRATION_DIR / f"{key.replace('/', '_')}_word_count_stats.json"
        with open(stats_file, "w") as f:
            json.dump(raw_counts, f, indent=2)
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate number_words thresholds per model"
    )
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument(
        "--backend", type=str, default="tinker",
        choices=["tinker", "openrouter"],
    )
    parser.add_argument("--model-path", type=str, default=None, help="Tinker checkpoint path")
    parser.add_argument(
        "--reasoning-effort", type=str, default="high",
        help="Reasoning effort (for gpt-oss models)",
    )
    parser.add_argument("--n-runs", type=int, default=3, help="Number of runs per sample")
    parser.add_argument("--max-concurrency", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=28000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--request-timeout", type=int, default=420)
    parser.add_argument("--dry-run", action="store_true", help="Print plan only")

    args = parser.parse_args()

    key = _model_key(args.model, args.reasoning_effort)
    n_samples = 300
    total = n_samples * args.n_runs

    if args.dry_run:
        print(f"Model:     {args.model}")
        print(f"Key:       {key}")
        print(f"Backend:   {args.backend}")
        print(f"Effort:    {args.reasoning_effort}")
        print(f"Runs:      {args.n_runs}")
        print(f"Total:     {n_samples} × {args.n_runs} = {total} requests")
        print(f"Conc:      {args.max_concurrency}")
        print(f"Output:    {REFERENCE_FILE}")
        return

    logging.info("Calibrating '%s' (%s)", key, args.model)

    calibration, rollout_records = asyncio.run(
        calibrate_model(
            model=args.model,
            backend=args.backend,
            model_path=args.model_path,
            reasoning_effort=args.reasoning_effort,
            n_runs=args.n_runs,
            max_concurrency=args.max_concurrency,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            request_timeout=args.request_timeout,
        )
    )

    save_calibration(
        model=args.model,
        calibration=calibration,
        reasoning_effort=args.reasoning_effort,
    )

    # Save rollouts as JSONL
    ROLLOUT_DIR.mkdir(parents=True, exist_ok=True)
    rollout_file = ROLLOUT_DIR / f"{key.replace('/', '_')}_calibration.jsonl"
    with open(rollout_file, "w") as f:
        for record in rollout_records:
            f.write(json.dumps(record) + "\n")
    logging.info("Saved %d rollout records to %s", len(rollout_records), rollout_file)

    print(f"\nCalibration for {key}:")
    for source, threshold in sorted(calibration.items()):
        print(f"  {source}: {threshold}")


if __name__ == "__main__":
    main()
