#!/usr/bin/env python3
"""Run an evaluation sweep."""

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from controllability.config import ExperimentConfig
from controllability.evals.runner import run_experiment

# Default modes per dataset family -- used when --modes all
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


def _resolve_modes(modes_str: str, dataset: str) -> list[str]:
    """Resolve mode string, expanding 'all' to dataset-appropriate modes."""
    if modes_str.strip().lower() == "all":
        family = dataset.split("/")[0]  # "cotcontrol/gpqa" -> "cotcontrol"
        if family in DATASET_MODES:
            return DATASET_MODES[family]
        raise ValueError(
            f"Don't know default modes for dataset '{dataset}'. "
            f"Known families: {list(DATASET_MODES.keys())}"
        )
    return [m.strip() for m in modes_str.split(",")]


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Run CoT controllability evaluation")

    parser.add_argument("--model", default="qwen/qwen3-235b-a22b", help="Model name (OpenRouter or Tinker)")
    parser.add_argument("--dataset", default="cotcontrol/gpqa", help="Dataset name")
    parser.add_argument("--modes", default="baseline", help="Comma-separated list of control modes, or 'all'")
    parser.add_argument("--split", default="test", choices=["train", "test", "all"], help="Dataset split")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of split to use (0-1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--backend", default="openrouter", choices=["openrouter", "tinker"], help="Inference backend")
    parser.add_argument("--max-concurrency", type=int, default=30, help="Max concurrent requests")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries per request")
    parser.add_argument("--max-tokens", type=int, default=16384, help="Max tokens per response")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--output-dir", default="results/rollouts", help="Output directory")
    parser.add_argument("--adversarial-pressure", default="none", help="Adversarial pressure type")
    parser.add_argument("--n-samples", type=int, default=None, help="Number of samples (proportional stratified sampling)")

    args = parser.parse_args()

    return ExperimentConfig(
        model=args.model,
        dataset=args.dataset,
        modes=_resolve_modes(args.modes, args.dataset),
        split=args.split,
        fraction=args.fraction,
        seed=args.seed,
        backend=args.backend,
        max_concurrency=args.max_concurrency,
        max_retries=args.max_retries,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        output_dir=args.output_dir,
        adversarial_pressure=args.adversarial_pressure,
        n_samples=args.n_samples,
    )


def main():
    config = parse_args()
    print(f"Experiment: {config.experiment_id}")
    print(f"  model={config.model}  dataset={config.dataset}")
    print(f"  modes={config.modes}  split={config.split}  fraction={config.fraction}")
    print(f"  backend={config.backend}  concurrency={config.max_concurrency}")
    print()

    asyncio.run(run_experiment(config))


if __name__ == "__main__":
    main()
