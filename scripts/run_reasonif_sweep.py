#!/usr/bin/env python3
"""Run ReasonIF evals across multiple models in parallel.

Parallelizes across models (and backends) while respecting per-model concurrency.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from controllability.config import ExperimentConfig
from controllability.evals.runner import run_experiment

# All ReasonIF instruction types
REASONIF_MODES = [
    "reasoning_language", "number_words", "english_capital",
    "end_checker", "json_format", "no_comma",
]

# Models to evaluate: (model_id, backend)
MODELS = [
    # Tinker models
    ("openai/gpt-oss-20b", "tinker"),
    ("openai/gpt-oss-120b", "tinker"),
    ("Qwen/Qwen3-32B", "tinker"),
    ("Qwen/Qwen3-8B", "tinker"),
    # OpenRouter models
    ("qwen/qwen3-235b-a22b-thinking-2507", "openrouter"),
    ("deepseek/deepseek-r1", "openrouter"),
]


async def run_model(model: str, backend: str) -> None:
    """Run full ReasonIF eval for a single model."""
    config = ExperimentConfig(
        model=model,
        dataset="reasonif",
        modes=REASONIF_MODES,
        split="all",
        fraction=1.0,
        backend=backend,
        max_concurrency=30,
        max_tokens=16384,
        temperature=1.0,
        output_dir="results/rollouts",
    )
    print(f"\n{'='*60}")
    print(f"Starting: {model} ({backend})")
    print(f"  experiment_id={config.experiment_id}")
    print(f"{'='*60}\n")

    try:
        await run_experiment(config)
    except Exception as e:
        print(f"\nERROR for {model}: {e}")


async def main():
    # Run all models in parallel -- each model gets its own client/connection
    tasks = [run_model(model, backend) for model, backend in MODELS]
    await asyncio.gather(*tasks)
    print("\n\nAll models complete. Summarize with:")
    print("  python scripts/summarize.py results/rollouts/*reasonif*.jsonl")


if __name__ == "__main__":
    asyncio.run(main())
