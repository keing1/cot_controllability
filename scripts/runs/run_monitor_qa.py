#!/usr/bin/env python3
"""Run a monitor_qa experiment.

Two-phase pipeline: a reasoning actor model generates responses with side tasks,
then a non-reasoning monitor model scores stripped versions of the interaction.

Usage:
    uv run python scripts/runs/run_monitor_qa.py \
        --dataset-path data/monitor_qa/example.jsonl \
        --actor-model qwen/qwen3-8b \
        --monitor-model openai/gpt-4o \
        --modes side_task,baseline \
        --n-samples 5
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from controllability.config import MonitorQAConfig
from controllability.evals.runner import run_monitor_experiment


def parse_args() -> MonitorQAConfig:
    parser = argparse.ArgumentParser(description="Run monitor_qa evaluation")

    # Actor model
    parser.add_argument("--actor-model", default="qwen/qwen3-8b", help="Actor model name")
    parser.add_argument("--actor-backend", default="tinker", choices=["openrouter", "tinker"])
    parser.add_argument("--actor-max-tokens", type=int, default=28000)
    parser.add_argument("--actor-temperature", type=float, default=1.0)
    parser.add_argument("--actor-concurrency", type=int, default=40)

    # Monitor model
    parser.add_argument("--monitor-model", default="openai/gpt-4o", help="Monitor model name")
    parser.add_argument("--monitor-max-tokens", type=int, default=1024)
    parser.add_argument("--monitor-temperature", type=float, default=0.0)
    parser.add_argument("--monitor-concurrency", type=int, default=60)
    parser.add_argument(
        "--monitor-reasoning-effort",
        default=None,
        help=(
            "Default reasoning effort for monitor calls ('low'/'medium'/'high'/"
            "'minimal'). Required for gpt-5.x / gpt-oss / o-series to think. "
            "Omit for no thinking (default)."
        ),
    )
    parser.add_argument(
        "--monitor-reasoning-effort-overrides",
        default="",
        help=(
            "Per-monitor-type reasoning effort overrides as "
            "'type1=effort1,type2=effort2' (e.g. 'metr_note_gpt54=high')."
        ),
    )

    # Dataset
    parser.add_argument("--dataset-path", default="results/monitorqa/gpqa_hle_w_math_monitorqa.jsonl",
                        help="Path to monitor_qa JSONL dataset")
    parser.add_argument("--n-samples", type=int, default=None, help="Subsample N samples")
    parser.add_argument("--seed", type=int, default=42)

    # Prompt types
    parser.add_argument("--prompt-type", default="default")
    parser.add_argument("--monitor-types", default="metr_note",
                        help="Comma-separated monitor types: metr_note,old_monitor")

    # Modes
    parser.add_argument("--modes", default="side_task", help="Comma-separated: side_task,baseline")

    # Inference
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--request-timeout", type=int, default=900)
    parser.add_argument("--model-path", default=None, help="Tinker checkpoint path")

    # Output
    parser.add_argument("--output-dir", default="results/rollouts/monitor_qa")

    args = parser.parse_args()

    overrides: dict[str, str] = {}
    if args.monitor_reasoning_effort_overrides:
        for pair in args.monitor_reasoning_effort_overrides.split(","):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                raise ValueError(
                    f"Bad --monitor-reasoning-effort-overrides entry {pair!r}; "
                    "expected 'type=effort'"
                )
            k, v = pair.split("=", 1)
            overrides[k.strip()] = v.strip()

    return MonitorQAConfig(
        actor_model=args.actor_model,
        actor_backend=args.actor_backend,
        actor_max_tokens=args.actor_max_tokens,
        actor_temperature=args.actor_temperature,
        actor_concurrency=args.actor_concurrency,
        monitor_model=args.monitor_model,
        monitor_max_tokens=args.monitor_max_tokens,
        monitor_temperature=args.monitor_temperature,
        monitor_concurrency=args.monitor_concurrency,
        dataset_path=args.dataset_path,
        n_samples=args.n_samples,
        seed=args.seed,
        prompt_type=args.prompt_type,
        monitor_types=[m.strip() for m in args.monitor_types.split(",")],
        modes=[m.strip() for m in args.modes.split(",")],
        reasoning_effort=args.reasoning_effort,
        monitor_reasoning_effort=args.monitor_reasoning_effort,
        monitor_reasoning_effort_overrides=overrides,
        request_timeout=args.request_timeout,
        model_path=args.model_path,
        output_dir=args.output_dir,
    )


def main():
    config = parse_args()
    print(f"Monitor QA Experiment: {config.experiment_id}")
    print(f"  actor={config.actor_model}  monitor={config.monitor_model}")
    print(f"  dataset={config.dataset_path}  modes={config.modes}")
    print(f"  actor_concurrency={config.actor_concurrency}  monitor_concurrency={config.monitor_concurrency}")
    print()

    asyncio.run(run_monitor_experiment(config))


if __name__ == "__main__":
    main()
