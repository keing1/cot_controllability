"""Reusable batch evaluation function for running model evals."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from controllability.config import ExperimentConfig
from controllability.evals.runner import run_experiment

# Default modes per dataset family (mirrors run_eval.py DATASET_MODES)
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

DEFAULT_DATASETS = ["reasonif", "cotcontrol"]


@dataclass
class ModelEvalSpec:
    """Specification for evaluating a single model across datasets."""

    model: str
    backend: str  # "openrouter" or "tinker"
    reasoning_effort: str = "high"
    max_tokens: int = 32768
    request_timeout: int = 420
    max_concurrency: int = 30
    temperature: float = 1.0
    seed: int = 42
    datasets: list[str] | None = None  # default: ["reasonif", "cotcontrol"]
    cotcontrol_n_samples: int = 100
    analysis_channel: bool = False  # Use "analysis channel" terminology for ReasonIF


async def run_model_eval(spec: ModelEvalSpec) -> list[Path]:
    """Run evaluation for one model across its configured datasets.

    Returns list of output file paths (one per dataset).
    """
    datasets = spec.datasets or DEFAULT_DATASETS
    output_paths: list[Path] = []

    for dataset in datasets:
        family = dataset.split("/")[0]
        modes = DATASET_MODES.get(family)
        if modes is None:
            raise ValueError(
                f"No modes defined for dataset family '{family}'. "
                f"Known: {list(DATASET_MODES.keys())}"
            )

        # ReasonIF: all 300 samples; CoTControl: n_samples subset
        n_samples = None if family == "reasonif" else spec.cotcontrol_n_samples

        # analysis_channel only applies to reasonif evals
        use_ac = spec.analysis_channel and family == "reasonif"

        config = ExperimentConfig(
            model=spec.model,
            dataset=dataset,
            modes=modes,
            split="all",
            fraction=1.0,
            seed=spec.seed,
            backend=spec.backend,
            max_concurrency=spec.max_concurrency,
            max_retries=3,
            max_tokens=spec.max_tokens,
            temperature=spec.temperature,
            output_dir="results/rollouts",
            reasoning_effort=spec.reasoning_effort,
            request_timeout=spec.request_timeout,
            n_samples=n_samples,
            analysis_channel=use_ac,
        )

        output_path = Path(config.output_dir) / config.output_filename

        print(f"\n{'='*60}")
        print(f"  Model: {spec.model} | Dataset: {dataset}")
        print(f"  Modes: {len(modes)} | Samples: {n_samples or 'all'}")
        print(f"  Backend: {spec.backend} | Timeout: {spec.request_timeout}s")
        print(f"  experiment_id={config.experiment_id}")
        print(f"{'='*60}\n")

        await run_experiment(config)
        output_paths.append(output_path)

    # Rebuild eval manifest after all datasets for this model are done
    _rebuild_manifest()

    return output_paths


def _rebuild_manifest() -> None:
    """Rebuild the eval manifest CSV."""
    try:
        subprocess.run(
            [sys.executable, "scripts/build_eval_manifest.py"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as e:
        print(f"  Warning: failed to rebuild manifest: {e}")
