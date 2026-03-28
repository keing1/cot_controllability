"""Configuration and settings for controllability evaluation."""

from __future__ import annotations

import glob
import hashlib
import json
import time
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Environment-based secrets and global settings."""

    openrouter_api_key: str = ""
    tinker_api_key: str = ""
    openai_api_key: str = ""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


class ExperimentConfig(BaseSettings):
    """Configuration for a single evaluation experiment. Built from CLI args."""

    model: str = "qwen/qwen3-235b-a22b"
    dataset: str = "cotcontrol/gpqa"
    modes: list[str] = Field(default_factory=lambda: ["baseline"])
    split: Literal["train", "test", "all"] = "test"
    fraction: float = 1.0
    seed: int = 42
    backend: Literal["openrouter", "tinker"] = "openrouter"
    max_concurrency: int = 30
    max_retries: int = 3
    max_tokens: int = 16384
    temperature: float = 1.0
    output_dir: str = "results/rollouts"
    adversarial_pressure: str = "none"
    n_samples: int | None = None
    model_path: str | None = None  # Tinker checkpoint path (tinker://...)
    reasoning_effort: str = "high"  # Reasoning effort for gpt-oss models
    request_timeout: int = 300  # Per-request timeout in seconds (not in experiment_id)
    analysis_channel: bool = False  # Use "analysis channel" instead of "reasoning" in ReasonIF prompts

    @property
    def experiment_id(self) -> str:
        """Deterministic experiment ID from config params."""
        key = json.dumps(
            {
                "model": self.model,
                "dataset": self.dataset,
                "modes": sorted(self.modes),
                "split": self.split,
                "fraction": self.fraction,
                "seed": self.seed,
                "adversarial_pressure": self.adversarial_pressure,
                "n_samples": self.n_samples,
                "model_path": self.model_path,
                "reasoning_effort": self.reasoning_effort,
                "analysis_channel": self.analysis_channel,
            },
            sort_keys=True,
        )
        return hashlib.sha256(key.encode()).hexdigest()[:12]

    @property
    def output_filename(self) -> str:
        """Generate output filename from config.

        Reuses an existing file matching the experiment_id (for resume support).
        Creates a new timestamped filename only on first run.
        """
        model_short = self.model.replace("/", "_")
        dataset_short = self.dataset.replace("/", "_")
        prefix = f"{model_short}_{dataset_short}_{self.split}_{self.experiment_id}"
        # Look for an existing file to resume into.
        pattern = f"{self.output_dir}/{prefix}_*.jsonl"
        existing = sorted(glob.glob(pattern))
        if existing:
            return Path(existing[-1]).name
        ts = int(time.time())
        return f"{prefix}_{ts}.jsonl"
