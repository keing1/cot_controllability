"""Configuration and settings for controllability evaluation."""

from __future__ import annotations

import hashlib
import json
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
            },
            sort_keys=True,
        )
        return hashlib.sha256(key.encode()).hexdigest()[:12]

    @property
    def output_filename(self) -> str:
        """Generate output filename from config."""
        model_short = self.model.replace("/", "_")
        dataset_short = self.dataset.replace("/", "_")
        return f"{model_short}_{dataset_short}_{self.split}_{self.experiment_id}.jsonl"
