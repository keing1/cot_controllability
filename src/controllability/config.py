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

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


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
    reasoning_effort: str = "medium"  # Reasoning effort for gpt-oss models
    request_timeout: int = 300  # Per-request timeout in seconds (not in experiment_id)
    analysis_channel: bool = True  # Use "analysis channel" instead of "reasoning" in ReasonIF prompts
    word_suppression_synonyms: bool = False  # Include synonyms in word_suppression prompt & grading (paper mode)

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
                "word_suppression_synonyms": self.word_suppression_synonyms,
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


class MonitorQAConfig(BaseSettings):
    """Configuration for a monitor_qa experiment (actor + monitor pipeline)."""

    model_config = {"extra": "ignore"}

    # Actor model
    actor_model: str = "qwen/qwen3-8b"
    actor_backend: Literal["openrouter", "tinker"] = "tinker"
    actor_max_tokens: int = 28000
    actor_temperature: float = 1.0
    actor_concurrency: int = 40
    actor_max_retries: int = 1

    # Monitor model (non-reasoning)
    monitor_model: str = "openai/gpt-4o"
    monitor_max_tokens: int = 1024
    monitor_temperature: float = 0.0
    monitor_concurrency: int = 60
    monitor_max_retries: int = 3

    # Dataset
    dataset_path: str = "results/monitorqa/gpqa_hle_w_math_monitorqa.jsonl"
    dataset_split: str = "val"  # "old", "val", or "test"
    n_samples: int | None = None
    seed: int = 42

    # Prompt types
    prompt_type: str = "default"
    monitor_types: list[str] = Field(default_factory=lambda: ["metr_note"])
    monitor_model_overrides: dict[str, str] = Field(default_factory=dict)  # monitor_type -> model name

    # Modes
    modes: list[str] = Field(default_factory=lambda: ["side_task"])

    # Inference settings
    reasoning_effort: str = "medium"
    request_timeout: int = 900
    model_path: str | None = None  # Tinker checkpoint path

    # Output
    output_dir: str = "results/rollouts/monitor_qa"
    save_every: int = 100  # Save rollouts to disk every N samples
    output_filename_override: str | None = None  # If set, use this filename instead of auto-generated

    @property
    def experiment_id(self) -> str:
        """Deterministic experiment ID from config params."""
        key = json.dumps(
            {
                "actor_model": self.actor_model,
                "monitor_model": self.monitor_model,
                "dataset_path": self.dataset_path,
                "dataset_split": self.dataset_split,
                "modes": sorted(self.modes),
                "n_samples": self.n_samples,
                "seed": self.seed,
                "prompt_type": self.prompt_type,
                "monitor_types": sorted(self.monitor_types),
                "model_path": self.model_path,
                "reasoning_effort": self.reasoning_effort,
            },
            sort_keys=True,
        )
        return hashlib.sha256(key.encode()).hexdigest()[:12]

    @property
    def output_filename(self) -> str:
        """Generate output filename with resume support."""
        if self.output_filename_override:
            return self.output_filename_override
        actor_short = self.actor_model.replace("/", "_")
        prefix = f"monitor_qa_{actor_short}_{self.experiment_id}"
        pattern = f"{self.output_dir}/{prefix}_*.jsonl"
        existing = sorted(glob.glob(pattern))
        if existing:
            return Path(existing[-1]).name
        ts = int(time.time())
        return f"{prefix}_{ts}.jsonl"
