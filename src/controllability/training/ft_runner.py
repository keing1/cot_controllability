"""Shared helpers for fine-tuning runs.

Extracted from ``scripts/runs/overnight_ft_eval_v4.py`` so the new
``launch_ft.py`` orchestrator and any future runners can reuse:
  * ``FTModelSpec`` — model/renderer/effort config
  * ``SFTDataset`` — loads an SFT parquet, emits Tinker Datum objects
  * ``Qwen3CoTRenderer`` — subclass of Qwen3Renderer that handles the
    'thinking' field for CoT SFT training
  * ``make_renderer`` / ``load_checkpoints_jsonl`` — small utility helpers
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import chz
import pandas as pd
import tinker
from tinker_cookbook.renderers import (
    GptOssRenderer,
    Message,
    Qwen3Renderer,
    TrainOnWhat,
)
from tinker_cookbook.supervised.common import datum_from_tokens_weights
from tinker_cookbook.supervised.types import SupervisedDataset, SupervisedDatasetBuilder


# ---------------------------------------------------------------------------
# Model specification
# ---------------------------------------------------------------------------


@dataclass
class FTModelSpec:
    """Describes a model to fine-tune."""

    model: str                         # API-facing name (e.g. openai/gpt-oss-20b)
    base_model: str                    # Tinker base model id (may differ in case)
    parquet: str                       # filename OR absolute path to SFT parquet
    log_path: str                      # per-run log-path stem (usually matches model)
    renderer_type: str                 # "gpt-oss" | "qwen3"
    reasoning_effort: str              # "low"/"medium"/"high"/"none"
    short_name: str = ""

    def __post_init__(self):
        if not self.short_name:
            self.short_name = self.model.split("/")[-1]


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


class Qwen3CoTRenderer(Qwen3Renderer):
    """Qwen3Renderer extended to handle the 'thinking' field for CoT SFT."""

    def _render_message(
        self, idx: int, message: Message
    ) -> tuple[list[int], list[int], list[int]]:
        maybe_newline = "\n" if idx > 0 else ""
        ob_str = f"{maybe_newline}<|im_start|>{message['role']}\n"

        thinking = message.get("thinking")
        content = message.get("content", "") or ""

        if message["role"] == "assistant" and thinking:
            ac_content = f"<think>\n{thinking}\n</think>\n{content}"
        elif message["role"] == "assistant" and "<think>" not in content:
            ob_str += "<think>\n"
            ac_content = content
        else:
            ac_content = content

        ac_content += "<|im_end|>"

        return (
            self.tokenizer.encode(ob_str, add_special_tokens=False),
            self.tokenizer.encode(ac_content, add_special_tokens=False),
            self.tokenizer.encode("", add_special_tokens=False),
        )


def make_renderer(spec: FTModelSpec, tokenizer):
    if spec.renderer_type == "gpt-oss":
        return GptOssRenderer(
            tokenizer=tokenizer,
            use_system_prompt=True,
            reasoning_effort=spec.reasoning_effort,
        )
    if spec.renderer_type == "qwen3":
        return Qwen3CoTRenderer(tokenizer=tokenizer)
    raise ValueError(f"Unknown renderer_type: {spec.renderer_type}")


# ---------------------------------------------------------------------------
# SFT dataset
# ---------------------------------------------------------------------------


class SFTDataset(SupervisedDataset):
    """Loads an SFT parquet and converts to Tinker Datum batches.

    Accepts either a single parquet path or a list (which will be concatenated
    row-wise). If ``n_examples`` is set, the result is truncated to the first N
    rows (after concatenation, before any shuffling).
    """

    def __init__(
        self,
        parquet_paths: str | Path | list[str | Path],
        batch_size: int,
        max_length: int,
        renderer,
        n_examples: int | None = None,
    ):
        if isinstance(parquet_paths, (str, Path)):
            parquet_paths = [parquet_paths]
        frames = [pd.read_parquet(p) for p in parquet_paths]
        df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        if n_examples is not None:
            df = df.iloc[:n_examples]

        self._conversations: list[list[dict]] = []
        for _, row in df.iterrows():
            messages = []
            for msg in row["messages"]:
                m = {"role": msg["role"], "content": msg.get("content", "") or ""}
                if msg.get("thinking"):
                    m["thinking"] = msg["thinking"]
                messages.append(m)
            self._conversations.append(messages)

        self._batch_size = batch_size
        self._max_length = max_length
        self._renderer = renderer
        self._indices = list(range(len(self._conversations)))

    def set_epoch(self, seed: int = 0):
        rng = random.Random(seed)
        self._indices = list(range(len(self._conversations)))
        rng.shuffle(self._indices)

    def get_batch(self, index: int) -> list[tinker.Datum]:
        start = index * self._batch_size
        end = min(start + self._batch_size, len(self._indices))
        batch_indices = self._indices[start:end]
        data = []
        for idx in batch_indices:
            conv = self._conversations[idx]
            tokens, weights = self._renderer.build_supervised_example(
                conv, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES
            )
            datum = datum_from_tokens_weights(tokens, weights, self._max_length)
            data.append(datum)
        return data

    def __len__(self) -> int:
        return len(self._indices) // self._batch_size


# Global handoff for chz-based SupervisedDatasetBuilder — set by the caller
# before invoking Tinker's training entrypoint, then read by ``_build_dataset``.
_ACTIVE_BUILDER_CONFIG: "DatasetBuildConfig | None" = None


@dataclass
class DatasetBuildConfig:
    parquet_paths: list[str]
    base_model: str
    renderer_type: str
    reasoning_effort: str
    batch_size: int
    max_length: int
    n_examples: int | None = None


def set_active_builder_config(cfg: DatasetBuildConfig | None) -> None:
    global _ACTIVE_BUILDER_CONFIG
    _ACTIVE_BUILDER_CONFIG = cfg


def get_active_builder_config() -> DatasetBuildConfig:
    if _ACTIVE_BUILDER_CONFIG is None:
        raise RuntimeError(
            "No active DatasetBuildConfig — set one via set_active_builder_config()"
        )
    return _ACTIVE_BUILDER_CONFIG


@chz.chz
class ParquetDatasetBuilder(SupervisedDatasetBuilder):
    """SupervisedDatasetBuilder that reads from ``_ACTIVE_BUILDER_CONFIG``."""

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        from transformers import AutoTokenizer

        cfg = get_active_builder_config()
        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
        spec_like = FTModelSpec(
            model=cfg.base_model,
            base_model=cfg.base_model,
            parquet="",
            log_path="",
            renderer_type=cfg.renderer_type,
            reasoning_effort=cfg.reasoning_effort,
        )
        renderer = make_renderer(spec_like, tokenizer)
        dataset = SFTDataset(
            parquet_paths=cfg.parquet_paths,
            batch_size=cfg.batch_size,
            max_length=cfg.max_length,
            renderer=renderer,
            n_examples=cfg.n_examples,
        )
        return dataset, None


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def load_checkpoints_jsonl(log_path: Path) -> list[dict]:
    """Read the ``checkpoints.jsonl`` file emitted by Tinker training.

    Returns only entries with a ``sampler_path`` field (i.e. saved checkpoints),
    preserving their on-disk order.
    """
    ckpt_file = log_path / "checkpoints.jsonl"
    if not ckpt_file.exists():
        return []
    out: list[dict] = []
    with open(ckpt_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out.append(rec)
    return [c for c in out if "sampler_path" in c]
