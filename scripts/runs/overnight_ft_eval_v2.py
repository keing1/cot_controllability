#!/usr/bin/env python3
"""Overnight FT + Eval v2: train 4 models at two learning rates, eval checkpoints.

Runs 8 total fine-tunes (4 models x 2 LRs):
  - lr=4e-4  ("high LR")
  - lr=2.5e-5 ("low LR")

Then evaluates every checkpoint on both eval suites (reasonif + cotcontrol).

Usage:
    python scripts/runs/overnight_ft_eval_v2.py --dry-run
    python scripts/runs/overnight_ft_eval_v2.py
    python scripts/runs/overnight_ft_eval_v2.py --delay 3600   # wait 1hr before starting
    python scripts/runs/overnight_ft_eval_v2.py --models gpt-oss-20b,qwen3-8b
    python scripts/runs/overnight_ft_eval_v2.py --lrs 4e-4     # only high LR
    python scripts/runs/overnight_ft_eval_v2.py --eval-only --run-id 20260320_XXXXXX
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import chz
import pandas as pd
import tinker
from transformers import AutoTokenizer

from tinker_cookbook.renderers import GptOssRenderer, Qwen3Renderer, TrainOnWhat, Message
from tinker_cookbook.supervised.common import datum_from_tokens_weights
from tinker_cookbook.supervised.train import Config, main as train_main
from tinker_cookbook.supervised.types import SupervisedDataset, SupervisedDatasetBuilder

from controllability.config import ExperimentConfig
from controllability.evals.runner import run_experiment

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
SFT_DIR = RESULTS_DIR / "sft"
ROLLOUTS_DIR = RESULTS_DIR / "rollouts"
LOG_FILE = RESULTS_DIR / "overnight_ft_eval_v2.log"

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

# Shared training hyperparams (same as before, except LR which is parameterized)
BATCH_SIZE = 4
MAX_LENGTH = 8192
NUM_EPOCHS = 1
LORA_RANK = 32
SAVE_EVERY = 60
EVAL_EVERY = 10

# Learning rates to sweep
LR_CONFIGS = [
    {"lr": 4e-4, "label": "lr4e-4"},
    {"lr": 2.5e-5, "label": "lr2.5e-5"},
]

# Eval config
EVAL_DATASETS = [
    {"dataset": "reasonif", "modes": "all", "n_samples": None},
    {"dataset": "cotcontrol", "modes": "all", "n_samples": 100},
]

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


# --------------------------------------------------------------------------- #
# Model Specs
# --------------------------------------------------------------------------- #

@dataclass
class FTModelSpec:
    model: str
    base_model: str
    parquet: str
    log_path: str
    renderer_type: str
    reasoning_effort: str
    short_name: str = ""

    def __post_init__(self):
        if not self.short_name:
            self.short_name = self.model.split("/")[-1]


FT_MODELS = [
    FTModelSpec(
        model="openai/gpt-oss-20b",
        base_model="openai/gpt-oss-20b",
        parquet="gpt-oss-20b-reasonif-sft.parquet",
        log_path="gpt-oss-20b-reasonif",
        renderer_type="gpt-oss",
        reasoning_effort="medium",
    ),
    FTModelSpec(
        model="openai/gpt-oss-120b",
        base_model="openai/gpt-oss-120b",
        parquet="gpt-oss-120b-reasonif-sft.parquet",
        log_path="gpt-oss-120b-reasonif",
        renderer_type="gpt-oss",
        reasoning_effort="medium",
    ),
    FTModelSpec(
        model="qwen/qwen3-8b",
        base_model="Qwen/Qwen3-8B",
        parquet="qwen3-8b-reasonif-sft.parquet",
        log_path="qwen3-8b-reasonif",
        renderer_type="qwen3",
        reasoning_effort="none",
    ),
    FTModelSpec(
        model="qwen/qwen3-32b",
        base_model="Qwen/Qwen3-32B",
        parquet="qwen3-32b-reasonif-sft.parquet",
        log_path="qwen3-32b-reasonif",
        renderer_type="qwen3",
        reasoning_effort="none",
    ),
]


# --------------------------------------------------------------------------- #
# Qwen3 CoT Renderer (same as v1)
# --------------------------------------------------------------------------- #

class Qwen3CoTRenderer(Qwen3Renderer):
    """Qwen3Renderer extended to handle the 'thinking' field for CoT SFT training."""

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


# --------------------------------------------------------------------------- #
# Dataset (same as v1)
# --------------------------------------------------------------------------- #

class ReasonIFDataset(SupervisedDataset):
    def __init__(self, parquet_path, batch_size, max_length, renderer):
        df = pd.read_parquet(parquet_path)
        self._conversations = []
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


_ACTIVE_SPEC: FTModelSpec | None = None


@chz.chz
class ReasonIFDatasetBuilder(SupervisedDatasetBuilder):
    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        spec = _ACTIVE_SPEC
        assert spec is not None, "No active FTModelSpec set"

        tokenizer = AutoTokenizer.from_pretrained(spec.base_model)
        renderer = _make_renderer(spec, tokenizer)

        parquet_path = SFT_DIR / spec.parquet
        dataset = ReasonIFDataset(
            parquet_path=parquet_path,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
            renderer=renderer,
        )
        logging.info(
            "Dataset: %d examples, %d batches of %d",
            len(dataset._conversations), len(dataset), BATCH_SIZE,
        )
        return dataset, None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _make_renderer(spec: FTModelSpec, tokenizer):
    if spec.renderer_type == "gpt-oss":
        return GptOssRenderer(
            tokenizer=tokenizer,
            use_system_prompt=True,
            reasoning_effort=spec.reasoning_effort,
        )
    elif spec.renderer_type == "qwen3":
        return Qwen3CoTRenderer(tokenizer=tokenizer)
    else:
        raise ValueError(f"Unknown renderer_type: {spec.renderer_type}")


def _resolve_modes(modes_str: str, dataset: str) -> list[str]:
    if modes_str == "all":
        family = dataset.split("/")[0]
        return DATASET_MODES[family]
    return [m.strip() for m in modes_str.split(",")]


def _load_checkpoints(log_path: Path) -> list[dict]:
    checkpoints_file = log_path / "checkpoints.jsonl"
    if not checkpoints_file.exists():
        logging.warning("No checkpoints found at %s", checkpoints_file)
        return []

    checkpoints = []
    with open(checkpoints_file) as f:
        for line in f:
            line = line.strip()
            if line:
                checkpoints.append(json.loads(line))

    return [c for c in checkpoints if "sampler_path" in c]


def _log_path_for(spec: FTModelSpec, lr_label: str, run_id: str) -> str:
    """Build log path that includes LR label + run ID to avoid collisions."""
    return str(SFT_DIR / f"{spec.log_path}-{lr_label}-{run_id}")


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def do_train(spec: FTModelSpec, lr: float, lr_label: str, run_id: str) -> None:
    global _ACTIVE_SPEC
    _ACTIVE_SPEC = spec

    log_path = _log_path_for(spec, lr_label, run_id)

    logging.info("=" * 60)
    logging.info("TRAINING: %s  lr=%s (%s)", spec.model, lr, lr_label)
    logging.info("=" * 60)
    logging.info("Base model:  %s", spec.base_model)
    logging.info("Parquet:     %s", SFT_DIR / spec.parquet)
    logging.info("Log path:    %s", log_path)
    logging.info("Renderer:    %s (effort=%s)", spec.renderer_type, spec.reasoning_effort)
    logging.info(
        "LR=%s  batch_size=%d  lora_rank=%d  save_every=%d",
        lr, BATCH_SIZE, LORA_RANK, SAVE_EVERY,
    )

    config = Config(
        log_path=log_path,
        model_name=spec.base_model,
        dataset_builder=ReasonIFDatasetBuilder(),
        learning_rate=lr,
        lr_schedule="linear",
        num_epochs=NUM_EPOCHS,
        lora_rank=LORA_RANK,
        save_every=SAVE_EVERY,
        eval_every=EVAL_EVERY,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
    )

    asyncio.run(train_main(config))
    logging.info("Training complete for %s lr=%s", spec.model, lr_label)


# --------------------------------------------------------------------------- #
# Checkpoint Evaluation
# --------------------------------------------------------------------------- #

def do_eval_checkpoints(spec: FTModelSpec, lr_label: str, run_id: str) -> None:
    log_path = Path(_log_path_for(spec, lr_label, run_id))
    checkpoints = _load_checkpoints(log_path)

    if not checkpoints:
        logging.warning("No checkpoints to evaluate for %s lr=%s", spec.model, lr_label)
        return

    logging.info("=" * 60)
    logging.info("EVALUATING CHECKPOINTS: %s lr=%s (%d checkpoints)",
                 spec.model, lr_label, len(checkpoints))
    logging.info("=" * 60)
    for c in checkpoints:
        logging.info("  %s -> %s", c["name"], c["sampler_path"])

    output_dir = str(ROLLOUTS_DIR)

    for ckpt in checkpoints:
        ckpt_name = ckpt["name"]
        sampler_path = ckpt["sampler_path"]

        for eval_cfg in EVAL_DATASETS:
            dataset = eval_cfg["dataset"]
            modes = _resolve_modes(eval_cfg["modes"], dataset)
            n_samples = eval_cfg["n_samples"]

            # Model label includes LR and checkpoint
            model_label = f"{spec.model}-rif-{lr_label}-{ckpt_name}"

            logging.info("Eval: %s on %s", model_label, dataset)

            config = ExperimentConfig(
                model=spec.model,
                dataset=dataset,
                modes=modes,
                split="all",
                fraction=1.0,
                seed=42,
                backend="tinker",
                max_concurrency=100,
                max_retries=3,
                max_tokens=28000,
                temperature=1.0,
                output_dir=output_dir,
                n_samples=n_samples,
                model_path=sampler_path,
                reasoning_effort=spec.reasoning_effort,
                request_timeout=420,
            )

            # Custom filename with LR label + run ID to guarantee uniqueness
            model_short = model_label.replace("/", "_")
            dataset_short = dataset.replace("/", "_")
            custom_filename = (
                f"{model_short}_{dataset_short}_all"
                f"_{run_id}_{config.experiment_id}.jsonl"
            )
            output_path = Path(output_dir) / custom_filename

            # Monkey-patch the config to use custom filename
            config.__class__.output_filename = property(
                lambda self, fn=custom_filename: fn
            )

            logging.info("  Output: %s", output_path)
            asyncio.run(run_experiment(config))

    logging.info("All checkpoint evals complete for %s lr=%s", spec.model, lr_label)


# --------------------------------------------------------------------------- #
# Dry Run
# --------------------------------------------------------------------------- #

def print_dry_run(specs: list[FTModelSpec], lr_configs: list[dict], run_id: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"DRY RUN — Execution Plan (run_id={run_id})")
    print(f"{'=' * 70}")

    total_ft = len(specs) * len(lr_configs)
    print(f"\n{total_ft} fine-tune runs ({len(specs)} models x {len(lr_configs)} LRs)\n")

    job_num = 0
    for lr_cfg in lr_configs:
        lr = lr_cfg["lr"]
        lr_label = lr_cfg["label"]
        print(f"  --- LR = {lr} ({lr_label}) ---")
        for spec in specs:
            job_num += 1
            parquet_path = SFT_DIR / spec.parquet
            log_path = _log_path_for(spec, lr_label, run_id)
            parquet_exists = parquet_path.exists()

            print(f"\n  {job_num}. {spec.model}")
            print(f"     Base model:  {spec.base_model}")
            print(f"     Renderer:    {spec.renderer_type} (effort={spec.reasoning_effort})")
            print(f"     Parquet:     {parquet_path} {'[OK]' if parquet_exists else '[MISSING]'}")
            print(f"     Log path:    {log_path}")
            print(f"     Training:    lr={lr}, batch_size={BATCH_SIZE}, "
                  f"lora_rank={LORA_RANK}, epochs={NUM_EPOCHS}")
            print(f"     Eval:        reasonif (all 300) + cotcontrol (100 stratified)")
        print()

    total_evals = total_ft * 4 * 2  # 4 checkpoints x 2 datasets
    print(f"Total: {total_ft} training runs, ~{total_evals} eval runs")
    print(f"  (4 checkpoints/model x 2 datasets x {total_ft} models)\n")


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

def _setup_logging() -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(LOG_FILE, mode="a")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Overnight FT + Eval v2: train models at two LRs, eval checkpoints"
    )
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, only eval existing checkpoints")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print execution plan without running")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model short names (e.g., gpt-oss-20b,qwen3-8b)")
    parser.add_argument("--lrs", type=str, default=None,
                        help="Comma-separated LR labels to run (e.g., lr4e-4,lr2.5e-5)")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Reuse a previous run ID for resume")
    parser.add_argument("--delay", type=int, default=0,
                        help="Seconds to wait before starting (e.g., 3600 for 1hr)")
    args = parser.parse_args()

    run_id = args.run_id or RUN_ID

    _setup_logging()

    # Filter models
    specs = list(FT_MODELS)
    if args.models:
        requested = {m.strip() for m in args.models.split(",")}
        specs = [s for s in specs if s.short_name in requested]
        if not specs:
            print(f"ERROR: No matching models for --models={args.models}")
            print(f"Available: {', '.join(s.short_name for s in FT_MODELS)}")
            sys.exit(1)

    # Filter LRs
    lr_configs = list(LR_CONFIGS)
    if args.lrs:
        requested_lrs = {l.strip() for l in args.lrs.split(",")}
        lr_configs = [c for c in lr_configs if c["label"] in requested_lrs]
        if not lr_configs:
            print(f"ERROR: No matching LRs for --lrs={args.lrs}")
            print(f"Available: {', '.join(c['label'] for c in LR_CONFIGS)}")
            sys.exit(1)

    if args.dry_run:
        print_dry_run(specs, lr_configs, run_id)
        return

    # Delay
    if args.delay > 0:
        logging.info("Waiting %d seconds before starting...", args.delay)
        time.sleep(args.delay)
        logging.info("Delay complete, starting now.")

    start_time = datetime.now(timezone.utc)
    total_jobs = len(specs) * len(lr_configs)
    logging.info(
        "Overnight FT + Eval v2: %d models x %d LRs = %d jobs, "
        "eval_only=%s, run_id=%s",
        len(specs), len(lr_configs), total_jobs, args.eval_only, run_id,
    )

    results: dict[str, str] = {}

    for lr_cfg in lr_configs:
        lr = lr_cfg["lr"]
        lr_label = lr_cfg["label"]

        for spec in specs:
            job_key = f"{spec.model} ({lr_label})"
            job_start = datetime.now(timezone.utc)
            logging.info("START: %s", job_key)

            try:
                if not args.eval_only:
                    parquet_path = SFT_DIR / spec.parquet
                    if not parquet_path.exists():
                        raise FileNotFoundError(f"Parquet not found: {parquet_path}")
                    do_train(spec, lr, lr_label, run_id)

                do_eval_checkpoints(spec, lr_label, run_id)

                elapsed = (datetime.now(timezone.utc) - job_start).total_seconds()
                results[job_key] = f"OK ({elapsed:.0f}s)"
                logging.info("DONE  %s — %s", job_key, results[job_key])

            except Exception as e:
                elapsed = (datetime.now(timezone.utc) - job_start).total_seconds()
                results[job_key] = f"FAIL ({type(e).__name__}: {e}, {elapsed:.0f}s)"
                logging.error("FAIL  %s — %s", job_key, results[job_key], exc_info=True)

    # Summary
    elapsed_total = (datetime.now(timezone.utc) - start_time).total_seconds()
    summary = [
        "",
        "=" * 70,
        f"OVERNIGHT FT + EVAL v2 COMPLETE — {elapsed_total:.0f}s total",
        "=" * 70,
    ]
    for key, status in results.items():
        summary.append(f"  {key:50s} {status}")

    n_ok = sum(1 for s in results.values() if s.startswith("OK"))
    n_fail = sum(1 for s in results.values() if s.startswith("FAIL"))
    summary.append(f"\n  {n_ok} succeeded, {n_fail} failed")

    for line in summary:
        logging.info(line)


if __name__ == "__main__":
    main()
