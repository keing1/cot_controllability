#!/usr/bin/env python3
"""Overnight FT + Eval: train 4 models, eval checkpoints, then reasoning effort sweep.

Part 1: Fine-tune each model on its ReasonIF SFT parquet, then eval every checkpoint
        on both eval suites (reasonif + cotcontrol).
Part 2: Run reasoning effort sweep on base GPT-OSS models.

Usage:
    python scripts/runs/overnight_ft_eval.py                          # full run
    python scripts/runs/overnight_ft_eval.py --eval-only              # skip training
    python scripts/runs/overnight_ft_eval.py --skip-effort-sweep      # skip Part 2
    python scripts/runs/overnight_ft_eval.py --dry-run                # print plan only
    python scripts/runs/overnight_ft_eval.py --models gpt-oss-20b,qwen3-8b  # subset
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
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


class Qwen3CoTRenderer(Qwen3Renderer):
    """Qwen3Renderer extended to handle the 'thinking' field for CoT SFT training.

    The upstream Qwen3Renderer asserts if a message has a 'thinking' field.
    This subclass merges thinking into content using Qwen3's native
    <think>...</think> format so the model is trained on its own reasoning.
    """

    def _render_message(
        self, idx: int, message: Message
    ) -> tuple[list[int], list[int], list[int]]:
        maybe_newline = "\n" if idx > 0 else ""
        ob_str = f"{maybe_newline}<|im_start|>{message['role']}\n"

        thinking = message.get("thinking")
        content = message.get("content", "") or ""

        if message["role"] == "assistant" and thinking:
            # Combine thinking + content in Qwen3's native format.
            # Both thinking and content end up in action_part (trained on).
            ac_content = f"<think>\n{thinking}\n</think>\n{content}"
        elif message["role"] == "assistant" and "<think>" not in content:
            # No thinking at all — add <think> prompt as observation (not trained on),
            # matching upstream Qwen3Renderer behavior.
            ob_str += "<think>\n"
            ac_content = content
        else:
            # Content already has <think> tags inline — keep as-is.
            ac_content = content

        ac_content += "<|im_end|>"

        return (
            self.tokenizer.encode(ob_str, add_special_tokens=False),
            self.tokenizer.encode(ac_content, add_special_tokens=False),
            self.tokenizer.encode("", add_special_tokens=False),
        )

from controllability.config import ExperimentConfig
from controllability.evals.runner import run_experiment

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
SFT_DIR = RESULTS_DIR / "sft"
ROLLOUTS_DIR = RESULTS_DIR / "rollouts"
LOG_FILE = RESULTS_DIR / "overnight_ft_eval.log"

# Unique run ID so training/eval never collide with prior runs
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

# Shared training hyperparams (same as train_reasonif_sft.py)
BATCH_SIZE = 4
MAX_LENGTH = 8192
LEARNING_RATE = 1e-4
NUM_EPOCHS = 1
LORA_RANK = 32
SAVE_EVERY = 60
EVAL_EVERY = 10

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
    model: str           # e.g., "openai/gpt-oss-20b"
    base_model: str      # HF name for tokenizer, e.g., "openai/gpt-oss-20b" or "Qwen/Qwen3-8B"
    parquet: str         # filename under results/sft/
    log_path: str        # checkpoint output dir name under results/sft/
    renderer_type: str   # "gpt-oss" or "qwen3"
    reasoning_effort: str  # "medium" for gpt, "high" for qwen
    short_name: str = ""   # for --models filtering

    def __post_init__(self):
        if not self.short_name:
            # "openai/gpt-oss-20b" -> "gpt-oss-20b"
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
        reasoning_effort="high",
    ),
    FTModelSpec(
        model="qwen/qwen3-32b",
        base_model="Qwen/Qwen3-32B",
        parquet="qwen3-32b-reasonif-sft.parquet",
        log_path="qwen3-32b-reasonif",
        renderer_type="qwen3",
        reasoning_effort="high",
    ),
]

EFFORT_SWEEP = [
    {"model": "openai/gpt-oss-20b", "efforts": ["low", "medium", "high"]},
    {"model": "openai/gpt-oss-120b", "efforts": ["low", "high"]},
]


# --------------------------------------------------------------------------- #
# Dataset (parameterized version of train_reasonif_sft.py)
# --------------------------------------------------------------------------- #

class ReasonIFDataset(SupervisedDataset):
    """Load the ReasonIF SFT parquet and convert to Tinker Datum objects."""

    def __init__(self, parquet_path: str | Path, batch_size: int, max_length: int,
                 renderer):
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


# We need a global to pass spec into the chz-decorated builder
_ACTIVE_SPEC: FTModelSpec | None = None


@chz.chz
class ReasonIFDatasetBuilder(SupervisedDatasetBuilder):
    """Builder that creates the ReasonIF SFT dataset using the active spec."""

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
    """Create the appropriate renderer for a model spec."""
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
    """Load checkpoints.jsonl, returning entries with sampler_path."""
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


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def do_train(spec: FTModelSpec) -> None:
    """Fine-tune a single model on its SFT parquet."""
    global _ACTIVE_SPEC
    _ACTIVE_SPEC = spec

    log_path = str(SFT_DIR / f"{spec.log_path}-{RUN_ID}")

    logging.info("=" * 60)
    logging.info("TRAINING: %s", spec.model)
    logging.info("=" * 60)
    logging.info("Base model:  %s", spec.base_model)
    logging.info("Parquet:     %s", SFT_DIR / spec.parquet)
    logging.info("Log path:    %s", log_path)
    logging.info("Renderer:    %s (effort=%s)", spec.renderer_type, spec.reasoning_effort)
    logging.info(
        "LR=%s  batch_size=%d  lora_rank=%d  save_every=%d",
        LEARNING_RATE, BATCH_SIZE, LORA_RANK, SAVE_EVERY,
    )

    config = Config(
        log_path=log_path,
        model_name=spec.base_model,
        dataset_builder=ReasonIFDatasetBuilder(),
        learning_rate=LEARNING_RATE,
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
    logging.info("Training complete for %s", spec.model)


# --------------------------------------------------------------------------- #
# Checkpoint Evaluation
# --------------------------------------------------------------------------- #

def do_eval_checkpoints(spec: FTModelSpec) -> None:
    """Evaluate all checkpoints for a trained model on both eval suites."""
    log_path = SFT_DIR / f"{spec.log_path}-{RUN_ID}"
    checkpoints = _load_checkpoints(log_path)

    if not checkpoints:
        logging.warning("No checkpoints to evaluate for %s", spec.model)
        return

    logging.info("=" * 60)
    logging.info("EVALUATING CHECKPOINTS: %s (%d checkpoints)", spec.model, len(checkpoints))
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

            # Model label includes checkpoint name
            model_label = f"{spec.model}-rif-{ckpt_name}"

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

            # Override output filename to include checkpoint name + run ID
            model_short = model_label.replace("/", "_")
            dataset_short = dataset.replace("/", "_")
            custom_filename = f"{model_short}_{dataset_short}_all_{RUN_ID}_{config.experiment_id}.jsonl"
            output_path = Path(output_dir) / custom_filename

            # Monkey-patch the config to use our custom filename
            config.__class__.output_filename = property(lambda self, fn=custom_filename: fn)

            logging.info("  Output: %s", output_path)
            asyncio.run(run_experiment(config))

    logging.info("All checkpoint evals complete for %s", spec.model)


# --------------------------------------------------------------------------- #
# Reasoning Effort Sweep (Part 2)
# --------------------------------------------------------------------------- #

def do_effort_sweep() -> None:
    """Run reasoning effort sweep on base GPT-OSS models."""
    logging.info("=" * 60)
    logging.info("PART 2: Reasoning Effort Sweep")
    logging.info("=" * 60)

    output_dir = str(ROLLOUTS_DIR)

    for sweep in EFFORT_SWEEP:
        model = sweep["model"]
        efforts = sweep["efforts"]

        for effort in efforts:
            logging.info("Effort sweep: %s effort=%s", model, effort)

            for eval_cfg in EVAL_DATASETS:
                dataset = eval_cfg["dataset"]
                modes = _resolve_modes(eval_cfg["modes"], dataset)
                n_samples = eval_cfg["n_samples"]

                config = ExperimentConfig(
                    model=model,
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
                    reasoning_effort=effort,
                    request_timeout=420,
                )

                # Use custom filename with effort suffix + run ID to avoid collisions
                model_short = model.replace("/", "_")
                dataset_short = dataset.replace("/", "_")
                custom_filename = (
                    f"{model_short}_{dataset_short}_all"
                    f"_effort-{effort}_{RUN_ID}_{config.experiment_id}.jsonl"
                )
                output_path = Path(output_dir) / custom_filename

                config.__class__.output_filename = property(lambda self, fn=custom_filename: fn)

                logging.info("  %s effort=%s -> %s", dataset, effort, output_path)
                asyncio.run(run_experiment(config))

    logging.info("Effort sweep complete.")


# --------------------------------------------------------------------------- #
# Dry Run
# --------------------------------------------------------------------------- #

def print_dry_run(specs: list[FTModelSpec], skip_effort_sweep: bool) -> None:
    """Print the execution plan without running anything."""
    print("\n" + "=" * 70)
    print(f"DRY RUN — Execution Plan (run_id={RUN_ID})")
    print("=" * 70)

    print("\n--- Part 1: Fine-Tune + Eval ---\n")
    for i, spec in enumerate(specs, 1):
        parquet_path = SFT_DIR / spec.parquet
        log_path = SFT_DIR / f"{spec.log_path}-{RUN_ID}"
        parquet_exists = parquet_path.exists()

        print(f"  {i}. {spec.model}")
        print(f"     Base model:  {spec.base_model}")
        print(f"     Renderer:    {spec.renderer_type} (effort={spec.reasoning_effort})")
        print(f"     Parquet:     {parquet_path} {'[OK]' if parquet_exists else '[MISSING]'}")
        print(f"     Log path:    {log_path}")
        print(f"     Training:    batch_size={BATCH_SIZE}, lr={LEARNING_RATE}, "
              f"lora_rank={LORA_RANK}, epochs={NUM_EPOCHS}")
        print(f"     Eval:        reasonif (all 300) + cotcontrol (100 stratified)")
        print()

    if not skip_effort_sweep:
        print("--- Part 2: Reasoning Effort Sweep ---\n")
        for sweep in EFFORT_SWEEP:
            efforts_str = ", ".join(sweep["efforts"])
            print(f"  {sweep['model']}: efforts=[{efforts_str}]")
            print(f"     Eval: reasonif + cotcontrol per effort level")
        total_sweep = sum(len(s["efforts"]) for s in EFFORT_SWEEP)
        print(f"\n  Total: {total_sweep} effort configs × 2 datasets = {total_sweep * 2} eval runs")
    else:
        print("--- Part 2: SKIPPED (--skip-effort-sweep) ---")

    print()


# --------------------------------------------------------------------------- #
# Logging Setup
# --------------------------------------------------------------------------- #

def _setup_logging() -> None:
    """Configure logging to both stdout and a log file."""
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
        description="Overnight FT + Eval: train models, eval checkpoints, effort sweep"
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training, only eval existing checkpoints",
    )
    parser.add_argument(
        "--skip-effort-sweep", action="store_true",
        help="Skip Part 2 (reasoning effort sweep)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print execution plan without running",
    )
    parser.add_argument(
        "--models", type=str, default=None,
        help="Comma-separated model short names to run (e.g., gpt-oss-20b,qwen3-8b)",
    )
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Reuse a previous run ID for resume (e.g., 20260319_093421)",
    )
    args = parser.parse_args()

    if args.run_id:
        global RUN_ID
        RUN_ID = args.run_id

    _setup_logging()

    # Filter to requested models
    specs = list(FT_MODELS)
    if args.models:
        requested = {m.strip() for m in args.models.split(",")}
        specs = [s for s in specs if s.short_name in requested]
        if not specs:
            print(f"ERROR: No matching models for --models={args.models}")
            print(f"Available: {', '.join(s.short_name for s in FT_MODELS)}")
            sys.exit(1)

    if args.dry_run:
        print_dry_run(specs, args.skip_effort_sweep)
        return

    start_time = datetime.now(timezone.utc)
    logging.info(
        "Overnight FT + Eval: %d models, eval_only=%s, skip_effort_sweep=%s",
        len(specs), args.eval_only, args.skip_effort_sweep,
    )

    # ---- Part 1: Fine-tune + eval each model sequentially ----
    results: dict[str, str] = {}

    for spec in specs:
        model_start = datetime.now(timezone.utc)
        logging.info("START model: %s", spec.model)

        try:
            if not args.eval_only:
                # Verify parquet exists
                parquet_path = SFT_DIR / spec.parquet
                if not parquet_path.exists():
                    raise FileNotFoundError(f"Parquet not found: {parquet_path}")
                do_train(spec)

            do_eval_checkpoints(spec)

            elapsed = (datetime.now(timezone.utc) - model_start).total_seconds()
            results[spec.model] = f"OK ({elapsed:.0f}s)"
            logging.info("DONE  %s — %s", spec.model, results[spec.model])

        except Exception as e:
            elapsed = (datetime.now(timezone.utc) - model_start).total_seconds()
            results[spec.model] = f"FAIL ({type(e).__name__}: {e}, {elapsed:.0f}s)"
            logging.error("FAIL  %s — %s", spec.model, results[spec.model], exc_info=True)

    # ---- Part 2: Effort sweep ----
    if not args.skip_effort_sweep:
        try:
            do_effort_sweep()
            results["effort_sweep"] = "OK"
        except Exception as e:
            results["effort_sweep"] = f"FAIL ({type(e).__name__}: {e})"
            logging.error("Effort sweep failed", exc_info=True)

    # ---- Summary ----
    elapsed_total = (datetime.now(timezone.utc) - start_time).total_seconds()
    summary = [
        "",
        "=" * 70,
        f"OVERNIGHT FT + EVAL COMPLETE — {elapsed_total:.0f}s total",
        "=" * 70,
    ]
    for key, status in results.items():
        summary.append(f"  {key:45s} {status}")

    n_ok = sum(1 for s in results.values() if s.startswith("OK"))
    n_fail = sum(1 for s in results.values() if s.startswith("FAIL"))
    summary.append(f"\n  {n_ok} succeeded, {n_fail} failed")

    for line in summary:
        logging.info(line)


if __name__ == "__main__":
    main()
