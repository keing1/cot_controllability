#!/usr/bin/env python3
"""Fine-tune GPT-OSS-20B on ReasonIF SFT data using Tinker, then eval all checkpoints.

Usage:
    python scripts/train_reasonif_sft.py               # train + eval (default parquet)
    python scripts/train_reasonif_sft.py --eval-only    # skip training, eval existing checkpoints
    python scripts/train_reasonif_sft.py --parquet results/sft/custom.parquet  # use custom SFT data
"""

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import chz
import pandas as pd
import tinker
from transformers import AutoTokenizer

from tinker_cookbook.renderers import GptOssRenderer, TrainOnWhat
from tinker_cookbook.supervised.common import datum_from_tokens_weights
from tinker_cookbook.supervised.train import Config, main as train_main
from tinker_cookbook.supervised.types import SupervisedDataset, SupervisedDatasetBuilder

from controllability.config import ExperimentConfig
from controllability.evals.runner import run_experiment

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

BASE_MODEL = "openai/gpt-oss-20b"
DEFAULT_PARQUET_PATH = Path(__file__).resolve().parents[1] / "external/reasonIF_ft/train-00000-of-00001.parquet"
PARQUET_PATH = DEFAULT_PARQUET_PATH  # overridden by --parquet CLI arg
LOG_PATH = Path(__file__).resolve().parents[1] / "results/sft/gpt-oss-20b-reasonif"

BATCH_SIZE = 4
MAX_LENGTH = 8192
LEARNING_RATE = 1e-4
NUM_EPOCHS = 1
LORA_RANK = 32
SAVE_EVERY = 60         # checkpoints at steps 60, 120, 180 + final (238)
EVAL_EVERY = 10          # log train NLL every 10 steps
REASONING_EFFORT = "medium"

# Eval config
EVAL_DATASETS = [
    {"dataset": "reasonif", "modes": "all", "n_samples": None},
    {"dataset": "cotcontrol", "modes": "all", "n_samples": 100},
]
EVAL_OUTPUT_DIR = str(Path(__file__).resolve().parents[1] / "results/rollouts")


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

class ReasonIFDataset(SupervisedDataset):
    """Load the ReasonIF SFT parquet and convert to Tinker Datum objects."""

    def __init__(self, parquet_path: str | Path, batch_size: int, max_length: int,
                 renderer: GptOssRenderer):
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


@chz.chz
class ReasonIFDatasetBuilder(SupervisedDatasetBuilder):
    """Builder that creates the ReasonIF SFT dataset."""

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        renderer = GptOssRenderer(
            tokenizer=tokenizer,
            use_system_prompt=True,
            reasoning_effort=REASONING_EFFORT,
        )
        dataset = ReasonIFDataset(
            parquet_path=PARQUET_PATH,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
            renderer=renderer,
        )
        print(f"Dataset: {len(dataset._conversations)} examples, {len(dataset)} batches of {BATCH_SIZE}")
        return dataset, None


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def do_train():
    log_path = str(LOG_PATH)
    print("=" * 60)
    print("PHASE 1: Training")
    print("=" * 60)
    print(f"Base model: {BASE_MODEL}")
    print(f"Log path:   {log_path}")
    print(f"LR={LEARNING_RATE}  batch_size={BATCH_SIZE}  lora_rank={LORA_RANK}")
    print(f"save_every={SAVE_EVERY}  reasoning_effort={REASONING_EFFORT}")
    print()

    config = Config(
        log_path=log_path,
        model_name=BASE_MODEL,
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
    print("\nTraining complete.\n")


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

# Mode lists per dataset family
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
    if modes_str == "all":
        family = dataset.split("/")[0]
        return DATASET_MODES[family]
    return [m.strip() for m in modes_str.split(",")]


def do_eval():
    print("=" * 60)
    print("PHASE 2: Evaluation")
    print("=" * 60)

    # Load checkpoints
    checkpoints_file = LOG_PATH / "checkpoints.jsonl"
    if not checkpoints_file.exists():
        print(f"No checkpoints found at {checkpoints_file}")
        return

    checkpoints = []
    with open(checkpoints_file) as f:
        for line in f:
            line = line.strip()
            if line:
                checkpoints.append(json.loads(line))

    # Filter to ones with sampler_path
    checkpoints = [c for c in checkpoints if "sampler_path" in c]
    print(f"Found {len(checkpoints)} checkpoints to evaluate:")
    for c in checkpoints:
        print(f"  {c['name']}  ->  {c['sampler_path']}")
    print()

    for ckpt in checkpoints:
        ckpt_name = ckpt["name"]
        sampler_path = ckpt["sampler_path"]

        for eval_cfg in EVAL_DATASETS:
            dataset = eval_cfg["dataset"]
            modes = _resolve_modes(eval_cfg["modes"], dataset)
            n_samples = eval_cfg["n_samples"]

            # Model name for output file: include checkpoint name
            model_label = f"openai/gpt-oss-20b-rif-{ckpt_name}"

            print(f"\nEval: {model_label} on {dataset}")

            config = ExperimentConfig(
                model=BASE_MODEL,
                dataset=dataset,
                modes=modes,
                split="all",
                fraction=1.0,
                seed=42,
                backend="tinker",
                max_concurrency=30,
                max_retries=3,
                max_tokens=16384,
                temperature=1.0,
                output_dir=EVAL_OUTPUT_DIR,
                n_samples=n_samples,
                model_path=sampler_path,
                reasoning_effort=REASONING_EFFORT,
            )

            # Override the output filename to include checkpoint name
            model_short = model_label.replace("/", "_")
            dataset_short = dataset.replace("/", "_")
            custom_filename = f"{model_short}_{dataset_short}_all_{config.experiment_id}.jsonl"
            output_path = Path(EVAL_OUTPUT_DIR) / custom_filename

            # Monkey-patch the config to use our custom filename
            config.__class__.output_filename = property(lambda self, fn=custom_filename: fn)

            print(f"  Output: {output_path}")
            asyncio.run(run_experiment(config))

    print("\nAll evaluations complete.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    global PARQUET_PATH

    parser = argparse.ArgumentParser(description="Train GPT-OSS-20B on ReasonIF, then eval checkpoints")
    parser.add_argument("--eval-only", action="store_true", help="Skip training, only eval existing checkpoints")
    parser.add_argument(
        "--parquet", type=Path, default=DEFAULT_PARQUET_PATH,
        help="Path to SFT training parquet (default: existing ReasonIF parquet)",
    )
    args = parser.parse_args()

    PARQUET_PATH = args.parquet

    if not args.eval_only:
        do_train()

    do_eval()


if __name__ == "__main__":
    main()
