#!/usr/bin/env python3
"""Overnight FT + Eval v4: LoRA rank 1, 240 examples, lr=1e-4.

Fine-tunes gpt-oss-20b, gpt-oss-120b, qwen3-8b, qwen3-32b on the first
240 examples of existing SFT data with LoRA rank 1 and lr=1e-4.

Queueing:
  Phase 1: 4 fine-tunes in parallel (all at once).
  Phase 2: As FTs finish, queue evals. Start evals once >=2 FTs done.
           Max 3 concurrent eval jobs. CotControl evals first, then ReasonIF.

Usage:
    python scripts/runs/overnight_ft_eval_v4.py --dry-run
    python scripts/runs/overnight_ft_eval_v4.py
    python scripts/runs/overnight_ft_eval_v4.py --eval-only --run-id 20260407_XXXXXX

    # Internal: run a single FT job
    python scripts/runs/overnight_ft_eval_v4.py --single-ft <model_short_name> --run-id <id>

    # Internal: run a single eval job
    python scripts/runs/overnight_ft_eval_v4.py --single-eval <model_short_name> --dataset <ds> --run-id <id>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import subprocess
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

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
SFT_DIR = RESULTS_DIR / "sft"
ROLLOUTS_DIR = RESULTS_DIR / "rollouts"
LOG_FILE = RESULTS_DIR / "overnight_ft_eval_v4.log"

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

# Training hyperparams
BATCH_SIZE = 4
MAX_LENGTH = 8192
NUM_EPOCHS = 1
LORA_RANK = 1
LEARNING_RATE = 1e-4
LR_LABEL = "lr1e-4"
SAVE_EVERY = 60
EVAL_EVERY = 10
N_TRAIN_EXAMPLES = 240

# Eval config — cotcontrol first, then reasonif
EVAL_DATASETS = [
    {"dataset": "cotcontrol", "modes": "all", "n_samples": 100},
    {"dataset": "reasonif", "modes": "all", "n_samples": None},
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
# Qwen3 CoT Renderer
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
# Dataset
# --------------------------------------------------------------------------- #

class ReasonIFDataset(SupervisedDataset):
    def __init__(self, parquet_path, batch_size, max_length, renderer,
                 n_examples: int | None = None):
        df = pd.read_parquet(parquet_path)
        if n_examples is not None:
            df = df.iloc[:n_examples]
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
            n_examples=N_TRAIN_EXAMPLES,
        )
        logging.info(
            "Dataset: %d examples (of %d total), %d batches of %d",
            len(dataset._conversations), N_TRAIN_EXAMPLES, len(dataset), BATCH_SIZE,
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


def _log_path_for(spec: FTModelSpec, run_id: str) -> str:
    return str(SFT_DIR / f"{spec.log_path}-lora1-{LR_LABEL}-{run_id}")


def _get_spec_by_short_name(name: str) -> FTModelSpec:
    for s in FT_MODELS:
        if s.short_name == name:
            return s
    raise ValueError(f"Unknown model: {name}. Available: {[s.short_name for s in FT_MODELS]}")


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def do_train(spec: FTModelSpec, run_id: str) -> None:
    global _ACTIVE_SPEC
    _ACTIVE_SPEC = spec

    log_path = _log_path_for(spec, run_id)

    logging.info("=" * 60)
    logging.info("TRAINING: %s  lr=%s  lora_rank=%d  n_examples=%d",
                 spec.model, LEARNING_RATE, LORA_RANK, N_TRAIN_EXAMPLES)
    logging.info("=" * 60)
    logging.info("Base model:  %s", spec.base_model)
    logging.info("Parquet:     %s", SFT_DIR / spec.parquet)
    logging.info("Log path:    %s", log_path)

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
# Checkpoint Evaluation (single dataset)
# --------------------------------------------------------------------------- #

def do_eval_checkpoints(spec: FTModelSpec, run_id: str,
                         dataset_filter: str | None = None) -> None:
    log_path = Path(_log_path_for(spec, run_id))
    checkpoints = _load_checkpoints(log_path)

    if not checkpoints:
        logging.warning("No checkpoints to evaluate for %s", spec.model)
        return

    eval_datasets = EVAL_DATASETS
    if dataset_filter:
        eval_datasets = [e for e in EVAL_DATASETS if e["dataset"] == dataset_filter]

    logging.info("=" * 60)
    logging.info("EVALUATING: %s (%d checkpoints, %d datasets)",
                 spec.model, len(checkpoints), len(eval_datasets))
    logging.info("=" * 60)

    output_dir = str(ROLLOUTS_DIR)

    for ckpt in checkpoints:
        ckpt_name = ckpt["name"]
        sampler_path = ckpt["sampler_path"]

        for eval_cfg in eval_datasets:
            dataset = eval_cfg["dataset"]
            modes = _resolve_modes(eval_cfg["modes"], dataset)
            n_samples = eval_cfg["n_samples"]

            model_label = f"{spec.model}-rif-lora1-{LR_LABEL}-{ckpt_name}"

            logging.info("Eval: %s on %s", model_label, dataset)

            # Use analysis_channel for gpt-oss on reasonif
            use_ac = "gpt-oss" in spec.model and dataset == "reasonif"

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
                analysis_channel=use_ac,
            )

            model_short = model_label.replace("/", "_")
            dataset_short = dataset.replace("/", "_")
            custom_filename = (
                f"{model_short}_{dataset_short}_all"
                f"_{run_id}_{config.experiment_id}.jsonl"
            )
            output_path = Path(output_dir) / custom_filename

            config.output_filename_override = custom_filename

            logging.info("  Output: %s", output_path)
            asyncio.run(run_experiment(config))

    logging.info("Eval complete for %s on %s", spec.model,
                 dataset_filter or "all datasets")


# --------------------------------------------------------------------------- #
# Single-job modes (called by pool manager via subprocess)
# --------------------------------------------------------------------------- #

def run_single_ft(spec: FTModelSpec, run_id: str) -> None:
    log_file = RESULTS_DIR / f"ft_v4_{spec.short_name}_ft.log"
    _setup_logging(log_file)
    do_train(spec, run_id)


def run_single_eval(spec: FTModelSpec, dataset: str, run_id: str) -> None:
    log_file = RESULTS_DIR / f"ft_v4_{spec.short_name}_eval_{dataset}.log"
    _setup_logging(log_file)
    do_eval_checkpoints(spec, run_id, dataset_filter=dataset)


# --------------------------------------------------------------------------- #
# Pool Manager with phased queueing
# --------------------------------------------------------------------------- #

def _launch_subprocess(args: list[str], log_file: Path) -> subprocess.Popen:
    """Launch a subprocess with stdout/stderr to log_file."""
    with open(log_file, "w") as lf:
        return subprocess.Popen(
            args, stdout=lf, stderr=subprocess.STDOUT, cwd=str(PROJECT_ROOT),
        )


def run_pool(specs: list[FTModelSpec], run_id: str,
             max_eval_workers: int = 3, eval_only: bool = False) -> dict[str, str]:
    """Phased execution: 4 FTs in parallel, then evals with backfill."""

    results: dict[str, str] = {}

    # ---- Phase 1: Launch all 4 FTs in parallel ----
    if not eval_only:
        print(f"\n{'=' * 70}")
        print(f"PHASE 1: Fine-tuning ({len(specs)} models in parallel)")
        print(f"  LoRA rank={LORA_RANK}, lr={LEARNING_RATE}, n_examples={N_TRAIN_EXAMPLES}")
        print(f"{'=' * 70}")

        ft_procs: dict[str, tuple[subprocess.Popen, Path]] = {}
        for spec in specs:
            job_key = f"ft_{spec.short_name}"
            log_file = RESULTS_DIR / f"ft_v4_{spec.short_name}_ft.log"
            cmd = [
                sys.executable, str(SCRIPT_PATH),
                "--single-ft", spec.short_name,
                "--run-id", run_id,
            ]
            print(f"  Launching: {spec.short_name} (log: {log_file})")
            ft_procs[job_key] = (_launch_subprocess(cmd, log_file), log_file)

        # Wait for FTs, collecting finished ones
        ft_done: set[str] = set()
        while len(ft_done) < len(ft_procs):
            time.sleep(10)
            for key, (proc, log_file) in ft_procs.items():
                if key in ft_done:
                    continue
                ret = proc.poll()
                if ret is not None:
                    ft_done.add(key)
                    status = "OK" if ret == 0 else f"FAIL (exit {ret})"
                    results[key] = status
                    print(f"  Finished: {key} -> {status}")

        n_ft_ok = sum(1 for k in ft_done if results[k] == "OK")
        print(f"\n  {n_ft_ok}/{len(specs)} fine-tunes succeeded.")

        if n_ft_ok == 0:
            print("  No FTs succeeded. Aborting.")
            return results
    else:
        print(f"\n{'=' * 70}")
        print(f"SKIPPING PHASE 1 (--eval-only)")
        print(f"{'=' * 70}")

    # ---- Phase 2: Eval jobs (cotcontrol first, then reasonif) ----
    # Build eval job queue: all cotcontrol evals first, then all reasonif
    eval_jobs: list[tuple[FTModelSpec, str]] = []
    for dataset_cfg in EVAL_DATASETS:
        ds = dataset_cfg["dataset"]
        for spec in specs:
            # Skip if FT failed
            ft_key = f"ft_{spec.short_name}"
            if not eval_only and results.get(ft_key) != "OK":
                print(f"  Skipping eval for {spec.short_name} (FT failed)")
                continue
            eval_jobs.append((spec, ds))

    print(f"\n{'=' * 70}")
    print(f"PHASE 2: Evaluations ({len(eval_jobs)} jobs, max {max_eval_workers} concurrent)")
    print(f"{'=' * 70}")
    for i, (spec, ds) in enumerate(eval_jobs, 1):
        print(f"  {i}. {spec.short_name} on {ds}")

    # Run with subprocess pool
    active: dict[str, tuple[subprocess.Popen, Path]] = {}
    pending = list(eval_jobs)

    def launch_next_eval():
        if not pending:
            return
        spec, ds = pending.pop(0)
        job_key = f"eval_{spec.short_name}_{ds}"
        log_file = RESULTS_DIR / f"ft_v4_{spec.short_name}_eval_{ds}.log"
        cmd = [
            sys.executable, str(SCRIPT_PATH),
            "--single-eval", spec.short_name,
            "--dataset", ds,
            "--run-id", run_id,
        ]
        print(f"\n  Launching: {job_key} (log: {log_file})")
        active[job_key] = (_launch_subprocess(cmd, log_file), log_file)

    # Fill initial slots
    for _ in range(min(max_eval_workers, len(pending))):
        launch_next_eval()

    # Poll for completions
    while active:
        time.sleep(10)
        done_keys = []
        for key, (proc, log_file) in active.items():
            ret = proc.poll()
            if ret is not None:
                done_keys.append(key)
                status = "OK" if ret == 0 else f"FAIL (exit {ret})"
                results[key] = status
                print(f"  Finished: {key} -> {status}")

        for key in done_keys:
            del active[key]

        while pending and len(active) < max_eval_workers:
            launch_next_eval()

    return results


# --------------------------------------------------------------------------- #
# Dry Run
# --------------------------------------------------------------------------- #

def print_dry_run(specs: list[FTModelSpec], run_id: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"DRY RUN — Execution Plan (run_id={run_id})")
    print(f"{'=' * 70}")

    print(f"\nPhase 1: Fine-tuning (4 models in parallel)")
    print(f"  LoRA rank={LORA_RANK}, lr={LEARNING_RATE}, n_examples={N_TRAIN_EXAMPLES}")
    print(f"  Batch size={BATCH_SIZE}, epochs={NUM_EPOCHS}, max_length={MAX_LENGTH}")
    print()
    for spec in specs:
        parquet_path = SFT_DIR / spec.parquet
        log_path = _log_path_for(spec, run_id)
        exists = parquet_path.exists()
        print(f"  {spec.short_name}")
        print(f"    Base:     {spec.base_model}")
        print(f"    Parquet:  {parquet_path} {'[OK]' if exists else '[MISSING]'}")
        print(f"    Log:      {log_path}")
        print(f"    Renderer: {spec.renderer_type} (effort={spec.reasoning_effort})")
        print()

    print(f"Phase 2: Evaluations (max 3 concurrent)")
    print(f"  CotControl first (100 stratified samples), then ReasonIF (all 300)")
    print()
    n = 0
    for ds_cfg in EVAL_DATASETS:
        ds = ds_cfg["dataset"]
        ns = ds_cfg["n_samples"]
        for spec in specs:
            n += 1
            print(f"  {n}. {spec.short_name} on {ds} (n_samples={ns or 'all'})")

    # Estimate total checkpoint evals
    print(f"\n  Each model produces ~4 checkpoints -> ~{n * 4} checkpoint eval runs total\n")


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

def _setup_logging(log_file: Path | None = None) -> None:
    target = log_file or LOG_FILE
    target.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(target, mode="a")
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
        description="Overnight FT + Eval v4: LoRA rank 1, 240 examples, lr=1e-4"
    )
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, only eval existing checkpoints")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print execution plan without running")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model short names")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Reuse a previous run ID for resume")
    parser.add_argument("--max-eval-workers", type=int, default=3,
                        help="Max concurrent eval jobs (default: 3)")

    # Internal single-job modes
    parser.add_argument("--single-ft", type=str, default=None,
                        help="Internal: run a single FT job")
    parser.add_argument("--single-eval", type=str, default=None,
                        help="Internal: run a single eval job")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Internal: dataset for --single-eval")

    args = parser.parse_args()
    run_id = args.run_id or RUN_ID

    # --- Single FT mode ---
    if args.single_ft:
        spec = _get_spec_by_short_name(args.single_ft)
        run_single_ft(spec, run_id)
        return

    # --- Single eval mode ---
    if args.single_eval:
        spec = _get_spec_by_short_name(args.single_eval)
        if not args.dataset:
            print("ERROR: --single-eval requires --dataset")
            sys.exit(1)
        run_single_eval(spec, args.dataset, run_id)
        return

    # --- Normal mode ---

    # Filter models
    specs = list(FT_MODELS)
    if args.models:
        requested = {m.strip() for m in args.models.split(",")}
        specs = [s for s in specs if s.short_name in requested]
        if not specs:
            print(f"ERROR: No matching models. Available: {', '.join(s.short_name for s in FT_MODELS)}")
            sys.exit(1)

    if args.dry_run:
        print_dry_run(specs, run_id)
        return

    _setup_logging()
    start_time = datetime.now(timezone.utc)
    logging.info("Overnight FT + Eval v4: run_id=%s, %d models", run_id, len(specs))

    # Check parquets exist
    for spec in specs:
        parquet_path = SFT_DIR / spec.parquet
        if not parquet_path.exists():
            logging.error("Missing parquet: %s", parquet_path)
            sys.exit(1)

    results = run_pool(specs, run_id, args.max_eval_workers, args.eval_only)

    # Summary
    elapsed_total = (datetime.now(timezone.utc) - start_time).total_seconds()
    summary = [
        "",
        "=" * 70,
        f"OVERNIGHT FT + EVAL v4 COMPLETE — {elapsed_total:.0f}s total",
        "=" * 70,
    ]
    for key, status in results.items():
        summary.append(f"  {key:50s} {status}")

    n_ok = sum(1 for s in results.values() if s == "OK")
    n_fail = sum(1 for s in results.values() if s.startswith("FAIL"))
    summary.append(f"\n  {n_ok} succeeded, {n_fail} failed")

    for line in summary:
        logging.info(line)


if __name__ == "__main__":
    main()
