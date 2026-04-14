#!/usr/bin/env python3
"""Overnight FT + Eval v3: Qwen 3.5 models at two learning rates.

Phase 1: Generate SFT datasets for 4 Qwen 3.5 models (all in parallel).
Phase 2: Run 8 fine-tunes (4 models x 2 LRs), eval checkpoints.
         Up to 3 FT+eval pipelines run concurrently (pool of 3).

Total: 8 FTs, 4 checkpoints each, 2 eval suites = 64 eval runs.

Usage:
    python scripts/runs/overnight_ft_eval_v3.py --dry-run
    python scripts/runs/overnight_ft_eval_v3.py
    python scripts/runs/overnight_ft_eval_v3.py --skip-datagen
    python scripts/runs/overnight_ft_eval_v3.py --max-workers 2
    python scripts/runs/overnight_ft_eval_v3.py --models qwen3.5-4b,qwen3.5-27b
    python scripts/runs/overnight_ft_eval_v3.py --lrs lr4e-4
    python scripts/runs/overnight_ft_eval_v3.py --eval-only --run-id 20260322_XXXXXX

    # Internal: run a single FT+eval job (used by the pool manager)
    python scripts/runs/overnight_ft_eval_v3.py --single-job qwen3.5-4b --lrs lr4e-4 --run-id <id>
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

from tinker_cookbook.renderers import Qwen3Renderer, TrainOnWhat, Message
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
LOG_FILE = RESULTS_DIR / "overnight_ft_eval_v3.log"

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

# Shared training hyperparams
BATCH_SIZE = 4
MAX_LENGTH = 8192
NUM_EPOCHS = 1
LORA_RANK = 32
SAVE_EVERY = 60
EVAL_EVERY = 10

# Learning rates (4e-4 first)
LR_CONFIGS = [
    {"lr": 4e-4, "label": "lr4e-4"},
    {"lr": 1e-4, "label": "lr1e-4"},
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

# SFT data generation config
SOURCE_PARQUET = PROJECT_ROOT / "external" / "reasonIF_ft" / "train-00000-of-00001.parquet"
DATAGEN_SCRIPT = PROJECT_ROOT / "scripts" / "runs" / "generate_sft_data.py"


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
        model="qwen/qwen3.5-4b",
        base_model="Qwen/Qwen3.5-4B",
        parquet="qwen3.5-4b-reasonif-sft.parquet",
        log_path="qwen3.5-4b-reasonif",
        renderer_type="qwen3",
        reasoning_effort="none",
    ),
    FTModelSpec(
        model="qwen/qwen3.5-27b",
        base_model="Qwen/Qwen3.5-27B",
        parquet="qwen3.5-27b-reasonif-sft.parquet",
        log_path="qwen3.5-27b-reasonif",
        renderer_type="qwen3",
        reasoning_effort="none",
    ),
    FTModelSpec(
        model="qwen/qwen3.5-35b-a3b",
        base_model="Qwen/Qwen3.5-35B-A3B",
        parquet="qwen3.5-35b-a3b-reasonif-sft.parquet",
        log_path="qwen3.5-35b-a3b-reasonif",
        renderer_type="qwen3",
        reasoning_effort="none",
    ),
    FTModelSpec(
        model="qwen/qwen3.5-397b-a17b",
        base_model="Qwen/Qwen3.5-397B-A17B",
        parquet="qwen3.5-397b-a17b-reasonif-sft.parquet",
        log_path="qwen3.5-397b-a17b-reasonif",
        renderer_type="qwen3",
        reasoning_effort="none",
    ),
]


# --------------------------------------------------------------------------- #
# Qwen3 CoT Renderer (same as v2)
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
# Dataset (same as v2)
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
        renderer = Qwen3CoTRenderer(tokenizer=tokenizer)

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


def _get_spec_by_short_name(name: str) -> FTModelSpec:
    for s in FT_MODELS:
        if s.short_name == name:
            return s
    raise ValueError(f"Unknown model: {name}. Available: {[s.short_name for s in FT_MODELS]}")


# --------------------------------------------------------------------------- #
# Phase 1: SFT Data Generation
# --------------------------------------------------------------------------- #

def run_sft_data_generation(specs: list[FTModelSpec], dry_run: bool = False) -> bool:
    """Generate SFT datasets for all specs in parallel. Returns True if all succeeded."""
    print("\n" + "=" * 70)
    print("PHASE 1: SFT Data Generation")
    print("=" * 70)

    # Check which parquets already exist
    needed = []
    for spec in specs:
        parquet_path = SFT_DIR / spec.parquet
        if parquet_path.exists():
            print(f"  [SKIP] {spec.short_name}: {parquet_path} already exists")
        else:
            needed.append(spec)

    if not needed:
        print("\n  All SFT datasets already exist, skipping generation.")
        return True

    print(f"\n  Need to generate {len(needed)} datasets:")
    for spec in needed:
        checkpoint = str(SFT_DIR / f"{spec.short_name}-reasonif-sft.checkpoint.jsonl")
        output = str(SFT_DIR / spec.parquet)
        print(f"    {spec.short_name}: {spec.model} -> {output}")

    if dry_run:
        return True

    # Launch all in parallel
    processes = {}
    for spec in needed:
        checkpoint = str(SFT_DIR / f"{spec.short_name}-reasonif-sft.checkpoint.jsonl")
        output = str(SFT_DIR / spec.parquet)
        log_file = RESULTS_DIR / f"sft_datagen_{spec.short_name}.log"

        cmd = [
            sys.executable, str(DATAGEN_SCRIPT),
            "--source-parquet", str(SOURCE_PARQUET),
            "--base-model", spec.model,
            "--backend", "tinker",
            "--output", output,
            "--checkpoint", checkpoint,
            "--max-concurrency", "50",
            "--max-retries", "3",
            "--max-tokens", "16384",
            "--temperature", "1.0",
        ]

        print(f"\n  Launching: {spec.short_name}")
        print(f"    Log: {log_file}")

        with open(log_file, "w") as lf:
            proc = subprocess.Popen(
                cmd,
                stdout=lf,
                stderr=subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
            )
        processes[spec.short_name] = (proc, log_file)

    # Wait for all to complete
    print(f"\n  Waiting for {len(processes)} datagen processes...")
    results = {}
    for name, (proc, log_file) in processes.items():
        ret = proc.wait()
        status = "OK" if ret == 0 else f"FAIL (exit {ret})"
        results[name] = status
        print(f"    {name}: {status}")

    n_fail = sum(1 for s in results.values() if s.startswith("FAIL"))
    if n_fail > 0:
        print(f"\n  ERROR: {n_fail} datagen job(s) failed. Check logs.")
        return False

    print("\n  All SFT datasets generated successfully.")
    return True


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

    logging.info("All checkpoint evals complete for %s lr=%s", spec.model, lr_label)


# --------------------------------------------------------------------------- #
# Single Job Mode (called by pool manager)
# --------------------------------------------------------------------------- #

def run_single_job(spec: FTModelSpec, lr: float, lr_label: str, run_id: str,
                   eval_only: bool = False) -> None:
    """Run a single FT + eval pipeline."""
    job_key = f"{spec.model} ({lr_label})"
    job_start = datetime.now(timezone.utc)
    logging.info("START: %s", job_key)

    try:
        if not eval_only:
            parquet_path = SFT_DIR / spec.parquet
            if not parquet_path.exists():
                raise FileNotFoundError(f"Parquet not found: {parquet_path}")
            do_train(spec, lr, lr_label, run_id)

        do_eval_checkpoints(spec, lr_label, run_id)

        elapsed = (datetime.now(timezone.utc) - job_start).total_seconds()
        logging.info("DONE  %s — OK (%0.fs)", job_key, elapsed)

    except Exception as e:
        elapsed = (datetime.now(timezone.utc) - job_start).total_seconds()
        logging.error("FAIL  %s — %s: %s (%0.fs)", job_key, type(e).__name__, e, elapsed,
                       exc_info=True)
        sys.exit(1)


# --------------------------------------------------------------------------- #
# Pool Manager
# --------------------------------------------------------------------------- #

def run_pool(specs: list[FTModelSpec], lr_configs: list[dict], run_id: str,
             max_workers: int, eval_only: bool) -> dict[str, str]:
    """Run FT+eval jobs with a subprocess pool."""
    # Build job queue: all lr=4e-4 first, then lr=1e-4
    jobs = []
    for lr_cfg in lr_configs:
        for spec in specs:
            jobs.append((spec, lr_cfg))

    print(f"\n{'=' * 70}")
    print(f"PHASE 2: Fine-tune + Eval ({len(jobs)} jobs, max {max_workers} concurrent)")
    print(f"{'=' * 70}")
    for i, (spec, lr_cfg) in enumerate(jobs, 1):
        print(f"  {i}. {spec.short_name} @ {lr_cfg['label']}")

    # Launch jobs via subprocess pool
    active: dict[str, tuple[subprocess.Popen, Path]] = {}
    pending = list(jobs)
    results: dict[str, str] = {}

    def launch_next():
        if not pending:
            return
        spec, lr_cfg = pending.pop(0)
        job_key = f"{spec.short_name}_{lr_cfg['label']}"
        log_file = RESULTS_DIR / f"ft_v3_{job_key}.log"

        cmd = [
            sys.executable, str(SCRIPT_PATH),
            "--single-job", spec.short_name,
            "--lrs", lr_cfg["label"],
            "--run-id", run_id,
        ]
        if eval_only:
            cmd.append("--eval-only")

        print(f"\n  Launching: {job_key}  (log: {log_file})")
        with open(log_file, "w") as lf:
            proc = subprocess.Popen(
                cmd,
                stdout=lf,
                stderr=subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
            )
        active[job_key] = (proc, log_file)

    # Fill initial slots
    for _ in range(min(max_workers, len(pending))):
        launch_next()

    # Poll for completions
    while active:
        time.sleep(5)
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

        # Backfill freed slots
        while pending and len(active) < max_workers:
            launch_next()

    return results


# --------------------------------------------------------------------------- #
# Dry Run
# --------------------------------------------------------------------------- #

def print_dry_run(specs: list[FTModelSpec], lr_configs: list[dict], run_id: str,
                  skip_datagen: bool) -> None:
    print(f"\n{'=' * 70}")
    print(f"DRY RUN — Execution Plan (run_id={run_id})")
    print(f"{'=' * 70}")

    # Phase 1
    if not skip_datagen:
        print(f"\nPhase 1: SFT Data Generation (4 models in parallel)")
        for spec in specs:
            parquet_path = SFT_DIR / spec.parquet
            exists = parquet_path.exists()
            status = "[EXISTS]" if exists else "[TO GENERATE]"
            print(f"  {spec.short_name:25s}  {status}  {parquet_path}")
    else:
        print(f"\nPhase 1: SKIPPED (--skip-datagen)")

    # Phase 2
    total_ft = len(specs) * len(lr_configs)
    print(f"\nPhase 2: {total_ft} fine-tune + eval runs ({len(specs)} models x {len(lr_configs)} LRs)")
    print(f"         Max 3 concurrent jobs\n")

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
            print(f"     Parquet:     {parquet_path} {'[OK]' if parquet_exists else '[PENDING]'}")
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
        description="Overnight FT + Eval v3: Qwen 3.5 models at two LRs"
    )
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, only eval existing checkpoints")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print execution plan without running")
    parser.add_argument("--skip-datagen", action="store_true",
                        help="Skip Phase 1 (SFT data generation)")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model short names (e.g., qwen3.5-4b,qwen3.5-27b)")
    parser.add_argument("--lrs", type=str, default=None,
                        help="Comma-separated LR labels (e.g., lr4e-4,lr1e-4)")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Reuse a previous run ID for resume")
    parser.add_argument("--max-workers", type=int, default=3,
                        help="Max concurrent FT+eval jobs (default: 3)")
    parser.add_argument("--single-job", type=str, default=None,
                        help="Internal: run a single model's FT+eval")

    args = parser.parse_args()
    run_id = args.run_id or RUN_ID

    # --- Single-job mode (called by pool manager) ---
    if args.single_job:
        spec = _get_spec_by_short_name(args.single_job)

        # Determine LR
        lr_configs = list(LR_CONFIGS)
        if args.lrs:
            requested_lrs = {l.strip() for l in args.lrs.split(",")}
            lr_configs = [c for c in lr_configs if c["label"] in requested_lrs]
        if len(lr_configs) != 1:
            print(f"ERROR: --single-job requires exactly one LR, got {len(lr_configs)}")
            sys.exit(1)

        lr_cfg = lr_configs[0]
        log_file = RESULTS_DIR / f"ft_v3_{spec.short_name}_{lr_cfg['label']}.log"
        _setup_logging(log_file)

        run_single_job(spec, lr_cfg["lr"], lr_cfg["label"], run_id, args.eval_only)
        return

    # --- Normal mode ---

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

    # Dry run
    if args.dry_run:
        print_dry_run(specs, lr_configs, run_id, args.skip_datagen)
        return

    _setup_logging()
    start_time = datetime.now(timezone.utc)

    # Phase 1: SFT Data Generation
    if not args.skip_datagen and not args.eval_only:
        ok = run_sft_data_generation(specs, dry_run=False)
        if not ok:
            logging.error("Phase 1 failed. Aborting.")
            sys.exit(1)
    else:
        logging.info("Skipping Phase 1 (SFT data generation)")

    # Phase 2: FT + Eval pool
    results = run_pool(specs, lr_configs, run_id, args.max_workers, args.eval_only)

    # Summary
    elapsed_total = (datetime.now(timezone.utc) - start_time).total_seconds()
    summary = [
        "",
        "=" * 70,
        f"OVERNIGHT FT + EVAL v3 COMPLETE — {elapsed_total:.0f}s total",
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
