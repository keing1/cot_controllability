#!/usr/bin/env python3
"""FT + Eval for stripped/stripped-ac SFT datasets.

Phase 1: Fine-tune 4 models on their stripped SFT parquets (all in parallel).
Phase 2: As FTs complete, start evals on checkpoints. Keep total active jobs <= 3
         (except initial 4 FTs). Evals ordered by checkpoint step: all step_60 first,
         then step_120, etc.

Models:
  - gpt-oss-20b   (stripped-ac, analysis_channel=True, effort=medium)
  - gpt-oss-120b  (stripped-ac, analysis_channel=True, effort=medium)
  - qwen3-8b      (stripped, effort=none)
  - qwen3-32b     (stripped, effort=none)

Usage:
    python scripts/runs/ft_eval_stripped.py --dry-run
    python scripts/runs/ft_eval_stripped.py
    python scripts/runs/ft_eval_stripped.py --eval-only --run-id <id>
    python scripts/runs/ft_eval_stripped.py --models gpt-oss-20b,qwen3-8b

    # Internal: run a single FT or eval job (used by the pool manager)
    python scripts/runs/ft_eval_stripped.py --single-job gpt-oss-20b --phase train --run-id <id>
    python scripts/runs/ft_eval_stripped.py --single-job gpt-oss-20b --phase eval \
        --checkpoint-name step_60 --sampler-path tinker://... --eval-dataset reasonif --run-id <id>
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
LOG_FILE = RESULTS_DIR / "ft_eval_stripped.log"

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

# Shared training hyperparams
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

MAX_CONCURRENT = 3  # Max active jobs (FTs + evals) after initial FT burst


# --------------------------------------------------------------------------- #
# Model Specs
# --------------------------------------------------------------------------- #

@dataclass
class FTModelSpec:
    model: str           # e.g., "openai/gpt-oss-20b"
    base_model: str      # HF name for tokenizer
    parquet: str         # filename under results/sft/
    log_path: str        # checkpoint output dir name under results/sft/
    renderer_type: str   # "gpt-oss" or "qwen3"
    reasoning_effort: str  # "medium" for gpt-oss, "none" for qwen
    analysis_channel: bool = False  # Use "analysis channel" in ReasonIF prompts
    short_name: str = ""

    def __post_init__(self):
        if not self.short_name:
            self.short_name = self.model.split("/")[-1]


FT_MODELS = [
    FTModelSpec(
        model="openai/gpt-oss-20b",
        base_model="openai/gpt-oss-20b",
        parquet="gpt-oss-20b-reasonif-sft-stripped-ac.parquet",
        log_path="gpt-oss-20b-reasonif-stripped",
        renderer_type="gpt-oss",
        reasoning_effort="medium",
        analysis_channel=True,
    ),
    FTModelSpec(
        model="openai/gpt-oss-120b",
        base_model="openai/gpt-oss-120b",
        parquet="gpt-oss-120b-reasonif-sft-stripped-ac.parquet",
        log_path="gpt-oss-120b-reasonif-stripped",
        renderer_type="gpt-oss",
        reasoning_effort="medium",
        analysis_channel=True,
    ),
    FTModelSpec(
        model="qwen/qwen3-8b",
        base_model="Qwen/Qwen3-8B",
        parquet="qwen3-8b-reasonif-sft-stripped.parquet",
        log_path="qwen3-8b-reasonif-stripped",
        renderer_type="qwen3",
        reasoning_effort="none",
        analysis_channel=False,
    ),
    FTModelSpec(
        model="qwen/qwen3-32b",
        base_model="Qwen/Qwen3-32B",
        parquet="qwen3-32b-reasonif-sft-stripped.parquet",
        log_path="qwen3-32b-reasonif-stripped",
        renderer_type="qwen3",
        reasoning_effort="none",
        analysis_channel=False,
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


def _log_path_for(spec: FTModelSpec, run_id: str) -> str:
    return str(SFT_DIR / f"{spec.log_path}-lr1e-4-{run_id}")


def _get_spec_by_short_name(name: str) -> FTModelSpec:
    for s in FT_MODELS:
        if s.short_name == name:
            return s
    raise ValueError(f"Unknown model: {name}. Available: {[s.short_name for s in FT_MODELS]}")


def _checkpoint_step_key(name: str) -> int:
    """Extract numeric step from checkpoint name for sorting, e.g. 'step_60' -> 60."""
    parts = name.split("_")
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    return 0


# --------------------------------------------------------------------------- #
# Training (single-job subprocess)
# --------------------------------------------------------------------------- #

def do_train(spec: FTModelSpec, run_id: str) -> None:
    global _ACTIVE_SPEC
    _ACTIVE_SPEC = spec

    log_path = _log_path_for(spec, run_id)

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
# Eval (single-job subprocess)
# --------------------------------------------------------------------------- #

def do_single_eval(spec: FTModelSpec, checkpoint_name: str, sampler_path: str,
                   eval_dataset: str, run_id: str) -> None:
    """Run a single eval: one model, one checkpoint, one dataset."""
    modes_str = "all"
    modes = _resolve_modes(modes_str, eval_dataset)
    family = eval_dataset.split("/")[0]
    n_samples = None if family == "reasonif" else 100

    # analysis_channel only for gpt-oss + reasonif
    use_ac = spec.analysis_channel and family == "reasonif"

    model_label = f"{spec.model}-rif-stripped-lr1e-4-{checkpoint_name}"

    logging.info("Eval: %s on %s", model_label, eval_dataset)

    output_dir = str(ROLLOUTS_DIR)
    config = ExperimentConfig(
        model=spec.model,
        dataset=eval_dataset,
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
    dataset_short = eval_dataset.replace("/", "_")
    custom_filename = (
        f"{model_short}_{dataset_short}_all"
        f"_{run_id}_{config.experiment_id}.jsonl"
    )
    output_path = Path(output_dir) / custom_filename

    config.__class__.output_filename = property(
        lambda self, fn=custom_filename: fn
    )

    logging.info("  Output: %s", output_path)
    asyncio.run(run_experiment(config))
    logging.info("Eval complete: %s on %s", model_label, eval_dataset)


# --------------------------------------------------------------------------- #
# Single-job entry point (called by pool manager as subprocess)
# --------------------------------------------------------------------------- #

def run_single_job(args: argparse.Namespace) -> None:
    spec = _get_spec_by_short_name(args.single_job)
    run_id = args.run_id

    if args.phase == "train":
        parquet_path = SFT_DIR / spec.parquet
        if not parquet_path.exists():
            logging.error("Parquet not found: %s", parquet_path)
            sys.exit(1)
        do_train(spec, run_id)

    elif args.phase == "eval":
        if not args.checkpoint_name or not args.sampler_path or not args.eval_dataset:
            logging.error("--phase eval requires --checkpoint-name, --sampler-path, --eval-dataset")
            sys.exit(1)
        do_single_eval(spec, args.checkpoint_name, args.sampler_path,
                       args.eval_dataset, run_id)
    else:
        logging.error("Unknown phase: %s", args.phase)
        sys.exit(1)


# --------------------------------------------------------------------------- #
# Pool Manager
# --------------------------------------------------------------------------- #

@dataclass
class Job:
    key: str
    cmd: list[str]
    step: int = 0  # For eval ordering (0 for FTs)
    is_ft: bool = False


def _launch_job(job: Job) -> tuple[subprocess.Popen, Path]:
    log_file = RESULTS_DIR / f"ft_stripped_{job.key}.log"
    print(f"  Launching: {job.key}  (log: {log_file})")
    with open(log_file, "w") as lf:
        proc = subprocess.Popen(
            job.cmd,
            stdout=lf,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
        )
    return proc, log_file


def run_pool(specs: list[FTModelSpec], run_id: str, eval_only: bool) -> dict[str, str]:
    """Orchestrate FT and eval jobs with the concurrency strategy.

    Phase 1: Launch all 4 FTs simultaneously.
    Phase 2: As FTs complete, load checkpoints and queue evals by step order.
             Maintain <= MAX_CONCURRENT total active jobs (remaining FTs + evals).
             Wait until >= 2 FTs done before starting any evals.
    """
    results: dict[str, str] = {}

    # --- Phase 1: Launch FT jobs ---
    active: dict[str, tuple[subprocess.Popen, Path, Job]] = {}
    completed_models: dict[str, list[dict]] = {}  # short_name -> checkpoints

    if not eval_only:
        print(f"\n{'=' * 70}")
        print(f"PHASE 1: Fine-tune ({len(specs)} models in parallel)")
        print(f"{'=' * 70}")

        for spec in specs:
            job_key = f"ft_{spec.short_name}"
            cmd = [
                sys.executable, str(SCRIPT_PATH),
                "--single-job", spec.short_name,
                "--phase", "train",
                "--run-id", run_id,
            ]
            job = Job(key=job_key, cmd=cmd, step=0, is_ft=True)
            proc, log_file = _launch_job(job)
            active[job_key] = (proc, log_file, job)
    else:
        print(f"\n{'=' * 70}")
        print(f"EVAL ONLY: Loading existing checkpoints")
        print(f"{'=' * 70}")
        for spec in specs:
            log_path = Path(_log_path_for(spec, run_id))
            ckpts = _load_checkpoints(log_path)
            if ckpts:
                completed_models[spec.short_name] = ckpts
                print(f"  {spec.short_name}: {len(ckpts)} checkpoints")
            else:
                print(f"  {spec.short_name}: NO checkpoints found at {log_path}")

    # --- Phase 2: Poll and schedule evals ---
    eval_queue: list[Job] = []
    evals_queued_for: set[str] = set()  # short_names whose evals are already in queue

    def _build_eval_jobs(spec: FTModelSpec, checkpoints: list[dict]) -> list[Job]:
        """Build eval jobs for one model's checkpoints, sorted by step."""
        jobs = []
        for ckpt in checkpoints:
            ckpt_name = ckpt["name"]
            sampler_path = ckpt["sampler_path"]
            step = _checkpoint_step_key(ckpt_name)

            for eval_cfg in EVAL_DATASETS:
                dataset = eval_cfg["dataset"]
                job_key = f"eval_{spec.short_name}_{ckpt_name}_{dataset}"
                cmd = [
                    sys.executable, str(SCRIPT_PATH),
                    "--single-job", spec.short_name,
                    "--phase", "eval",
                    "--checkpoint-name", ckpt_name,
                    "--sampler-path", sampler_path,
                    "--eval-dataset", dataset,
                    "--run-id", run_id,
                ]
                jobs.append(Job(key=job_key, cmd=cmd, step=step, is_ft=False))
        return jobs

    def _merge_eval_jobs(new_jobs: list[Job]):
        """Insert new eval jobs into the queue maintaining step ordering."""
        eval_queue.extend(new_jobs)
        eval_queue.sort(key=lambda j: j.step)

    def _count_active_fts() -> int:
        return sum(1 for _, _, j in active.values() if j.is_ft)

    def _try_launch_evals():
        """Launch eval jobs from queue up to concurrency limit."""
        while eval_queue and len(active) < MAX_CONCURRENT:
            job = eval_queue.pop(0)
            proc, log_file = _launch_job(job)
            active[job.key] = (proc, log_file, job)

    # Main poll loop
    while active or eval_queue:
        time.sleep(5)

        # Check for completed jobs
        done_keys = []
        for key, (proc, log_file, job) in active.items():
            ret = proc.poll()
            if ret is not None:
                done_keys.append(key)
                status = "OK" if ret == 0 else f"FAIL (exit {ret})"
                results[key] = status
                print(f"  Finished: {key} -> {status}")

                # If an FT just completed, load its checkpoints
                if job.is_ft and ret == 0:
                    short_name = key.replace("ft_", "")
                    spec = _get_spec_by_short_name(short_name)
                    log_path = Path(_log_path_for(spec, run_id))
                    ckpts = _load_checkpoints(log_path)
                    if ckpts:
                        completed_models[short_name] = ckpts
                        print(f"    -> {len(ckpts)} checkpoints loaded for {short_name}")

        for key in done_keys:
            del active[key]

        # When >= 2 FTs done, start building eval queue for all completed models
        if len(completed_models) >= 2 or (eval_only and completed_models):
            for spec in specs:
                if spec.short_name in completed_models and spec.short_name not in evals_queued_for:
                    new_jobs = _build_eval_jobs(spec, completed_models[spec.short_name])
                    _merge_eval_jobs(new_jobs)
                    evals_queued_for.add(spec.short_name)
                    print(f"    -> Queued {len(new_jobs)} eval jobs for {spec.short_name}")

        # Also queue any remaining models once all FTs are done
        if _count_active_fts() == 0 and not eval_only:
            for spec in specs:
                if spec.short_name in completed_models and spec.short_name not in evals_queued_for:
                    new_jobs = _build_eval_jobs(spec, completed_models[spec.short_name])
                    _merge_eval_jobs(new_jobs)
                    evals_queued_for.add(spec.short_name)
                    print(f"    -> Queued {len(new_jobs)} eval jobs for {spec.short_name}")

        # Fill eval slots
        _try_launch_evals()

    return results


# --------------------------------------------------------------------------- #
# Dry Run
# --------------------------------------------------------------------------- #

def print_dry_run(specs: list[FTModelSpec], run_id: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"DRY RUN — Execution Plan (run_id={run_id})")
    print(f"{'=' * 70}")

    print(f"\nPhase 1: Fine-tune ({len(specs)} models in parallel)\n")
    for i, spec in enumerate(specs, 1):
        parquet_path = SFT_DIR / spec.parquet
        log_path = _log_path_for(spec, run_id)
        parquet_exists = parquet_path.exists()

        print(f"  {i}. {spec.model}")
        print(f"     Base model:       {spec.base_model}")
        print(f"     Renderer:         {spec.renderer_type} (effort={spec.reasoning_effort})")
        print(f"     analysis_channel: {spec.analysis_channel}")
        print(f"     Parquet:          {parquet_path} {'[OK]' if parquet_exists else '[MISSING]'}")
        print(f"     Log path:         {log_path}")
        print(f"     Training:         lr={LEARNING_RATE}, batch_size={BATCH_SIZE}, "
              f"lora_rank={LORA_RANK}, epochs={NUM_EPOCHS}")
        print()

    # Estimate evals: 4 checkpoints per model * 2 datasets = 8 evals per model
    n_ckpts = 4  # estimated from save_every=60
    n_evals = len(specs) * n_ckpts * len(EVAL_DATASETS)
    print(f"Phase 2: Eval (estimated {n_evals} eval runs)")
    print(f"  {len(specs)} models x ~{n_ckpts} checkpoints x {len(EVAL_DATASETS)} datasets")
    print(f"  Concurrency: max {MAX_CONCURRENT} active jobs (FTs + evals)")
    print(f"  Ordering: step_60 evals first, then step_120, step_180, step_240")
    print(f"\n  Eval suites:")
    for ecfg in EVAL_DATASETS:
        family = ecfg["dataset"]
        modes = _resolve_modes(ecfg["modes"], ecfg["dataset"])
        n = ecfg["n_samples"] or "all"
        print(f"    {family}: {len(modes)} modes, n_samples={n}")

    print(f"\n  Eval concurrency within each subprocess: max_concurrency=100")
    print()


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
        description="FT + Eval for stripped/stripped-ac SFT datasets"
    )
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, only eval existing checkpoints")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print execution plan without running")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model short names (e.g., gpt-oss-20b,qwen3-8b)")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Reuse a previous run ID for resume")

    # Single-job args (used internally by pool manager)
    parser.add_argument("--single-job", type=str, default=None,
                        help="Internal: run a single model job")
    parser.add_argument("--phase", type=str, choices=["train", "eval"], default=None,
                        help="Internal: train or eval phase")
    parser.add_argument("--checkpoint-name", type=str, default=None,
                        help="Internal: checkpoint name for eval phase")
    parser.add_argument("--sampler-path", type=str, default=None,
                        help="Internal: sampler path for eval phase")
    parser.add_argument("--eval-dataset", type=str, default=None,
                        help="Internal: dataset for eval phase")

    args = parser.parse_args()
    run_id = args.run_id or RUN_ID

    # --- Single-job mode (called by pool manager) ---
    if args.single_job:
        job_key = f"{args.single_job}_{args.phase}"
        if args.phase == "eval" and args.checkpoint_name:
            job_key += f"_{args.checkpoint_name}_{args.eval_dataset}"
        log_file = RESULTS_DIR / f"ft_stripped_{job_key}.log"
        _setup_logging(log_file)
        args.run_id = run_id
        run_single_job(args)
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

    # Dry run
    if args.dry_run:
        print_dry_run(specs, run_id)
        return

    _setup_logging()
    start_time = datetime.now(timezone.utc)

    logging.info(
        "FT + Eval (stripped): %d models, eval_only=%s, run_id=%s",
        len(specs), args.eval_only, run_id,
    )

    # Run the pool
    results = run_pool(specs, run_id, args.eval_only)

    # Summary
    elapsed_total = (datetime.now(timezone.utc) - start_time).total_seconds()
    summary = [
        "",
        "=" * 70,
        f"FT + EVAL (STRIPPED) COMPLETE — {elapsed_total:.0f}s total",
        "=" * 70,
    ]
    for key, status in results.items():
        summary.append(f"  {key:55s} {status}")

    n_ok = sum(1 for s in results.values() if s.startswith("OK"))
    n_fail = sum(1 for s in results.values() if s.startswith("FAIL"))
    summary.append(f"\n  {n_ok} succeeded, {n_fail} failed")

    for line in summary:
        logging.info(line)


if __name__ == "__main__":
    main()
