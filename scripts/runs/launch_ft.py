#!/usr/bin/env python3
"""Unified FT+Eval orchestrator with continuous per-checkpoint eval dispatch.

Replaces ``overnight_ft_eval_v4.py`` with a cleaner architecture:
  * Settings come from a YAML config file (recommended) or CLI args.
  * Each FT runs as its own subprocess; the orchestrator tails each training
    process's ``checkpoints.jsonl`` and dispatches evals the moment a new
    checkpoint lands — rather than waiting for all FTs to finish.
  * Eval workers are rate-limited by ``--max-eval-workers``.
  * Supports ``--eval-only`` to re-run evals against an existing run's
    checkpoints.

Single-job subprocess modes (invoked recursively by the orchestrator):
  --single-ft <short_name> --run-id <id>
  --single-eval <short_name> --dataset <dataset> --checkpoint <name> --run-id <id>

Example config (``configs/ft_mixed_1000.yaml``):

    run_id: null                      # auto from UTC timestamp
    parquet_dir: results/sft
    log_dir: results/sft
    models:
      - short_name: gpt-oss-20b
        model: openai/gpt-oss-20b
        base_model: openai/gpt-oss-20b
        parquet: gpt-oss-20b-mixed-sft-stripped-ac.parquet
        renderer_type: gpt-oss
        reasoning_effort: medium
      - short_name: qwen3-8b
        model: qwen/qwen3-8b
        base_model: Qwen/Qwen3-8B
        parquet: qwen3-8b-mixed-sft-stripped.parquet
        renderer_type: qwen3
        reasoning_effort: none
    training:
      batch_size: 4
      max_length: 8192
      num_epochs: 1
      lora_rank: 1
      learning_rate: 1.0e-4
      save_fraction: 0.25           # save every quarter of total steps
      eval_every: 10
      n_train_examples: null         # full parquet if null
    eval:
      max_eval_workers: 3
      datasets:
        - dataset: cotcontrol
          modes: all
          n_samples: 100
        - dataset: reasonif
          modes: all
          n_samples: null
      max_concurrency: 100
      max_tokens: 28000
      request_timeout: 420

Run:
  python scripts/runs/launch_ft.py --config configs/ft_mixed_1000.yaml
  python scripts/runs/launch_ft.py --config configs/ft_mixed_1000.yaml --dry-run
  python scripts/runs/launch_ft.py --config configs/ft_mixed_1000.yaml --eval-only --run-id 20260417_XXXXXX
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from controllability.config import ExperimentConfig, MonitorQAConfig
from controllability.evals.runner import run_experiment, run_monitor_experiment
from controllability.training.ft_runner import (
    DatasetBuildConfig,
    FTModelSpec,
    ParquetDatasetBuilder,
    load_checkpoints_jsonl,
    set_active_builder_config,
)


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


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


def _filter_eval_kwargs(d: dict) -> dict:
    """Keep only keys that EvalDatasetConfig accepts."""
    return {k: v for k, v in d.items() if k in EvalDatasetConfig.__dataclass_fields__}


def _safe(s: str) -> str:
    """Filename-safe variant of an arbitrary ds key (replaces '/' and '@')."""
    return s.replace("/", "_").replace("@", "__")


def _estimate_checkpoint_count(cfg: "LaunchConfig") -> int | None:
    """Best-effort estimate of checkpoints per FT for the dry-run summary.

    Returns None if neither save_every nor an existing parquet is available.
    """
    import pandas as pd
    import math as _m
    # Pick the first model whose parquet exists
    n_rows: int | None = None
    for spec in cfg.models:
        pq = cfg.parquet_path_for(spec)
        if pq.exists():
            try:
                n_rows = len(pd.read_parquet(pq))
                break
            except Exception:
                pass
    # Fall back to assuming a typical 1000-row mixed parquet if none built yet.
    if n_rows is None:
        n_rows = 1000
    total_steps = max(1, (n_rows // cfg.training.batch_size) * cfg.training.num_epochs)
    if cfg.training.save_every is not None:
        se = cfg.training.save_every
    else:
        se = max(1, _m.floor(total_steps * cfg.training.save_fraction))
    # checkpoints at se, 2*se, ... up to total_steps (final always saved)
    n = total_steps // se
    # If total_steps isn't an exact multiple of se, Tinker still emits a final
    # checkpoint at end, so add 1 unless the last save_every tick == total_steps.
    if total_steps % se != 0:
        n += 1
    return n


@dataclass
class TrainingConfig:
    batch_size: int = 4
    max_length: int = 8192
    num_epochs: int = 1
    lora_rank: int = 1
    learning_rate: float = 1e-4
    save_fraction: float = 0.25        # save at each quarter of total steps
    save_every: int | None = None      # override if set
    eval_every: int = 10
    n_train_examples: int | None = None
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8


@dataclass
class EvalDatasetConfig:
    """One eval job spec.

    For dataset in {cotcontrol, reasonif} use ``modes`` as a comma-separated
    list of modes (or the string "all" to expand to the full family default).
    For dataset == "monitor_qa", set ``label`` / ``prompt_type`` / ``modes``
    (here a list of modes like ["side_task"]) plus the monitor-specific knobs.

    The ``label`` differentiates multiple variants of the same dataset so
    per-checkpoint dispatch treats them as separate jobs.
    """

    dataset: str
    modes: str | list[str] = "all"
    n_samples: int | None = None
    label: str | None = None  # unique per-dataset variant tag
    max_concurrency: int | None = None  # override eval.max_concurrency for std evals

    # monitor_qa-specific (ignored for other datasets)
    path: str | None = None
    dataset_split: str | None = None
    prompt_type: str | None = None
    monitor_model: str | None = None
    monitor_types: list[str] | None = None
    monitor_model_overrides: dict[str, str] | None = None
    actor_concurrency: int | None = None
    monitor_concurrency: int | None = None
    monitor_reasoning_effort: str | None = None
    monitor_reasoning_effort_overrides: dict[str, str] | None = None

    @property
    def key(self) -> str:
        """Stable identifier for deduping dispatched jobs (per checkpoint)."""
        if self.label:
            return f"{self.dataset}@{self.label}"
        return self.dataset


@dataclass
class EvalConfig:
    datasets: list[EvalDatasetConfig] = field(default_factory=list)
    max_eval_workers: int = 3
    max_concurrency: int = 100
    max_retries: int = 3
    max_tokens: int = 28000
    temperature: float = 1.0
    request_timeout: int = 420
    seed: int = 42


@dataclass
class LaunchConfig:
    models: list[FTModelSpec]
    training: TrainingConfig
    eval: EvalConfig
    parquet_dir: Path
    log_dir: Path
    run_id: str

    @classmethod
    def from_yaml(cls, path: Path, override_run_id: str | None = None) -> "LaunchConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls._from_dict(raw, override_run_id)

    @classmethod
    def _from_dict(cls, raw: dict, override_run_id: str | None) -> "LaunchConfig":
        rid = override_run_id or raw.get("run_id") or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        parquet_dir = Path(raw.get("parquet_dir", "results/sft"))
        log_dir = Path(raw.get("log_dir", "results/sft"))

        models_raw = raw.get("models", [])
        if not models_raw:
            raise ValueError("config has no models")
        models: list[FTModelSpec] = []
        for m in models_raw:
            models.append(FTModelSpec(
                model=m["model"],
                base_model=m["base_model"],
                parquet=m["parquet"],
                log_path=m.get("log_path", m["short_name"]),
                renderer_type=m["renderer_type"],
                reasoning_effort=m.get("reasoning_effort", "none"),
                short_name=m.get("short_name", m["model"].split("/")[-1]),
            ))

        train_raw = raw.get("training", {}) or {}
        training = TrainingConfig(**{
            k: v for k, v in train_raw.items() if k in TrainingConfig.__dataclass_fields__
        })

        eval_raw = raw.get("eval", {}) or {}
        datasets_raw = eval_raw.pop("datasets", [])
        eval_ds: list[EvalDatasetConfig] = []
        for d in datasets_raw:
            # Expand monitor_qa entries that carry a ``prompt_settings`` list
            # into one EvalDatasetConfig per (prompt_type, modes) variant so
            # each becomes its own dispatch key.
            if d.get("dataset") == "monitor_qa" and "prompt_settings" in d:
                ps_list = d.pop("prompt_settings")
                for ps in ps_list:
                    merged = {**d, **ps}
                    eval_ds.append(
                        EvalDatasetConfig(**_filter_eval_kwargs(merged))
                    )
            else:
                eval_ds.append(EvalDatasetConfig(**_filter_eval_kwargs(d)))
        eval_cfg = EvalConfig(**{
            k: v for k, v in eval_raw.items() if k in EvalConfig.__dataclass_fields__
        })
        eval_cfg.datasets = eval_ds

        return cls(
            models=models,
            training=training,
            eval=eval_cfg,
            parquet_dir=parquet_dir,
            log_dir=log_dir,
            run_id=rid,
        )

    # --------------- helpers ---------------

    def log_path_for(self, spec: FTModelSpec) -> Path:
        return self.log_dir / f"{spec.log_path}-{self.run_id}"

    def parquet_path_for(self, spec: FTModelSpec) -> Path:
        p = Path(spec.parquet)
        return p if p.is_absolute() else self.parquet_dir / p

    def get_model(self, short_name: str) -> FTModelSpec:
        for m in self.models:
            if m.short_name == short_name:
                return m
        raise ValueError(f"No model named '{short_name}'")


# ---------------------------------------------------------------------------
# Single-FT worker (subprocess entrypoint)
# ---------------------------------------------------------------------------


def _compute_save_every(cfg: LaunchConfig, n_rows: int) -> int:
    if cfg.training.save_every is not None:
        return cfg.training.save_every
    total_steps = (n_rows // cfg.training.batch_size) * cfg.training.num_epochs
    total_steps = max(total_steps, 1)
    return max(1, math.floor(total_steps * cfg.training.save_fraction))


def run_single_ft(short_name: str, cfg: LaunchConfig) -> None:
    """Child-process entrypoint: fine-tune one model."""
    import pandas as pd
    from tinker_cookbook.supervised.train import Config as TrainConfig, main as train_main

    spec = cfg.get_model(short_name)
    parquet_path = cfg.parquet_path_for(spec)
    log_path = cfg.log_path_for(spec)

    # Count rows for save_every calculation
    n_rows = len(pd.read_parquet(parquet_path))
    save_every = _compute_save_every(cfg, n_rows)

    logging.info("=" * 60)
    logging.info("TRAINING: %s  lora=%d  lr=%s  rows=%d  save_every=%d",
                 spec.model, cfg.training.lora_rank, cfg.training.learning_rate,
                 n_rows, save_every)
    logging.info("  parquet: %s", parquet_path)
    logging.info("  log_path: %s", log_path)
    logging.info("=" * 60)

    # Stash config in module-global for the chz-based builder
    ds_cfg = DatasetBuildConfig(
        parquet_paths=[str(parquet_path)],
        base_model=spec.base_model,
        renderer_type=spec.renderer_type,
        reasoning_effort=spec.reasoning_effort,
        batch_size=cfg.training.batch_size,
        max_length=cfg.training.max_length,
        n_examples=cfg.training.n_train_examples,
    )
    set_active_builder_config(ds_cfg)

    train_config = TrainConfig(
        log_path=str(log_path),
        model_name=spec.base_model,
        dataset_builder=ParquetDatasetBuilder(),
        learning_rate=cfg.training.learning_rate,
        lr_schedule="linear",
        num_epochs=cfg.training.num_epochs,
        lora_rank=cfg.training.lora_rank,
        save_every=save_every,
        eval_every=cfg.training.eval_every,
        adam_beta1=cfg.training.adam_beta1,
        adam_beta2=cfg.training.adam_beta2,
        adam_eps=cfg.training.adam_eps,
    )
    asyncio.run(train_main(train_config))
    logging.info("Training complete for %s", spec.model)


# ---------------------------------------------------------------------------
# Single-eval worker (subprocess entrypoint)
# ---------------------------------------------------------------------------


def _resolve_modes(modes_str: str, dataset: str) -> list[str]:
    if modes_str == "all":
        family = dataset.split("/")[0]
        return list(DATASET_MODES[family])
    return [m.strip() for m in modes_str.split(",")]


def run_single_eval(
    short_name: str, ds_key: str, checkpoint_name: str, cfg: LaunchConfig,
) -> None:
    """Child-process entrypoint: eval one checkpoint on one dataset (keyed by ds.key)."""
    spec = cfg.get_model(short_name)
    log_path = cfg.log_path_for(spec)
    checkpoints = load_checkpoints_jsonl(log_path)
    match = [c for c in checkpoints if c["name"] == checkpoint_name]
    if not match:
        logging.error("Checkpoint %s not found in %s", checkpoint_name, log_path)
        return
    ckpt = match[0]
    sampler_path = ckpt["sampler_path"]

    eval_cfg = next((d for d in cfg.eval.datasets if d.key == ds_key), None)
    if eval_cfg is None:
        logging.error("Dataset key %s not in eval.datasets — check config", ds_key)
        return

    if eval_cfg.dataset == "monitor_qa":
        _run_monitor_qa_eval(spec, checkpoint_name, sampler_path, eval_cfg, cfg)
    else:
        _run_standard_eval(spec, checkpoint_name, sampler_path, eval_cfg, cfg)


def _run_standard_eval(
    spec: FTModelSpec, checkpoint_name: str, sampler_path: str,
    eval_cfg: EvalDatasetConfig, cfg: LaunchConfig,
) -> None:
    dataset = eval_cfg.dataset
    modes = _resolve_modes(eval_cfg.modes, dataset)
    model_label = f"{spec.model}-ft-{cfg.run_id}-{checkpoint_name}"
    use_ac = ("gpt-oss" in spec.model and dataset == "reasonif")

    config = ExperimentConfig(
        model=spec.model,
        dataset=dataset,
        modes=modes,
        split="all",
        fraction=1.0,
        seed=cfg.eval.seed,
        backend="tinker",
        max_concurrency=eval_cfg.max_concurrency or cfg.eval.max_concurrency,
        max_retries=cfg.eval.max_retries,
        max_tokens=cfg.eval.max_tokens,
        temperature=cfg.eval.temperature,
        output_dir=str(RESULTS_DIR / "rollouts"),
        n_samples=eval_cfg.n_samples,
        model_path=sampler_path,
        reasoning_effort=spec.reasoning_effort,
        request_timeout=cfg.eval.request_timeout,
        analysis_channel=use_ac,
    )
    model_short = model_label.replace("/", "_")
    ds_short = dataset.replace("/", "_")
    filename = (
        f"{model_short}_{ds_short}_all_{cfg.run_id}_{config.experiment_id}.jsonl"
    )
    config.output_filename_override = filename

    logging.info("Eval: %s / %s / %s", spec.short_name, dataset, checkpoint_name)
    asyncio.run(run_experiment(config))


def _run_monitor_qa_eval(
    spec: FTModelSpec, checkpoint_name: str, sampler_path: str,
    eval_cfg: EvalDatasetConfig, cfg: LaunchConfig,
) -> None:
    if not eval_cfg.path:
        raise ValueError(
            f"monitor_qa eval '{eval_cfg.label}' missing 'path' (JSONL dataset path)"
        )
    modes = eval_cfg.modes if isinstance(eval_cfg.modes, list) else [
        m.strip() for m in str(eval_cfg.modes).split(",")
    ]
    prompt_type = eval_cfg.prompt_type or "default"
    label = eval_cfg.label or f"{prompt_type}_{modes[0]}"

    mqa_config = MonitorQAConfig(
        actor_model=spec.model,
        actor_backend="tinker",
        actor_max_tokens=cfg.eval.max_tokens,
        actor_temperature=cfg.eval.temperature,
        actor_concurrency=eval_cfg.actor_concurrency or 40,
        actor_max_retries=1,
        monitor_model=eval_cfg.monitor_model or "openai/gpt-4o",
        monitor_max_tokens=1024,
        monitor_temperature=0.0,
        monitor_concurrency=eval_cfg.monitor_concurrency or 60,
        monitor_max_retries=cfg.eval.max_retries,
        dataset_path=eval_cfg.path,
        dataset_split=eval_cfg.dataset_split or "val",
        n_samples=eval_cfg.n_samples,
        seed=cfg.eval.seed,
        prompt_type=prompt_type,
        monitor_types=eval_cfg.monitor_types or ["metr_note"],
        monitor_model_overrides=eval_cfg.monitor_model_overrides or {},
        monitor_reasoning_effort=eval_cfg.monitor_reasoning_effort,
        monitor_reasoning_effort_overrides=eval_cfg.monitor_reasoning_effort_overrides or {},
        modes=modes,
        reasoning_effort=spec.reasoning_effort,
        request_timeout=cfg.eval.request_timeout,
        model_path=sampler_path,
        output_dir=str(RESULTS_DIR / "rollouts" / "monitor_qa"),
    )
    actor_short = spec.model.replace("/", "_")
    filename = (
        f"monitor_qa_{actor_short}-ft-{cfg.run_id}-{checkpoint_name}"
        f"_{label}_{mqa_config.experiment_id}.jsonl"
    )
    mqa_config.output_filename_override = filename

    logging.info("Eval (monitor_qa): %s / %s / %s", spec.short_name, label, checkpoint_name)
    asyncio.run(run_monitor_experiment(mqa_config))


# ---------------------------------------------------------------------------
# Orchestrator with continuous per-checkpoint dispatch
# ---------------------------------------------------------------------------


async def _spawn_subprocess(args: list[str], log_file: Path) -> asyncio.subprocess.Process:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text("")  # truncate
    log_fp = open(log_file, "a", buffering=1)  # line-buffered
    return await asyncio.create_subprocess_exec(
        *args,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        cwd=str(PROJECT_ROOT),
    )


def _ft_cmd(short_name: str, config_path: Path, run_id: str) -> list[str]:
    return [
        sys.executable, str(SCRIPT_PATH),
        "--single-ft", short_name,
        "--config", str(config_path),
        "--run-id", run_id,
    ]


def _eval_cmd(
    short_name: str, ds_key: str, ckpt_name: str,
    config_path: Path, run_id: str,
) -> list[str]:
    return [
        sys.executable, str(SCRIPT_PATH),
        "--single-eval", short_name,
        "--dataset", ds_key,
        "--checkpoint", ckpt_name,
        "--config", str(config_path),
        "--run-id", run_id,
    ]


async def orchestrate(
    cfg: LaunchConfig, config_path: Path,
    eval_only: bool = False,
    models_filter: list[str] | None = None,
    poll_interval: float = 10.0,
) -> dict[str, str]:
    """Run the full pipeline: FTs in parallel, evals dispatched per-checkpoint."""
    results: dict[str, str] = {}
    specs = cfg.models
    if models_filter:
        specs = [s for s in specs if s.short_name in set(models_filter)]
    if not specs:
        raise ValueError("No models selected")

    # -- Start FT subprocesses (unless eval-only) --
    ft_procs: dict[str, asyncio.subprocess.Process] = {}
    if not eval_only:
        for spec in specs:
            log_file = RESULTS_DIR / "logs" / f"{cfg.run_id}_{spec.short_name}_ft.log"
            cmd = _ft_cmd(spec.short_name, config_path, cfg.run_id)
            logging.info("Launching FT: %s (log: %s)", spec.short_name, log_file)
            proc = await _spawn_subprocess(cmd, log_file)
            ft_procs[spec.short_name] = proc
    else:
        logging.info("--eval-only: skipping FT phase, using existing checkpoints")

    # -- Track dispatched (short_name, ckpt_name, dataset) triples --
    dispatched: set[tuple[str, str, str]] = set()
    eval_tasks: set[asyncio.Task] = set()
    ft_done: dict[str, int] = {}
    sem = asyncio.Semaphore(cfg.eval.max_eval_workers)

    async def run_one_eval(short_name: str, ds_key: str, ckpt_name: str) -> str:
        """Single eval subprocess with semaphore."""
        async with sem:
            log_file = RESULTS_DIR / "logs" / f"{cfg.run_id}_{short_name}_{_safe(ds_key)}_{ckpt_name}.log"
            cmd = _eval_cmd(short_name, ds_key, ckpt_name, config_path, cfg.run_id)
            logging.info("Launching eval: %s / %s / %s (log: %s)",
                         short_name, ds_key, ckpt_name, log_file)
            proc = await _spawn_subprocess(cmd, log_file)
            ret = await proc.wait()
            status = "OK" if ret == 0 else f"FAIL({ret})"
            logging.info("Eval done: %s / %s / %s -> %s",
                         short_name, ds_key, ckpt_name, status)
            return status

    def queue_evals_for(spec: FTModelSpec) -> None:
        """Inspect this model's checkpoints.jsonl and queue any unqueued evals."""
        log_path = cfg.log_path_for(spec)
        ckpts = load_checkpoints_jsonl(log_path)
        for ckpt in ckpts:
            ckpt_name = ckpt["name"]
            for ds in cfg.eval.datasets:
                key = (spec.short_name, ckpt_name, ds.key)
                if key in dispatched:
                    continue
                dispatched.add(key)
                task = asyncio.create_task(run_one_eval(spec.short_name, ds.key, ckpt_name))
                eval_tasks.add(task)
                task.add_done_callback(lambda t, k=key: _record_eval_result(k, t, results))

    # -- Main loop: poll + dispatch --
    while True:
        # Poll each FT's checkpoints.jsonl
        for spec in specs:
            if (not eval_only) and spec.short_name in ft_done:
                # already queued everything for this model
                pass
            queue_evals_for(spec)

        # Check FT process exits
        if not eval_only:
            for short_name, proc in list(ft_procs.items()):
                if short_name in ft_done:
                    continue
                if proc.returncode is not None:
                    ft_done[short_name] = proc.returncode
                    results[f"ft_{short_name}"] = "OK" if proc.returncode == 0 else f"FAIL({proc.returncode})"
                    logging.info("FT exited: %s -> %s", short_name, results[f"ft_{short_name}"])
                    # Final poll to catch the last checkpoint
                    queue_evals_for(cfg.get_model(short_name))

        # Prune finished eval tasks
        still_running = {t for t in eval_tasks if not t.done()}
        eval_tasks = still_running

        all_fts_done = eval_only or len(ft_done) == len(ft_procs)
        no_more_evals = not eval_tasks
        if all_fts_done and no_more_evals:
            break

        await asyncio.sleep(poll_interval)

    # Wait for any in-flight evals
    if eval_tasks:
        await asyncio.gather(*eval_tasks, return_exceptions=True)

    return results


def _record_eval_result(
    key: tuple[str, str, str], task: asyncio.Task, results: dict,
) -> None:
    short_name, ckpt, ds = key
    try:
        status = task.result()
    except Exception as e:  # noqa: BLE001
        status = f"EXC({e})"
    results[f"eval_{short_name}_{ds}_{ckpt}"] = status


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _setup_logging(log_file: Path | None = None) -> None:
    target = log_file or (RESULTS_DIR / "launch_ft.log")
    target.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(target, mode="a")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    root.addHandler(fh)
    root.addHandler(sh)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_dry_run(cfg: LaunchConfig) -> None:
    print(f"\nRun ID:        {cfg.run_id}")
    print(f"Parquet dir:   {cfg.parquet_dir}")
    print(f"Log dir:       {cfg.log_dir}")
    print("\nModels:")
    for spec in cfg.models:
        pq = cfg.parquet_path_for(spec)
        lp = cfg.log_path_for(spec)
        exists = "OK" if pq.exists() else "MISSING"
        print(f"  {spec.short_name}")
        print(f"    base:    {spec.base_model}")
        print(f"    parquet: {pq} [{exists}]")
        print(f"    log:     {lp}")
        print(f"    renderer: {spec.renderer_type} (effort={spec.reasoning_effort})")
    t = cfg.training
    print("\nTraining:")
    print(f"  batch_size={t.batch_size}  max_length={t.max_length}  epochs={t.num_epochs}")
    print(f"  lora_rank={t.lora_rank}  lr={t.learning_rate}")
    if t.save_every is not None:
        print(f"  save_every={t.save_every} (explicit)")
    else:
        print(f"  save_fraction={t.save_fraction} (auto save_every per model)")
    print(f"  n_train_examples={t.n_train_examples}")
    e = cfg.eval
    print("\nEval:")
    print(f"  max_eval_workers={e.max_eval_workers}  max_concurrency={e.max_concurrency}")
    print(f"  max_tokens={e.max_tokens}  request_timeout={e.request_timeout}")
    for ds in e.datasets:
        if ds.dataset == "monitor_qa":
            print(f"  - monitor_qa[{ds.label}]  prompt={ds.prompt_type}  modes={ds.modes}"
                  f"  n={ds.n_samples}")
            print(f"      monitors: {ds.monitor_types}")
            if ds.monitor_model_overrides:
                for t, m in ds.monitor_model_overrides.items():
                    print(f"        {t:20s} -> {m}")
        else:
            print(f"  - {ds.dataset}  modes={ds.modes}  n={ds.n_samples}")
    # Estimate checkpoint count per FT based on save_every and a typical
    # parquet size. Reads the first model's parquet if it exists.
    n_ckpts = _estimate_checkpoint_count(cfg)
    print(f"\n  Total eval jobs per checkpoint: {len(e.datasets)}")
    if n_ckpts:
        print(f"  Expected jobs per FT (~{n_ckpts} checkpoints): {len(e.datasets) * n_ckpts}")
        print(f"  Total across {len(cfg.models)} models: "
              f"{len(e.datasets) * n_ckpts * len(cfg.models)}")
    else:
        print("  Expected checkpoints per FT: depends on parquet size "
              "(not yet built).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified FT+Eval orchestrator with continuous checkpoint dispatch",
    )
    parser.add_argument("--config", type=Path, help="YAML config path")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--models", default=None,
                        help="Comma-separated short names to restrict to")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--poll-interval", type=float, default=10.0)

    # Internal single-job modes (recursively invoked by the orchestrator)
    parser.add_argument("--single-ft", type=str, default=None)
    parser.add_argument("--single-eval", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)

    args = parser.parse_args()

    if args.config is None:
        parser.error("--config is required")

    config_path = args.config.resolve()
    cfg = LaunchConfig.from_yaml(config_path, override_run_id=args.run_id)

    # --- single-ft subprocess mode ---
    if args.single_ft:
        log_file = RESULTS_DIR / "logs" / f"{cfg.run_id}_{args.single_ft}_ft.log"
        _setup_logging(log_file)
        run_single_ft(args.single_ft, cfg)
        return

    # --- single-eval subprocess mode ---
    if args.single_eval:
        if not args.dataset or not args.checkpoint:
            parser.error("--single-eval requires --dataset and --checkpoint")
        log_file = RESULTS_DIR / "logs" / f"{cfg.run_id}_{args.single_eval}_{args.dataset}_{args.checkpoint}.log"
        _setup_logging(log_file)
        run_single_eval(args.single_eval, args.dataset, args.checkpoint, cfg)
        return

    # --- orchestrator mode ---
    if args.dry_run:
        _print_dry_run(cfg)
        return

    _setup_logging()
    logging.info("Starting run %s (%d models)", cfg.run_id, len(cfg.models))

    models_filter = args.models.split(",") if args.models else None
    results = asyncio.run(orchestrate(
        cfg, config_path,
        eval_only=args.eval_only,
        models_filter=models_filter,
        poll_interval=args.poll_interval,
    ))

    # Summary
    logging.info("=" * 60)
    logging.info("RUN %s COMPLETE", cfg.run_id)
    logging.info("=" * 60)
    for k, v in sorted(results.items()):
        logging.info("  %-60s %s", k, v)
    n_ok = sum(1 for v in results.values() if v == "OK")
    n_fail = sum(1 for v in results.values() if not v.startswith("OK"))
    logging.info("Total: %d OK, %d not-OK", n_ok, n_fail)


if __name__ == "__main__":
    main()
