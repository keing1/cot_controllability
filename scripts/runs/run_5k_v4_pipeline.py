#!/usr/bin/env python3
"""Fourth iteration 5K-mixed FT pipeline (v4).

Two parts:

  PART 1 — main pipeline:
    Phase 0  Generate SFT parquets for all 7 models concurrently
             (in-process; ``build_sft_dataset`` per model).
    Phase 1a Baseline evals on the 4 phase-1 datasets across all 7 models
             (in-process; 7-wide eval semaphore; uses base weights, no FT).
    Phase 1b FT + per-checkpoint evals via ``launch_ft.py``. The launcher
             trains all 7 models in parallel (Tinker subprocesses) and
             dispatches evals as checkpoints land. Eval ordering is a
             *soft* priority — earlier ckpts preferred when multiple are
             simultaneously available, but slow models don't block fast
             ones from progressing to later ckpts.

  PART 2 — auto-kicked off after PART 1 finishes:
    Phase 6a Baseline evals for the 2 phase-2 datasets (vocab_control,
             dont_think_v2).
    Phase 6b ``launch_ft.py --eval-only`` for FT'd-checkpoint evals on
             those same 2 datasets.

Sample sizes per (model, ckpt, dataset) eval are scaled to ~500 rollouts:
    cotcontrol  56 × 9 modes  ≈ 504
    reasonif    84 × 6 modes  =  504
    monitor_qa  500 (single mode/setting)

Allowed eval checkpoints: 000050, 000100, 000300, final (matches v3).

Concurrency:
    7-wide within each phase; FT subprocesses run in the background on
    Tinker side and don't share the eval semaphore. Phases run
    sequentially within each part — datasets, then baselines, then FT
    + ckpt evals.

Usage:
    uv run python scripts/runs/run_5k_v4_pipeline.py
    uv run python scripts/runs/run_5k_v4_pipeline.py --dry-run
    uv run python scripts/runs/run_5k_v4_pipeline.py --skip-datasets --run-id <id>
    uv run python scripts/runs/run_5k_v4_pipeline.py --skip-part-2  # part 1 only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from controllability.config import ExperimentConfig, MonitorQAConfig, Settings  # noqa: E402
from controllability.evals.runner import run_experiment, run_monitor_experiment  # noqa: E402
from controllability.training.sft_builder import (  # noqa: E402
    BuildConfig, SourceRequest, build_sft_dataset,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

CONFIG_PATH = PROJECT_ROOT / "configs" / "ft_mixed_5k_v4_7models.yaml"
LAUNCH_FT = PROJECT_ROOT / "scripts" / "runs" / "launch_ft.py"

DEFAULT_QUESTION_POOL = PROJECT_ROOT / "external" / "question_pools" / "mixed_5k.parquet"
REASONIF_TEMPLATE_SOURCE = PROJECT_ROOT / "external" / "reasonIF_ft" / "train-00000-of-00001.parquet"

# Eval-pool size for orchestrator-managed phases (baseline evals). The
# launch_ft.py phases use their own semaphore at the same size from the YAML.
EVAL_CONCURRENCY = 7

# Phase grouping. Dataset keys here match launch_ft.py's --datasets-filter.
PART_1_DATASETS = [
    "cotcontrol",
    "reasonif",
    "monitor_qa@baseline",
    "monitor_qa@less_reasoning_v2",
]
PART_2_DATASETS = [
    "monitor_qa@vocab_control",
    "monitor_qa@dont_think_v2",
]


# ---------------------------------------------------------------------------
# Helpers — config + paths
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _is_gpt_oss(spec: dict) -> bool:
    return "gpt-oss" in spec["base_model"].lower()


def _analysis_channel_for(spec: dict) -> bool:
    return _is_gpt_oss(spec)


# ---------------------------------------------------------------------------
# Phase 0 — dataset generation (in-process, async, 7 parallel)
# ---------------------------------------------------------------------------


async def _generate_dataset(spec: dict, parquet_dir: Path, dry_run: bool, max_length: int) -> Path:
    parquet_path = parquet_dir / spec["parquet"]
    checkpoint_path = parquet_path.with_suffix(parquet_path.suffix + ".checkpoint.jsonl")
    if parquet_path.exists():
        logging.info("[%s] Parquet exists, skipping dataset gen: %s",
                     spec["short_name"], parquet_path.name)
        # Still apply the post-stage-3 length filter in case the parquet
        # was generated before the filter was added.
        _maybe_filter_parquet(spec, parquet_path, max_length)
        return parquet_path
    if dry_run:
        logging.info("[%s] DRY-RUN dataset gen → %s", spec["short_name"], parquet_path.name)
        return parquet_path
    logging.info("[%s] Generating SFT parquet → %s",
                 spec["short_name"], parquet_path.name)
    cfg = BuildConfig(
        output_path=parquet_path,
        checkpoint_path=checkpoint_path,
        question_source_parquet=DEFAULT_QUESTION_POOL,
        reasonif_template_source=REASONIF_TEMPLATE_SOURCE,
        base_model=spec["base_model"],
        backend="tinker",
        reasoning_effort=spec.get("reasoning_effort"),
        analysis_channel=_analysis_channel_for(spec),
        sources=[
            SourceRequest(name="reasonif",   count=2500),
            SourceRequest(name="cotcontrol", count=2500),
        ],
        seed=42,
        max_concurrency=2000,
        request_timeout=1800,
        max_retries=3,
        # Built-in post-Stage-3 validation (length filter + content validator).
        train_max_length=max_length,
        renderer_type=spec["renderer_type"],
    )
    await build_sft_dataset(cfg)
    logging.info("[%s] Dataset done.", spec["short_name"])
    return parquet_path


def _maybe_filter_parquet(spec: dict, parquet_path: Path, max_length: int) -> None:
    """Render each parquet row and drop rows over `max_length` tokens.

    Then run a content validator that drops rows with empty content,
    space-stripped thinking, or leaked special tokens.

    Together these guarantee training never sees a silently-truncated row or
    a row produced by a decoder bug.
    """
    if not parquet_path.exists() or max_length is None or max_length <= 0:
        return
    try:
        from transformers import AutoTokenizer
        from controllability.training.ft_runner import FTModelSpec
        from _v4_filter_helpers import (
            filter_parquet_by_max_length, validate_parquet_content,
        )

        ft_spec = FTModelSpec(
            model=spec["model"],
            base_model=spec["base_model"],
            parquet=spec["parquet"],
            log_path=spec.get("log_path") or spec["short_name"],
            renderer_type=spec["renderer_type"],
            reasoning_effort=spec.get("reasoning_effort", "none"),
            short_name=spec["short_name"],
        )
        trc = "moonshotai" in ft_spec.base_model.lower()
        tok = AutoTokenizer.from_pretrained(ft_spec.base_model, trust_remote_code=trc)
        stats = filter_parquet_by_max_length(parquet_path, ft_spec, tok, max_length)
        logging.info(
            "[%s] Length filter (max_length=%d): kept=%d dropped=%d max_seen=%d",
            spec["short_name"], max_length,
            stats["kept"], stats["dropped"], stats["max_seen"],
        )
        vstats = validate_parquet_content(parquet_path)
        logging.info(
            "[%s] Content validator: kept=%d dropped=%d reasons=%s",
            spec["short_name"], vstats["kept"], vstats["dropped"],
            vstats["drop_reasons"],
        )
    except Exception as e:
        logging.exception("[%s] Filter/validate failed: %s", spec["short_name"], e)


# ---------------------------------------------------------------------------
# Phase 1a / 6a — baseline evals (orchestrator-controlled, 7-wide)
# ---------------------------------------------------------------------------


def _expand_ds(eval_section: dict) -> list[dict]:
    """Expand monitor_qa.prompt_settings into one entry per (label)."""
    out: list[dict] = []
    for d in eval_section.get("datasets", []):
        if d.get("dataset") == "monitor_qa" and "prompt_settings" in d:
            base = {k: v for k, v in d.items() if k != "prompt_settings"}
            for ps in d["prompt_settings"]:
                out.append({**base, **ps})
        else:
            out.append(dict(d))
    return out


def _ds_key(d: dict) -> str:
    if d["dataset"] == "monitor_qa":
        return f"monitor_qa@{d.get('label', '')}"
    return d["dataset"]


def _filter(entries: list[dict], keep: list[str]) -> list[dict]:
    return [d for d in entries if _ds_key(d) in keep]


async def _run_baseline_eval(
    spec: dict, ds_entry: dict, eval_cfg: dict, run_id: str, dry_run: bool,
) -> tuple[str, str, str]:
    actor_short = spec["model"].replace("/", "_")
    ds_key = _ds_key(ds_entry)
    if dry_run:
        return (spec["short_name"], ds_key, "DRY_OK")
    if ds_entry["dataset"] == "monitor_qa":
        modes = ds_entry.get("modes") or ["side_task"]
        prompt_type = ds_entry.get("prompt_type") or "default"
        label = ds_entry.get("label") or f"{prompt_type}_{modes[0]}"
        cfg = MonitorQAConfig(
            actor_model=spec["model"],
            actor_backend="tinker",
            actor_max_tokens=eval_cfg["max_tokens"],
            actor_temperature=eval_cfg["temperature"],
            actor_concurrency=ds_entry.get("actor_concurrency") or 40,
            actor_max_retries=1,
            monitor_model=ds_entry.get("monitor_model") or "openai/gpt-4o",
            monitor_max_tokens=1024,
            monitor_temperature=0.0,
            monitor_concurrency=ds_entry.get("monitor_concurrency") or 60,
            monitor_max_retries=eval_cfg["max_retries"],
            dataset_path=ds_entry["path"],
            dataset_split="val",
            n_samples=ds_entry["n_samples"],
            seed=eval_cfg["seed"],
            prompt_type=prompt_type,
            monitor_types=ds_entry.get("monitor_types") or ["metr_note"],
            monitor_model_overrides=ds_entry.get("monitor_model_overrides") or {},
            modes=modes,
            reasoning_effort=spec.get("reasoning_effort", "none"),
            request_timeout=ds_entry.get("request_timeout") or eval_cfg["request_timeout"],
            model_path=None,  # base model: TinkerClient will use base_model
            output_dir=str(PROJECT_ROOT / "results" / "rollouts" / "monitor_qa"),
        )
        cfg.output_filename_override = (
            f"monitor_qa_{actor_short}-ft-{run_id}-base_{label}_{cfg.experiment_id}.jsonl"
        )
        await run_monitor_experiment(cfg)
    else:
        modes_str = ds_entry.get("modes", "all")
        modes_list = (
            modes_str if isinstance(modes_str, list)
            else [m.strip() for m in str(modes_str).split(",")]
        )
        if modes_list == ["all"]:
            from launch_ft import DATASET_MODES  # type: ignore
            modes_list = list(DATASET_MODES[ds_entry["dataset"]])
        excl = set(ds_entry.get("exclude_modes") or [])
        modes_list = [m for m in modes_list if m not in excl]
        cfg = ExperimentConfig(
            model=spec["model"],
            dataset=ds_entry["dataset"],
            modes=modes_list,
            backend="tinker",
            n_samples=ds_entry["n_samples"],
            max_tokens=eval_cfg["max_tokens"],
            temperature=eval_cfg["temperature"],
            max_concurrency=ds_entry.get("max_concurrency") or eval_cfg["max_concurrency"],
            max_retries=eval_cfg["max_retries"],
            request_timeout=ds_entry.get("request_timeout") or eval_cfg["request_timeout"],
            seed=eval_cfg["seed"],
            reasoning_effort=spec.get("reasoning_effort", "none"),
            model_path=None,
            split="all",
        )
        cfg.output_filename_override = (
            f"{actor_short}-ft-{run_id}-base_{ds_entry['dataset']}_all_{run_id}_"
            f"{cfg.experiment_id}.jsonl"
        )
        await run_experiment(cfg)
    return (spec["short_name"], ds_key, "OK")


async def _run_baseline_phase(
    label: str,
    specs: list[dict],
    ds_entries: list[dict],
    eval_cfg: dict,
    run_id: str,
    dry_run: bool,
) -> list[tuple]:
    pairs = [(s, d) for s in specs for d in ds_entries]
    sem = asyncio.Semaphore(EVAL_CONCURRENCY)
    logging.info("=== BASELINE EVAL PHASE %s — %d evals (7-wide) ===",
                 label, len(pairs))

    async def _bound(spec, ds):
        async with sem:
            t0 = time.time()
            try:
                res = await _run_baseline_eval(spec, ds, eval_cfg, run_id, dry_run)
            except Exception as e:
                logging.exception("    base eval FAILED [%s/%s]: %s",
                                  spec["short_name"], _ds_key(ds), e)
                return (spec["short_name"], _ds_key(ds), f"FAIL({type(e).__name__})")
            dt = time.time() - t0
            logging.info("    base eval done [%s/%s] -> %s (%.1fs)",
                         res[0], res[1], res[2], dt)
            return res

    results = await asyncio.gather(*[_bound(s, d) for s, d in pairs])
    n_ok = sum(1 for r in results if r[2] in ("OK", "DRY_OK"))
    logging.info("=== BASELINE %s done — %d/%d OK ===",
                 label, n_ok, len(results))
    return results


# ---------------------------------------------------------------------------
# Phase 1b / 6b — FT + checkpoint evals via launch_ft.py
# ---------------------------------------------------------------------------


async def _run_launch_ft(
    run_id: str,
    datasets_filter: list[str],
    eval_only: bool,
    log_dir: Path,
    dry_run: bool,
    label: str,
) -> int:
    """Spawn launch_ft.py with --datasets-filter and (optionally) --eval-only.

    Returns the subprocess exit code.
    """
    log_file = log_dir / "logs" / f"{run_id}_launch_ft_{label}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(LAUNCH_FT),
        "--config", str(CONFIG_PATH),
        "--run-id", run_id,
        "--datasets-filter", ",".join(datasets_filter),
    ]
    if eval_only:
        cmd.append("--eval-only")
    if dry_run:
        cmd.append("--dry-run")

    logging.info("Spawning launch_ft.py [%s]: %s", label, " ".join(cmd))
    if dry_run:
        # In dry-run we still want to invoke launch_ft.py --dry-run so it
        # prints its own plan; that exits cleanly without doing real work.
        pass

    fp = open(log_file, "a", buffering=1)
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=fp, stderr=subprocess.STDOUT, cwd=str(PROJECT_ROOT),
    )
    rc = await proc.wait()
    fp.close()
    logging.info("launch_ft.py [%s] exited with code %d (log: %s)",
                 label, rc, log_file)
    return rc


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


async def main(args) -> None:
    config = _load_yaml(CONFIG_PATH)
    parquet_dir = PROJECT_ROOT / config.get("parquet_dir", "results/sft")
    log_dir = PROJECT_ROOT / config.get("log_dir", "results/sft")
    parquet_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    print(f"=== Run ID: {run_id} ===")
    print(f"Config:    {CONFIG_PATH}")
    print(f"Models:    {[s['short_name'] for s in config['models']]}")
    print(f"Allowed ckpts: {config['eval'].get('allowed_checkpoints')}")

    eval_cfg = config["eval"]
    ds_entries_all = _expand_ds(eval_cfg)
    p1_entries = _filter(ds_entries_all, PART_1_DATASETS)
    p2_entries = _filter(ds_entries_all, PART_2_DATASETS)

    print(f"\nPart 1 datasets ({len(p1_entries)}):")
    for d in p1_entries:
        print(f"  - {_ds_key(d):35s} n={d.get('n_samples')}")
    print(f"Part 2 datasets ({len(p2_entries)}):")
    for d in p2_entries:
        print(f"  - {_ds_key(d):35s} n={d.get('n_samples')}")

    if args.dry_run:
        print("\n[DRY RUN] No FT subprocesses spawned, no API calls made (except in launch_ft.py --dry-run sub-invocations which print only).")

    specs = config["models"]

    # ============== PART 1 ==============

    # Phase 0: dataset gen (parallel async)
    train_max_length = config.get("training", {}).get("max_length", 8192)
    if not args.skip_datasets:
        print("\n--- PART 1 / Phase 0: dataset generation (7 parallel) ---")
        await asyncio.gather(*[
            _generate_dataset(s, parquet_dir, args.dry_run, train_max_length)
            for s in specs
        ])
    else:
        print("\n--skip-datasets: assuming SFT parquets are already built")

    # Phase 1a + 1b run in parallel: FT subprocesses don't depend on baseline
    # evals completing; both only need Phase 0 parquets. Starting FT
    # immediately after dataset gen minimizes wall-clock time.
    print("\n--- PART 1 / Phase 1a + 1b in parallel: baseline evals + FT/ckpt evals ---")

    async def _phase_1a():
        print("  [1a] baseline evals starting")
        return await _run_baseline_phase(
            "part1", specs, p1_entries, eval_cfg, run_id, args.dry_run,
        )

    async def _phase_1b():
        print("  [1b] FT + checkpoint evals (launch_ft.py) starting")
        return await _run_launch_ft(
            run_id,
            datasets_filter=PART_1_DATASETS,
            eval_only=False,
            log_dir=log_dir,
            dry_run=args.dry_run,
            label="part1",
        )

    base_p1, rc1 = await asyncio.gather(_phase_1a(), _phase_1b())

    if rc1 != 0 and not args.dry_run:
        print(f"\nPart 1 FT/eval exited non-zero ({rc1}). Aborting before Part 2.")
        sys.exit(rc1)

    # ============== PART 2 ==============

    if args.skip_part_2:
        print("\n--skip-part-2: stopping after Part 1.")
        return

    # Phase 6a: baseline evals on the 2 phase-2 datasets
    print("\n--- PART 2 / Phase 6a: baseline evals (vocab_control + dont_think_v2) ---")
    base_p2 = await _run_baseline_phase(
        "part2", specs, p2_entries, eval_cfg, run_id, args.dry_run,
    )

    # Phase 6b: --eval-only ckpt evals on the 2 phase-2 datasets
    print("\n--- PART 2 / Phase 6b: --eval-only checkpoint evals ---")
    rc2 = await _run_launch_ft(
        run_id,
        datasets_filter=PART_2_DATASETS,
        eval_only=True,
        log_dir=log_dir,
        dry_run=args.dry_run,
        label="part2",
    )

    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETE — Run ID: {run_id}")
    print("=" * 80)
    print(f"Part 1 baseline evals: {sum(1 for r in base_p1 if r[2] in ('OK','DRY_OK'))}/{len(base_p1)} OK")
    print(f"Part 2 baseline evals: {sum(1 for r in base_p2 if r[2] in ('OK','DRY_OK'))}/{len(base_p2)} OK")
    print(f"Part 1 launch_ft exit: {rc1}")
    print(f"Part 2 launch_ft exit: {rc2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-datasets", action="store_true",
                        help="Skip Phase 0 (assume parquets already built).")
    parser.add_argument("--skip-part-2", action="store_true",
                        help="Stop after Part 1; don't run vocab_control + dont_think_v2.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    asyncio.run(main(args))
