#!/usr/bin/env python3
"""Fifth iteration FT pipeline (v5) — 14 fine-tunes, with vs without
instruction at Stage-1 generation.

  PART 1
    Phase 0  Generate 14 SFT parquets concurrently (in-process).
             Each model has an ``include_instruction_at_generation`` flag
             from the YAML; with-instr runs additionally do meta-removal,
             pre-pick suppression keywords, and the new editor prompts.
    Phase 1b FT + per-checkpoint evals via ``launch_ft.py``. The launcher
             trains all 14 models in parallel (Tinker subprocesses) and
             dispatches evals as checkpoints land. Cross-model checkpoint
             priority: every model's ckpt 50 evals fire before any model's
             ckpt 100 evals.

  Pre-FT baseline evals are SKIPPED — those are already covered by the v4
  run.

  PART 2 (auto-kicked off after Part 1):
    Phase 2  ``launch_ft.py --eval-only`` for vocab_control + dont_think_v2.

Usage:
    uv run python scripts/runs/run_v5_14models_pipeline.py
    uv run python scripts/runs/run_v5_14models_pipeline.py --dry-run
    uv run python scripts/runs/run_v5_14models_pipeline.py --skip-datasets --run-id <id>
    uv run python scripts/runs/run_v5_14models_pipeline.py --skip-part-2
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

from controllability.training.sft_builder import (  # noqa: E402
    BuildConfig, SourceRequest, build_sft_dataset,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

CONFIG_PATH = PROJECT_ROOT / "configs" / "ft_v5_14models.yaml"
LAUNCH_FT = PROJECT_ROOT / "scripts" / "runs" / "launch_ft.py"

DEFAULT_QUESTION_POOL = PROJECT_ROOT / "external" / "question_pools" / "mixed_5k.parquet"
REASONIF_TEMPLATE_SOURCE = PROJECT_ROOT / "external" / "reasonIF_ft" / "train-00000-of-00001.parquet"

# Per-dataset SFT generation concurrency. Bumped from v4 (2000) per
# operator preference.
DATASET_CONCURRENCY = 1000

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
# Helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _is_gpt_oss(spec: dict) -> bool:
    return "gpt-oss" in spec["base_model"].lower()


def _analysis_channel_for(spec: dict) -> bool:
    return _is_gpt_oss(spec)


# ---------------------------------------------------------------------------
# Phase 0 — dataset generation (in-process, 14 parallel)
# ---------------------------------------------------------------------------


async def _generate_dataset(spec: dict, parquet_dir: Path, dry_run: bool, max_length: int) -> Path:
    parquet_path = parquet_dir / spec["parquet"]
    checkpoint_path = parquet_path.with_suffix(parquet_path.suffix + ".checkpoint.jsonl")
    if parquet_path.exists():
        logging.info("[%s] Parquet exists, skipping dataset gen: %s",
                     spec["short_name"], parquet_path.name)
        _maybe_filter_parquet(spec, parquet_path, max_length)
        return parquet_path
    if dry_run:
        logging.info("[%s] DRY-RUN dataset gen → %s (include_instruction=%s)",
                     spec["short_name"], parquet_path.name,
                     spec.get("include_instruction_at_generation", False))
        return parquet_path
    include_instr = bool(spec.get("include_instruction_at_generation", False))
    logging.info(
        "[%s] Generating SFT parquet → %s (include_instruction=%s)",
        spec["short_name"], parquet_path.name, include_instr,
    )
    cfg = BuildConfig(
        output_path=parquet_path,
        checkpoint_path=checkpoint_path,
        question_source_parquet=DEFAULT_QUESTION_POOL,
        reasonif_template_source=REASONIF_TEMPLATE_SOURCE,
        base_model=spec["base_model"],
        backend="tinker",
        reasoning_effort=spec.get("reasoning_effort"),
        analysis_channel=_analysis_channel_for(spec),
        include_instruction_at_generation=include_instr,
        sources=[
            # 1500 + 1500 = 3000 total per dataset (per operator preference;
            # downsized from prior 5K target). Comfortably under the
            # ~4536 unique questions available in the mixed_5k pool after
            # the non-Latin filter.
            SourceRequest(name="reasonif",   count=1500),
            SourceRequest(name="cotcontrol", count=1500),
        ],
        seed=42,
        max_concurrency=DATASET_CONCURRENCY,
        request_timeout=1800,
        max_retries=3,
        train_max_length=max_length,
        renderer_type=spec["renderer_type"],
    )
    await build_sft_dataset(cfg)
    logging.info("[%s] Dataset done.", spec["short_name"])
    return parquet_path


def _maybe_filter_parquet(spec: dict, parquet_path: Path, max_length: int) -> None:
    """Re-render + filter rows that exceed max_length, plus content validation."""
    if not parquet_path.exists() or max_length is None or max_length <= 0:
        return
    try:
        from transformers import AutoTokenizer
        from controllability.training.ft_runner import FTModelSpec
        sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "runs"))
        from _v4_filter_helpers import (  # noqa: E402
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
    except Exception as e:  # noqa: BLE001
        logging.exception("[%s] Filter/validate failed: %s", spec["short_name"], e)


# ---------------------------------------------------------------------------
# launch_ft subprocess wrapper
# ---------------------------------------------------------------------------


async def _run_launch_ft(
    run_id: str,
    datasets_filter: list[str],
    eval_only: bool,
    log_dir: Path,
    dry_run: bool,
    label: str,
) -> int:
    log_file = log_dir / "logs" / f"{run_id}_launch_ft_v5_{label}.log"
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
    print(f"Models ({len(config['models'])}):")
    for s in config['models']:
        print(f"  - {s['short_name']:30s} include_instr={s.get('include_instruction_at_generation', False)}")
    print(f"Allowed ckpts: {config['eval'].get('allowed_checkpoints')}")
    print(f"Dataset concurrency: {DATASET_CONCURRENCY}")
    print(f"Eval workers: {config['eval'].get('max_eval_workers')}")

    if args.dry_run:
        print("\n[DRY RUN] No FT subprocesses spawned, no real API calls "
              "(launch_ft.py --dry-run prints its own plan).")

    specs = config["models"]
    train_max_length = config.get("training", {}).get("max_length", 8192)

    # ============== PART 1 ==============
    # Dataset gen + FT/eval orchestration run CONCURRENTLY. launch_ft.py
    # waits per-spec for each parquet to land before spawning that spec's
    # FT subprocess — so a fast model's FT can start while a slow model
    # is still finishing its dataset. Fastest path to first checkpoint.

    async def _phase_0_datasets():
        if args.skip_datasets:
            print("\n--skip-datasets: assuming SFT parquets are already built")
            return
        print(f"\n--- PART 1 / Phase 0: dataset generation ({len(specs)} parallel) ---")
        await asyncio.gather(*[
            _generate_dataset(s, parquet_dir, args.dry_run, train_max_length)
            for s in specs
        ])
        print(f"--- Phase 0 done — all {len(specs)} datasets written ---")

    async def _phase_1b_ft_evals():
        print(f"\n--- PART 1 / Phase 1b: FT + checkpoint evals (launch_ft.py)")
        print(f"    starts NOW; per-spec waits for its parquet then spawns FT ---")
        return await _run_launch_ft(
            run_id,
            datasets_filter=PART_1_DATASETS,
            eval_only=False,
            log_dir=log_dir,
            dry_run=args.dry_run,
            label="part1",
        )

    # Run both phases concurrently. Phase 0 returns None; phase 1b returns rc.
    _, rc1 = await asyncio.gather(_phase_0_datasets(), _phase_1b_ft_evals())

    if rc1 != 0 and not args.dry_run:
        print(f"\nPart 1 FT/eval exited non-zero ({rc1}). Aborting before Part 2.")
        sys.exit(rc1)

    # ============== PART 2 ==============

    if args.skip_part_2:
        print("\n--skip-part-2: stopping after Part 1.")
        return

    print(f"\n--- PART 2 / Phase 2: --eval-only checkpoint evals on {PART_2_DATASETS} ---")
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
    if rc2 != 0 and not args.dry_run:
        print(f"Part 2 exit code: {rc2}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", default=None,
                   help="Reuse a prior run-id (e.g. with --skip-datasets)")
    p.add_argument("--skip-datasets", action="store_true",
                   help="Skip Phase 0 (assume parquets already built)")
    p.add_argument("--skip-part-2", action="store_true",
                   help="Run Part 1 only (skip vocab_control + dont_think_v2)")
    p.add_argument("--dry-run", action="store_true",
                   help="Plan only — no subprocesses, no API calls")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(main(parse_args()))
