#!/usr/bin/env python3
"""Retry the v5 FTs that failed (Tinker JWT 401 / corrupted parquet).

For each model spec in the v5 yaml that does NOT have a "final" record in
its checkpoints.jsonl:
  * Verify the parquet is valid; if it's corrupt (read fails), regenerate.
    Stage-1 is cached so this is fast.
  * Auto-generate a retry yaml that includes ONLY the missing-final specs,
    pointed at a fresh log_path so checkpoints don't collide with the prior
    (partial) v5 run.
  * Spawn launch_ft.py with the retry yaml — trains from scratch (no
    load_checkpoint_path) and dispatches all evals on each new ckpt.

Idempotent: re-runnable; skips models that already have a final.

Usage:
    uv run python scripts/runs/retry_failed_v5_fts.py
    uv run python scripts/runs/retry_failed_v5_fts.py --dry-run
    uv run python scripts/runs/retry_failed_v5_fts.py --run-id <id>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import subprocess
import sys
import yaml
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(PROJECT_ROOT / ".env")

from controllability.training.sft_builder import (  # noqa: E402
    BuildConfig, SourceRequest, build_sft_dataset,
)


V5_RUN_ID = "v5_14m_20260505_051845"
V5_CONFIG = PROJECT_ROOT / "configs" / "ft_v5_14models.yaml"
LAUNCH_FT = PROJECT_ROOT / "scripts" / "runs" / "launch_ft.py"

QUESTION_POOL = (
    PROJECT_ROOT / "external" / "question_pools" / "mixed_5k.parquet"
)
REASONIF_TEMPLATE_SOURCE = (
    PROJECT_ROOT / "external" / "reasonIF_ft" / "train-00000-of-00001.parquet"
)

DATASET_CONCURRENCY = 1000
PART_1_DATASETS = [
    "cotcontrol", "reasonif",
    "monitor_qa@baseline", "monitor_qa@less_reasoning_v2",
]
PART_2_DATASETS = [
    "monitor_qa@vocab_control", "monitor_qa@dont_think_v2",
]


def _has_final(spec: dict) -> bool:
    log_dir = (PROJECT_ROOT / "results" / "sft"
               / f"{spec['short_name']}-{V5_RUN_ID}")
    ckpts = log_dir / "checkpoints.jsonl"
    if not ckpts.exists():
        return False
    for line in ckpts.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("name") == "final":
            return True
    return False


def _parquet_is_valid(parquet_path: Path) -> bool:
    if not parquet_path.exists():
        return False
    try:
        # Lightweight validation — just open the metadata.
        import pyarrow.parquet as pq
        pq.read_metadata(parquet_path)
        return True
    except Exception as e:  # noqa: BLE001
        logging.warning("parquet %s is corrupt: %s", parquet_path.name, e)
        return False


def _is_gpt_oss(spec: dict) -> bool:
    return "gpt-oss" in spec["base_model"].lower()


async def _regenerate_parquet(spec: dict, parquet_path: Path,
                                max_length: int) -> None:
    """Regenerate a corrupt parquet. Stage-1 cache (.checkpoint.jsonl) is
    typically intact and reused, so this is mostly Stage-2 work."""
    # Delete the bad file so build_sft_dataset re-creates it.
    bad_path = parquet_path.with_suffix(parquet_path.suffix + ".corrupt.bak")
    if parquet_path.exists():
        parquet_path.rename(bad_path)
        logging.info("[%s] moved corrupt parquet to %s",
                     spec["short_name"], bad_path.name)
    include_instr = bool(spec.get("include_instruction_at_generation", False))
    cfg = BuildConfig(
        output_path=parquet_path,
        checkpoint_path=parquet_path.with_suffix(parquet_path.suffix + ".checkpoint.jsonl"),
        question_source_parquet=QUESTION_POOL,
        reasonif_template_source=REASONIF_TEMPLATE_SOURCE,
        base_model=spec["base_model"],
        backend="tinker",
        reasoning_effort=spec.get("reasoning_effort"),
        analysis_channel=_is_gpt_oss(spec),
        include_instruction_at_generation=include_instr,
        sources=[
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
    logging.info("[%s] parquet regenerated.", spec["short_name"])


def _build_retry_yaml(v5_cfg: dict, retry_specs: list[dict],
                       run_id: str, dst_path: Path) -> Path:
    """Clone the v5 yaml with only the specs in ``retry_specs``, fresh
    log_paths/short_names, and the standard 3-checkpoint allowlist."""
    new_cfg = dict(v5_cfg)
    new_models = []
    for spec in retry_specs:
        # Fresh short_name + log_path → checkpoints written to a different
        # directory so we don't conflict with the prior (partial) v5 run.
        retry_short = spec["short_name"] + "-retry"
        new_models.append({
            **spec,
            "short_name": retry_short,
            "log_path": retry_short,
            # No load_checkpoint_path → fresh LoRA training from base.
        })
    new_cfg["models"] = new_models
    new_cfg["run_id"] = run_id
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_path, "w") as f:
        yaml.safe_dump(new_cfg, f, sort_keys=False)
    return dst_path


async def _run_launch_ft(run_id: str, config_path: Path,
                          datasets_filter: list[str], eval_only: bool,
                          log_dir: Path, dry_run: bool, label: str) -> int:
    log_file = log_dir / "logs" / f"{run_id}_launch_ft_v5retry_{label}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(LAUNCH_FT),
        "--config", str(config_path),
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
    logging.info("launch_ft.py [%s] exited %d (log: %s)", label, rc, log_file)
    return rc


async def main(args) -> None:
    with open(V5_CONFIG) as f:
        v5_cfg = yaml.safe_load(f)
    parquet_dir = PROJECT_ROOT / v5_cfg.get("parquet_dir", "results/sft")
    log_dir = PROJECT_ROOT / v5_cfg.get("log_dir", "results/sft")
    train_max_length = v5_cfg.get("training", {}).get("max_length", 8192)

    # Identify which specs need a retry (no "final" in checkpoints.jsonl).
    retry_specs = [s for s in v5_cfg["models"] if not _has_final(s)]
    if not retry_specs:
        print("No failed v5 models — every spec has a final ckpt. Nothing to do.")
        return

    print(f"=== {len(retry_specs)} models need retry ===")
    for s in retry_specs:
        print(f"  - {s['short_name']:30s} (parquet: {s['parquet']})")

    # Validate / regenerate parquets for retry candidates.
    print("\n--- Validating parquets ---")
    regen_tasks = []
    for spec in retry_specs:
        pq = parquet_dir / spec["parquet"]
        if _parquet_is_valid(pq):
            print(f"  ✓ {spec['short_name']:30s} parquet OK")
        else:
            print(f"  ✗ {spec['short_name']:30s} parquet INVALID — regenerating")
            if not args.dry_run:
                regen_tasks.append(_regenerate_parquet(spec, pq, train_max_length))
    if regen_tasks:
        print(f"\nRegenerating {len(regen_tasks)} parquets in parallel...")
        await asyncio.gather(*regen_tasks)
        print("Parquet regeneration complete.")

    # Build the retry yaml.
    run_id = args.run_id or ("v5retry_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    retry_yaml = (
        PROJECT_ROOT / "configs" / "auto_generated" / f"ft_v5retry_{run_id}.yaml"
    )
    _build_retry_yaml(v5_cfg, retry_specs, run_id, retry_yaml)
    print(f"\nRetry yaml: {retry_yaml}")
    print(f"Retry run-id: {run_id}")

    # Phase 1b: FT + per-checkpoint evals
    print(f"\n--- Phase 1b: FT + ckpt evals (launch_ft.py) ---")
    rc1 = await _run_launch_ft(
        run_id, retry_yaml,
        datasets_filter=PART_1_DATASETS, eval_only=False,
        log_dir=log_dir, dry_run=args.dry_run, label="part1",
    )

    if rc1 != 0 and not args.dry_run:
        print(f"\nPart 1 launch_ft exited non-zero ({rc1}); skipping Part 2.")
        sys.exit(rc1)

    if args.skip_part_2:
        print("\n--skip-part-2: stopping after Part 1.")
        return

    print(f"\n--- Phase 2: --eval-only on {PART_2_DATASETS} ---")
    rc2 = await _run_launch_ft(
        run_id, retry_yaml,
        datasets_filter=PART_2_DATASETS, eval_only=True,
        log_dir=log_dir, dry_run=args.dry_run, label="part2",
    )

    print("\n" + "=" * 80)
    print(f"RETRY PIPELINE COMPLETE — Run ID: {run_id}")
    print("=" * 80)
    if rc2 != 0 and not args.dry_run:
        print(f"Part 2 exit code: {rc2}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", default=None,
                   help="Override the auto-generated run-id "
                        "(default: v5retry_<UTC timestamp>)")
    p.add_argument("--skip-part-2", action="store_true",
                   help="Run Part 1 only")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(main(parse_args()))
