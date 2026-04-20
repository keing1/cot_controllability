#!/usr/bin/env python3
"""End-to-end driver for the 4-model × 1000-row mixed FT run.

Wraps the two phases described in ``configs/ft_mixed_1000.yaml``:

  Phase 0 — build the four mixed SFT parquets (500 reasonif + 500 cotcontrol),
            one per base model, using ``build_sft_dataset.py`` internally.
            Skipped for any model whose parquet already exists unless
            ``--force-rebuild`` is passed.

  Phase 1 — launch training + continuous per-checkpoint evals via
            ``launch_ft.py``, with the same eval task set used by the
            existing reasonif fine-tunes (cotcontrol n=100 + reasonif all).

The goal is a single command that matches the spec:
  * qwen3-8b, qwen3-32b, gpt-oss-20b, gpt-oss-120b
  * 500 reasonif + 500 cotcontrol rows
  * gpt-oss uses medium reasoning effort; qwen uses none
  * save_fraction=0.25 → four checkpoints per training run
  * 1 epoch

Usage:
  python scripts/runs/run_mixed_1000_ft.py
  python scripts/runs/run_mixed_1000_ft.py --dry-run
  python scripts/runs/run_mixed_1000_ft.py --skip-build           # parquets ready
  python scripts/runs/run_mixed_1000_ft.py --force-rebuild        # regenerate SFT
  python scripts/runs/run_mixed_1000_ft.py --models qwen3-8b      # subset
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import yaml  # noqa: E402

from controllability.training.sft_builder import (  # noqa: E402
    BuildConfig,
    SourceRequest,
    build_sft_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "ft_mixed_1000.yaml"
DEFAULT_QUESTION_POOL = PROJECT_ROOT / "external" / "reasonIF_ft" / "train-00000-of-00001.parquet"


def _analysis_channel_for(base_model: str) -> bool:
    return "gpt-oss" in base_model.lower()


def _sft_build_config(
    spec: dict, parquet_path: Path, run_id: str,
    reasonif_count: int, cotcontrol_count: int,
    seed: int,
) -> BuildConfig:
    ac = _analysis_channel_for(spec["model"])
    checkpoint = parquet_path.with_suffix(".checkpoint.jsonl")
    reasoning_effort = spec.get("reasoning_effort", "none")
    if reasoning_effort == "none":
        reasoning_effort = None
    return BuildConfig(
        output_path=parquet_path,
        checkpoint_path=checkpoint,
        question_source_parquet=DEFAULT_QUESTION_POOL,
        base_model=spec["model"],
        backend="tinker",
        reasoning_effort=reasoning_effort,
        sources=[
            SourceRequest(name="reasonif", count=reasonif_count),
            SourceRequest(name="cotcontrol", count=cotcontrol_count),
        ],
        analysis_channel=ac,
        seed=seed,
        run_id=run_id,
    )


async def _build_phase(
    cfg: dict, models_filter: set[str] | None,
    reasonif_count: int, cotcontrol_count: int,
    seed: int, force_rebuild: bool, dry_run: bool,
) -> None:
    parquet_dir = Path(cfg.get("parquet_dir", "results/sft"))
    parquet_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    for spec in cfg["models"]:
        short = spec.get("short_name") or spec["model"].split("/")[-1]
        if models_filter and short not in models_filter:
            continue
        pq_path = parquet_dir / spec["parquet"]
        if pq_path.exists() and not force_rebuild:
            print(f"[build] {short}: {pq_path.name} exists — skipping (use --force-rebuild to regenerate)")
            continue
        print(f"\n[build] {short}: generating {pq_path.name}")
        if dry_run:
            print(f"  DRY RUN — would call build_sft_dataset for {spec['model']}")
            continue
        build_cfg = _sft_build_config(
            spec, pq_path, run_id,
            reasonif_count, cotcontrol_count, seed,
        )
        await build_sft_dataset(build_cfg)


def _launch_phase(
    config_path: Path, models_filter: set[str] | None,
    eval_only: bool, dry_run: bool, extra_args: list[str],
) -> int:
    import subprocess
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "runs" / "launch_ft.py"),
        "--config", str(config_path),
    ]
    if models_filter:
        cmd += ["--models", ",".join(sorted(models_filter))]
    if eval_only:
        cmd.append("--eval-only")
    if dry_run:
        cmd.append("--dry-run")
    cmd += extra_args
    print("\n[launch] $", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Driver: build 4 mixed SFT parquets, then run FT + evals")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="YAML config (default: configs/ft_mixed_1000.yaml)")
    parser.add_argument("--models", default=None,
                        help="Comma-separated short names; omit to use all in config")
    parser.add_argument("--reasonif-count", type=int, default=500)
    parser.add_argument("--cotcontrol-count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-build", action="store_true",
                        help="Assume SFT parquets already exist")
    parser.add_argument("--force-rebuild", action="store_true",
                        help="Rebuild SFT parquets even if present")
    parser.add_argument("--eval-only", action="store_true",
                        help="Pass --eval-only to launch_ft (uses existing checkpoints)")
    parser.add_argument("--dry-run", action="store_true")

    args, extra = parser.parse_known_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    models_filter = set(args.models.split(",")) if args.models else None

    # Phase 0: build SFT parquets
    if not args.skip_build:
        asyncio.run(_build_phase(
            cfg, models_filter,
            args.reasonif_count, args.cotcontrol_count,
            args.seed, args.force_rebuild, args.dry_run,
        ))

    # Phase 1: launch training + evals
    rc = _launch_phase(
        args.config, models_filter,
        args.eval_only, args.dry_run, extra,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
