#!/usr/bin/env python3
"""Smoke test the new include_instruction flag end-to-end.

Builds 14 small SFT datasets:
  7 base models × {with-instruction, without-instruction}
  5 rows per mode × 15 modes (6 reasonif + 9 cotcontrol) = 75 rows per dataset.

Outputs go to results/sft/smoke_with_without/.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Match other production runners (e.g. run_5k_v4_pipeline.py, launch_ft.py):
# load .env so TINKER_API_KEY + OPENROUTER_API_KEY are visible.
from dotenv import load_dotenv  # noqa: E402
load_dotenv(PROJECT_ROOT / ".env")

from controllability.training.sft_builder import (  # noqa: E402
    BuildConfig,
    SourceRequest,
    build_sft_dataset,
)


DEFAULT_QUESTION_POOL = (
    PROJECT_ROOT / "external" / "reasonIF_ft" / "train-00000-of-00001.parquet"
)


@dataclass
class ModelSpec:
    short: str
    base_model: str           # what to send to inference backend
    backend: str              # "openrouter" | "tinker"
    reasoning_effort: str | None = None
    analysis_channel: bool | None = None  # None → auto-detect


# Mirrors configs/ft_mixed_5k_v4_7models.yaml.
#
# The v4 production pipeline (scripts/runs/run_5k_v4_pipeline.py) uses
# backend="tinker" for SFT generation across all 7 models — Tinker calls the
# raw HF model directly, so the slugs below are HF paths for everything that
# isn't natively on OpenRouter. gpt-oss + qwen3 stay on openrouter where they
# work cleanly and are quicker.
MODELS: list[ModelSpec] = [
    ModelSpec("gpt-oss-20b",    "openai/gpt-oss-20b",                              "openrouter", reasoning_effort="medium"),
    ModelSpec("gpt-oss-120b",   "openai/gpt-oss-120b",                             "openrouter", reasoning_effort="medium"),
    ModelSpec("qwen3-8b",       "qwen/qwen3-8b",                                   "openrouter"),
    ModelSpec("qwen3-32b",      "qwen/qwen3-32b",                                  "openrouter"),
    ModelSpec("deepseek-v3.1",  "deepseek-ai/DeepSeek-V3.1",                       "tinker"),
    ModelSpec("kimi-k2.5",      "moonshotai/Kimi-K2.5",                            "tinker"),
    ModelSpec("nemotron3-nano", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",      "tinker"),
]


def _config_for(
    model: ModelSpec, include_instruction: bool, output_dir: Path,
    rows_per_mode: int,
) -> BuildConfig:
    n_reasonif = rows_per_mode * 6   # 6 reasonif modes
    n_cotcontrol = rows_per_mode * 9 # 9 cotcontrol modes
    sources = [
        SourceRequest(name="reasonif", count=n_reasonif),
        SourceRequest(name="cotcontrol", count=n_cotcontrol),
    ]
    variant = "with" if include_instruction else "without"
    stem = f"{model.short}-smoke-{variant}-instr"
    return BuildConfig(
        output_path=output_dir / f"{stem}.parquet",
        checkpoint_path=output_dir / f"{stem}.checkpoint.jsonl",
        question_source_parquet=DEFAULT_QUESTION_POOL,
        base_model=model.base_model,
        backend=model.backend,
        reasoning_effort=model.reasoning_effort,
        max_tokens=12000,
        request_timeout=600,
        max_concurrency=80,
        sources=sources,
        editor_model="openai/gpt-5.4",
        analysis_channel=model.analysis_channel,
        include_instruction_at_generation=include_instruction,
        seed=42,
        max_transform_retries=2,
        run_id=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
    )


async def _run_one(model: ModelSpec, include: bool, output_dir: Path,
                   rows_per_mode: int) -> tuple[str, dict | str]:
    label = f"{model.short} ({'with' if include else 'without'} instr)"
    cfg = _config_for(model, include, output_dir, rows_per_mode)
    t0 = time.monotonic()
    print(f"\n>>>>>> START {label}")
    try:
        summary = await build_sft_dataset(cfg)
    except Exception as e:  # noqa: BLE001
        elapsed = time.monotonic() - t0
        print(f"<<<<<< FAIL  {label} ({elapsed:.0f}s): {type(e).__name__}: {e}")
        return label, f"ERROR: {type(e).__name__}: {e}"
    elapsed = time.monotonic() - t0
    print(f"<<<<<< DONE  {label} ({elapsed:.0f}s)")
    return label, summary


async def main_async(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.only:
        wanted = set(args.only.split(","))
        models = [m for m in MODELS if m.short in wanted]
    else:
        models = list(MODELS)

    if args.with_only:
        variants = [True]
    elif args.without_only:
        variants = [False]
    else:
        variants = [False, True]

    print(f"Models: {[m.short for m in models]}")
    print(f"Variants: {['with' if v else 'without' for v in variants]}")
    print(f"Rows per mode: {args.rows_per_mode}  (so {args.rows_per_mode * 15} per dataset)")
    print(f"Output: {output_dir}")

    summaries: list[tuple[str, dict | str]] = []
    if args.parallel:
        coros = [
            _run_one(m, v, output_dir, args.rows_per_mode)
            for m in models for v in variants
        ]
        summaries = await asyncio.gather(*coros, return_exceptions=False)
    else:
        for m in models:
            for v in variants:
                summaries.append(await _run_one(m, v, output_dir, args.rows_per_mode))

    print("\n\n===== SMOKE SUMMARY =====")
    for label, s in summaries:
        if isinstance(s, str):
            print(f"  {label:50s}  {s}")
        else:
            n = s.get("n_rows", "?")
            comp = s.get("n_compliant", "?")
            print(f"  {label:50s}  rows={n}  compliant={comp}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="results/sft/smoke_with_without")
    p.add_argument("--rows-per-mode", type=int, default=5)
    p.add_argument("--only", default=None,
                   help="Comma-separated subset of model short names")
    p.add_argument("--parallel", action="store_true",
                   help="Run all builds concurrently (heavy on quota)")
    p.add_argument("--with-only", action="store_true",
                   help="Only run the with-instruction variant")
    p.add_argument("--without-only", action="store_true",
                   help="Only run the without-instruction variant")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
