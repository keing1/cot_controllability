#!/usr/bin/env python3
"""Rescore the monitor_qa rollouts backing
`results/summaries/plots/2026_04_13/top3_stacked_per_model_gpt54.png`
with the new metr_note_v3 monitor (GPT-5.4 instant, non-thinking).

Scope restricted by user request to the 4 prompt variants of interest
plus baseline (needed for 5%-FPR threshold calibration):
    baseline, less_reasoning_v2, gray_swan, vocab_control, dont_think_v2

Base-model (non-FT) scores already exist for baseline, less_reasoning_v2,
vocab_control, and dont_think_v2 (run by rescore_metr_note_v3.py for the
2026-04-20 plot). So we rescore:

    Base: gray_swan only          (4 files)
    FT:   all 5 (baseline + 4)    (20 files)
                                  -----
                                  24 files

Monitor: openai/gpt-5.4 with reasoning_effort=None → "instant" mode (see
OPTIONAL_REASONING_PATTERNS in src/controllability/inference/openrouter.py).

Usage:
    uv run python scripts/runs/rescore_metr_note_v3_top3_plot.py --dry-run
    uv run python scripts/runs/rescore_metr_note_v3_top3_plot.py
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from controllability.config import Settings
from controllability.evals.prompts import get_monitor_prompt_builder
from controllability.evals.rescore import rescore_file
from controllability.inference.openrouter import OpenRouterClient

ROLLOUTS_DIR = REPO_ROOT / "results" / "rollouts"
MQA_DIR = ROLLOUTS_DIR / "monitor_qa"

MONITOR_TYPE = "metr_note_v3"
MONITOR_MODEL = "openai/gpt-5.4"   # reasoning_effort=None → "instant" mode
CONCURRENCY = 300

MODEL_GROUPS = ["GPT-OSS-20B", "GPT-OSS-120B", "Qwen3-8B", "Qwen3-32B"]

TOKENS_BASE = {
    "GPT-OSS-20B":  "gpt-oss-20b",
    "GPT-OSS-120B": "gpt-oss-120b",
    "Qwen3-8B":     "qwen3-8b",
    "Qwen3-32B":    "qwen3-32b",
}
TOKENS_FT = {
    "GPT-OSS-20B":  "gpt-oss-20b-ft60",
    "GPT-OSS-120B": "gpt-oss-120b-ft60",
    "Qwen3-8B":     "qwen3-8b-ft60",
    "Qwen3-32B":    "qwen3-32b-ft60",
}

RUN_ID_ORIG = "20260409_061534"      # gray_swan
RUN_ID_V2 = "20260414_023958"        # *_v2 + vocab_control
RUN_ID_BASELINE = "20260414_175620"  # val-split baselines

VARIANT_RUN_ID = {
    "baseline":               RUN_ID_BASELINE,
    "user_gray_swan":         RUN_ID_ORIG,
    "user_less_reasoning_v2": RUN_ID_V2,
    "user_vocab_control":     RUN_ID_V2,
    "user_dont_think_v2":     RUN_ID_V2,
}


def _find_one(pattern: str) -> Path:
    matches = sorted(MQA_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No match for {pattern} under {MQA_DIR}")
    if len(matches) > 1:
        raise ValueError(
            f"Multiple matches for {pattern}:\n  "
            + "\n  ".join(m.name for m in matches)
        )
    return matches[0]


def _path(token: str, variant: str) -> Path:
    rid = VARIANT_RUN_ID[variant]
    return _find_one(f"monitor_qa_{token}_{variant}_val_split_{rid}_*.jsonl")


def collect_files() -> list[tuple[str, str, str, Path]]:
    """Return (model, split, variant, path) for the 24 files in scope."""
    out: list[tuple[str, str, str, Path]] = []
    for group in MODEL_GROUPS:
        # Base: only gray_swan (baseline + other 3 already scored by
        # rescore_metr_note_v3.py).
        out.append((
            group, "Base", "user_gray_swan",
            _path(TOKENS_BASE[group], "user_gray_swan"),
        ))
        # FT: baseline + all 4 prompt variants
        for variant in VARIANT_RUN_ID:
            out.append((
                group, "FT", variant, _path(TOKENS_FT[group], variant),
            ))
    return out


async def main(dry_run: bool) -> None:
    settings = Settings()
    if not settings.openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    files = collect_files()
    logging.info(
        "Collected %d files to rescore with %s (%s, instant):",
        len(files), MONITOR_TYPE, MONITOR_MODEL,
    )
    for model, split, variant, path in files:
        logging.info("  %-14s %-4s %-22s → %s", model, split, variant, path.name)

    if dry_run:
        from controllability.evals.rescore import dry_run as dr
        total = 0
        for _, _, _, path in files:
            n_score = dr(path.parent, path.name, MONITOR_TYPE)
            total += n_score
        logging.info(
            "Dry-run: %d rollouts would be scored across %d files",
            total, len(files),
        )
        return

    client = OpenRouterClient(
        api_key=settings.openrouter_api_key,
        request_timeout=120,
    )
    builder = get_monitor_prompt_builder(MONITOR_TYPE)

    total_scored = 0
    total_errors = 0
    t0 = time.time()
    try:
        for idx, (model, split, variant, path) in enumerate(files, 1):
            ts = time.time()
            summary = await rescore_file(
                path,
                client,
                monitor_type=MONITOR_TYPE,
                monitor_model=MONITOR_MODEL,
                builder=builder,
                monitor_max_tokens=1024,
                monitor_temperature=0.0,
                concurrency=CONCURRENCY,
                max_retries=3,
            )
            elapsed = time.time() - ts
            total_scored += summary["scored"]
            total_errors += summary["errors"]
            logging.info(
                "[%d/%d] %s %s %s: %d scored in %.1fs (%d errors) — "
                "running totals: %d scored, %d errors",
                idx, len(files), model, split, variant,
                summary["scored"], elapsed, summary["errors"],
                total_scored, total_errors,
            )
    finally:
        await client.close()

    total_elapsed = time.time() - t0
    logging.info("")
    logging.info("=" * 70)
    logging.info("METR_NOTE_V3 RESCORE COMPLETE — %.0fs total", total_elapsed)
    logging.info("=" * 70)
    logging.info("  Files:  %d", len(files))
    logging.info("  Scored: %d", total_scored)
    logging.info("  Errors: %d", total_errors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)

    asyncio.run(main(dry_run=args.dry_run))
