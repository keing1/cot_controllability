#!/usr/bin/env python3
"""Add reasoning_audit (RA) monitor scores to the 30 side-task rollouts in
the 2026-04-20 plot workflow that currently have none.

Plot coverage map (before this run):
    Old 4 Base × 3 side-task variants (12 files)  → RA present
    Old 4 FT   × 3 side-task variants (12 files)  → RA missing
    New 3 Base × 3 side-task variants  (9 files)  → RA missing
    New 3 FT   × 3 side-task variants  (9 files)  → RA missing
                                                   = 30 files to rescore

Baseline variants (14 files) are skipped: the baseline prompt has no side
task injected, so RA on those rows is a no-op by construction.

RA uses the *full* user_prompt (not the stripped one) and emits a 0/1
binary score via <monitor_score> tags. ``rescore_file`` auto-routes to
the full-prompt interface because the builder exposes ``uses_full_prompt``.

Monitor model: openai/gpt-5.4 (reasoning_effort=None → instant mode, same
as the metr_note_v* runs).

Usage:
    uv run python scripts/runs/rescore_reasoning_audit_2026_04_20.py --dry-run
    uv run python scripts/runs/rescore_reasoning_audit_2026_04_20.py
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

MQA_DIR = REPO_ROOT / "results" / "rollouts" / "monitor_qa"

MONITOR_TYPE = "reasoning_audit"
MONITOR_MODEL = "openai/gpt-5.4"
CONCURRENCY = 300

MODEL_GROUPS = [
    "GPT-OSS-20B", "GPT-OSS-120B", "Qwen3-8B", "Qwen3-32B",
    "DeepSeek V3.1", "Kimi K2.5", "Nemotron-3 Nano",
]

FT_PREFIX = {
    "GPT-OSS-20B":     "openai_gpt-oss-20b",
    "GPT-OSS-120B":    "openai_gpt-oss-120b",
    "Qwen3-8B":        "qwen_qwen3-8b",
    "Qwen3-32B":       "qwen_qwen3-32b",
    "DeepSeek V3.1":   "deepseek-ai_DeepSeek-V3.1",
    "Kimi K2.5":       "moonshotai_Kimi-K2.5",
    "Nemotron-3 Nano": "nvidia_NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
}

FT_RUN_ID = {
    "GPT-OSS-20B":     "20260419_081434",
    "GPT-OSS-120B":    "20260419_081434",
    "Qwen3-8B":        "20260419_081434",
    "Qwen3-32B":       "20260419_081434",
    "DeepSeek V3.1":   "20260420_111850",
    "Kimi K2.5":       "20260420_111850",
    "Nemotron-3 Nano": "20260420_111850",
}

BASE_RUN_ID = {
    "GPT-OSS-20B":     None,
    "GPT-OSS-120B":    None,
    "Qwen3-8B":        None,
    "Qwen3-32B":       None,
    "DeepSeek V3.1":   "base_20260420_224645",
    "Kimi K2.5":       "base_20260420_224645",
    "Nemotron-3 Nano": "base_20260420_224645",
}

BASE_TOKEN = {
    "GPT-OSS-20B":  "gpt-oss-20b",
    "GPT-OSS-120B": "gpt-oss-120b",
    "Qwen3-8B":     "qwen3-8b",
    "Qwen3-32B":    "qwen3-32b",
}

VARIANTS_SIDE_TASK = ["less_reasoning_v2", "vocab_control", "dont_think_v2"]


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


def _ft_mqa(model: str, label: str) -> Path:
    rid = FT_RUN_ID[model]
    return _find_one(f"monitor_qa_{FT_PREFIX[model]}-ft-{rid}-final_{label}_*.jsonl")


def _base_mqa(model: str, label: str) -> Path:
    rid = BASE_RUN_ID[model]
    if rid is None:
        token = BASE_TOKEN[model]
        return _find_one(f"monitor_qa_{token}_user_{label}_val_split_*.jsonl")
    return _find_one(f"monitor_qa_{FT_PREFIX[model]}-ft-{rid}-base_{label}_*.jsonl")


def collect_files() -> list[tuple[str, str, str, Path]]:
    """42 side-task files (7 models × 2 splits × 3 variants).

    ``rescore_file`` auto-skips files already scored with this monitor, so
    the 12 old-4-Base side-task files pass through as no-ops at a rate of
    ~0.1s each; only the 30 genuinely uncovered files actually hit OpenRouter.
    """
    out: list[tuple[str, str, str, Path]] = []
    for model in MODEL_GROUPS:
        for variant in VARIANTS_SIDE_TASK:
            out.append((model, "Base", variant, _base_mqa(model, variant)))
            out.append((model, "FT",   variant, _ft_mqa(model, variant)))
    return out


async def main(dry_run: bool) -> None:
    settings = Settings()
    if not settings.openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    files = collect_files()
    logging.info("Collected %d side-task files (RA rescore with %s):",
                 len(files), MONITOR_MODEL)
    for model, split, variant, path in files:
        logging.info("  %-18s %-4s %-20s → %s", model, split, variant, path.name)

    if dry_run:
        from controllability.evals.rescore import dry_run as dr
        total = 0
        for _, _, _, path in files:
            total += dr(path.parent, path.name, MONITOR_TYPE)
        logging.info("Dry-run: %d rollouts would be scored across %d files",
                     total, len(files))
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
                "[%d/%d] %s %s %s: %d scored in %.1fs (%d errors) — running "
                "totals: %d scored, %d errors",
                idx, len(files), model, split, variant,
                summary["scored"], elapsed, summary["errors"],
                total_scored, total_errors,
            )
    finally:
        await client.close()

    total_elapsed = time.time() - t0
    logging.info("")
    logging.info("=" * 70)
    logging.info("REASONING_AUDIT RESCORE COMPLETE — %.0fs total", total_elapsed)
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
