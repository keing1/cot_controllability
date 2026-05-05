#!/usr/bin/env python3
"""Rescore all monitor_qa rollouts from the 2026-04-30 v4 run with the new
``metr_note_v3`` monitor (graded by gpt-5.4 instant).

Coverage:
    7 models × {base + 000050 + 000100 + 000300 + final} × {baseline,
    less_reasoning_v2, vocab_control, dont_think_v2}
    = 7 × 5 × 4 = 140 files

Usage:
    uv run python scripts/runs/rescore_metr_note_v3_2026_04_30.py --dry-run
    uv run python scripts/runs/rescore_metr_note_v3_2026_04_30.py
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

# Total in-flight budget = PARALLEL_FILES × PER_FILE_CONCURRENCY ≈ 5000
PARALLEL_FILES = 10
PER_FILE_CONCURRENCY = 500

RUN_ID = "20260430_113831"
CKPTS = ["base", "000050", "000100", "000300", "final"]
VARIANTS = ["baseline", "less_reasoning_v2", "vocab_control", "dont_think_v2"]

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


def _find_one(pattern: str, root: Path) -> Path | None:
    matches = sorted(root.glob(pattern))
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(
            f"Multiple matches for {pattern} under {root}:\n  "
            + "\n  ".join(m.name for m in matches)
        )
    return matches[0]


def _mqa_path(model: str, ckpt: str, label: str) -> Path | None:
    """All v4 monitor_qa files share one filename pattern keyed on ckpt name."""
    return _find_one(
        f"monitor_qa_{FT_PREFIX[model]}-ft-{RUN_ID}-{ckpt}_{label}_*.jsonl",
        MQA_DIR,
    )


def collect_files() -> tuple[list[tuple[str, str, str, Path]], list[str]]:
    """Return (matched_files, missing_descriptions)."""
    out: list[tuple[str, str, str, Path]] = []
    missing: list[str] = []
    for model in MODEL_GROUPS:
        for ckpt in CKPTS:
            for variant in VARIANTS:
                p = _mqa_path(model, ckpt, variant)
                if p is None:
                    missing.append(f"{model}/{ckpt}/{variant}")
                else:
                    out.append((model, ckpt, variant, p))
    return out, missing


async def main(dry_run: bool, parallel_files: int, per_file_concurrency: int) -> None:
    settings = Settings()
    if not settings.openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    files, missing = collect_files()
    logging.info(
        "Collected %d / %d files. Rescore: %s (%s, instant). "
        "Per-file concurrency=%d, parallel_files=%d → in-flight budget≈%d",
        len(files), len(MODEL_GROUPS) * len(CKPTS) * len(VARIANTS),
        MONITOR_TYPE, MONITOR_MODEL,
        per_file_concurrency, parallel_files,
        per_file_concurrency * parallel_files,
    )
    if missing:
        logging.warning("%d expected files were missing:", len(missing))
        for m in missing[:25]:
            logging.warning("  - %s", m)
        if len(missing) > 25:
            logging.warning("  ... and %d more", len(missing) - 25)

    if dry_run:
        from controllability.evals.rescore import dry_run as dr
        total = 0
        for _, _, _, path in files:
            n_score = dr(path.parent, path.name, MONITOR_TYPE)
            total += n_score
        logging.info("Dry-run: %d rollouts would be scored across %d files",
                     total, len(files))
        return

    client = OpenRouterClient(
        api_key=settings.openrouter_api_key,
        request_timeout=120,
    )
    builder = get_monitor_prompt_builder(MONITOR_TYPE)

    file_sem = asyncio.Semaphore(parallel_files)
    total_scored = 0
    total_errors = 0
    finished = 0
    t0 = time.time()

    async def process_one(idx: int, model: str, ckpt: str, variant: str,
                           path: Path) -> dict:
        nonlocal total_scored, total_errors, finished
        async with file_sem:
            ts = time.time()
            summary = await rescore_file(
                path,
                client,
                monitor_type=MONITOR_TYPE,
                monitor_model=MONITOR_MODEL,
                builder=builder,
                monitor_max_tokens=1024,
                monitor_temperature=0.0,
                concurrency=per_file_concurrency,
                max_retries=3,
            )
            elapsed = time.time() - ts
            total_scored += summary["scored"]
            total_errors += summary["errors"]
            finished += 1
            logging.info(
                "[%3d/%3d done] %-18s %-7s %-20s: %d scored in %.1fs "
                "(%d errors) — totals: %d scored, %d errors",
                finished, len(files), model, ckpt, variant,
                summary["scored"], elapsed, summary["errors"],
                total_scored, total_errors,
            )
            return summary

    try:
        tasks = [
            process_one(idx, model, ckpt, variant, path)
            for idx, (model, ckpt, variant, path) in enumerate(files, 1)
        ]
        await asyncio.gather(*tasks)
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
    parser.add_argument("--per-file-concurrency", type=int,
                        default=PER_FILE_CONCURRENCY,
                        help=f"In-flight calls per file (default: {PER_FILE_CONCURRENCY})")
    parser.add_argument("--parallel-files", type=int, default=PARALLEL_FILES,
                        help=f"Files processed in parallel (default: {PARALLEL_FILES})")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    asyncio.run(main(
        dry_run=args.dry_run,
        parallel_files=args.parallel_files,
        per_file_concurrency=args.per_file_concurrency,
    ))
