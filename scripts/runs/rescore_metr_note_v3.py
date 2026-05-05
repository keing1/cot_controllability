#!/usr/bin/env python3
"""Rescore the monitor_qa rollouts that feed mqa_stacked_per_model_v2
(2026-04-20 plot) with the new metr_note_v3 monitor.

Covers:
    7 models × {Base, FT} × {baseline, less_reasoning_v2, vocab_control,
                              dont_think_v2}
    = 56 files total.

Baseline files are also included because the plotting pipeline uses them
for 5%-FPR threshold calibration — without v3 scores on them,
threshold_5pct_fpr() returns None and all downstream evasion rates collapse.

Monitor model: openai/gpt-5.4 with reasoning_effort=None → the *instant*
(non-thinking) gpt-5.4 variant (see OPTIONAL_REASONING_PATTERNS in
src/controllability/inference/openrouter.py).

Usage:
    uv run python scripts/runs/rescore_metr_note_v3.py --dry-run
    uv run python scripts/runs/rescore_metr_note_v3.py
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

# None → use hand-picked BASE_TOKEN-based paths (old 4)
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

# Hand-picked base-model baseline files for old 4 (old baselines don't use
# the `user_` prefix pattern used by the side-task variants).
BASE_MQA_BASELINE = {
    "GPT-OSS-20B":  "monitor_qa_gpt-oss-20b_baseline_val_split_20260414_175620_a025af2725fd.jsonl",
    "GPT-OSS-120B": "monitor_qa_gpt-oss-120b_baseline_val_split_20260414_175620_32b2af202ff9.jsonl",
    "Qwen3-8B":     "monitor_qa_qwen3-8b_baseline_val_split_20260414_175620_e8ca076915e8.jsonl",
    "Qwen3-32B":    "monitor_qa_qwen3-32b_baseline_val_split_20260414_175620_2b53e2ef1d8c.jsonl",
}

VARIANTS_ALL = ["baseline", "less_reasoning_v2", "vocab_control", "dont_think_v2"]


def _find_one(pattern: str, root: Path) -> Path:
    matches = sorted(root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No match for {pattern} under {root}")
    if len(matches) > 1:
        raise ValueError(
            f"Multiple matches for {pattern} under {root}:\n  "
            + "\n  ".join(m.name for m in matches)
        )
    return matches[0]


def _ft_mqa(model: str, label: str) -> Path:
    rid = FT_RUN_ID[model]
    return _find_one(
        f"monitor_qa_{FT_PREFIX[model]}-ft-{rid}-final_{label}_*.jsonl",
        MQA_DIR,
    )


def _base_mqa(model: str, label: str) -> Path:
    rid = BASE_RUN_ID[model]
    if rid is None:
        if label == "baseline":
            p = MQA_DIR / BASE_MQA_BASELINE[model]
            if not p.exists():
                raise FileNotFoundError(p)
            return p
        token = BASE_TOKEN[model]
        return _find_one(
            f"monitor_qa_{token}_user_{label}_val_split_*.jsonl", MQA_DIR,
        )
    return _find_one(
        f"monitor_qa_{FT_PREFIX[model]}-ft-{rid}-base_{label}_*.jsonl",
        MQA_DIR,
    )


def collect_files() -> list[tuple[str, str, str, Path]]:
    """Return (model, split, variant, path) for the 56 files."""
    out: list[tuple[str, str, str, Path]] = []
    for model in MODEL_GROUPS:
        for variant in VARIANTS_ALL:
            out.append((model, "Base", variant, _base_mqa(model, variant)))
            out.append((model, "FT",   variant, _ft_mqa(model, variant)))
    return out


async def main(dry_run: bool) -> None:
    settings = Settings()
    if not settings.openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    files = collect_files()
    logging.info("Collected %d files to rescore with %s (%s, instant):",
                 len(files), MONITOR_TYPE, MONITOR_MODEL)
    for model, split, variant, path in files:
        logging.info("  %-18s %-4s %-20s → %s", model, split, variant, path.name)

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
