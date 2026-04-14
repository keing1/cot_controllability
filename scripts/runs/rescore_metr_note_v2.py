#!/usr/bin/env python3
"""Rescore all user-prompt val-split rollouts with metr_note_v2 monitor.

Runs 3 groups concurrently. Processes the 24 newest rollouts (RUN_ID_V2)
first, then the 64 older rollouts (RUN_ID).

Automatically resumes — skips rollouts that already have metr_note_v2 score.

Usage:
    uv run python scripts/runs/rescore_metr_note_v2.py --dry-run
    uv run python scripts/runs/rescore_metr_note_v2.py
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from controllability.config import Settings
from controllability.evals.prompts import get_monitor_prompt_builder
from controllability.evals.rescore import rescore_file
from controllability.inference.openrouter import OpenRouterClient

ROLLOUT_DIR = Path(__file__).resolve().parents[2] / "results" / "rollouts" / "monitor_qa"

MONITOR_TYPE = "metr_note_v2"
MONITOR_MODEL = "openai/gpt-5.4"

RUN_ID_V2 = "20260414"
RUN_ID = "20260409"

# Process newest first, then older
MODEL_GROUPS_NEW = [
    ["gpt-oss-120b-ft60", "gpt-oss-120b", "gpt-oss-20b"],
    ["gpt-oss-20b-ft60", "qwen3-32b", "qwen3-32b-ft60"],
    ["qwen3-8b", "qwen3-8b-ft60"],
]


def get_files_for_models(models: list[str], run_id: str) -> list[Path]:
    files = []
    for model in models:
        pattern = f"monitor_qa_{model}_user_*val_split*{run_id}*.jsonl"
        files.extend(sorted(ROLLOUT_DIR.glob(pattern)))
    return [f for f in files if f.is_file()]


async def run_group(
    group_id: int,
    models: list[str],
    run_id: str,
    phase: str,
    settings: Settings,
    concurrency: int,
) -> dict:
    files = get_files_for_models(models, run_id)
    if not files:
        logging.info("[%s Group %d] No files found for %s", phase, group_id, models)
        return {"group": group_id, "phase": phase, "scored": 0, "errors": 0}

    client = OpenRouterClient(
        api_key=settings.openrouter_api_key,
        request_timeout=120,
    )

    builder = get_monitor_prompt_builder(MONITOR_TYPE)
    total_scored = 0
    total_errors = 0

    try:
        for path in files:
            t0 = time.time()
            summary = await rescore_file(
                path,
                client,
                monitor_type=MONITOR_TYPE,
                monitor_model=MONITOR_MODEL,
                builder=builder,
                monitor_max_tokens=1024,
                monitor_temperature=0.0,
                concurrency=concurrency,
                max_retries=3,
            )
            elapsed = time.time() - t0
            total_scored += summary["scored"]
            total_errors += summary["errors"]
            if summary["scored"] > 0:
                logging.info(
                    "[%s Group %d] %s: %d scored in %.1fs (%d errors)",
                    phase, group_id, summary["file"], summary["scored"],
                    elapsed, summary["errors"],
                )
    finally:
        await client.close()

    return {"group": group_id, "phase": phase, "scored": total_scored, "errors": total_errors}


async def main(dry_run: bool = False):
    settings = Settings()
    if not settings.openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    concurrency_per_group = 200

    # Show plan
    for phase, run_id in [("NEW", RUN_ID_V2), ("OLD", RUN_ID)]:
        for i, models in enumerate(MODEL_GROUPS_NEW):
            files = get_files_for_models(models, run_id)
            logging.info("[%s Group %d] %s → %d files", phase, i, models, len(files))

    if dry_run:
        from controllability.evals.rescore import dry_run as dr
        logging.info("\n--- NEW rollouts (RUN_ID_V2) ---")
        dr(ROLLOUT_DIR, f"monitor_qa_*_user_*val_split*{RUN_ID_V2}*.jsonl", MONITOR_TYPE)
        logging.info("\n--- OLD rollouts (RUN_ID) ---")
        dr(ROLLOUT_DIR, f"monitor_qa_*_user_*val_split*{RUN_ID}*.jsonl", MONITOR_TYPE)
        return

    # Phase 1: newest 24 rollouts
    logging.info("\n=== Phase 1: Scoring %s rollouts (newest 24) ===", RUN_ID_V2)
    t0 = time.time()
    results_new = await asyncio.gather(*[
        run_group(i, models, RUN_ID_V2, "NEW", settings, concurrency_per_group)
        for i, models in enumerate(MODEL_GROUPS_NEW)
    ])
    elapsed_new = time.time() - t0
    scored_new = sum(r["scored"] for r in results_new)
    errors_new = sum(r["errors"] for r in results_new)
    logging.info("Phase 1 done: %d scored, %d errors in %.0fs", scored_new, errors_new, elapsed_new)

    # Phase 2: older 64 rollouts
    logging.info("\n=== Phase 2: Scoring %s rollouts (older 64) ===", RUN_ID)
    t1 = time.time()
    results_old = await asyncio.gather(*[
        run_group(i, models, RUN_ID, "OLD", settings, concurrency_per_group)
        for i, models in enumerate(MODEL_GROUPS_NEW)
    ])
    elapsed_old = time.time() - t1
    scored_old = sum(r["scored"] for r in results_old)
    errors_old = sum(r["errors"] for r in results_old)
    logging.info("Phase 2 done: %d scored, %d errors in %.0fs", scored_old, errors_old, elapsed_old)

    # Summary
    total_elapsed = time.time() - t0
    logging.info("")
    logging.info("=" * 60)
    logging.info("METR NOTE V2 SCORING COMPLETE (%.0fs total)", total_elapsed)
    logging.info("=" * 60)
    logging.info("  Phase 1 (new): %d scored, %d errors", scored_new, errors_new)
    logging.info("  Phase 2 (old): %d scored, %d errors", scored_old, errors_old)
    logging.info("  Total: %d scored, %d errors", scored_new + scored_old, errors_new + errors_old)


if __name__ == "__main__":
    import argparse
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
