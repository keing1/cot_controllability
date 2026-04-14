#!/usr/bin/env python3
"""Run reasoning_audit + reasoning_audit_mini in parallel across file groups.

Splits 64 val-split user-prompt files into groups by model, runs 3 groups
concurrently. Within each group, files are processed serially and both
monitor types are scored before moving to the next file (avoids two
processes writing the same file).

Automatically resumes — skips rollouts that already have the monitor score.
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

MONITOR_CONFIGS = {
    "reasoning_audit": "openai/gpt-5.4",
    "reasoning_audit_mini": "openai/gpt-5.4-mini",
}

MODEL_GROUPS = [
    ["gpt-oss-120b-ft60", "gpt-oss-120b", "gpt-oss-20b"],
    ["gpt-oss-20b-ft60", "qwen3-32b", "qwen3-32b-ft60"],
    ["qwen3-8b", "qwen3-8b-ft60"],
]


def get_files_for_models(models: list[str]) -> list[Path]:
    files = []
    for model in models:
        pattern = f"monitor_qa_{model}_user_*val_split*.jsonl"
        files.extend(sorted(ROLLOUT_DIR.glob(pattern)))
    return [f for f in files if f.is_file()]


async def run_group(
    group_id: int,
    models: list[str],
    settings: Settings,
    concurrency: int,
) -> dict:
    files = get_files_for_models(models)
    if not files:
        logging.info("[Group %d] No files found for %s", group_id, models)
        return {"group": group_id, "scored": 0, "errors": 0}

    client = OpenRouterClient(
        api_key=settings.openrouter_api_key,
        request_timeout=120,
    )

    total_scored = 0
    total_errors = 0

    try:
        for path in files:
            for mt, mm in MONITOR_CONFIGS.items():
                builder = get_monitor_prompt_builder(mt)
                t0 = time.time()
                summary = await rescore_file(
                    path,
                    client,
                    monitor_type=mt,
                    monitor_model=mm,
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
                        "[Group %d] %s [%s]: %d scored in %.1fs (%d errors)",
                        group_id, summary["file"], mt, summary["scored"],
                        elapsed, summary["errors"],
                    )
    finally:
        await client.close()

    return {"group": group_id, "scored": total_scored, "errors": total_errors}


async def main():
    settings = Settings()
    if not settings.openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    concurrency_per_group = 200

    # Show plan
    for i, models in enumerate(MODEL_GROUPS):
        files = get_files_for_models(models)
        logging.info("Group %d: %s → %d files", i, models, len(files))

    t0 = time.time()
    results = await asyncio.gather(*[
        run_group(i, models, settings, concurrency_per_group)
        for i, models in enumerate(MODEL_GROUPS)
    ])

    elapsed = time.time() - t0
    total_scored = sum(r["scored"] for r in results)
    total_errors = sum(r["errors"] for r in results)

    logging.info("")
    logging.info("=" * 60)
    logging.info("REASONING AUDIT PARALLEL SCORING COMPLETE (%.0fs)", elapsed)
    logging.info("=" * 60)
    for r in results:
        logging.info("  Group %d: %d scored, %d errors", r["group"], r["scored"], r["errors"])
    logging.info("  Total: %d scored, %d errors", total_scored, total_errors)


if __name__ == "__main__":
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)

    asyncio.run(main())
