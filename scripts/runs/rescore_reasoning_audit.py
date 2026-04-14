#!/usr/bin/env python3
"""Add reasoning_audit and reasoning_audit_mini scores to existing monitor_qa rollouts.

Thin wrapper around the generic rescore module that runs both monitor types
in a single pass over each file.

Usage:
    uv run python scripts/runs/rescore_reasoning_audit.py --dry-run
    uv run python scripts/runs/rescore_reasoning_audit.py --file-glob "*user_*val_split*.jsonl"
    uv run python scripts/runs/rescore_reasoning_audit.py --concurrency 500
    uv run python scripts/runs/rescore_reasoning_audit.py --monitors reasoning_audit
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from controllability.config import Settings
from controllability.evals.prompts import get_monitor_prompt_builder
from controllability.evals.rescore import dry_run, rescore_file
from controllability.inference.openrouter import OpenRouterClient

ROLLOUT_DIR = Path(__file__).resolve().parents[2] / "results" / "rollouts" / "monitor_qa"

MONITOR_CONFIGS = {
    "reasoning_audit": "openai/gpt-5.4",
    "reasoning_audit_mini": "openai/gpt-5.4-mini",
}


async def main_async(args):
    settings = Settings()
    if not settings.openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY not set. Check your .env file.")

    monitor_client = OpenRouterClient(
        api_key=settings.openrouter_api_key,
        request_timeout=args.timeout,
    )

    monitor_types = [m.strip() for m in args.monitors.split(",")]
    for mt in monitor_types:
        if mt not in MONITOR_CONFIGS:
            raise ValueError(f"Unknown monitor type: {mt}. Available: {list(MONITOR_CONFIGS.keys())}")

    files = sorted(ROLLOUT_DIR.glob(args.file_glob))
    files = [f for f in files if f.is_file()]
    if not files:
        logging.error("No files match %s in %s", args.file_glob, ROLLOUT_DIR)
        return

    logging.info(
        "Scoring %d file(s) with %s (concurrency=%d)",
        len(files), monitor_types, args.concurrency,
    )

    try:
        summaries = []
        for path in files:
            for mt in monitor_types:
                model = MONITOR_CONFIGS[mt]
                builder = get_monitor_prompt_builder(mt)
                t0 = time.time()
                summary = await rescore_file(
                    path,
                    monitor_client,
                    monitor_type=mt,
                    monitor_model=model,
                    builder=builder,
                    monitor_max_tokens=args.monitor_max_tokens,
                    monitor_temperature=args.monitor_temperature,
                    concurrency=args.concurrency,
                    max_retries=args.max_retries,
                    backup_dir=Path(args.backup_dir) if args.backup_dir else None,
                    rescore_le=args.rescore_le,
                )
                elapsed = time.time() - t0
                summary["elapsed_s"] = round(elapsed, 1)
                summary["monitor_type"] = mt
                summaries.append(summary)
                logging.info(
                    "  %s [%s]: %d scored in %.1fs (%d errors)",
                    summary["file"], mt, summary["scored"], elapsed, summary["errors"],
                )
    finally:
        await monitor_client.close()

    logging.info("")
    logging.info("=" * 60)
    logging.info("REASONING AUDIT SCORING COMPLETE")
    logging.info("=" * 60)
    total_scored = sum(s["scored"] for s in summaries)
    total_errors = sum(s["errors"] for s in summaries)
    logging.info("  Total: %d scored, %d errors", total_scored, total_errors)


def main():
    parser = argparse.ArgumentParser(
        description="Add reasoning_audit monitor scores to existing monitor_qa rollouts"
    )
    parser.add_argument("--monitors", type=str, default="reasoning_audit,reasoning_audit_mini",
                        help="Comma-separated monitor types to score")
    parser.add_argument("--monitor-max-tokens", type=int, default=1024)
    parser.add_argument("--monitor-temperature", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=500)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--backup-dir", type=str, default=None)
    parser.add_argument("--file-glob", type=str, default="*user_*val_split*.jsonl",
                        help="Glob pattern for files to process")
    parser.add_argument("--rescore-le", type=float, default=None)
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

    if args.dry_run:
        monitor_types = [m.strip() for m in args.monitors.split(",")]
        for mt in monitor_types:
            print(f"\n--- {mt} ({MONITOR_CONFIGS[mt]}) ---")
            dry_run(ROLLOUT_DIR, args.file_glob, mt, args.rescore_le)
        return

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
