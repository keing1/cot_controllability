#!/usr/bin/env python3
"""CLI wrapper for generic rescore — add any monitor type to existing rollouts.

Usage:
    uv run python scripts/runs/rescore.py --monitor-type reasoning_audit --monitor-model openai/gpt-5.4 --dry-run
    uv run python scripts/runs/rescore.py --monitor-type reasoning_audit --monitor-model openai/gpt-5.4 --file-glob "*user_*val_split*.jsonl"
    uv run python scripts/runs/rescore.py --monitor-type reasoning_audit_mini --monitor-model openai/gpt-5.4-mini --concurrency 500
    uv run python scripts/runs/rescore.py --monitor-type metr_note_gpt54 --monitor-model openai/gpt-5.4 --rescore-le 10
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


async def main_async(args):
    settings = Settings()
    if not settings.openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY not set. Check your .env file.")

    monitor_client = OpenRouterClient(
        api_key=settings.openrouter_api_key,
        request_timeout=args.timeout,
    )

    builder = get_monitor_prompt_builder(args.monitor_type)

    files = sorted(ROLLOUT_DIR.glob(args.file_glob))
    files = [f for f in files if f.is_file()]
    if not files:
        logging.error("No files match %s in %s", args.file_glob, ROLLOUT_DIR)
        return

    logging.info(
        "Scoring %d file(s) with %s (model=%s, concurrency=%d)",
        len(files), args.monitor_type, args.monitor_model, args.concurrency,
    )

    try:
        summaries = []
        for path in files:
            t0 = time.time()
            summary = await rescore_file(
                path,
                monitor_client,
                monitor_type=args.monitor_type,
                monitor_model=args.monitor_model,
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
            summaries.append(summary)
            logging.info(
                "  %s: %d scored in %.1fs (%d errors)",
                summary["file"], summary["scored"], elapsed, summary["errors"],
            )
    finally:
        await monitor_client.close()

    logging.info("")
    logging.info("=" * 60)
    logging.info("%s SCORING COMPLETE", args.monitor_type.upper())
    logging.info("=" * 60)
    total_scored = sum(s["scored"] for s in summaries)
    total_errors = sum(s["errors"] for s in summaries)
    logging.info("  Total: %d scored, %d errors", total_scored, total_errors)


def main():
    parser = argparse.ArgumentParser(
        description="Generic rescore — add any monitor type to existing monitor_qa rollouts"
    )
    parser.add_argument("--monitor-type", type=str, required=True,
                        help="Monitor type name (e.g. reasoning_audit, metr_note_gpt54)")
    parser.add_argument("--monitor-model", type=str, required=True,
                        help="Model to use for scoring (e.g. openai/gpt-5.4)")
    parser.add_argument("--monitor-max-tokens", type=int, default=1024)
    parser.add_argument("--monitor-temperature", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=500)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--backup-dir", type=str, default=None)
    parser.add_argument("--file-glob", type=str, default="*.jsonl")
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

    try:
        get_monitor_prompt_builder(args.monitor_type)
    except ValueError as e:
        print(f"Error: {e}")
        return

    if args.dry_run:
        dry_run(ROLLOUT_DIR, args.file_glob, args.monitor_type, args.rescore_le)
        return

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
