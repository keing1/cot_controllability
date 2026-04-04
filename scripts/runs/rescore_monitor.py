#!/usr/bin/env python3
"""Re-score existing monitor_qa rollouts with an updated monitor prompt.

Reads rollout JSONL files, re-runs only the monitor scoring step (no actor
inference), and writes updated files. Old monitor scores are preserved in a
``monitor_score_old`` field on each rollout.

Usage:
    uv run python scripts/runs/rescore_monitor.py results/rollouts/monitor_qa/monitor_qa_*.jsonl
    uv run python scripts/runs/rescore_monitor.py --concurrency 120 results/rollouts/monitor_qa/*.jsonl
    uv run python scripts/runs/rescore_monitor.py --dry-run results/rollouts/monitor_qa/*.jsonl
    uv run python scripts/runs/rescore_monitor.py --backup-dir results/rollouts/monitor_qa/old_scores/ ...
"""

import argparse
import asyncio
import json
import logging
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from controllability.config import Settings
from controllability.evals.grading import parse_monitor_score
from controllability.evals.prompts import get_monitor_prompt_builder
from controllability.inference.openrouter import OpenRouterClient
from controllability.types import InferenceRequest


async def rescore_file(
    path: Path,
    monitor_client: OpenRouterClient,
    *,
    monitor_model: str,
    monitor_max_tokens: int,
    monitor_temperature: float,
    concurrency: int,
    max_retries: int,
    backup_dir: Path | None,
) -> dict:
    """Re-score all non-error rollouts in a single JSONL file.

    Returns a summary dict with counts.
    """
    from controllability.inference.batch import _single_with_retry

    monitor_prompt_builder = get_monitor_prompt_builder("old_monitor")
    sem = asyncio.Semaphore(concurrency)

    # Read all lines
    header_lines: list[str] = []
    rollout_lines: list[dict] = []
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                header_lines.append(line)
                continue
            try:
                rollout_lines.append(json.loads(stripped))
            except json.JSONDecodeError:
                continue

    # Identify which rollouts need rescoring (have stripped fields, no actor error)
    to_rescore: list[int] = []
    for i, r in enumerate(rollout_lines):
        if r.get("actor_error"):
            continue
        # Need stripped fields to build monitor prompt
        if not r.get("stripped_prompt") and not r.get("stripped_reasoning"):
            continue
        to_rescore.append(i)

    logging.info(
        "  %s: %d rollouts, %d to rescore",
        path.name, len(rollout_lines), len(to_rescore),
    )

    if not to_rescore:
        return {"file": path.name, "total": len(rollout_lines), "rescored": 0, "errors": 0}

    # Build monitor requests
    async def rescore_one(idx: int) -> tuple[int, float | None, str, str | None]:
        r = rollout_lines[idx]
        messages = monitor_prompt_builder.build_monitor_prompt(
            r.get("stripped_prompt", ""),
            r.get("stripped_reasoning", ""),
            r.get("stripped_response", ""),
        )
        request = InferenceRequest(
            messages=messages,
            model=monitor_model,
            max_tokens=monitor_max_tokens,
            temperature=monitor_temperature,
        )
        resp = await _single_with_retry(monitor_client, request, sem, max_retries)
        score = parse_monitor_score(resp.content)
        return idx, score, resp.content, resp.error

    # Run all rescoring concurrently
    completed = 0
    total = len(to_rescore)
    errors = 0

    tasks = [rescore_one(i) for i in to_rescore]
    results = await asyncio.gather(*tasks)

    for idx, score, response_text, error in results:
        r = rollout_lines[idx]
        # Preserve old scores
        r["monitor_score_old"] = r.get("monitor_score")
        r["monitor_response_old"] = r.get("monitor_response")
        # Set new scores
        r["monitor_score"] = score
        r["monitor_response"] = response_text
        r["monitor_error"] = error
        if error:
            errors += 1
        completed += 1
        if completed % 100 == 0 or completed == total:
            logging.info("    %s: %d/%d rescored", path.name, completed, total)

    # Backup original file
    if backup_dir:
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / path.name
        shutil.copy2(path, backup_path)
        logging.info("  Backed up to %s", backup_path)

    # Write updated file
    with open(path, "w") as f:
        for line in header_lines:
            f.write(line if line.endswith("\n") else line + "\n")
        for r in rollout_lines:
            f.write(json.dumps(r) + "\n")

    return {"file": path.name, "total": len(rollout_lines), "rescored": completed, "errors": errors}


async def main_async(args):
    settings = Settings()
    if not settings.openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY not set. Check your .env file.")

    monitor_client = OpenRouterClient(
        api_key=settings.openrouter_api_key,
        request_timeout=args.timeout,
    )

    files = [Path(f) for f in args.files]
    files = [f for f in files if f.exists() and f.suffix == ".jsonl"]

    if not files:
        logging.error("No valid .jsonl files provided")
        return

    logging.info(
        "Rescoring %d file(s) with model=%s, concurrency=%d",
        len(files), args.monitor_model, args.concurrency,
    )

    try:
        summaries = []
        for path in files:
            summary = await rescore_file(
                path,
                monitor_client,
                monitor_model=args.monitor_model,
                monitor_max_tokens=args.monitor_max_tokens,
                monitor_temperature=args.monitor_temperature,
                concurrency=args.concurrency,
                max_retries=args.max_retries,
                backup_dir=Path(args.backup_dir) if args.backup_dir else None,
            )
            summaries.append(summary)
    finally:
        await monitor_client.close()

    # Print summary
    logging.info("")
    logging.info("=" * 60)
    logging.info("RESCORE COMPLETE")
    logging.info("=" * 60)
    total_rescored = 0
    total_errors = 0
    for s in summaries:
        logging.info(
            "  %s: %d/%d rescored (%d errors)",
            s["file"], s["rescored"], s["total"], s["errors"],
        )
        total_rescored += s["rescored"]
        total_errors += s["errors"]
    logging.info("  Total: %d rescored, %d errors", total_rescored, total_errors)


def main():
    parser = argparse.ArgumentParser(description="Re-score monitor_qa rollouts with updated prompt")
    parser.add_argument("files", nargs="+", help="JSONL rollout files to rescore")
    parser.add_argument("--monitor-model", type=str, default="openai/gpt-4o")
    parser.add_argument("--monitor-max-tokens", type=int, default=1024)
    parser.add_argument("--monitor-temperature", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=100)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--backup-dir", type=str, default=None,
                        help="Directory to save backup copies of original files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be rescored without running")
    args = parser.parse_args()

    # Setup logging
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)

    if args.dry_run:
        files = [Path(f) for f in args.files if Path(f).exists() and Path(f).suffix == ".jsonl"]
        for path in files:
            total = 0
            scoreable = 0
            with open(path) as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    try:
                        r = json.loads(s)
                    except json.JSONDecodeError:
                        continue
                    total += 1
                    if not r.get("actor_error"):
                        scoreable += 1
            print(f"  {path.name}: {total} total, {scoreable} to rescore")
        return

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
