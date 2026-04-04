#!/usr/bin/env python3
"""Add metr_note_gpt54 monitor scores to existing monitor_qa rollouts.

Same metr_note prompt template but scored by gpt-5.4 (no reasoning).
Appends a MonitorResult entry with monitor_type="metr_note_gpt54".

Skips rollouts that already have a metr_note_gpt54 entry or have actor errors.

Usage:
    uv run python scripts/runs/rescore_metr_note_gpt54.py
    uv run python scripts/runs/rescore_metr_note_gpt54.py --concurrency 200 --dry-run
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
from controllability.evals.prompts import get_monitor_prompt_builder
from controllability.inference.batch import _single_with_retry
from controllability.inference.openrouter import OpenRouterClient
from controllability.types import InferenceRequest

ROLLOUT_DIR = Path(__file__).resolve().parents[2] / "results" / "rollouts" / "monitor_qa"
MONITOR_TYPE = "metr_note_gpt54"


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
    """Add metr_note_gpt54 scores to all eligible rollouts in a single JSONL file."""
    builder = get_monitor_prompt_builder("metr_note")
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

    # Identify which rollouts need scoring
    to_score: list[int] = []
    for i, r in enumerate(rollout_lines):
        if r.get("actor_error"):
            continue
        existing_types = {
            mr.get("monitor_type")
            for mr in r.get("monitor_results", [])
        }
        if MONITOR_TYPE in existing_types:
            continue
        if not r.get("stripped_prompt") and not r.get("stripped_reasoning"):
            continue
        to_score.append(i)

    logging.info(
        "  %s: %d rollouts, %d to score with %s",
        path.name, len(rollout_lines), len(to_score), MONITOR_TYPE,
    )

    if not to_score:
        return {"file": path.name, "total": len(rollout_lines), "scored": 0, "errors": 0}

    completed = [0]
    total = len(to_score)
    errors = 0

    async def score_one(idx: int) -> tuple[int, float | None, str, str | None, float, dict]:
        r = rollout_lines[idx]
        messages = builder.build_monitor_prompt(
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
        score = builder.parse_score(resp.content)
        completed[0] += 1
        c = completed[0]
        if c % 200 == 0 or c == total:
            logging.info("    %s: %d/%d scored", path.name, c, total)
        return idx, score, resp.content, resp.error, resp.latency_ms, resp.usage

    tasks = [score_one(i) for i in to_score]
    results = await asyncio.gather(*tasks)

    for idx, score, response_text, error, latency_ms, usage in results:
        r = rollout_lines[idx]
        if "monitor_results" not in r or r["monitor_results"] is None:
            r["monitor_results"] = []
        r["monitor_results"].append({
            "monitor_type": MONITOR_TYPE,
            "monitor_model": monitor_model,
            "monitor_prompt": [],
            "monitor_response": response_text,
            "monitor_score": score,
            "monitor_latency_ms": latency_ms,
            "monitor_usage": usage,
            "monitor_error": error,
        })
        if error:
            errors += 1

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

    return {"file": path.name, "total": len(rollout_lines), "scored": len(to_score), "errors": errors}


async def main_async(args):
    settings = Settings()
    if not settings.openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY not set. Check your .env file.")

    monitor_client = OpenRouterClient(
        api_key=settings.openrouter_api_key,
        request_timeout=args.timeout,
    )

    files = sorted(f for f in ROLLOUT_DIR.iterdir() if f.suffix == ".jsonl" and f.is_file())
    if not files:
        logging.error("No .jsonl files found in %s", ROLLOUT_DIR)
        return

    logging.info(
        "Scoring %d file(s) with %s monitor (model=%s, concurrency=%d)",
        len(files), MONITOR_TYPE, args.monitor_model, args.concurrency,
    )

    try:
        summaries = []
        for path in files:
            t0 = time.time()
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
    logging.info("%s SCORING COMPLETE", MONITOR_TYPE.upper())
    logging.info("=" * 60)
    total_scored = 0
    total_errors = 0
    for s in summaries:
        total_scored += s["scored"]
        total_errors += s["errors"]
    logging.info("  Total: %d scored, %d errors", total_scored, total_errors)


def main():
    parser = argparse.ArgumentParser(
        description=f"Add {MONITOR_TYPE} monitor scores to existing monitor_qa rollouts"
    )
    parser.add_argument("--monitor-model", type=str, default="openai/gpt-5.4")
    parser.add_argument("--monitor-max-tokens", type=int, default=1024)
    parser.add_argument("--monitor-temperature", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=200)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--backup-dir", type=str, default=None,
                        help="Directory to save backup copies before scoring")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be scored without running")
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
        files = sorted(f for f in ROLLOUT_DIR.iterdir() if f.suffix == ".jsonl" and f.is_file())
        for path in files:
            total = 0
            scoreable = 0
            already_done = 0
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
                    if r.get("actor_error"):
                        continue
                    existing_types = {
                        mr.get("monitor_type")
                        for mr in r.get("monitor_results", [])
                    }
                    if MONITOR_TYPE in existing_types:
                        already_done += 1
                        continue
                    scoreable += 1
            print(f"  {path.name}: {total} total, {scoreable} to score, {already_done} already done")
        return

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
