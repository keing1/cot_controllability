#!/usr/bin/env python3
"""Add main_task_grading scores to existing monitor_qa rollouts.

Uses GPT-5.4 to judge whether the model's main task answer is correct.
Appends a MonitorResult entry with monitor_type="main_task_grading".

For baseline mode: the "student's answer" is the full response text.
For side_task mode: the "student's answer" is the extracted <main_task_answer> content.

Requires that the monitorqa dataset file has main_task_answer fields, or that
the correct answer can be looked up by main_id from source datasets.

Skips rollouts that already have a main_task_grading entry or have actor errors.

Usage:
    uv run python scripts/runs/rescore_main_task.py --dry-run
    uv run python scripts/runs/rescore_main_task.py --concurrency 400
    uv run python scripts/runs/rescore_main_task.py --backup-dir results/rollouts/monitor_qa/pre_main_task_grading/
"""

import argparse
import asyncio
import json
import logging
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from controllability.config import Settings
from controllability.evals.grading import MainTaskGradingPromptBuilder
from controllability.inference.batch import _single_with_retry
from controllability.inference.openrouter import OpenRouterClient
from controllability.types import InferenceRequest

ROLLOUT_DIR = Path(__file__).resolve().parents[2] / "results" / "rollouts" / "monitor_qa"
MONITOR_TYPE = "main_task_grading"

# ---------------------------------------------------------------------------
# Correct answer lookup
# ---------------------------------------------------------------------------

_ANSWER_LOOKUP: dict[str, str] | None = None


def _build_answer_lookup() -> dict[str, str]:
    """Build main_id -> correct_answer mapping from all monitorqa dataset files."""
    global _ANSWER_LOOKUP
    if _ANSWER_LOOKUP is not None:
        return _ANSWER_LOOKUP

    lookup = {}
    monitorqa_dir = Path(__file__).resolve().parents[2] / "results" / "monitorqa"
    for jsonl_path in monitorqa_dir.glob("*.jsonl"):
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                mid = d.get("metadata", {}).get("main_id", "")
                answer = d.get("main_task_answer", "")
                if mid and answer:
                    lookup[mid] = answer

    logging.info("Built answer lookup: %d entries", len(lookup))
    _ANSWER_LOOKUP = lookup
    return lookup


def _get_correct_answer(rollout: dict) -> str:
    """Get the correct answer for a rollout's main task."""
    # Check if the sample already has it
    sample = rollout.get("sample", {})
    meta = sample.get("metadata", {})
    answer = meta.get("main_task_answer", "") or sample.get("correct_answer", "")
    if answer:
        return answer

    # Fall back to lookup by main_id
    main_id = meta.get("main_id", "")
    if main_id:
        lookup = _build_answer_lookup()
        return lookup.get(main_id, "")

    return ""


def _get_model_answer(rollout: dict) -> str:
    """Get the model's main task answer from a rollout."""
    mode = rollout.get("mode", "")
    if mode == "baseline":
        # Full response for baseline
        return rollout.get("response", "")
    else:
        # Extract from main_task_response or from tags
        main_resp = rollout.get("main_task_response", "")
        if main_resp:
            return main_resp
        # Fallback: extract from response tags
        response = rollout.get("response", "")
        match = re.search(
            r"<main_task_answer>(.*?)</main_task_answer>", response, re.DOTALL,
        )
        if match:
            return match.group(1).strip()
        return response


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


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
    """Add main_task_grading scores to all eligible rollouts in a JSONL file."""
    builder = MainTaskGradingPromptBuilder()
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
    skipped_no_answer = 0
    for i, r in enumerate(rollout_lines):
        if r.get("actor_error"):
            continue
        has_valid = any(
            mr.get("monitor_type") == MONITOR_TYPE and mr.get("monitor_score") is not None
            for mr in r.get("monitor_results", [])
        )
        if has_valid:
            continue
        # Need both correct answer and model answer
        correct = _get_correct_answer(r)
        model_ans = _get_model_answer(r)
        if not correct:
            skipped_no_answer += 1
            continue
        if not model_ans:
            continue
        to_score.append(i)

    logging.info(
        "  %s: %d rollouts, %d to score, %d skipped (no correct answer)",
        path.name, len(rollout_lines), len(to_score), skipped_no_answer,
    )

    if not to_score:
        return {"file": path.name, "total": len(rollout_lines), "scored": 0, "errors": 0}

    completed = [0]
    total = len(to_score)

    async def score_one(idx: int) -> tuple[int, float | None, str, str | None, float, dict]:
        r = rollout_lines[idx]
        question = r.get("sample", {}).get("question", "")
        correct_answer = _get_correct_answer(r)
        model_answer = _get_model_answer(r)

        messages = builder.build_grading_prompt(question, correct_answer, model_answer)
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
        if c % 500 == 0 or c == total:
            logging.info("    %s: %d/%d scored", path.name, c, total)
        return idx, score, resp.content, resp.error, resp.latency_ms, resp.usage

    tasks = [score_one(i) for i in to_score]
    results = await asyncio.gather(*tasks)

    errors = 0
    for idx, score, response_text, error, latency_ms, usage in results:
        r = rollout_lines[idx]
        if "monitor_results" not in r or r["monitor_results"] is None:
            r["monitor_results"] = []
        # Drop any existing stale (None-score) entry for this monitor type
        r["monitor_results"] = [
            mr for mr in r["monitor_results"]
            if mr.get("monitor_type") != MONITOR_TYPE
        ]
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
        # Set main_task_correct on the rollout
        if score is not None:
            r["main_task_correct"] = score == 1.0
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

    files = sorted(ROLLOUT_DIR.glob(args.file_glob))
    files = [f for f in files if f.is_file()]
    if not files:
        logging.error("No files match %s in %s", args.file_glob, ROLLOUT_DIR)
        return

    logging.info(
        "Scoring %d file(s) with %s (model=%s, concurrency=%d)",
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
    logging.info("MAIN TASK GRADING COMPLETE")
    logging.info("=" * 60)
    total_scored = 0
    total_errors = 0
    for s in summaries:
        total_scored += s["scored"]
        total_errors += s["errors"]
    logging.info("  Total: %d scored, %d errors", total_scored, total_errors)


def main():
    parser = argparse.ArgumentParser(
        description="Add main_task_grading scores to existing monitor_qa rollouts"
    )
    parser.add_argument("--monitor-model", type=str, default="openai/gpt-5.4")
    parser.add_argument("--monitor-max-tokens", type=int, default=16384)
    parser.add_argument("--monitor-temperature", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=400)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--backup-dir", type=str, default=None)
    parser.add_argument("--file-glob", type=str, default="*.jsonl",
                        help="Glob pattern (relative to rollouts dir) to restrict which files are processed")
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
        _build_answer_lookup()  # noqa: preload for lookup
        files = sorted(ROLLOUT_DIR.glob(args.file_glob))
        files = [f for f in files if f.is_file()]
        total_to_score = 0
        for path in files:
            total = 0
            scoreable = 0
            already_done = 0
            no_answer = 0
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
                    has_valid = any(
                        mr.get("monitor_type") == MONITOR_TYPE
                        and mr.get("monitor_score") is not None
                        for mr in r.get("monitor_results", [])
                    )
                    if has_valid:
                        already_done += 1
                        continue
                    correct = _get_correct_answer(r)
                    if not correct:
                        no_answer += 1
                        continue
                    scoreable += 1
            total_to_score += scoreable
            if scoreable:
                print(f"  {path.name}: {total} total, {scoreable} to score, {already_done} done, {no_answer} no answer")
        print(f"\n  Total to score: {total_to_score}")
        return

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
