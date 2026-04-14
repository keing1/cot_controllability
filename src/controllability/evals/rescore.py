"""Reusable rescore logic — add any monitor type to existing monitor_qa rollouts.

Automatically detects the monitor's interface (stripped prompt, full prompt,
or grading) and passes the right fields.
"""

import asyncio
import json
import logging
import shutil
from pathlib import Path

from controllability.evals.prompts import get_monitor_prompt_builder
from controllability.inference.batch import _single_with_retry
from controllability.inference.openrouter import OpenRouterClient
from controllability.types import InferenceRequest


def build_monitor_messages(builder, rollout: dict) -> list[dict[str, str]]:
    """Build monitor messages using the right interface for the builder type."""
    if getattr(builder, "is_grading_monitor", False):
        correct_answer = rollout.get("sample", {}).get("metadata", {}).get("main_task_answer", "")
        mode = rollout.get("mode", "")
        if mode == "baseline":
            model_answer = rollout.get("response", "")
        elif mode == "baseline_formatted":
            model_answer = rollout.get("main_task_response", "") or rollout.get("response", "")
        else:
            model_answer = rollout.get("main_task_response", "")
        question = rollout.get("sample", {}).get("question", "")
        return builder.build_grading_prompt(question, correct_answer, model_answer)
    elif getattr(builder, "uses_full_prompt", False):
        return builder.build_monitor_prompt(
            rollout.get("user_prompt", ""),
            rollout.get("reasoning", ""),
        )
    else:
        return builder.build_monitor_prompt(
            rollout.get("stripped_prompt", ""),
            rollout.get("reasoning", ""),
            rollout.get("stripped_response", ""),
        )


async def rescore_file(
    path: Path,
    monitor_client: OpenRouterClient,
    *,
    monitor_type: str,
    monitor_model: str,
    builder=None,
    monitor_max_tokens: int = 1024,
    monitor_temperature: float = 0.0,
    concurrency: int = 500,
    max_retries: int = 3,
    backup_dir: Path | None = None,
    rescore_le: float | None = None,
) -> dict:
    """Add monitor scores to all eligible rollouts in a single JSONL file.

    Returns a summary dict with keys: file, total, scored, errors.
    """
    if builder is None:
        builder = get_monitor_prompt_builder(monitor_type)

    sem = asyncio.Semaphore(concurrency)

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

    to_score: list[int] = []
    for i, r in enumerate(rollout_lines):
        if r.get("actor_error"):
            continue
        existing_score = None
        has_any = False
        for mr in r.get("monitor_results", []):
            if mr.get("monitor_type") == monitor_type:
                has_any = True
                existing_score = mr.get("monitor_score")
                break
        if has_any and existing_score is not None:
            if rescore_le is None or existing_score > rescore_le:
                continue
        to_score.append(i)

    logging.info(
        "  %s: %d rollouts, %d to score with %s (%s)",
        path.name, len(rollout_lines), len(to_score), monitor_type, monitor_model,
    )

    if not to_score:
        return {"file": path.name, "total": len(rollout_lines), "scored": 0, "errors": 0}

    completed = [0]
    total = len(to_score)
    errors = 0

    async def score_one(idx: int) -> tuple[int, float | None, str, str | None, float, dict]:
        r = rollout_lines[idx]
        messages = build_monitor_messages(builder, r)
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
        r["monitor_results"] = [
            mr for mr in r["monitor_results"]
            if mr.get("monitor_type") != monitor_type
        ]
        r["monitor_results"].append({
            "monitor_type": monitor_type,
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

    if backup_dir:
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / path.name
        shutil.copy2(path, backup_path)
        logging.info("  Backed up to %s", backup_path)

    with open(path, "w") as f:
        for line in header_lines:
            f.write(line if line.endswith("\n") else line + "\n")
        for r in rollout_lines:
            f.write(json.dumps(r) + "\n")

    return {"file": path.name, "total": len(rollout_lines), "scored": len(to_score), "errors": errors}


def dry_run(
    rollout_dir: Path,
    file_glob: str,
    monitor_type: str,
    rescore_le: float | None = None,
) -> int:
    """Print scoring plan and return total count of rollouts to score."""
    files = sorted(rollout_dir.glob(file_glob))
    files = [f for f in files if f.is_file()]
    grand_score = 0
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
                existing_score = None
                has_any = False
                for mr in r.get("monitor_results", []):
                    if mr.get("monitor_type") == monitor_type:
                        has_any = True
                        existing_score = mr.get("monitor_score")
                        break
                if has_any and existing_score is not None:
                    if rescore_le is None or existing_score > rescore_le:
                        already_done += 1
                        continue
                scoreable += 1
        grand_score += scoreable
        if scoreable:
            print(f"  {path.name}: {total} total, {scoreable} to score, {already_done} already done")
    print(f"\nTotal to score: {grand_score}")
    return grand_score
