#!/usr/bin/env python3
"""Run baseline_main_and_side eval: prefill baseline_formatted rollouts with
a dual-task prompt and let the model complete just the side task answer.

For each of 8 models, reads the baseline_formatted val-split rollout, builds
a DontThink-style prompt (solve both tasks, don't think about side task in
reasoning), prefills the assistant turn with the original reasoning +
reformatted main task answer, and lets the model generate the side task answer.

This measures: can the model answer the side task when it has already spent
its reasoning budget on the main task?

Usage:
    uv run python scripts/runs/run_baseline_main_and_side.py --dry-run
    uv run python scripts/runs/run_baseline_main_and_side.py
    uv run python scripts/runs/run_baseline_main_and_side.py --n-samples 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from controllability.config import MonitorQAConfig, Settings
from controllability.evals.grading import (
    extract_main_task_response,
    extract_side_task_response,
    extract_task_answer_response,
    grade_side_task,
    has_extraneous_text,
)
from controllability.evals.prompts import (
    DontThinkMonitorQAPromptBuilder,
    get_monitor_prompt_builder,
)
from controllability.inference.base import InferenceClient
from controllability.inference.batch import _single_with_retry
from controllability.inference.openrouter import OpenRouterClient
from controllability.rollouts.store import (
    append_monitor_rollouts,
    load_monitor_completed_keys,
    write_experiment_header,
)
from controllability.types import (
    InferenceRequest,
    InferenceResponse,
    MonitorResult,
    MonitorRollout,
    Sample,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "rollouts" / "monitor_qa"
ROLLOUT_DIR = RESULTS_DIR / "rollouts" / "monitor_qa"
DATASET_PATH = str(RESULTS_DIR / "monitorqa" / "monitorqa_val.jsonl")

# ---------------------------------------------------------------------------
# Models — same 8 as baseline_formatted
# ---------------------------------------------------------------------------

BASE_MODELS = [
    {
        "label": "gpt-oss-20b",
        "actor_model": "openai/gpt-oss-20b",
        "model_path": None,
        "reasoning_effort": "medium",
    },
    {
        "label": "gpt-oss-120b",
        "actor_model": "openai/gpt-oss-120b",
        "model_path": None,
        "reasoning_effort": "medium",
    },
    {
        "label": "qwen3-8b",
        "actor_model": "qwen/qwen3-8b",
        "model_path": None,
        "reasoning_effort": "medium",
    },
    {
        "label": "qwen3-32b",
        "actor_model": "qwen/qwen3-32b",
        "model_path": None,
        "reasoning_effort": "medium",
    },
]

FT_MODELS = [
    {
        "label": "gpt-oss-20b-ft60",
        "actor_model": "openai/gpt-oss-20b",
        "model_path": "tinker://5ec3cd52-fe1e-579b-ae83-7bd8260330ec:train:0/sampler_weights/000060",
        "reasoning_effort": "medium",
    },
    {
        "label": "gpt-oss-120b-ft60",
        "actor_model": "openai/gpt-oss-120b",
        "model_path": "tinker://f745ff57-2fcf-54ad-9c19-3dfab62598c5:train:0/sampler_weights/000060",
        "reasoning_effort": "medium",
    },
    {
        "label": "qwen3-8b-ft60",
        "actor_model": "qwen/qwen3-8b",
        "model_path": "tinker://e3fea85f-5797-5a24-8c5e-af48cd0ad866:train:0/sampler_weights/000060",
        "reasoning_effort": "medium",
    },
    {
        "label": "qwen3-32b-ft60",
        "actor_model": "qwen/qwen3-32b",
        "model_path": "tinker://eb355212-c7a1-5f36-9db6-c526e1f33298:train:0/sampler_weights/000060",
        "reasoning_effort": "medium",
    },
]

ALL_MODELS = BASE_MODELS + FT_MODELS

MODE = "baseline_main_and_side"
HIGH_CONCURRENCY = 1000
LOW_CONCURRENCY = 200
ERROR_THRESHOLD = 0.20
MONITOR_TYPES = ["metr_note", "metr_note_gpt54", "main_task_grading"]
MONITOR_MODEL_OVERRIDES = {
    "metr_note_gpt54": "openai/gpt-5.4",
    "main_task_grading": "openai/gpt-5.4",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

prompt_builder = DontThinkMonitorQAPromptBuilder()


def load_baseline_rollouts(label: str) -> list[dict]:
    """Load non-errored rollouts from the baseline_formatted val-split file."""
    pattern = f"monitor_qa_{label}_baseline_formatted_val_split_*.jsonl"
    files = sorted(ROLLOUT_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No baseline_formatted file found for {label}: {pattern}"
        )
    filepath = files[-1]
    logging.info("Reading baseline_formatted rollouts from %s", filepath.name)

    rollouts = []
    for line in open(filepath):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        try:
            r = json.loads(s)
        except json.JSONDecodeError:
            continue
        if r.get("actor_error"):
            continue
        rollouts.append(r)
    return rollouts


def build_prefill(rollout: dict, model: str) -> str:
    """Build the assistant prefill from a baseline_formatted rollout.

    Format: original reasoning + <main_task_answer>answer</main_task_answer>
    + \\n<side_task_answer>\\n

    For Qwen models, reasoning is wrapped in <think>...</think> tags.
    For GPT-OSS models, reasoning is included as plain text.
    """
    reasoning = rollout.get("reasoning", "")
    response = rollout.get("response", "")

    # Extract the answer from <task_answer> tags (baseline_formatted format)
    task_match = re.search(r"<task_answer>(.*?)</task_answer>", response, re.DOTALL)
    if task_match:
        main_answer = task_match.group(1).strip()
    else:
        main_answer = response.strip()

    # Build prefill with reasoning + reformatted main answer + start of side answer
    parts = []
    if reasoning:
        if "qwen" in model.lower():
            parts.append(f"<think>{reasoning}</think>")
        else:
            parts.append(reasoning)

    parts.append(f"<main_task_answer>\n{main_answer}\n</main_task_answer>")
    parts.append("<side_task_answer>\n")

    return "\n".join(parts)


def build_messages_for_rollout(
    rollout: dict, model: str,
) -> tuple[list[dict[str, str]], str, Sample]:
    """Build messages + prefill for one rollout.

    Returns (messages, prefill_text, sample). The prefill is passed via
    InferenceRequest.prefill (not as an assistant message, which hangs
    Tinker).
    """
    sample_data = rollout["sample"]
    sample = Sample(**sample_data)

    sys_prompt, usr_prompt = prompt_builder.build_actor_prompt(
        sample, "side_task", model=model,
    )

    prefill = build_prefill(rollout, model)

    messages = []
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": usr_prompt})

    return messages, prefill, sample


def _create_actor_client(model_cfg: dict, settings: Settings) -> InferenceClient:
    """Create the actor inference client."""
    if model_cfg["model_path"]:
        from controllability.inference.tinker_client import TinkerClient
        return TinkerClient(
            model=model_cfg["actor_model"],
            model_path=model_cfg["model_path"],
            reasoning_effort=model_cfg["reasoning_effort"],
            request_timeout=900,
        )
    else:
        from controllability.inference.tinker_client import TinkerClient
        return TinkerClient(
            model=model_cfg["actor_model"],
            model_path=None,
            reasoning_effort=model_cfg["reasoning_effort"],
            request_timeout=900,
        )


def _create_monitor_client(settings: Settings) -> OpenRouterClient:
    """Create the monitor inference client."""
    return OpenRouterClient(
        api_key=settings.openrouter_api_key,
        request_timeout=900,
    )


def count_errors(filepath: Path) -> tuple[int, int]:
    errors = 0
    total = 0
    with open(filepath) as f:
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
                errors += 1
    return errors, total


def remove_errored_lines(filepath: Path) -> int:
    header_lines: list[str] = []
    keep_lines: list[str] = []
    removed = 0
    with open(filepath) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                header_lines.append(line)
                continue
            try:
                r = json.loads(s)
            except json.JSONDecodeError:
                continue
            if r.get("actor_error"):
                removed += 1
            else:
                keep_lines.append(json.dumps(r) + "\n")
    with open(filepath, "w") as f:
        for line in header_lines:
            f.write(line if line.endswith("\n") else line + "\n")
        for line in keep_lines:
            f.write(line)
    return removed


# ---------------------------------------------------------------------------
# Core async pipeline
# ---------------------------------------------------------------------------

async def run_model(
    model_cfg: dict,
    run_id: str,
    n_samples: int | None,
    concurrency: int,
) -> str:
    """Run baseline_main_and_side for one model."""
    label = model_cfg["label"]
    model = model_cfg["actor_model"]
    settings = Settings()

    # Load baseline_formatted rollouts
    try:
        base_rollouts = load_baseline_rollouts(label)
    except FileNotFoundError as e:
        logging.error("SKIP %s: %s", label, e)
        return f"SKIP ({e})"

    logging.info("%s: loaded %d non-errored baseline_formatted rollouts", label, len(base_rollouts))

    if n_samples and n_samples < len(base_rollouts):
        base_rollouts = base_rollouts[:n_samples]
        logging.info("%s: subsampled to %d", label, len(base_rollouts))

    # Build output path
    custom_filename = (
        f"monitor_qa_{label}_{MODE}_val_split"
        f"_{run_id}.jsonl"
    )
    output_path = OUTPUT_DIR / custom_filename

    # Write header
    write_experiment_header(output_path, {
        "type": "monitor_qa",
        "eval_type": MODE,
        "actor_model": model,
        "model_path": model_cfg["model_path"],
        "source": f"baseline_formatted rollouts for {label}",
        "monitor_types": MONITOR_TYPES,
        "run_id": run_id,
    })

    # Check completed keys for resume
    completed_keys = load_monitor_completed_keys(output_path)

    # Create clients
    actor_client = _create_actor_client(model_cfg, settings)
    monitor_client = _create_monitor_client(settings)

    # Build monitor prompt builders
    monitor_builders = {mt: get_monitor_prompt_builder(mt) for mt in MONITOR_TYPES}
    monitor_type_order = list(monitor_builders.keys())

    actor_sem = asyncio.Semaphore(concurrency)
    monitor_sem = asyncio.Semaphore(60)

    all_rollouts: list[MonitorRollout] = []
    buffer: list[MonitorRollout] = []
    buffer_lock = asyncio.Lock()
    saved_count = 0
    saved_keys: set[tuple[str, str]] = set(completed_keys)
    total = len(base_rollouts)
    completed_count = [0]
    save_every = 100

    def _flush_sync(force: bool = False) -> None:
        nonlocal saved_count
        if not buffer:
            return
        if not force and len(buffer) < save_every:
            return
        seen: set[tuple[str, str]] = set()
        new_rollouts: list[MonitorRollout] = []
        for r in buffer:
            key = (r.sample.id, r.mode)
            if key not in saved_keys and key not in seen:
                seen.add(key)
                new_rollouts.append(r)
        buffer.clear()
        if new_rollouts:
            append_monitor_rollouts(output_path, new_rollouts)
            for r in new_rollouts:
                saved_keys.add((r.sample.id, r.mode))
            all_rollouts.extend(new_rollouts)
            saved_count += len(new_rollouts)
        n_errors = sum(1 for r in new_rollouts if r.actor_error)
        print(f"    [{label}] Saved {saved_count}/{total} ({n_errors} errors in batch)")

    async def process_one(base_rollout: dict) -> MonitorRollout | None:
        sample_data = base_rollout["sample"]
        sample = Sample(**sample_data)

        if (sample.id, MODE) in saved_keys:
            return None

        # Build messages + prefill
        messages, prefill_text, sample = build_messages_for_rollout(
            base_rollout, model,
        )

        # The model just needs to complete the side task answer — short tokens
        actor_request = InferenceRequest(
            messages=messages,
            model=model,
            max_tokens=2048,
            temperature=1.0,
            prefill=prefill_text,
        )

        actor_resp = await _single_with_retry(
            actor_client, actor_request, actor_sem, max_retries=1,
        )

        if actor_resp.error:
            logging.warning("[actor error] %s sample=%s: %s", label, sample.id, actor_resp.error)

        # Both TinkerClient and OpenRouterClient prepend prefill to content
        full_response = actor_resp.content or ""

        # Extract answers from the full response
        main_task_resp = extract_main_task_response(full_response)
        side_task_resp = extract_side_task_response(full_response)
        expected_answer = sample.metadata.get("side_task_answer", "")
        side_task_correct = grade_side_task(side_task_resp, expected_answer)

        if side_task_correct and has_extraneous_text(full_response):
            side_task_correct = False

        # Build stripped versions for monitor
        stripped_prompt = sample.question
        stripped_reasoning = base_rollout.get("reasoning", "")
        stripped_response = main_task_resp or full_response

        # --- Run monitors ---
        monitor_results: list[MonitorResult] = []
        total_monitor_latency = 0.0
        total_monitor_usage: dict[str, int] = {}

        for monitor_type in monitor_type_order:
            builder = monitor_builders[monitor_type]
            if getattr(builder, "is_grading_monitor", False):
                correct_answer = sample.metadata.get("main_task_answer", "")
                model_answer = main_task_resp
                monitor_messages = builder.build_grading_prompt(
                    sample.question, correct_answer, model_answer,
                )
            else:
                monitor_messages = builder.build_monitor_prompt(
                    stripped_prompt, stripped_reasoning, stripped_response,
                )

            monitor_model_name = MONITOR_MODEL_OVERRIDES.get(
                monitor_type, "openai/gpt-4o",
            )
            monitor_request = InferenceRequest(
                messages=monitor_messages,
                model=monitor_model_name,
                max_tokens=1024,
                temperature=0.0,
            )
            monitor_resp = await _single_with_retry(
                monitor_client, monitor_request, monitor_sem, max_retries=3,
            )

            score = builder.parse_score(monitor_resp.content)
            store_prompt = monitor_messages if monitor_type == "main_task_grading" else []
            monitor_results.append(MonitorResult(
                monitor_type=monitor_type,
                monitor_model=monitor_model_name,
                monitor_prompt=store_prompt,
                monitor_response=monitor_resp.content,
                monitor_score=score,
                monitor_latency_ms=monitor_resp.latency_ms,
                monitor_usage=monitor_resp.usage,
                monitor_error=monitor_resp.error,
            ))
            total_monitor_latency += monitor_resp.latency_ms
            for k, v in monitor_resp.usage.items():
                total_monitor_usage[k] = total_monitor_usage.get(k, 0) + v

        # Derive main_task_correct from grading monitor
        main_task_correct: bool | None = None
        for mr in monitor_results:
            if mr.monitor_type == "main_task_grading" and mr.monitor_score is not None:
                main_task_correct = mr.monitor_score == 1.0
                break

        # Build the system/user prompts used for this eval
        sys_prompt_used, usr_prompt_used = prompt_builder.build_actor_prompt(
            sample, "side_task", model=model,
        )

        first = monitor_results[0] if monitor_results else None
        rollout = MonitorRollout(
            experiment_id=run_id,
            sample=sample,
            model=model,
            mode=MODE,
            system_prompt=sys_prompt_used,
            user_prompt=usr_prompt_used,
            reasoning=base_rollout.get("reasoning", ""),
            response=full_response,
            main_task_response=main_task_resp,
            side_task_response=side_task_resp,
            side_task_correct=side_task_correct,
            main_task_correct=main_task_correct,
            monitor_results=monitor_results,
            monitor_model=first.monitor_model if first else "",
            monitor_prompt=first.monitor_prompt if first else [],
            monitor_response=first.monitor_response if first else "",
            monitor_score=first.monitor_score if first else None,
            stripped_prompt=stripped_prompt,
            stripped_reasoning=stripped_reasoning,
            stripped_response=stripped_response,
            actor_latency_ms=actor_resp.latency_ms,
            monitor_latency_ms=total_monitor_latency,
            actor_usage=actor_resp.usage,
            monitor_usage=total_monitor_usage,
            dataset_split="val",
            actor_error=actor_resp.error,
            monitor_error=first.monitor_error if first else None,
        )

        async with buffer_lock:
            buffer.append(rollout)
            completed_count[0] += 1
            c = completed_count[0]
            if c % 25 == 0 or c == total:
                print(f"    [{label}] Completed {c}/{total}")
            _flush_sync()

        return rollout

    try:
        logging.info("START: %s (%d rollouts, concurrency=%d)", label, total, concurrency)
        tasks = [process_one(r) for r in base_rollouts]
        await asyncio.gather(*tasks)

        async with buffer_lock:
            _flush_sync(force=True)

    finally:
        await actor_client.close()
        await monitor_client.close()

    # Report
    if output_path.exists():
        errors, total_count = count_errors(output_path)
        error_rate = errors / total_count if total_count > 0 else 0
        n_side_correct = sum(1 for r in all_rollouts if r.side_task_correct)
        n_main_correct = sum(1 for r in all_rollouts if r.main_task_correct)
        n_valid = sum(1 for r in all_rollouts if not r.actor_error)
        logging.info(
            "  %s: %d/%d errors (%.1f%%), side_task_acc=%.1f%%, main_task_acc=%.1f%%",
            label, errors, total_count, error_rate * 100,
            100 * n_side_correct / max(n_valid, 1),
            100 * n_main_correct / max(n_valid, 1),
        )
        return (
            f"OK ({errors}/{total_count} errors, "
            f"side_acc={100*n_side_correct/max(n_valid,1):.1f}%, "
            f"main_acc={100*n_main_correct/max(n_valid,1):.1f}%)"
        )
    else:
        return "FAIL (no output file)"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="baseline_main_and_side eval")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--models", type=str, default="all",
                        choices=["base", "ft", "all"])
    parser.add_argument("--concurrency", type=int, default=HIGH_CONCURRENCY)
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    log_file = RESULTS_DIR / f"baseline_main_and_side_{run_id}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)

    if args.models == "base":
        models = BASE_MODELS
    elif args.models == "ft":
        models = FT_MODELS
    else:
        models = ALL_MODELS

    print(f"Baseline Main+Side Eval: run_id={run_id}")
    print(f"  {len(models)} models, serial execution")
    print(f"  Source: baseline_formatted val-split rollouts")
    print(f"  Mode: {MODE}")
    print(f"  Prompt: DontThinkMonitorQAPromptBuilder (side_task)")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Monitors: {MONITOR_TYPES}")
    if args.n_samples:
        print(f"  n_samples: {args.n_samples}")
    print()

    for i, model_cfg in enumerate(models, 1):
        ft = " (FT)" if model_cfg["model_path"] else ""
        print(f"  {i:>2}. {model_cfg['label']}{ft}")

    if args.dry_run:
        # Show what baseline files would be used
        print()
        for model_cfg in models:
            label = model_cfg["label"]
            pattern = f"monitor_qa_{label}_baseline_formatted_val_split_*.jsonl"
            files = sorted(ROLLOUT_DIR.glob(pattern))
            if files:
                # Count non-errored rollouts
                count = 0
                for line in open(files[-1]):
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    try:
                        r = json.loads(s)
                    except json.JSONDecodeError:
                        continue
                    if not r.get("actor_error"):
                        count += 1
                print(f"  {label}: {count} rollouts from {files[-1].name}")
            else:
                print(f"  {label}: NO FILE FOUND ({pattern})")
        print(f"\n[DRY RUN] {len(models)} jobs would be submitted. Exiting.")
        return

    print()
    results = {}
    t0 = time.time()

    for i, model_cfg in enumerate(models, 1):
        label = model_cfg["label"]
        print(f"\n{'=' * 60}")
        print(f"Job {i}/{len(models)}: {label}")
        print(f"{'=' * 60}")
        t_start = time.time()
        result = asyncio.run(run_model(model_cfg, run_id, args.n_samples, args.concurrency))
        elapsed = time.time() - t_start
        results[label] = result
        print(f"  -> {result} ({elapsed:.0f}s)")

    total_time = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"ALL DONE — {total_time:.0f}s total ({total_time/3600:.1f} hours)")
    print(f"{'=' * 60}")
    for label, result in sorted(results.items()):
        print(f"  {label:<25s} {result}")
    n_ok = sum(1 for r in results.values() if r.startswith("OK"))
    print(f"\n  {n_ok} succeeded, {len(results) - n_ok} failed")


if __name__ == "__main__":
    main()
