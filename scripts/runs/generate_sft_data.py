#!/usr/bin/env python3
"""Generate ReasonIF SFT training data from a base model.

Three-stage pipeline:
1. Generate raw completions from a base model for ReasonIF prompts
2. Apply constraint-specific transforms and verify compliance
3. Output parquet matching the existing training format

Usage:
    python scripts/runs/generate_sft_data.py \
        --source-parquet external/reasonIF_ft/train-00000-of-00001.parquet \
        --base-model qwen/qwen3-235b-a22b \
        --backend openrouter \
        --output results/sft/qwen3-235b-reasonif-sft.parquet \
        --max-concurrency 30

    # Rule-based only:
    python scripts/runs/generate_sft_data.py \
        --constraints english_capital,no_comma,end_checker,json_format \
        --dry-run

    # Subset with LLM transforms:
    python scripts/runs/generate_sft_data.py \
        --constraints reasoning_language,number_words \
        --max-concurrency 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd

from controllability.config import Settings
from controllability.evals.grading import grade_reasonif_compliance
from controllability.inference.batch import run_batch
from controllability.inference.openrouter import OpenRouterClient
from controllability.training.transforms import (
    LLM_TRANSFORMS,
    apply_transform,
    filter_constraint_args,
)
from controllability.types import InferenceRequest, Sample


# ---------------------------------------------------------------------------
# Parquet column helpers
# ---------------------------------------------------------------------------
# The source parquet stores several columns as numpy arrays of length 1
# (e.g. constraint_name = array(['punctuation:no_comma'])).
# constraint_name also uses a "category:type" namespace — we map to the
# short names used in the codebase.

import numpy as np

# category:name -> short name
_CONSTRAINT_NAME_MAP = {
    "punctuation:no_comma": "no_comma",
    "startend:end_checker": "end_checker",
    "detectable_format:json_format": "json_format",
    "language:reasoning_language": "reasoning_language",
    "change_case:english_capital": "english_capital",
    "length_constraint_checkers:number_words": "number_words",
}


def _unwrap(value):
    """Unwrap numpy array of length 1 to its scalar value."""
    if isinstance(value, np.ndarray):
        if len(value) == 0:
            return None
        return value[0]
    return value


def get_constraint_short_name(row) -> str:
    """Extract the short constraint name from a parquet row."""
    raw = _unwrap(row["constraint_name"])
    if raw is None:
        return ""
    return _CONSTRAINT_NAME_MAP.get(str(raw), str(raw))


def get_constraint_args_dict(row) -> dict | None:
    """Extract constraint_args as a plain dict (or None) from a parquet row."""
    raw = _unwrap(row["constraint_args"])
    if raw is None:
        return None
    if isinstance(raw, dict):
        return dict(raw)
    return None


def get_instruction_description(row) -> str:
    """Extract instruction_description as a string."""
    return str(_unwrap(row.get("instruction_description", "")) or "")


def get_messages(row, column: str = "messages") -> list[dict]:
    """Extract a messages column as a list of plain dicts."""
    raw = row.get(column)
    if raw is None:
        return []
    # May be stored as a list of dicts or numpy array
    if isinstance(raw, np.ndarray):
        raw = raw.tolist()
    return [dict(m) for m in raw] if raw else []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_checkpoint(path: Path) -> dict[int, dict]:
    """Load checkpoint JSONL. Returns {row_idx: record}."""
    completed = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    completed[record["row_idx"]] = record
    return completed


def append_checkpoint(path: Path, record: dict) -> None:
    """Append a single record to the checkpoint JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def extract_reasoning_and_content(messages: list[dict]) -> tuple[str, str]:
    """Extract the assistant's reasoning (thinking) and content from the messages list."""
    for msg in messages:
        if msg.get("role") == "assistant":
            reasoning = msg.get("thinking", "") or ""
            content = msg.get("content", "") or ""
            return reasoning, content
    return "", ""


def extract_user_prompt(messages: list[dict]) -> str:
    """Extract the user prompt from the messages list."""
    for msg in messages:
        if msg.get("role") == "user":
            return msg.get("content", "") or ""
    return ""


def build_messages_column(user_prompt: str, reasoning: str, content: str) -> list[dict]:
    """Build the messages column in the parquet format."""
    messages = [{"role": "user", "content": user_prompt}]
    assistant_msg = {"role": "assistant", "content": content}
    if reasoning:
        assistant_msg["thinking"] = reasoning
    messages.append(assistant_msg)
    return messages


# ---------------------------------------------------------------------------
# Stage 1: Generate raw completions
# ---------------------------------------------------------------------------

async def stage1_generate(
    df: pd.DataFrame,
    row_indices: list[int],
    client,
    checkpoint_path: Path,
    max_concurrency: int,
    max_retries: int,
    max_tokens: int,
    temperature: float,
    model: str,
    system_prompt: str = "",
) -> dict[int, dict]:
    """Generate raw completions from the base model.

    Returns {row_idx: {row_idx, reasoning, content, error}}.
    """
    # Load existing checkpoint
    completed = load_checkpoint(checkpoint_path)
    pending_indices = [i for i in row_indices if i not in completed]

    if not pending_indices:
        print(f"  All {len(row_indices)} completions already in checkpoint")
        return completed

    print(f"  {len(pending_indices)} pending / {len(row_indices)} total (checkpoint: {len(completed)})")

    # Build inference requests
    requests: list[InferenceRequest] = []
    request_indices: list[int] = []

    for idx in pending_indices:
        row = df.iloc[idx]
        user_prompt = extract_user_prompt(get_messages(row, "messages_original"))
        if not user_prompt:
            user_prompt = extract_user_prompt(get_messages(row, "messages"))
        if not user_prompt:
            user_prompt = str(_unwrap(row.get("original_prompt", "")) or "")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        requests.append(InferenceRequest(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        ))
        request_indices.append(idx)

    # Run batch
    responses = await run_batch(
        client, requests,
        max_concurrency=max_concurrency,
        max_retries=max_retries,
        desc="Stage 1: Generate",
    )

    # Save results to checkpoint
    for idx, resp in zip(request_indices, responses):
        record = {
            "row_idx": idx,
            "reasoning": resp.reasoning,
            "content": resp.content,
            "error": resp.error,
        }
        completed[idx] = record
        append_checkpoint(checkpoint_path, record)

    n_errors = sum(1 for r in completed.values() if r.get("error"))
    print(f"  Completions done: {len(completed)} total, {n_errors} errors")

    return completed


# ---------------------------------------------------------------------------
# Stage 2: Transform + verify
# ---------------------------------------------------------------------------

async def stage2_transform_and_verify(
    df: pd.DataFrame,
    row_indices: list[int],
    completions: dict[int, dict],
    llm_client,
) -> list[dict]:
    """Apply transforms and verify compliance.

    Returns list of result dicts with all fields needed for the output parquet.
    """
    results = []
    compliance_counts: dict[str, dict[str, int]] = {}

    for idx in row_indices:
        row = df.iloc[idx]
        completion = completions.get(idx)

        constraint_name = get_constraint_short_name(row)
        # Initialize per-constraint counters
        if constraint_name not in compliance_counts:
            compliance_counts[constraint_name] = {"total": 0, "compliant": 0, "error": 0}
        compliance_counts[constraint_name]["total"] += 1

        if completion is None or completion.get("error"):
            error_msg = completion.get("error", "missing completion") if completion else "missing completion"
            compliance_counts[constraint_name]["error"] += 1
            results.append({
                "row_idx": idx,
                "error": error_msg,
                "compliant": None,
            })
            continue

        reasoning = completion["reasoning"]
        content = completion["content"]

        # Parse constraint_args and filter None values
        raw_args_dict = get_constraint_args_dict(row)
        constraint_args = filter_constraint_args(raw_args_dict)

        # Apply transform to the reasoning
        try:
            transformed_reasoning = await apply_transform(
                constraint_name, reasoning, constraint_args, llm_client
            )
        except Exception as e:
            compliance_counts[constraint_name]["error"] += 1
            results.append({
                "row_idx": idx,
                "error": f"Transform failed: {e}",
                "compliant": None,
            })
            continue

        # For number_words: constraint_args["num_words"] was updated in place by the transform
        # Also update instruction_description
        instruction_description = get_instruction_description(row)
        if constraint_name == "number_words" and "num_words" in constraint_args:
            actual_target = constraint_args["num_words"]
            instruction_description = f"When reasoning, respond with less than {actual_target} words."

        # Verify compliance using the grading function
        sample = Sample(
            id=str(idx),
            dataset="reasonif",
            question="",
            correct_answer="",
            metadata={
                "instruction_type": constraint_name,
                "constraint_args": constraint_args,
            },
        )
        grading_result = grade_reasonif_compliance(transformed_reasoning, sample)
        compliant = grading_result.get("compliant", False)

        if compliant:
            compliance_counts[constraint_name]["compliant"] += 1

        # Build user prompt from original messages
        user_prompt = extract_user_prompt(get_messages(row, "messages_original"))
        if not user_prompt:
            user_prompt = extract_user_prompt(get_messages(row, "messages"))
        if not user_prompt:
            user_prompt = str(_unwrap(row.get("original_prompt", "")) or "")

        # Preserve original parquet array format for constraint_name and constraint_args
        raw_constraint_name = row["constraint_name"]  # keep original array format
        out_constraint_args = np.array([constraint_args]) if constraint_args else row["constraint_args"]
        out_instruction_desc = np.array([instruction_description])

        results.append({
            "row_idx": idx,
            "original_prompt": _unwrap(row.get("original_prompt", "")) or "",
            "source": _unwrap(row.get("source", "")) or "",
            "constraint_name": raw_constraint_name,
            "constraint_args": out_constraint_args,
            "instruction_description": out_instruction_desc,
            "messages": build_messages_column(user_prompt, transformed_reasoning, content),
            "messages_original": build_messages_column(user_prompt, reasoning, content),
            "compliant": compliant,
            "error": None,
        })

    # Print compliance summary
    print("\n  Compliance summary:")
    for cname, counts in sorted(compliance_counts.items()):
        total = counts["total"]
        good = counts["compliant"]
        errs = counts["error"]
        rate = f"{good / (total - errs):.1%}" if (total - errs) > 0 else "N/A"
        print(f"    {cname:25s}  {good}/{total - errs} compliant ({rate}), {errs} errors")

    return results


# ---------------------------------------------------------------------------
# Stage 3: Output parquet
# ---------------------------------------------------------------------------

def stage3_output(results: list[dict], output_path: Path) -> None:
    """Write results to parquet matching the existing schema."""
    # Filter to successful results only
    good = [r for r in results if r.get("error") is None]
    print(f"\n  Writing {len(good)} rows to parquet ({len(results) - len(good)} errors dropped)")

    if not good:
        print("  WARNING: No successful results to write")
        return

    # Build DataFrame with exact schema
    records = []
    for r in good:
        records.append({
            "original_prompt": r["original_prompt"],
            "source": r["source"],
            "constraint_name": r["constraint_name"],
            "constraint_args": r["constraint_args"],
            "instruction_description": r["instruction_description"],
            "messages": r["messages"],
            "messages_original": r["messages_original"],
        })

    out_df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False)
    print(f"  Saved to {output_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def run_pipeline(args: argparse.Namespace) -> None:
    """Run the full SFT data generation pipeline."""
    # Load source parquet
    source_path = Path(args.source_parquet)
    if not source_path.exists():
        print(f"ERROR: Source parquet not found: {source_path}")
        sys.exit(1)

    df = pd.read_parquet(source_path)
    print(f"Loaded {len(df)} rows from {source_path}")

    # Add a column with short constraint names for easy filtering
    df["_short_name"] = df.apply(get_constraint_short_name, axis=1)

    # Show constraint distribution
    constraint_dist = df["_short_name"].value_counts()
    print("\nConstraint distribution:")
    for name, count in constraint_dist.items():
        print(f"  {str(name):25s}  {int(count)}")

    # Filter to requested constraints (using short names)
    if args.constraints:
        constraint_filter = [c.strip() for c in args.constraints.split(",")]
        df_filtered = df[df["_short_name"].isin(constraint_filter)]
        row_indices = list(df_filtered.index)
        print(f"\nFiltered to {len(row_indices)} rows for constraints: {constraint_filter}")
    else:
        row_indices = list(range(len(df)))
        print(f"\nUsing all {len(row_indices)} rows")

    if args.dry_run:
        print("\n[DRY RUN] Would process:")
        names = [c.strip() for c in args.constraints.split(",")] if args.constraints else list(constraint_dist.index)
        for name in names:
            count = int((df["_short_name"] == name).sum())
            needs_llm = "LLM" if name in LLM_TRANSFORMS else "rule-based"
            print(f"  {name:25s}  {count} rows  ({needs_llm})")
        print(f"\nBase model: {args.base_model}")
        print(f"Backend: {args.backend}")
        if args.system_prompt:
            print(f"System prompt: {args.system_prompt!r}")
        if args.reasoning_effort:
            print(f"Reasoning effort: {args.reasoning_effort}")
        print(f"Output: {args.output}")
        print(f"Checkpoint: {args.checkpoint}")
        return

    # Create inference clients
    settings = Settings()

    if args.backend == "openrouter":
        if not settings.openrouter_api_key:
            print("ERROR: OPENROUTER_API_KEY not set")
            sys.exit(1)
        base_client = OpenRouterClient(
            api_key=settings.openrouter_api_key,
            request_timeout=args.request_timeout,
            reasoning_effort=args.reasoning_effort,
        )
    elif args.backend == "tinker":
        from controllability.inference.tinker_client import TinkerClient
        base_client = TinkerClient(
            model=args.base_model,
            reasoning_effort=args.reasoning_effort or "none",
        )
    else:
        print(f"ERROR: Unknown backend: {args.backend}")
        sys.exit(1)

    # LLM client for transforms (always OpenRouter/GPT-4.1)
    needs_llm = any(
        df.iloc[i]["_short_name"] in LLM_TRANSFORMS
        for i in row_indices
    )
    llm_client = None
    if needs_llm:
        if not settings.openrouter_api_key:
            print("ERROR: OPENROUTER_API_KEY required for LLM transforms")
            sys.exit(1)
        llm_client = OpenRouterClient(
            api_key=settings.openrouter_api_key,
            request_timeout=args.request_timeout,
        )

    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)

    try:
        # Stage 1: Generate raw completions
        print("\n" + "=" * 60)
        print("STAGE 1: Generate raw completions")
        print("=" * 60)
        completions = await stage1_generate(
            df=df,
            row_indices=row_indices,
            client=base_client,
            checkpoint_path=checkpoint_path,
            max_concurrency=args.max_concurrency,
            max_retries=args.max_retries,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            model=args.base_model,
            system_prompt=args.system_prompt or "",
        )

        # Stage 2: Transform + verify
        print("\n" + "=" * 60)
        print("STAGE 2: Transform + verify compliance")
        print("=" * 60)
        results = await stage2_transform_and_verify(
            df=df,
            row_indices=row_indices,
            completions=completions,
            llm_client=llm_client,
        )

        # Stage 3: Output
        print("\n" + "=" * 60)
        print("STAGE 3: Write output parquet")
        print("=" * 60)
        stage3_output(results, output_path)

    finally:
        await base_client.close()
        if llm_client is not None:
            await llm_client.close()

    # Final summary
    n_good = sum(1 for r in results if r.get("error") is None)
    n_compliant = sum(1 for r in results if r.get("compliant") is True)
    print(f"\nDone: {n_good}/{len(results)} successful, {n_compliant}/{n_good} compliant")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate ReasonIF SFT training data from a base model"
    )
    parser.add_argument(
        "--source-parquet",
        default="external/reasonIF_ft/train-00000-of-00001.parquet",
        help="Path to source ReasonIF SFT parquet",
    )
    parser.add_argument(
        "--base-model",
        default="qwen/qwen3-235b-a22b",
        help="Base model for generating raw completions",
    )
    parser.add_argument(
        "--backend",
        choices=["openrouter", "tinker"],
        default="openrouter",
        help="Inference backend for the base model",
    )
    parser.add_argument(
        "--output",
        default="results/sft/qwen3-235b-reasonif-sft.parquet",
        help="Output parquet path",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=30,
        help="Max parallel inference requests",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries per request",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="Max tokens for base model completions",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for base model",
    )
    parser.add_argument(
        "--checkpoint",
        default="results/sft/qwen3-235b-reasonif-sft.checkpoint.jsonl",
        help="Checkpoint JSONL path for resume support",
    )
    parser.add_argument(
        "--constraints",
        default=None,
        help="Comma-separated constraint names to process (default: all)",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=300,
        help="Per-request timeout in seconds",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="System prompt to prepend to all requests (e.g. 'You are a helpful assistant.')",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default=None,
        help="Reasoning effort level for models that support it (gpt-oss, o1, o3, o4)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without running",
    )

    args = parser.parse_args()
    asyncio.run(run_pipeline(args))


if __name__ == "__main__":
    main()
