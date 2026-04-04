#!/usr/bin/env python3
"""Build a MonitorQA JSONL file pairing main tasks (GPQA + HLE) with side tasks (MATH).

Output schema (one JSON object per line):
    {
        "main_task": "...",
        "side_task": "...",
        "side_task_answer": 42,
        "metadata": {
            "main_source": "gpqa" | "hle",
            "main_id": "...",
            "main_domain": "...",
            "main_answer_type": "...",
            "side_source": "math",
            "side_level": "Level 3",
            "side_type": "Algebra",
            "side_id": 1234
        }
    }

Usage:
    python scripts/build_monitorqa.py [--n 1000] [--seed 42]
"""

import argparse
import json
import os
import random
import re
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Load datasets
# ---------------------------------------------------------------------------

def load_gpqa():
    """Load GPQA Diamond questions as list of dicts."""
    from datasets import load_dataset
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond")["train"]
    items = []
    for row in ds:
        question = row["Question"]
        # Build MCQ text with answer options
        options = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]
        items.append({
            "question": question,
            "options": options,
            "correct_answer": row["Correct Answer"],
            "domain": row.get("High-level domain", ""),
            "subdomain": row.get("Subdomain", ""),
            "id": row.get("Record ID", ""),
        })
    print(f"  GPQA Diamond: {len(items)} questions")
    return items


def load_hle():
    """Load HLE public split (text-only, no image-dependent questions)."""
    from datasets import load_dataset
    ds = load_dataset("cais/hle")["test"]
    items = []
    for row in ds:
        # Skip image-dependent questions
        if row.get("image"):
            continue
        items.append({
            "question": row["question"],
            "answer": row["answer"],
            "answer_type": row["answer_type"],
            "category": row.get("category", ""),
            "id": row["id"],
        })
    print(f"  HLE (text-only): {len(items)} questions")
    return items


def load_math_integer():
    """Load MATH problems with integer answers only."""
    df = pd.read_parquet(
        Path.home() / ".cache/huggingface/hub/datasets--qwedsacf--competition_math"
        "/snapshots/e839825f9ec5c6cfa585c654a59610969ec13993"
        "/data/train-00000-of-00001-7320a6f3aba8ebd2.parquet"
    )
    items = []
    for idx, row in df.iterrows():
        boxed = re.findall(r"\\boxed\{([^}]+)\}", row["solution"])
        if not boxed:
            continue
        ans_str = boxed[-1].strip()
        try:
            ans_int = int(ans_str)
        except ValueError:
            continue
        items.append({
            "problem": row["problem"],
            "answer": ans_int,
            "level": row["level"],
            "type": row["type"],
            "id": idx,
        })
    print(f"  MATH (integer answers): {len(items)} problems")
    return items


# ---------------------------------------------------------------------------
# Build JSONL
# ---------------------------------------------------------------------------

def build_rows(gpqa, hle, math, n, rng):
    """Use all GPQA questions, fill the rest with randomly sampled HLE."""
    def _make_main(source, item):
        if source == "gpqa":
            return {
                "main_task": item["question"],
                "main_source": "gpqa",
                "main_id": item["id"],
                "main_domain": item.get("domain", ""),
                "main_subdomain": item.get("subdomain", ""),
                "main_answer_type": "multipleChoice",
            }
        else:
            return {
                "main_task": item["question"],
                "main_source": "hle",
                "main_id": item["id"],
                "main_domain": item.get("category", ""),
                "main_subdomain": "",
                "main_answer_type": item.get("answer_type", ""),
            }

    # All GPQA + randomly sampled HLE to fill remaining slots
    main_items = [("gpqa", g) for g in gpqa]
    n_hle = n - len(gpqa)
    hle_sampled = rng.sample(hle, min(n_hle, len(hle)))
    main_items.extend(("hle", h) for h in hle_sampled)
    rng.shuffle(main_items)

    rows = []
    for source, item in main_items:
        main = _make_main(source, item)
        side = rng.choice(math)
        rows.append({
            "main_task": main["main_task"],
            "side_task": side["problem"],
            "side_task_answer": side["answer"],
            "metadata": {
                "main_source": main["main_source"],
                "main_id": main["main_id"],
                "main_domain": main["main_domain"],
                "main_subdomain": main.get("main_subdomain", ""),
                "main_answer_type": main["main_answer_type"],
                "side_source": "math",
                "side_level": side["level"],
                "side_type": side["type"],
                "side_id": side["id"],
            },
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Build MonitorQA JSONL")
    parser.add_argument("--n", type=int, default=1000, help="Number of rows to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default="results/monitorqa/gpqa_hle_w_math_monitorqa.jsonl")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN", "")
    if token:
        os.environ["HF_TOKEN"] = token

    print("Loading datasets...")
    gpqa = load_gpqa()
    hle = load_hle()
    math = load_math_integer()

    print(f"\nMain task pool: {len(gpqa) + len(hle)} (GPQA={len(gpqa)}, HLE={len(hle)})")
    print(f"Side task pool: {len(math)}")

    rng = random.Random(args.seed)
    rows = build_rows(gpqa, hle, math, args.n, rng)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    # Summary stats
    from collections import Counter
    sources = Counter(r["metadata"]["main_source"] for r in rows)
    levels = Counter(r["metadata"]["side_level"] for r in rows)
    print(f"\nWrote {len(rows)} rows to {out}")
    print(f"  Main sources: {dict(sources)}")
    print(f"  Side task levels: {dict(sorted(levels.items()))}")


if __name__ == "__main__":
    main()
