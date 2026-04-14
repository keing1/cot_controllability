#!/usr/bin/env python3
"""Build MonitorQA JSONL files pairing main tasks with side tasks (MATH).

Splits:
  old     — 1000 examples: GPQA Diamond + HLE for main tasks, MATH for side tasks.
            (This is the original dataset used for all prior runs.)
  val     — 1000 examples: GPQA Extended + HLE + MMLU-Pro, MATH for side tasks.
  test    — 1000 examples: GPQA Extended + HLE + MMLU-Pro, MATH for side tasks.

Val and test use disjoint problems. GPQA Extended (546) is split 50/50 between
val and test. The remainder is filled evenly by HLE and MMLU-Pro. HLE problems
used in the default split are excluded from val/test.

Output schema (one JSON object per line):
    {
        "main_task": "...",
        "side_task": "...",
        "side_task_answer": 42,
        "metadata": {
            "main_source": "gpqa" | "hle" | "mmlu_pro",
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
    python scripts/build_monitorqa.py                      # build all splits
    python scripts/build_monitorqa.py --split old
    python scripts/build_monitorqa.py --split val
    python scripts/build_monitorqa.py --split test
    python scripts/build_monitorqa.py --split val --n 500
"""

import argparse
import json
import os
import random
import re
from pathlib import Path

import pandas as pd

OUTDIR = Path("results/monitorqa")

# ---------------------------------------------------------------------------
# Load datasets
# ---------------------------------------------------------------------------

def _load_gpqa_config(config_name: str, label: str):
    """Load a GPQA config as list of dicts with correct_answer."""
    from datasets import load_dataset
    ds = load_dataset("Idavidrein/gpqa", config_name)["train"]
    items = []
    for row in ds:
        items.append({
            "question": row["Question"],
            "options": [
                row["Correct Answer"],
                row["Incorrect Answer 1"],
                row["Incorrect Answer 2"],
                row["Incorrect Answer 3"],
            ],
            "correct_answer": row["Correct Answer"],
            "domain": row.get("High-level domain", ""),
            "subdomain": row.get("Subdomain", ""),
            "id": row.get("Record ID", ""),
        })
    print(f"  GPQA {label}: {len(items)} questions")
    return items


def load_gpqa_diamond():
    return _load_gpqa_config("gpqa_diamond", "Diamond")


def load_gpqa_extended():
    return _load_gpqa_config("gpqa_extended", "Extended")


def load_hle():
    """Load HLE public split (text-only, no image-dependent questions)."""
    from datasets import load_dataset
    ds = load_dataset("cais/hle")["test"]
    items = []
    for row in ds:
        if row.get("image"):
            continue
        items.append({
            "question": row["question"],
            "correct_answer": row["answer"],
            "answer_type": row["answer_type"],
            "category": row.get("category", ""),
            "subject": row.get("raw_subject", ""),
            "id": row["id"],
        })
    print(f"  HLE (text-only): {len(items)} questions")
    return items


def load_mmlu_pro():
    """Load MMLU-Pro test split."""
    from datasets import load_dataset
    ds = load_dataset("TIGER-Lab/MMLU-Pro")["test"]
    items = []
    for row in ds:
        # Format question with answer choices
        question = row["question"]
        options = row["options"]
        # Build MCQ text
        option_labels = "ABCDEFGHIJ"
        option_lines = []
        for i, opt in enumerate(options):
            if i < len(option_labels):
                option_lines.append(f"{option_labels[i]}. {opt}")
        question_with_options = question + "\n\nAnswer Choices:\n" + "\n".join(option_lines)

        items.append({
            "question": question_with_options,
            "correct_answer": row["answer"],  # letter like "I"
            "category": row.get("category", ""),
            "src": row.get("src", ""),
            "id": str(row.get("question_id", "")),
        })
    print(f"  MMLU-Pro: {len(items)} questions")
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
# Row builders
# ---------------------------------------------------------------------------

def _make_main(source: str, item: dict) -> dict:
    """Convert a source item to main-task metadata."""
    if source == "gpqa":
        return {
            "main_task": item["question"],
            "main_task_answer": item["correct_answer"],
            "main_source": "gpqa",
            "main_id": item["id"],
            "main_domain": item.get("domain", ""),
            "main_subdomain": item.get("subdomain", ""),
            "main_answer_type": "multipleChoice",
        }
    elif source == "hle":
        return {
            "main_task": item["question"],
            "main_task_answer": item["correct_answer"],
            "main_source": "hle",
            "main_id": item["id"],
            "main_domain": item.get("category", ""),
            "main_subdomain": item.get("subject", ""),
            "main_answer_type": item.get("answer_type", ""),
        }
    elif source == "mmlu_pro":
        return {
            "main_task": item["question"],
            "main_task_answer": item["correct_answer"],
            "main_source": "mmlu_pro",
            "main_id": item["id"],
            "main_domain": item.get("category", ""),
            "main_subdomain": item.get("src", ""),
            "main_answer_type": "multipleChoice",
        }
    else:
        raise ValueError(f"Unknown source: {source}")


def _build_rows(main_items: list[tuple[str, dict]], math: list[dict],
                rng: random.Random) -> list[dict]:
    """Pair main-task items with random MATH side tasks."""
    rng.shuffle(main_items)
    rows = []
    for source, item in main_items:
        main = _make_main(source, item)
        side = rng.choice(math)
        rows.append({
            "main_task": main["main_task"],
            "main_task_answer": main["main_task_answer"],
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


# ---------------------------------------------------------------------------
# Split builders
# ---------------------------------------------------------------------------

def build_default(gpqa_diamond, hle, math, n, rng):
    """Default split: all GPQA Diamond + fill with HLE."""
    main_items = [("gpqa", g) for g in gpqa_diamond]
    n_hle = n - len(gpqa_diamond)
    hle_sampled = rng.sample(hle, min(n_hle, len(hle)))
    main_items.extend(("hle", h) for h in hle_sampled)
    return _build_rows(main_items, math, rng)


def build_val_test(gpqa_extended, hle, mmlu_pro, math, n_per_split, rng,
                   exclude_hle_ids: set[str]):
    """Build val and test splits.

    GPQA Extended split 50/50. Remainder filled evenly by HLE and MMLU-Pro.
    HLE IDs from the default split are excluded.
    """
    # Shuffle and split GPQA
    gpqa_shuffled = list(gpqa_extended)
    rng.shuffle(gpqa_shuffled)
    mid = len(gpqa_shuffled) // 2
    gpqa_val = gpqa_shuffled[:mid]
    gpqa_test = gpqa_shuffled[mid:]

    # Filter HLE: exclude IDs used in old split
    hle_available = [h for h in hle if h["id"] not in exclude_hle_ids]
    rng.shuffle(hle_available)

    # For each split, fill remainder evenly with HLE and MMLU-Pro
    splits = {}
    mmlu_used_ids: set[str] = set()
    for split_name, gpqa_subset in [("val", gpqa_val), ("test", gpqa_test)]:
        # Cap GPQA if n_per_split is smaller than the GPQA allocation
        if len(gpqa_subset) > n_per_split:
            gpqa_subset = gpqa_subset[:n_per_split]
        n_remaining = max(0, n_per_split - len(gpqa_subset))
        n_hle_needed = n_remaining // 2
        n_mmlu_needed = n_remaining - n_hle_needed

        if split_name == "val":
            hle_subset = hle_available[:n_hle_needed]
            hle_available = hle_available[n_hle_needed:]  # consume for test
        else:
            hle_subset = hle_available[:n_hle_needed]

        mmlu_shuffled = list(mmlu_pro)
        rng.shuffle(mmlu_shuffled)
        if split_name == "val":
            # Use first half of shuffled MMLU-Pro for val
            mmlu_subset = mmlu_shuffled[:n_mmlu_needed]
            mmlu_used_ids.update(m["id"] for m in mmlu_subset)
        else:
            # Exclude val's MMLU-Pro IDs for test
            mmlu_remaining = [m for m in mmlu_shuffled if m["id"] not in mmlu_used_ids]
            mmlu_subset = mmlu_remaining[:n_mmlu_needed]

        main_items = (
            [("gpqa", g) for g in gpqa_subset]
            + [("hle", h) for h in hle_subset]
            + [("mmlu_pro", m) for m in mmlu_subset]
        )
        splits[split_name] = _build_rows(main_items, math, rng)

    return splits["val"], splits["test"]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_jsonl(rows: list[dict], path: Path, overwrite: bool = False) -> bool:
    """Write rows to JSONL. Returns True if written, False if skipped."""
    if path.exists() and not overwrite:
        print(f"  SKIP: {path} already exists (use --overwrite to replace)")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    from collections import Counter
    sources = Counter(r["metadata"]["main_source"] for r in rows)
    levels = Counter(r["metadata"]["side_level"] for r in rows)
    print(f"  Wrote {len(rows)} rows to {path}")
    print(f"    Main sources: {dict(sources)}")
    print(f"    Side task levels: {dict(sorted(levels.items()))}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build MonitorQA JSONL")
    parser.add_argument("--split", choices=["old", "val", "test", "all"],
                        default="all", help="Which split to build")
    parser.add_argument("--n", type=int, default=1000, help="Number of rows per split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing split files (default: skip if exists)")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN", "")
    if token:
        os.environ["HF_TOKEN"] = token

    build_default_split = args.split in ("old", "all")
    build_val_test_splits = args.split in ("val", "test", "all")

    print("Loading datasets...")
    math = load_math_integer()

    # Default split lives at the original file path — never overwrite it.
    # We only need to load it to get HLE IDs for exclusion from val/test.
    DEFAULT_PATH = OUTDIR / "gpqa_hle_w_math_monitorqa.jsonl"

    if build_default_split and not build_val_test_splits:
        print(f"\n--- Old split already exists at {DEFAULT_PATH} ---")
        print("  The old split is the original dataset used for all prior runs.")
        print("  To rebuild it, delete the file and re-run.")

    # Load default HLE IDs for exclusion (needed by val/test)
    default_hle_ids = set()
    if DEFAULT_PATH.exists():
        with open(DEFAULT_PATH) as f:
            for line in f:
                d = json.loads(line.strip())
                if d["metadata"]["main_source"] == "hle":
                    default_hle_ids.add(d["metadata"]["main_id"])
        print(f"  Loaded {len(default_hle_ids)} HLE IDs from old split to exclude")

    if build_val_test_splits:
        hle = load_hle()
        gpqa_extended = load_gpqa_extended()
        mmlu_pro = load_mmlu_pro()

        rng = random.Random(args.seed + 1)  # different seed from default
        print(f"\n--- Building val/test splits (n={args.n} each) ---")
        val_rows, test_rows = build_val_test(
            gpqa_extended, hle, mmlu_pro, math, args.n, rng,
            exclude_hle_ids=default_hle_ids,
        )

        if args.split in ("val", "all"):
            write_jsonl(val_rows, OUTDIR / "monitorqa_val.jsonl", overwrite=args.overwrite)
        if args.split in ("test", "all"):
            write_jsonl(test_rows, OUTDIR / "monitorqa_test.jsonl", overwrite=args.overwrite)

        # Verify no overlap between val and test
        val_ids = {r["metadata"]["main_id"] for r in val_rows}
        test_ids = {r["metadata"]["main_id"] for r in test_rows}
        overlap = val_ids & test_ids
        if overlap:
            print(f"  WARNING: {len(overlap)} overlapping main IDs between val and test!")
        else:
            print(f"  No overlap between val ({len(val_ids)}) and test ({len(test_ids)}) main IDs")


if __name__ == "__main__":
    main()
