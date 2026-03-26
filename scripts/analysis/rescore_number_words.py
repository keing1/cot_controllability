#!/usr/bin/env python3
"""Rescore ReasonIF number_words rollouts with per-model calibrated thresholds.

Reads calibration from results/calibration/number_words_reference.json and
re-grades all number_words samples in reasonif rollout files, updating
the `compliant` field, `grading_details.constraint_args.num_words`, and
`sample.metadata.constraint_args.num_words` in place.

Usage:
    python scripts/analysis/rescore_number_words.py              # rescore all
    python scripts/analysis/rescore_number_words.py --dry-run    # preview only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import nltk.tokenize

REPO_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_FILE = REPO_ROOT / "results" / "calibration" / "number_words_reference.json"
ROLLOUT_DIR = REPO_ROOT / "results" / "rollouts"

# Must match the ReasonIF checker exactly
_TOKENIZER = nltk.tokenize.RegexpTokenizer(r"\w+")


def count_words(text: str) -> int:
    return len(_TOKENIZER.tokenize(text))


def calibration_key(model: str, reasoning_effort: str) -> str:
    """Build lookup key: gpt-oss-20b@medium or qwen3-8b."""
    # Strip FT suffixes: "openai/gpt-oss-20b-rif-000060" -> "openai/gpt-oss-20b"
    short = model.split("/")[-1]
    # Remove -rif-* suffix (FT checkpoint labels)
    short = re.sub(r"-rif-.*$", "", short)
    if "gpt-oss" in model and reasoning_effort and reasoning_effort != "none":
        return f"{short}@{reasoning_effort}"
    return short


def rescore_file(
    fpath: Path,
    calibration: dict,
    dry_run: bool = False,
) -> dict:
    """Rescore a single rollout file. Returns stats dict."""
    with open(fpath) as f:
        lines = f.readlines()

    if not lines or not lines[0].startswith("#"):
        return {"skipped": True, "reason": "no header"}

    header = json.loads(lines[0].lstrip("# "))
    model = header.get("model", "")
    effort = header.get("reasoning_effort", "")

    key = calibration_key(model, effort)
    if key not in calibration:
        return {"skipped": True, "reason": f"no calibration for {key}"}

    thresholds = calibration[key]

    n_number_words = 0
    n_changed = 0
    n_compliant_flipped = 0
    changed = False

    for i in range(1, len(lines)):
        r = json.loads(lines[i])
        if r.get("control_mode") != "number_words":
            continue

        n_number_words += 1

        sample = r.get("sample", {})
        meta = sample.get("metadata", {})
        source = meta.get("source", "")
        if source not in thresholds:
            continue

        new_threshold = int(thresholds[source])
        old_threshold = meta.get("constraint_args", {}).get("num_words", 0)

        if old_threshold == new_threshold:
            continue

        # Recount words in reasoning
        reasoning = r.get("reasoning", "")
        wc = count_words(reasoning)
        new_compliant = wc < new_threshold
        old_compliant = r.get("compliant")

        # Update fields
        r["compliant"] = new_compliant

        gd = r.get("grading_details", {})
        if "constraint_args" in gd:
            gd["constraint_args"]["num_words"] = new_threshold
        r["grading_details"] = gd

        if "constraint_args" in meta:
            meta["constraint_args"]["num_words"] = new_threshold

        lines[i] = json.dumps(r) + "\n"
        changed = True
        n_changed += 1
        if new_compliant != old_compliant:
            n_compliant_flipped += 1

    if changed and not dry_run:
        with open(fpath, "w") as f:
            f.writelines(lines)

    return {
        "skipped": False,
        "key": key,
        "n_number_words": n_number_words,
        "n_changed": n_changed,
        "n_compliant_flipped": n_compliant_flipped,
        "written": changed and not dry_run,
    }


def main():
    parser = argparse.ArgumentParser(description="Rescore number_words with calibrated thresholds")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    args = parser.parse_args()

    if not CALIBRATION_FILE.exists():
        print(f"ERROR: Calibration file not found: {CALIBRATION_FILE}")
        sys.exit(1)

    with open(CALIBRATION_FILE) as f:
        calibration = json.load(f)

    print(f"Loaded calibration for {len(calibration)} model keys")
    if args.dry_run:
        print("DRY RUN — no files will be modified\n")

    rollout_files = sorted(
        p for p in ROLLOUT_DIR.iterdir()
        if "reasonif" in p.name and p.suffix == ".jsonl"
    )
    print(f"Found {len(rollout_files)} reasonif rollout files\n")

    total_changed = 0
    total_flipped = 0
    files_modified = 0

    for fpath in rollout_files:
        stats = rescore_file(fpath, calibration, dry_run=args.dry_run)
        if stats.get("skipped"):
            continue
        if stats["n_changed"] > 0:
            files_modified += 1
            total_changed += stats["n_changed"]
            total_flipped += stats["n_compliant_flipped"]
            action = "would write" if args.dry_run else "wrote"
            print(
                f"  {fpath.name}\n"
                f"    key={stats['key']}  "
                f"rescored={stats['n_changed']}/{stats['n_number_words']}  "
                f"flipped={stats['n_compliant_flipped']}  "
                f"{action}"
            )

    print(f"\nSummary: {files_modified} files, {total_changed} samples rescored, "
          f"{total_flipped} compliance flips")
    if args.dry_run:
        print("(dry run — nothing written)")


if __name__ == "__main__":
    main()
