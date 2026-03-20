#!/usr/bin/env python3
"""Aggregate results from rollout JSONL files."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from controllability.evals.metrics import compute_metrics, format_metrics
from controllability.rollouts.store import load_rollouts


def main():
    parser = argparse.ArgumentParser(description="Summarize evaluation results")
    parser.add_argument("paths", nargs="+", help="JSONL rollout file(s)")
    parser.add_argument("--output", "-o", help="Write summary JSON to file")
    parser.add_argument("--by-dataset", action="store_true", help="Break down by dataset")
    args = parser.parse_args()

    all_rollouts = []
    for p in args.paths:
        rollouts = load_rollouts(p)
        print(f"Loaded {len(rollouts)} rollouts from {p}")
        all_rollouts.extend(rollouts)

    if not all_rollouts:
        print("No rollouts found.")
        return

    print(f"\nTotal: {len(all_rollouts)} rollouts")
    print()

    if args.by_dataset:
        from collections import defaultdict

        by_ds = defaultdict(list)
        for r in all_rollouts:
            by_ds[r.sample.dataset].append(r)

        for ds, ds_rollouts in sorted(by_ds.items()):
            print(f"=== {ds} ===")
            metrics = compute_metrics(ds_rollouts)
            print(format_metrics(metrics))
            print()
    else:
        metrics = compute_metrics(all_rollouts)
        print(format_metrics(metrics))

    if args.output:
        metrics = compute_metrics(all_rollouts)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nSummary written to: {output_path}")


if __name__ == "__main__":
    main()
