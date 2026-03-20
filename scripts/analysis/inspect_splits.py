#!/usr/bin/env python3
"""View dataset split statistics."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from controllability.datasets.loader import list_datasets, load_dataset
from controllability.datasets.splits import get_split_stats, split_dataset


def main():
    parser = argparse.ArgumentParser(description="Inspect dataset split statistics")
    parser.add_argument("--dataset", default=None, help="Dataset name (default: all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    parser.add_argument("--fraction", type=float, default=1.0, help="Subsample fraction to test")
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else list_datasets()

    for ds_name in datasets:
        print(f"=== {ds_name} ===")
        samples = load_dataset(ds_name)
        stats = get_split_stats(samples, seed=args.seed)
        print(f"  Total: {stats['total']}")
        print(f"  Train: {stats['train']}")
        print(f"  Test:  {stats['test']}")

        # Show type breakdowns if present
        for key in ["train_by_type", "test_by_type"]:
            if key in stats:
                print(f"  {key}:")
                for t, count in sorted(stats[key].items()):
                    print(f"    {t}: {count}")

        # Show effect of fraction
        if args.fraction < 1.0:
            train_sub = split_dataset(samples, split="train", seed=args.seed, fraction=args.fraction)
            test_sub = split_dataset(samples, split="test", seed=args.seed, fraction=args.fraction)
            print(f"  With fraction={args.fraction}:")
            print(f"    Train subsample: {len(train_sub)}")
            print(f"    Test subsample:  {len(test_sub)}")

        print()


if __name__ == "__main__":
    main()
