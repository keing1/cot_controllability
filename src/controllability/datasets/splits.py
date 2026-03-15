"""Dataset splitting: 50/50 train/test with stratification."""

from __future__ import annotations

import random
from typing import Literal

from sklearn.model_selection import train_test_split

from controllability.types import Sample


def split_dataset(
    samples: list[Sample],
    split: Literal["train", "test", "all"] = "all",
    seed: int = 42,
    fraction: float = 1.0,
) -> list[Sample]:
    """Split dataset into train/test and optionally subsample.

    - 50/50 train/test split
    - CoTControl datasets: random split (no obvious sub-categories)
    - ReasonIF: stratified on instruction_type
    - fraction < 1.0: randomly subsample the selected split

    Returns the selected split (or all data if split="all").
    """
    if split == "all":
        selected = samples
    else:
        # Determine if we should stratify
        is_reasonif = any(s.dataset == "reasonif" for s in samples)

        if is_reasonif:
            # Stratify on instruction_type
            labels = [s.metadata.get("instruction_type", "") for s in samples]
            train, test = train_test_split(
                samples, test_size=0.5, random_state=seed, stratify=labels
            )
        else:
            train, test = train_test_split(
                samples, test_size=0.5, random_state=seed
            )

        selected = train if split == "train" else test

    # Apply fraction subsampling
    if 0.0 < fraction < 1.0:
        rng = random.Random(seed)
        k = max(1, int(len(selected) * fraction))
        selected = rng.sample(selected, k)

    return selected


def proportional_sample(
    samples: list[Sample],
    n: int,
    seed: int = 42,
    require_valid_keywords: bool = False,
) -> list[Sample]:
    """Draw n samples proportional to dataset group sizes, deterministic.

    Groups samples by their `dataset` field and draws proportionally.
    If require_valid_keywords is True, filters to samples with non-empty
    valid_keywords before sampling.
    """
    if require_valid_keywords:
        samples = [
            s for s in samples
            if s.metadata.get("valid_keywords")
        ]

    # Group by dataset
    groups: dict[str, list[Sample]] = {}
    for s in samples:
        groups.setdefault(s.dataset, []).append(s)

    total = len(samples)
    if total == 0 or n <= 0:
        return []

    rng = random.Random(seed)
    selected: list[Sample] = []

    # Calculate proportional counts using largest remainder method
    raw_counts = {k: (len(v) / total) * n for k, v in groups.items()}
    floor_counts = {k: int(v) for k, v in raw_counts.items()}
    remainders = {k: raw_counts[k] - floor_counts[k] for k in raw_counts}

    allocated = sum(floor_counts.values())
    remaining = n - allocated

    # Distribute remaining slots to groups with largest remainders
    for k in sorted(remainders, key=lambda k: remainders[k], reverse=True):
        if remaining <= 0:
            break
        floor_counts[k] += 1
        remaining -= 1

    # Sample from each group
    for dataset_name in sorted(groups.keys()):
        group_samples = groups[dataset_name]
        count = min(floor_counts.get(dataset_name, 0), len(group_samples))
        selected.extend(rng.sample(group_samples, count))

    return selected


def get_split_stats(samples: list[Sample], seed: int = 42) -> dict:
    """Get statistics about the train/test split."""
    train = split_dataset(samples, split="train", seed=seed)
    test = split_dataset(samples, split="test", seed=seed)

    stats = {
        "total": len(samples),
        "train": len(train),
        "test": len(test),
    }

    # Add per-instruction-type breakdown for ReasonIF
    if any(s.dataset == "reasonif" for s in samples):
        for split_name, split_samples in [("train", train), ("test", test)]:
            by_type: dict[str, int] = {}
            for s in split_samples:
                itype = s.metadata.get("instruction_type", "unknown")
                by_type[itype] = by_type.get(itype, 0) + 1
            stats[f"{split_name}_by_type"] = by_type

    return stats
