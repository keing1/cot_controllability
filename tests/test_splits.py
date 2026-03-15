"""Tests for dataset splitting."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from controllability.datasets.loader import load_dataset, list_datasets
from controllability.datasets.splits import split_dataset, get_split_stats


class TestListDatasets:
    def test_all_registered(self):
        datasets = list_datasets()
        assert "cotcontrol/gpqa" in datasets
        assert "cotcontrol/hle" in datasets
        assert "cotcontrol/mmlu_pro" in datasets
        assert "reasonif" in datasets


class TestLoadDataset:
    def test_load_gpqa(self):
        samples = load_dataset("cotcontrol/gpqa")
        assert len(samples) > 0
        assert samples[0].dataset == "cotcontrol/gpqa"
        assert samples[0].question
        assert samples[0].correct_answer

    def test_load_reasonif(self):
        samples = load_dataset("reasonif")
        assert len(samples) == 300
        assert samples[0].dataset == "reasonif"
        assert "instruction_type" in samples[0].metadata

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset("nonexistent")


class TestSplitDataset:
    def test_split_sizes(self):
        samples = load_dataset("cotcontrol/gpqa")
        train = split_dataset(samples, split="train", seed=42)
        test = split_dataset(samples, split="test", seed=42)
        assert len(train) + len(test) == len(samples)
        # 50/50 split, allow rounding
        assert abs(len(train) - len(test)) <= 1

    def test_split_deterministic(self):
        samples = load_dataset("cotcontrol/gpqa")
        train1 = split_dataset(samples, split="train", seed=42)
        train2 = split_dataset(samples, split="train", seed=42)
        assert [s.id for s in train1] == [s.id for s in train2]

    def test_split_all_returns_everything(self):
        samples = load_dataset("cotcontrol/gpqa")
        all_samples = split_dataset(samples, split="all")
        assert len(all_samples) == len(samples)

    def test_fraction_subsamples(self):
        samples = load_dataset("cotcontrol/gpqa")
        subsampled = split_dataset(samples, split="test", fraction=0.1)
        full = split_dataset(samples, split="test")
        assert len(subsampled) < len(full)
        assert len(subsampled) >= 1

    def test_reasonif_stratified(self):
        samples = load_dataset("reasonif")
        stats = get_split_stats(samples)
        assert "train_by_type" in stats
        assert "test_by_type" in stats
        # All instruction types should be present in both splits
        assert set(stats["train_by_type"].keys()) == set(stats["test_by_type"].keys())
