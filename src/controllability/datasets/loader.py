"""Dataset loading from CoTControl and ReasonIF submodules."""

from __future__ import annotations

import ast
import csv
import json
import random
from pathlib import Path

from controllability.types import Sample

# Root of the repo (3 levels up from this file)
_REPO_ROOT = Path(__file__).resolve().parents[3]

_COTCONTROL_DATA = _REPO_ROOT / "external" / "CoTControl" / "CoT-Control-QA" / "datasets"
_REASONIF_DATA = _REPO_ROOT / "external" / "reasonIF" / "data"

# Registry: dataset name -> loader function
_LOADERS: dict[str, callable] = {}


def _register(name: str):
    def decorator(fn):
        _LOADERS[name] = fn
        return fn
    return decorator


def list_datasets() -> list[str]:
    """Return all registered dataset names."""
    return sorted(_LOADERS.keys())


def load_dataset(name: str) -> list[Sample]:
    """Load a dataset by registered name."""
    if name not in _LOADERS:
        available = ", ".join(list_datasets())
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")
    return _LOADERS[name]()


def _parse_options(raw: str | None) -> list[str] | None:
    """Parse answer_options from CSV string like "['A', 'B', 'C', 'D']"."""
    if not raw or raw.strip() == "":
        return None
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return parsed
    except (ValueError, SyntaxError):
        pass
    return None


def _parse_keywords_with_synonyms(raw: str | None) -> list[dict]:
    """Parse keywords_with_synonyms from CSV string."""
    if not raw or raw.strip() == "":
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return parsed
    except (ValueError, SyntaxError):
        pass
    return []


def _load_cotcontrol_csv(csv_path: Path, dataset_name: str) -> list[Sample]:
    """Load a CoTControl CSV file into Sample objects."""
    samples = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            keywords_data = _parse_keywords_with_synonyms(row.get("keywords_with_synonyms"))

            # Extract flat keyword list and synonym mapping
            keywords = []
            synonyms_map = {}
            for kw_entry in keywords_data:
                kw = kw_entry.get("keyword", "")
                syns = kw_entry.get("synonyms", [])
                if kw:
                    keywords.append(kw)
                    if syns:
                        synonyms_map[kw] = syns

            valid_keywords_raw = row.get("valid_keywords", "")
            try:
                valid_keywords = ast.literal_eval(valid_keywords_raw) if valid_keywords_raw else []
            except (ValueError, SyntaxError):
                valid_keywords = []

            sample = Sample(
                id=f"{dataset_name}_{i}",
                dataset=dataset_name,
                question=row["question"],
                correct_answer=row["answer"],
                options=_parse_options(row.get("answer_options")),
                metadata={
                    "source": row.get("source", ""),
                    "domain": row.get("domain", ""),
                    "keywords": keywords,
                    "valid_keywords": valid_keywords,
                    "synonyms_map": synonyms_map,
                },
            )
            samples.append(sample)
    return samples


@_register("cotcontrol/gpqa")
def _load_gpqa() -> list[Sample]:
    return _load_cotcontrol_csv(_COTCONTROL_DATA / "gpqa_w_keyword.csv", "cotcontrol/gpqa")


@_register("cotcontrol/hle")
def _load_hle() -> list[Sample]:
    return _load_cotcontrol_csv(_COTCONTROL_DATA / "hle_w_keyword.csv", "cotcontrol/hle")


@_register("cotcontrol/mmlu_pro")
def _load_mmlu_pro() -> list[Sample]:
    return _load_cotcontrol_csv(_COTCONTROL_DATA / "mmlu_pro_mini_w_keyword.csv", "cotcontrol/mmlu_pro")


@_register("cotcontrol")
def _load_cotcontrol_combined() -> list[Sample]:
    """Load all CoTControl QA datasets combined (GPQA + HLE + MMLU-Pro)."""
    gpqa = _load_cotcontrol_csv(_COTCONTROL_DATA / "gpqa_w_keyword.csv", "cotcontrol/gpqa")
    hle = _load_cotcontrol_csv(_COTCONTROL_DATA / "hle_w_keyword.csv", "cotcontrol/hle")
    mmlu = _load_cotcontrol_csv(_COTCONTROL_DATA / "mmlu_pro_mini_w_keyword.csv", "cotcontrol/mmlu_pro")
    return gpqa + hle + mmlu


@_register("reasonif")
def _load_reasonif() -> list[Sample]:
    """Load ReasonIF dataset from JSON."""
    data_path = _REASONIF_DATA / "reasonIF_dataset.json"
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for i, entry in enumerate(data):
        # constraint_name is a list like ["punctuation:no_comma"]
        constraint_names = entry.get("constraint_name", [])
        constraint_args = entry.get("constraint_args", [])

        # Map full constraint name to short name
        # e.g. "punctuation:no_comma" -> "no_comma"
        instruction_type = constraint_names[0].split(":")[-1] if constraint_names else ""

        sample = Sample(
            id=f"reasonif_{i}",
            dataset="reasonif",
            question=entry["question"],
            correct_answer=entry["answer"],
            options=None,
            metadata={
                "source": entry.get("source", ""),
                "hf_id": entry.get("hf_id", ""),
                "instruction_type": instruction_type,
                "constraint_name": constraint_names[0] if constraint_names else "",
                "constraint_args": constraint_args[0] if constraint_args else None,
                "original_prompt": entry.get("prompt", ""),
            },
        )
        samples.append(sample)
    return samples


# ---------------------------------------------------------------------------
# Monitor QA dataset loader (not in the registry — takes a path)
# ---------------------------------------------------------------------------


def load_monitor_dataset(path: str | Path) -> list[Sample]:
    """Load a monitor_qa dataset from a JSONL file.

    Each line should be a JSON object with:
        main_task, side_task, side_task_answer, id (optional), source (optional), metadata (optional)

    Maps to Sample with question=main_task, correct_answer="",
    and side task info stored in metadata.
    """
    path = Path(path)
    samples = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            entry = json.loads(line)
            extra_meta = entry.get("metadata", {})
            sample_id = entry.get("id", f"mqa_{i}")
            sample = Sample(
                id=str(sample_id),
                dataset="monitor_qa",
                question=entry["main_task"],
                correct_answer="",
                metadata={
                    "side_task": entry["side_task"],
                    "side_task_answer": str(entry.get("side_task_answer", "")),
                    "source": entry.get("source", ""),
                    **extra_meta,
                },
            )
            samples.append(sample)
    return samples


def subsample_dataset(
    samples: list[Sample], n: int | None, seed: int = 42,
) -> list[Sample]:
    """Randomly subsample a dataset. Returns all samples if n is None or >= len."""
    if n is None or n >= len(samples):
        return samples
    rng = random.Random(seed)
    return rng.sample(samples, n)
