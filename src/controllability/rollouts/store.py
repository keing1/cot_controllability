"""JSONL-based rollout persistence."""

from __future__ import annotations

import json
from pathlib import Path

from controllability.types import Rollout


def append_rollout(path: str | Path, rollout: Rollout) -> None:
    """Append a single rollout to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(_sanitize_jsonl(rollout.model_dump_json()) + "\n")


def _sanitize_jsonl(line: str) -> str:
    """Escape Unicode line/paragraph separators that break JSONL."""
    return line.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def append_rollouts(path: str | Path, rollouts: list[Rollout]) -> None:
    """Append multiple rollouts to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for rollout in rollouts:
            f.write(_sanitize_jsonl(rollout.model_dump_json()) + "\n")


def load_rollouts(path: str | Path) -> list[Rollout]:
    """Load all rollouts from a JSONL file."""
    path = Path(path)
    if not path.exists():
        return []
    rollouts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                rollouts.append(Rollout.model_validate_json(line))
    return rollouts


def load_completed_keys(path: str | Path) -> set[tuple[str, str]]:
    """Load set of (sample_id, control_mode) pairs already completed.

    Used for resume support -- skip samples that are already done.
    """
    path = Path(path)
    if not path.exists():
        return set()
    keys = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            data = json.loads(line)
            sample_id = data.get("sample", {}).get("id", "")
            mode = data.get("control_mode", "")
            if sample_id and mode:
                keys.add((sample_id, mode))
    return keys


def write_experiment_header(path: str | Path, config_dict: dict) -> None:
    """Write experiment config as a comment-like header line (prefixed with #)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Only write header if file doesn't exist yet
    if not path.exists() or path.stat().st_size == 0:
        with open(path, "w") as f:
            f.write(f"# {json.dumps(config_dict)}\n")
