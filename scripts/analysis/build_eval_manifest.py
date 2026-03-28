"""Build a CSV manifest of all eval rollout files."""

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import glob

OUTPATH = Path("results/eval_manifest.csv")

COLUMNS = [
    "base_model",
    "checkpoint",
    "eval_type",
    "n_samples",
    "n_rollouts",
    "vendor",
    "reasoning_effort",
    "analysis_channel",
    "max_tokens",
    "timestamp",
    "file_name",
]

EXCLUDE_MODELS = {"deepseek/deepseek-r1"}

rows = []
for f in sorted(glob.glob("results/rollouts/*.jsonl")):
    if "debug" in f:
        continue
    p = Path(f)
    lines = p.read_text().splitlines()
    header = lines[0]
    if not header.startswith("#"):
        continue
    config = json.loads(header[2:])

    if config.get("model", "") in EXCLUDE_MODELS:
        continue

    # Count actual rollouts (non-empty, non-comment lines)
    n_rollouts = sum(1 for line in lines if line.strip() and not line.startswith("#"))

    # Get timestamp from first rollout
    timestamp = ""
    for line in lines:
        if not line.strip() or line.startswith("#"):
            continue
        try:
            r = json.loads(line)
            ts = r.get("timestamp", "")
            if ts:
                # Parse and format with second precision
                timestamp = ts[:19]
            break
        except json.JSONDecodeError:
            continue

    # If no rollout timestamp, fall back to file mtime
    if not timestamp:
        mtime = os.path.getmtime(f)
        timestamp = datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    checkpoint = config.get("model_path", "")
    n_samples = config.get("n_samples")
    if n_samples is None:
        n_samples = ""

    reasoning_effort = config.get("reasoning_effort", "")
    analysis_channel = config.get("analysis_channel", False)
    max_tokens = config.get("max_tokens", "")

    rows.append({
        "base_model": config.get("model", ""),
        "checkpoint": checkpoint,
        "eval_type": config.get("dataset", ""),
        "n_samples": n_samples,
        "n_rollouts": n_rollouts,
        "vendor": config.get("backend", ""),
        "reasoning_effort": reasoning_effort,
        "analysis_channel": analysis_channel,
        "max_tokens": max_tokens,
        "timestamp": timestamp,
        "file_name": p.name,
    })

OUTPATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPATH, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {OUTPATH}")
