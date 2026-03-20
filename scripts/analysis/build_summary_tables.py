"""Build summary CSV tables from eval_manifest.csv and rollout JSONL files.

Produces two tables in results/summaries/tables/:
  1. summary_by_run.csv       — one row per eval run (JSONL file)
  2. summary_by_instruction.csv — one row per (eval run, instruction_type)

Reasoning error classification:
  - "reasoning_only":  reasoning present, response empty (token budget exhausted)
  - "response_only":   response present, reasoning empty (no </think> tag emitted)
  - "both_empty":      neither reasoning nor response (total API failure)

Metrics:
  - reasoning_error_rate: fraction of rollouts with any reasoning error
  - correct_rate, compliant_rate, meta_discussion_rate: computed on clean rows only
  - correct_rate_all, compliant_rate_all, meta_discussion_rate_all: computed on all rows

Usage:
    python scripts/analysis/build_summary_tables.py
"""

import csv
import json
from pathlib import Path

MANIFEST_PATH = Path("results/eval_manifest.csv")
ROLLOUTS_DIR = Path("results/rollouts")
OUT_DIR = Path("results/summaries/tables")


def load_manifest() -> list[dict]:
    with open(MANIFEST_PATH) as f:
        return list(csv.DictReader(f))


def load_rollouts(file_name: str) -> list[dict]:
    path = ROLLOUTS_DIR / file_name
    rollouts = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        try:
            rollouts.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rollouts


def has_reasoning_error(r: dict) -> bool:
    reasoning = (r.get("reasoning") or "").strip()
    response = (r.get("response") or "").strip()
    return not reasoning or not response


def classify_error(r: dict) -> str | None:
    """Return error type or None if clean."""
    reasoning = (r.get("reasoning") or "").strip()
    response = (r.get("response") or "").strip()
    has_rsn = len(reasoning) > 0
    has_rsp = len(response) > 0
    if has_rsn and has_rsp:
        return None
    if has_rsn and not has_rsp:
        return "reasoning_only"
    if not has_rsn and has_rsp:
        return "response_only"
    return "both_empty"


def compute_metrics(rollouts: list[dict]) -> dict:
    """Compute metrics for a group of rollouts."""
    n_total = len(rollouts)
    if n_total == 0:
        return {
            "n_rollouts_actual": 0,
            "reasoning_error_rate": None,
            "n_reasoning_errors": 0,
            "n_clean": 0,
            "correct_rate": None,
            "compliant_rate": None,
            "meta_discussion_rate": None,
            "correct_rate_all": None,
            "compliant_rate_all": None,
            "meta_discussion_rate_all": None,
        }

    clean = [r for r in rollouts if not has_reasoning_error(r)]
    n_errors = n_total - len(clean)

    # --- All-rows metrics (treat None as False) ---
    correct_all = sum(1 for r in rollouts if r.get("correct") is True) / n_total
    compliant_all = sum(1 for r in rollouts if r.get("compliant") is True) / n_total
    meta_all = sum(1 for r in rollouts if r.get("meta_discussion") is True) / n_total

    # --- Clean-rows metrics ---
    if clean:
        n_clean = len(clean)
        correct_clean = sum(1 for r in clean if r.get("correct") is True) / n_clean
        # For compliant/meta, only count rows where the value is not None
        compliant_graded = [r for r in clean if r.get("compliant") is not None]
        meta_graded = [r for r in clean if r.get("meta_discussion") is not None]
        compliant_clean = (
            sum(1 for r in compliant_graded if r["compliant"]) / len(compliant_graded)
            if compliant_graded
            else None
        )
        meta_clean = (
            sum(1 for r in meta_graded if r["meta_discussion"]) / len(meta_graded)
            if meta_graded
            else None
        )
    else:
        n_clean = 0
        correct_clean = None
        compliant_clean = None
        meta_clean = None

    return {
        "n_rollouts_actual": n_total,
        "reasoning_error_rate": n_errors / n_total,
        "n_reasoning_errors": n_errors,
        "n_clean": n_clean,
        "correct_rate": correct_clean,
        "compliant_rate": compliant_clean,
        "meta_discussion_rate": meta_clean,
        "correct_rate_all": correct_all,
        "compliant_rate_all": compliant_all,
        "meta_discussion_rate_all": meta_all,
    }


def fmt(val) -> str:
    if val is None:
        return ""
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


MANIFEST_COLS = [
    "base_model",
    "checkpoint",
    "eval_type",
    "n_samples",
    "n_rollouts",
    "vendor",
    "reasoning_effort",
    "max_tokens",
    "timestamp",
    "file_name",
]

METRIC_COLS = [
    "n_rollouts_actual",
    "reasoning_error_rate",
    "n_reasoning_errors",
    "n_clean",
    "correct_rate",
    "compliant_rate",
    "meta_discussion_rate",
    "correct_rate_all",
    "compliant_rate_all",
    "meta_discussion_rate_all",
]


def build_tables():
    manifest = load_manifest()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Table 1: one row per run
    table1_cols = MANIFEST_COLS + METRIC_COLS
    table1_rows = []

    # Table 2: one row per (run, instruction_type)
    table2_cols = MANIFEST_COLS + ["instruction_type"] + METRIC_COLS
    table2_rows = []

    for entry in manifest:
        file_name = entry["file_name"]
        rollout_path = ROLLOUTS_DIR / file_name
        if not rollout_path.exists():
            continue

        rollouts = load_rollouts(file_name)
        # Exclude baseline rows from run-level summary (not part of the experiment)
        rollouts_no_baseline = [
            r for r in rollouts if r.get("control_mode") != "baseline"
        ]
        metrics = compute_metrics(rollouts_no_baseline)

        row1 = {col: entry[col] for col in MANIFEST_COLS}
        for col in METRIC_COLS:
            row1[col] = fmt(metrics[col])
        table1_rows.append(row1)

        # Group by instruction type (control_mode key)
        groups: dict[str, list[dict]] = {}
        for r in rollouts:
            mode = r.get("control_mode", "unknown")
            groups.setdefault(mode, []).append(r)

        for mode in sorted(groups):
            group_metrics = compute_metrics(groups[mode])
            row2 = {col: entry[col] for col in MANIFEST_COLS}
            row2["instruction_type"] = mode
            for col in METRIC_COLS:
                row2[col] = fmt(group_metrics[col])
            table2_rows.append(row2)

    # Write table 1
    out1 = OUT_DIR / "summary_by_run.csv"
    with open(out1, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=table1_cols)
        writer.writeheader()
        writer.writerows(table1_rows)
    print(f"Wrote {len(table1_rows)} rows to {out1}")

    # Write table 2
    out2 = OUT_DIR / "summary_by_instruction.csv"
    with open(out2, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=table2_cols)
        writer.writeheader()
        writer.writerows(table2_rows)
    print(f"Wrote {len(table2_rows)} rows to {out2}")


if __name__ == "__main__":
    build_tables()
