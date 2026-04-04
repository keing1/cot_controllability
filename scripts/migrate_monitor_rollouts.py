#!/usr/bin/env python3
"""Migrate existing monitor_qa rollout files to include monitor_results.

Reads JSONL files in the top-level results/rollouts/monitor_qa/ directory
(not subfolders) and backfills the new ``monitor_results`` list from the
existing flat monitor fields. The monitor type is set to "old_monitor".

If ``monitor_score_old`` / ``monitor_response_old`` fields are present
(from prior rescoring), a second entry is added with those values.

Usage:
    python scripts/migrate_monitor_rollouts.py
    python scripts/migrate_monitor_rollouts.py --backup-dir results/rollouts/monitor_qa/pre_migration/
    python scripts/migrate_monitor_rollouts.py --dry-run
"""

import argparse
import json
import shutil
import sys
from pathlib import Path


ROLLOUT_DIR = Path(__file__).resolve().parents[1] / "results" / "rollouts" / "monitor_qa"


def migrate_file(path: Path, backup_dir: Path | None = None) -> dict:
    """Migrate a single JSONL file. Returns a summary dict."""
    header_lines: list[str] = []
    data_lines: list[str] = []
    migrated = 0
    skipped = 0

    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                header_lines.append(line)
                continue

            try:
                data = json.loads(stripped)
            except json.JSONDecodeError:
                data_lines.append(line)
                continue

            # Skip if already migrated
            if data.get("monitor_results"):
                data_lines.append(line)
                skipped += 1
                continue

            # Build monitor_results from flat fields
            results = []

            # Current monitor scores
            results.append({
                "monitor_type": "old_monitor",
                "monitor_model": data.get("monitor_model", ""),
                "monitor_prompt": data.get("monitor_prompt", []),
                "monitor_response": data.get("monitor_response", ""),
                "monitor_score": data.get("monitor_score"),
                "monitor_latency_ms": data.get("monitor_latency_ms", 0.0),
                "monitor_usage": data.get("monitor_usage", {}),
                "monitor_error": data.get("monitor_error"),
            })

            # Old scores from prior rescoring (if present)
            if "monitor_score_old" in data:
                results.append({
                    "monitor_type": "old_monitor_pre_rescore",
                    "monitor_model": data.get("monitor_model", ""),
                    "monitor_prompt": [],
                    "monitor_response": data.get("monitor_response_old", ""),
                    "monitor_score": data.get("monitor_score_old"),
                    "monitor_latency_ms": 0.0,
                    "monitor_usage": {},
                    "monitor_error": None,
                })

            data["monitor_results"] = results
            data_lines.append(json.dumps(data) + "\n")
            migrated += 1

    if migrated > 0:
        # Backup if requested
        if backup_dir:
            backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup_dir / path.name)

        # Write in place
        with open(path, "w") as f:
            for line in header_lines:
                f.write(line if line.endswith("\n") else line + "\n")
            for line in data_lines:
                f.write(line if line.endswith("\n") else line + "\n")

    return {
        "file": path.name,
        "migrated": migrated,
        "skipped": skipped,
        "total": migrated + skipped,
    }


def main():
    parser = argparse.ArgumentParser(description="Migrate monitor_qa rollouts to multi-monitor format")
    parser.add_argument("--backup-dir", type=str, default=None,
                        help="Directory to save backup copies before migration")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be migrated without modifying files")
    args = parser.parse_args()

    if not ROLLOUT_DIR.exists():
        print(f"Rollout directory not found: {ROLLOUT_DIR}")
        sys.exit(1)

    # Top-level JSONL files only (not subfolders)
    files = sorted(f for f in ROLLOUT_DIR.iterdir() if f.suffix == ".jsonl" and f.is_file())

    if not files:
        print("No .jsonl files found in", ROLLOUT_DIR)
        return

    print(f"Found {len(files)} JSONL file(s) in {ROLLOUT_DIR}")

    if args.dry_run:
        for path in files:
            total = 0
            already_migrated = 0
            needs_migration = 0
            with open(path) as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    try:
                        r = json.loads(s)
                    except json.JSONDecodeError:
                        continue
                    total += 1
                    if r.get("monitor_results"):
                        already_migrated += 1
                    else:
                        needs_migration += 1
            print(f"  {path.name}: {total} total, {needs_migration} to migrate, {already_migrated} already done")
        return

    summaries = []
    for path in files:
        summary = migrate_file(path, backup_dir=Path(args.backup_dir) if args.backup_dir else None)
        summaries.append(summary)
        if summary["migrated"] > 0:
            print(f"  {summary['file']}: migrated {summary['migrated']}/{summary['total']}")
        else:
            print(f"  {summary['file']}: nothing to migrate ({summary['skipped']} already done)")

    total_migrated = sum(s["migrated"] for s in summaries)
    print(f"\nDone. Migrated {total_migrated} rollouts across {len(files)} files.")


if __name__ == "__main__":
    main()
