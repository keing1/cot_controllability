#!/usr/bin/env python3
"""Chain driver: run the 4-new-models 5K FT+eval pipeline, then re-do the
qwen3-32b run from 2026-04-19 (which dropped ~28% of rows to the 300s
request_timeout bug, since fixed).

Sequential by design — Tinker concurrency is already high inside each
phase, and running them in parallel would compete for the same backend.

Total expected wall time: ~8-12 hours end-to-end.

Usage:
    uv run python scripts/runs/run_chain_newmodels_qwen32b_redo.py
    uv run python scripts/runs/run_chain_newmodels_qwen32b_redo.py --dry-run
    uv run python scripts/runs/run_chain_newmodels_qwen32b_redo.py --skip-newmodels
    uv run python scripts/runs/run_chain_newmodels_qwen32b_redo.py --skip-qwen32b
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: list[str]) -> int:
    print(f"\n[chain] $ {' '.join(cmd)}\n", flush=True)
    return subprocess.call(cmd, cwd=str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chain: 4-new-models 5K run → qwen3-32b redo",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-newmodels", action="store_true")
    parser.add_argument("--skip-qwen32b", action="store_true")
    args, extra = parser.parse_known_args()

    start = time.monotonic()

    # -------- Phase A: new-models 5K run --------
    if not args.skip_newmodels:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "runs" / "run_mixed_5k_ft_newmodels.py"),
        ]
        if args.dry_run:
            cmd.append("--dry-run")
        cmd += extra
        rc = _run(cmd)
        if rc != 0:
            print(f"[chain] ERROR: new-models phase exited with code {rc}. "
                  "Aborting before qwen3-32b redo.")
            sys.exit(rc)
        print(f"[chain] new-models phase OK ({(time.monotonic() - start) / 60:.1f} min so far)\n")

    # -------- Phase B: qwen3-32b redo --------
    if not args.skip_qwen32b:
        # --force-rebuild regenerates the SFT parquet with the fixed 1800s
        # timeout + 1:1 question assignment plan. The eval YAML (ft_mixed_5k_lr3e-5)
        # already uses exclude_modes + cotcontrol n_samples=112 per our recent
        # updates, so this rerun picks up the current settings.
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "runs" / "run_mixed_5k_ft.py"),
            "--models", "qwen3-32b",
            "--force-rebuild",
        ]
        if args.dry_run:
            cmd.append("--dry-run")
        rc = _run(cmd)
        if rc != 0:
            print(f"[chain] ERROR: qwen3-32b redo exited with code {rc}.")
            sys.exit(rc)
        print(f"[chain] qwen3-32b redo OK")

    elapsed = time.monotonic() - start
    print(f"\n[chain] complete in {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
