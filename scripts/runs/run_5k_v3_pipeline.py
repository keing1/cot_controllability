#!/usr/bin/env python3
"""Two-phase pipeline driver for the third 5K-mixed FT iteration
(``configs/ft_mixed_5k_v3_8models.yaml``).

Phase A: launch all 8 FTs in parallel + run Phase-1 evals
         (cotcontrol, reasonif, monitor_qa@baseline, monitor_qa@less_reasoning_v2)
         on checkpoints {000050, 000100, 000300, final}, priority order
         [000100, 000300, final, 000050].

Phase B: once Phase A is fully complete, run Phase-2 evals
         (monitor_qa@vocab_control, monitor_qa@dont_think_v2) on the same
         checkpoints — re-using the Phase-A run_id so checkpoints resolve.

Both phases use launch_ft.py under the hood; this script just chains them
and forwards a stable run_id.

Usage:
    uv run python scripts/runs/run_5k_v3_pipeline.py --dry-run
    uv run python scripts/runs/run_5k_v3_pipeline.py
    uv run python scripts/runs/run_5k_v3_pipeline.py --skip-phase-a --run-id 20260428_XXXXXX
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = PROJECT_ROOT / "scripts" / "runs" / "launch_ft.py"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "ft_mixed_5k_v3_8models.yaml"

PHASE_A_DATASETS = ",".join([
    "cotcontrol",
    "reasonif",
    "monitor_qa@baseline",
    "monitor_qa@less_reasoning_v2",
])
PHASE_B_DATASETS = ",".join([
    "monitor_qa@vocab_control",
    "monitor_qa@dont_think_v2",
])


def _run(cmd: list[str], dry_run: bool) -> int:
    print(">>> " + " ".join(str(c) for c in cmd))
    if dry_run:
        return 0
    return subprocess.call([str(c) for c in cmd], cwd=str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id", default=None,
                        help="Override run_id (must be reused across phases). "
                             "Auto-generated if omitted.")
    parser.add_argument("--skip-phase-a", action="store_true",
                        help="Skip Phase A (FTs + Phase-1 evals) and only run "
                             "Phase B. Requires --run-id.")
    parser.add_argument("--skip-phase-b", action="store_true",
                        help="Run Phase A only; do not start Phase B.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    print(f"Pipeline run_id: {run_id}")
    print(f"Config:         {args.config}")
    print()

    if args.skip_phase_a and not args.run_id:
        parser.error("--skip-phase-a requires an explicit --run-id")

    # ---- Phase A: FTs + Phase-1 evals ----
    if not args.skip_phase_a:
        print("=" * 70)
        print("PHASE A — FTs + Phase-1 evals (cotcontrol, reasonif, "
              "mqa@baseline, mqa@less_reasoning_v2)")
        print("=" * 70)
        cmd = [
            sys.executable, str(LAUNCHER),
            "--config", str(args.config),
            "--run-id", run_id,
            "--datasets-filter", PHASE_A_DATASETS,
        ]
        if args.dry_run:
            cmd.append("--dry-run")
        rc = _run(cmd, dry_run=False)  # _run() already handles printing; we want
                                        # to pass --dry-run so launch_ft prints
                                        # its plan, not no-op the whole thing.
        if rc != 0:
            print(f"Phase A exited with code {rc} — aborting before Phase B")
            sys.exit(rc)
    else:
        print("Skipping Phase A (--skip-phase-a)")

    # ---- Phase B: Phase-2 evals only (eval-only, reuses Phase-A run_id) ----
    if args.skip_phase_b:
        print("Skipping Phase B (--skip-phase-b)")
        return

    print()
    print("=" * 70)
    print("PHASE B — Phase-2 evals (mqa@vocab_control, mqa@dont_think_v2)")
    print("=" * 70)
    cmd = [
        sys.executable, str(LAUNCHER),
        "--config", str(args.config),
        "--run-id", run_id,
        "--eval-only",
        "--datasets-filter", PHASE_B_DATASETS,
    ]
    if args.dry_run:
        cmd.append("--dry-run")
    rc = _run(cmd, dry_run=False)
    sys.exit(rc)


if __name__ == "__main__":
    main()
