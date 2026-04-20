#!/usr/bin/env python3
"""Rerun eval files with elevated error/timeout rates.

For a given run, finds any rollout JSONL whose error rate (including stealth
timeouts with empty response+reasoning) is >= ``--threshold``, strips the
errored rows in-place (preserving ``#`` header lines), then dispatches a
fresh ``launch_ft.py --single-eval`` subprocess per affected file. The eval
runner's resume logic picks up where we left off, so only the stripped rows
get regenerated.

Relies on:
  * ``config.experiment_id`` being stable under timeout/concurrency changes
    (it is — those fields aren't hashed). So the new run writes to the same
    ``output_filename_override`` and the resume path deduplicates.

Usage:
  python scripts/runs/rerun_elevated_errors.py \\
    --config configs/ft_mixed_1000.yaml \\
    --run-id 20260417_034641

  # Preview without touching files / dispatching:
  python scripts/runs/rerun_elevated_errors.py ... --dry-run

  # Only rerun files above a different cutoff (default 10%):
  python scripts/runs/rerun_elevated_errors.py ... --threshold 5

  # Limit concurrency of the rerun subprocesses themselves:
  python scripts/runs/rerun_elevated_errors.py ... --max-workers 3
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
ROLLOUTS_DIR = RESULTS_DIR / "rollouts"
MQA_DIR = ROLLOUTS_DIR / "monitor_qa"


# ---------------------------------------------------------------------------
# Error detection (mirrors the analysis we've been running ad-hoc)
# ---------------------------------------------------------------------------


def row_is_errored(record: dict, is_monitor_qa: bool) -> bool:
    """Classify a rollout row as errored/timed-out.

    Explicit error field OR empty response+reasoning (stealth timeouts from
    the batch wrapper's sample_timeout path which returns empty content).
    """
    if is_monitor_qa:
        if record.get("actor_error"):
            return True
    else:
        if record.get("error"):
            return True
    resp = (record.get("response") or "").strip()
    reason = (record.get("reasoning") or "").strip()
    if not resp and not reason:
        return True
    return False


def scan_file(path: Path, is_monitor_qa: bool) -> tuple[int, int]:
    total, errors = 0, 0
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
            if row_is_errored(r, is_monitor_qa):
                errors += 1
    return errors, total


def strip_errored_rows(path: Path, is_monitor_qa: bool) -> int:
    """Remove errored rows in-place. Returns number removed."""
    headers: list[str] = []
    keep: list[str] = []
    removed = 0
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                headers.append(line if line.endswith("\n") else line + "\n")
                continue
            try:
                r = json.loads(s)
            except json.JSONDecodeError:
                continue
            if row_is_errored(r, is_monitor_qa):
                removed += 1
            else:
                keep.append(json.dumps(r) + "\n")
    with open(path, "w") as f:
        for h in headers:
            f.write(h)
        for k in keep:
            f.write(k)
    return removed


# ---------------------------------------------------------------------------
# Parse (model, dataset, checkpoint) out of a rollout filename
# ---------------------------------------------------------------------------


# Standard eval: {actor_short}-ft-{run_id}-{ckpt}_{dataset}_all_{run_id}_{hash}.jsonl
# actor_short has "/" replaced with "_", e.g. "openai_gpt-oss-20b"
_STD_RE = re.compile(
    r"^(?P<actor>.+)-ft-(?P<run>\d{8}_\d{6})-(?P<ckpt>\w+?)_(?P<dataset>cotcontrol|reasonif)_all_\2_[0-9a-f]+\.jsonl$"
)

# Monitor_qa: monitor_qa_{actor_short}-ft-{run_id}-{ckpt}_{label}_{hash}.jsonl
_MQA_RE = re.compile(
    r"^monitor_qa_(?P<actor>.+)-ft-(?P<run>\d{8}_\d{6})-(?P<ckpt>\w+?)_(?P<label>[a-z_0-9]+?)_[0-9a-f]+\.jsonl$"
)


def parse_filename(path: Path, run_id: str, short_name_by_actor: dict[str, str]) -> tuple[str, str, str] | None:
    """Return (short_name, ds_key, checkpoint_name) or None if not parseable."""
    name = path.name
    m = _STD_RE.match(name)
    if m and m.group("run") == run_id:
        short = short_name_by_actor.get(m.group("actor"))
        if not short:
            return None
        return short, m.group("dataset"), m.group("ckpt")
    m = _MQA_RE.match(name)
    if m and m.group("run") == run_id:
        short = short_name_by_actor.get(m.group("actor"))
        if not short:
            return None
        ds_key = f"monitor_qa@{m.group('label')}"
        return short, ds_key, m.group("ckpt")
    return None


def build_short_name_lookup(cfg) -> dict[str, str]:
    """``openai_gpt-oss-20b`` -> ``gpt-oss-20b`` (etc.) based on config."""
    out: dict[str, str] = {}
    for m in cfg.models:
        actor_file_form = m.model.replace("/", "_")
        out[actor_file_form] = m.short_name
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rerun elevated-error eval files from a launch_ft run"
    )
    parser.add_argument("--config", type=Path, required=True, help="YAML config")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--threshold", type=float, default=10.0,
                        help="Min error rate %% to trigger rerun (default 10)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-workers", type=int, default=3,
                        help="Max concurrent rerun subprocesses (default 3)")
    args = parser.parse_args()

    # Load launch_ft.py as a module so we can reuse LaunchConfig
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "runs"))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "launch_ft", PROJECT_ROOT / "scripts" / "runs" / "launch_ft.py"
    )
    launch_ft = importlib.util.module_from_spec(spec)
    sys.modules["launch_ft"] = launch_ft
    spec.loader.exec_module(launch_ft)

    cfg = launch_ft.LaunchConfig.from_yaml(args.config, override_run_id=args.run_id)
    short_lookup = build_short_name_lookup(cfg)

    # Scan both directories
    candidates: list[tuple[Path, bool]] = []
    for p in sorted(ROLLOUTS_DIR.glob(f"*ft-{args.run_id}*.jsonl")):
        candidates.append((p, False))
    for p in sorted(MQA_DIR.glob(f"*{args.run_id}*.jsonl")):
        candidates.append((p, True))

    print(f"Scanning {len(candidates)} rollout files for run {args.run_id}...")
    elevated: list[tuple[Path, bool, int, int, float]] = []
    for p, is_mqa in candidates:
        errs, total = scan_file(p, is_mqa)
        if total == 0:
            continue
        rate = errs / total * 100
        if rate >= args.threshold:
            elevated.append((p, is_mqa, errs, total, rate))

    if not elevated:
        print(f"No files >= {args.threshold}%. Nothing to do.")
        return

    print(f"\n{len(elevated)} file(s) above {args.threshold}% error rate:")
    for p, is_mqa, errs, total, rate in elevated:
        print(f"  {errs:>4d}/{total:<4d} ({rate:5.1f}%)  {p.name}")

    # Parse each filename to get (short_name, ds_key, ckpt)
    jobs: list[tuple[Path, bool, str, str, str]] = []
    for p, is_mqa, _, _, _ in elevated:
        parsed = parse_filename(p, args.run_id, short_lookup)
        if parsed is None:
            print(f"  WARN: could not parse filename, skipping: {p.name}")
            continue
        jobs.append((p, is_mqa, *parsed))

    print(f"\nWill strip + redispatch {len(jobs)} eval(s):")
    for p, is_mqa, short, ds_key, ckpt in jobs:
        print(f"  {short:>14s}  {ds_key:<40s}  ckpt={ckpt}  ({p.name})")

    if args.dry_run:
        print("\n[DRY RUN] no files stripped, no subprocesses launched")
        return

    # Strip errored rows in-place
    print("\nStripping errored rows...")
    is_mqa_by_path = {p: is_mqa for p, is_mqa, _, _, _ in elevated}
    for p, is_mqa, short, ds_key, ckpt in jobs:
        removed = strip_errored_rows(p, is_mqa_by_path[p])
        print(f"  stripped {removed} rows from {p.name}")

    # Dispatch in parallel (bounded by --max-workers)
    logs_dir = RESULTS_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    def dispatch(job: tuple[Path, bool, str, str, str]) -> tuple[str, int]:
        p, _, short, ds_key, ckpt = job
        safe_ds = ds_key.replace("/", "_").replace("@", "__")
        log_file = logs_dir / f"{args.run_id}_{short}_{safe_ds}_{ckpt}_rerun.log"
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "runs" / "launch_ft.py"),
            "--config", str(args.config),
            "--single-eval", short,
            "--dataset", ds_key,
            "--checkpoint", ckpt,
            "--run-id", args.run_id,
        ]
        tag = f"{short}/{ds_key}/{ckpt}"
        print(f"  launching: {tag} -> {log_file.name}")
        with open(log_file, "w") as lf:
            proc = subprocess.Popen(
                cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(PROJECT_ROOT),
            )
        ret = proc.wait()
        return tag, ret

    print(f"\nDispatching with max {args.max_workers} concurrent subprocesses...")
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futs = [pool.submit(dispatch, j) for j in jobs]
        for fut in as_completed(futs):
            tag, ret = fut.result()
            status = "OK" if ret == 0 else f"FAIL({ret})"
            print(f"  done: {tag} -> {status}")

    print("\nRerun sweep complete.")


if __name__ == "__main__":
    main()
