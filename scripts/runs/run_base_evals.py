#!/usr/bin/env python3
"""Run eval suite against BASE (pre-FT) model weights for a set of models.

Uses ``launch_ft.py --single-eval ... --checkpoint base`` per (model, dataset)
combination — the ``base`` sentinel tells launch_ft to skip checkpoint lookup
and hand ``model_path=None`` to the inference client, so Tinker samples from
the base model directly.

Usage:
    # Default: run all 3 new base models through the full eval suite
    uv run python scripts/runs/run_base_evals.py

    # Dry-run:
    uv run python scripts/runs/run_base_evals.py --dry-run

    # Only specific models:
    uv run python scripts/runs/run_base_evals.py --models kimi-k2.5,deepseek-v3.1
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "ft_mixed_5k_lr3e-5_newmodels.yaml"
DEFAULT_MODELS = "deepseek-v3.1,kimi-k2.5,nemotron3-nano"


def _launch_eval(model: str, ds_key: str, config_path: Path,
                 run_id: str, log_dir: Path) -> tuple[str, int]:
    """Spawn one ``launch_ft.py --single-eval`` subprocess for (model, ds)."""
    tag = f"{model}/{ds_key}"
    safe_ds = ds_key.replace("/", "_").replace("@", "__")
    log = log_dir / f"{run_id}_{model}_{safe_ds}_base.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "runs" / "launch_ft.py"),
        "--config", str(config_path),
        "--single-eval", model,
        "--dataset", ds_key,
        "--checkpoint", "base",
        "--run-id", run_id,
    ]
    print(f"  launching: {tag} → {log.name}")
    with open(log, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                cwd=str(PROJECT_ROOT))
    ret = proc.wait()
    status = "OK" if ret == 0 else f"FAIL({ret})"
    print(f"  done:      {tag} → {status}")
    return tag, ret


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run base-model evals for 3 new models across the full eval suite"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="YAML with models + eval.datasets (default: newmodels config)")
    parser.add_argument("--models", default=DEFAULT_MODELS,
                        help="Comma-separated short_names (default: 3 new models, no Super)")
    parser.add_argument("--max-workers", type=int, default=6,
                        help="Max concurrent eval subprocesses (default: 6)")
    parser.add_argument("--run-id", default=None,
                        help="Run ID for filename tagging (default: 'base_' + UTC timestamp)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id or ("base_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    models_filter = set(args.models.split(","))

    cfg = yaml.safe_load(open(args.config))

    # Filter model list
    model_shorts = [m["short_name"] for m in cfg["models"]
                    if m.get("short_name") in models_filter]
    if not model_shorts:
        print(f"ERROR: no models in config match {models_filter}")
        sys.exit(1)

    # Build the (model, ds_key) matrix. ds_key follows launch_ft's convention:
    #   standard: 'cotcontrol', 'reasonif'
    #   monitor_qa with prompt_settings: 'monitor_qa@<label>'
    eval_datasets: list[str] = []
    for d in cfg["eval"]["datasets"]:
        if d.get("dataset") == "monitor_qa" and "prompt_settings" in d:
            for ps in d["prompt_settings"]:
                eval_datasets.append(f"monitor_qa@{ps['label']}")
        else:
            eval_datasets.append(d["dataset"])

    jobs = [(m, d) for m in model_shorts for d in eval_datasets]

    print(f"Run ID: {run_id}")
    print(f"Models: {model_shorts}")
    print(f"Eval datasets: {eval_datasets}")
    print(f"Total jobs: {len(jobs)}  (max {args.max_workers} concurrent)")
    print()
    if args.dry_run:
        for m, d in jobs:
            print(f"  DRY RUN: {m} / {d}")
        return

    log_dir = PROJECT_ROOT / "results" / "logs"
    t0 = time.time()
    results: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futs = {pool.submit(_launch_eval, m, d, args.config, run_id, log_dir): f"{m}/{d}"
                for m, d in jobs}
        for fut in as_completed(futs):
            tag = futs[fut]
            try:
                _, ret = fut.result()
                results[tag] = "OK" if ret == 0 else f"FAIL({ret})"
            except Exception as e:  # noqa: BLE001
                results[tag] = f"EXC({e})"

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}\nBASE EVAL COMPLETE — {elapsed / 60:.1f} min\n{'=' * 60}")
    for k, v in sorted(results.items()):
        print(f"  {k:50s} {v}")
    n_ok = sum(1 for v in results.values() if v == "OK")
    print(f"\n  {n_ok}/{len(results)} succeeded")


if __name__ == "__main__":
    main()
