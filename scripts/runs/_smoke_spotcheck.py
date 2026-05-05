#!/usr/bin/env python3
"""Quick spot-check of the smoke sweep parquets.

Prints per-mode compliance + a couple of with-instruction samples (showing
that meta-removal cleaned the rollout and the constraint is satisfied).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

OUT_DIR = Path("results/sft/smoke_with_without")


def _short_name(p: Path) -> str:
    return p.stem.replace("-smoke-with-instr", "/with").replace(
        "-smoke-without-instr", "/without"
    )


def main() -> None:
    if not OUT_DIR.exists():
        print(f"No output dir: {OUT_DIR}", file=sys.stderr)
        return
    parquets = sorted(OUT_DIR.glob("*.parquet"))
    if not parquets:
        print("No parquets yet")
        return

    print("=" * 78)
    print(f"{'Build':<38} {'rows':>5} {'compliant':>10} {'rate':>8}")
    print("-" * 78)
    summaries: list[tuple[str, pd.DataFrame]] = []
    for p in parquets:
        try:
            df = pd.read_parquet(p)
        except Exception as e:  # noqa: BLE001
            print(f"{_short_name(p):<38} ERROR: {e}")
            continue
        n = len(df)
        comp = int(df["compliant"].sum())
        rate = f"{comp / n:.0%}" if n else "N/A"
        print(f"{_short_name(p):<38} {n:>5} {comp:>10} {rate:>8}")
        summaries.append((_short_name(p), df))

    # Per-mode breakdown across all builds
    print()
    print("Per-mode compliance (across all builds):")
    rows: list[dict] = []
    for name, df in summaries:
        for _, r in df.iterrows():
            mode = str(r["constraint_name"]).split("'")[1] if "'" in str(r["constraint_name"]) else "?"
            rows.append({"build": name, "mode": mode, "compliant": bool(r["compliant"])})
    if not rows:
        return
    big = pd.DataFrame(rows)
    by_mode = big.groupby("mode").agg(n=("compliant", "size"), ok=("compliant", "sum"))
    by_mode["rate"] = (by_mode["ok"] / by_mode["n"] * 100).round(0).astype(int).astype(str) + "%"
    print(by_mode.to_string())


if __name__ == "__main__":
    main()
