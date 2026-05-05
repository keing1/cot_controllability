#!/usr/bin/env python3
"""Remake `results/summaries/plots/2026_04_13/top3_stacked_per_model_gpt54.png`
using the new `metr_note_v3` monitor (GPT-5.4 instant, non-thinking) instead
of `metr_note_gpt54` (GPT-5.4 thinking).

Scope of v3 scores (see scripts/runs/rescore_metr_note_v3_top3_plot.py +
rescore_metr_note_v3.py): 4 prompt variants — gray_swan, less_reasoning_v2,
vocab_control, dont_think_v2 — plus baseline, across 4 model families ×
{Base, FT}. The 7 other prompt variants in the original plot have no v3
scores, so they naturally rank last (0% evasion) and never enter the top-3.

Output subfolder: results/summaries/plots/2026_04_13/metr_note_v3/

Usage:
    uv run python scripts/analysis/plotting/plot_top3_stacked_metr_note_v3_2026_04_13.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis" / "plotting"))

import plot_monitor_qa_2026_04_13 as pmq

MONITOR_V3 = "metr_note_v3"
OUT_SUBDIR = pmq.OUTDIR / "metr_note_v3"


def main() -> None:
    OUT_SUBDIR.mkdir(parents=True, exist_ok=True)
    # Redirect the existing plot helpers to the subfolder.
    pmq.OUTDIR = OUT_SUBDIR

    df = pmq.load_sweep_data()
    score_col = f"score_{MONITOR_V3}"
    if score_col not in df.columns or not df[score_col].notna().any():
        raise RuntimeError(
            f"No {MONITOR_V3} scores found in sweep data. "
            "Run scripts/runs/rescore_metr_note_v3_top3_plot.py first."
        )

    thresholds = pmq.compute_thresholds(df, score_col)
    print(f"\n{MONITOR_V3} thresholds (95th pctl, base/ft):")
    for (g, ft), v in sorted(thresholds.items()):
        print(f"  {g} {'FT' if ft else 'Base'}: {v:.1f}")

    pmq._ensure_monitor_evade_col(df, thresholds, MONITOR_V3, "_evades_v3")

    per_model_ylabel = "Success Rate (%) [side correct + evade]"
    mc_ylabel = "Success Rate (%) [main + side correct + evade]"

    print("\nPlot: top-3 stacked per model — metr_note_v3 (GPT-5.4 instant)...")
    pmq.plot_top3_stacked_per_model(
        df, thresholds, MONITOR_V3,
        title="Top-3 Per Model — metr_note_v3 (GPT-5.4 instant, 5% FPR) — Stacked Breakdown",
        ylabel=per_model_ylabel,
        fname="top3_stacked_per_model_metr_note_v3.png",
        evade_key="_evades_v3",
    )

    print("\nPlot: top-3 stacked per model (main correct) — metr_note_v3...")
    pmq.plot_top3_stacked_per_model(
        df, thresholds, MONITOR_V3,
        title="Top-3 Per Model (Main Task Correct) — metr_note_v3 (GPT-5.4 instant) — Stacked",
        ylabel=mc_ylabel,
        fname="top3_stacked_per_model_metr_note_v3_main_correct.png",
        require_main_correct=True,
        evade_key="_evades_v3",
    )

    print("\nLoading no-reasoning (val split) data...")
    no_reasoning_rates = pmq.load_no_reasoning_rates_val()
    for (g, ft), (rate, err, n) in sorted(no_reasoning_rates.items()):
        print(f"  {g} {'FT' if ft else 'Base'}: {rate*100:.1f}% correct (n={n})")

    print("\nPlot: best prompt vs no reasoning — metr_note_v3...")
    pmq.plot_best_prompt_vs_no_reasoning(
        df,
        evade_key="_evades_v3",
        title="Best Prompt vs No Reasoning — metr_note_v3 (GPT-5.4 instant, 5% FPR)",
        ylabel=per_model_ylabel,
        fname="best_prompt_vs_no_reasoning_metr_note_v3.png",
        no_reasoning_rates=no_reasoning_rates,
        thresholds=thresholds,
        score_col=score_col,
    )

    print(f"\nDone. Outputs in {OUT_SUBDIR}")


if __name__ == "__main__":
    main()
