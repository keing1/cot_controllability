#!/usr/bin/env python3
"""Remake `results/summaries/plots/2026_04_20/mqa_clear_giveaway_grouped.png`
(stacked Base/FT grouped bars, best side-task prompt per group, 4-category
Fully-clean / RA-flagged / Main-only-regex / Clear-giveaway breakdown with
80% Wald CIs) but using:

    * the ft60 FT rollouts + Base val-split rollouts from the 2026-04-13
      monitor_qa sweep (4 models: GPT-OSS-20B, -120B, Qwen3-8B, Qwen3-32B),
    * the metr_note_v3 monitor scores (GPT-5.4 instant, rescored by
      scripts/runs/rescore_metr_note_v3_top3_plot.py + rescore_metr_note_v3.py),
    * the 4 side-task prompt variants we have v3 scores for:
          gray_swan, less_reasoning_v2, vocab_control, dont_think_v2.

Output: results/summaries/plots/2026_04_13/metr_note_v3/
    mqa_clear_giveaway_grouped_metr_note_v3.png
    mqa_clear_giveaway_grouped_metr_note_v3_main_and_side.png

Usage:
    uv run python scripts/analysis/plotting/plot_mqa_clear_giveaway_2026_04_13_metr_note_v3.py
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis" / "plotting"))

# Re-use the 04-20 helpers as a library (stacked_breakdown, threshold, Wald CI, STACK_CATS)
import plot_summary_2026_04_20 as psum

MONITOR = "metr_note_v3"
# Make psum's _monitor_score lookup use v3 instead of v2.
psum.MONITOR = MONITOR

MQA_DIR = REPO_ROOT / "results" / "rollouts" / "monitor_qa"
OUTDIR = REPO_ROOT / "results" / "summaries" / "plots" / "2026_04_13" / "metr_note_v3"

MODEL_GROUPS = ["GPT-OSS-20B", "GPT-OSS-120B", "Qwen3-8B", "Qwen3-32B"]

TOKENS_BASE = {
    "GPT-OSS-20B":  "gpt-oss-20b",
    "GPT-OSS-120B": "gpt-oss-120b",
    "Qwen3-8B":     "qwen3-8b",
    "Qwen3-32B":    "qwen3-32b",
}
TOKENS_FT = {k: v + "-ft60" for k, v in TOKENS_BASE.items()}

RUN_ID_ORIG = "20260409_061534"      # gray_swan
RUN_ID_V2 = "20260414_023958"        # *_v2 + vocab_control
RUN_ID_BASELINE = "20260414_175620"  # val-split baselines

VARIANTS = [
    ("user_gray_swan",         RUN_ID_ORIG),
    ("user_less_reasoning_v2", RUN_ID_V2),
    ("user_vocab_control",     RUN_ID_V2),
    ("user_dont_think_v2",     RUN_ID_V2),
]


def _find_one(pattern: str) -> Path:
    matches = sorted(glob.glob(str(MQA_DIR / pattern)))
    if len(matches) != 1:
        raise FileNotFoundError(f"{len(matches)} matches for {pattern}")
    return Path(matches[0])


def _baseline_path(model: str, is_ft: bool) -> Path:
    tok = (TOKENS_FT if is_ft else TOKENS_BASE)[model]
    return _find_one(f"monitor_qa_{tok}_baseline_val_split_{RUN_ID_BASELINE}_*.jsonl")


def _variant_path(model: str, is_ft: bool, variant: str, rid: str) -> Path:
    tok = (TOKENS_FT if is_ft else TOKENS_BASE)[model]
    return _find_one(f"monitor_qa_{tok}_{variant}_val_split_{rid}_*.jsonl")


def plot_grouped(require_main_task: bool) -> None:
    x = np.arange(len(MODEL_GROUPS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(13, len(MODEL_GROUPS) * 1.8), 6.5))

    base_bd: list[dict[str, float]] = []
    ft_bd: list[dict[str, float]] = []
    base_ns: list[int] = []
    ft_ns: list[int] = []
    best_variant_per_cell: list[tuple[str, str, str]] = []  # (model, split, variant)

    for model in MODEL_GROUPS:
        for is_ft, stash_bd, stash_n in (
            (False, base_bd, base_ns),
            (True,  ft_bd,   ft_ns),
        ):
            thr = psum.threshold_5pct_fpr(
                psum.load_mqa_rows(_baseline_path(model, is_ft))
            )
            best_rate = -1.0
            best_bd = {k: 0.0 for k, _, _, _ in psum.STACK_CATS}
            best_n = 0
            best_var = "(none)"
            for variant, rid in VARIANTS:
                rows = psum.load_mqa_rows(_variant_path(model, is_ft, variant, rid))
                rate, _ = psum.evasion_rate_and_n(
                    rows, thr, require_main_task=require_main_task,
                )
                if not np.isnan(rate) and rate > best_rate:
                    best_rate = rate
                    best_bd, best_n = psum.stacked_breakdown(
                        rows, thr, require_main_task=require_main_task,
                    )
                    best_var = variant
            stash_bd.append(best_bd)
            stash_n.append(best_n)
            best_variant_per_cell.append(
                (model, "FT" if is_ft else "Base", best_var)
            )

    top_values: list[float] = []
    for side_idx, (bds, ns, offset, label) in enumerate([
        (base_bd, base_ns, -w / 2, "Base"),
        (ft_bd,   ft_ns,    w / 2, "FT"),
    ]):
        bottoms = np.zeros(len(MODEL_GROUPS))
        for cat_key, cat_label, color, hatch in psum.STACK_CATS:
            heights = np.array([b[cat_key] * 100 for b in bds])
            ax.bar(x + offset, heights, w, bottom=bottoms,
                   color=color, hatch=hatch,
                   label=cat_label if side_idx == 0 else None,
                   edgecolor="black", linewidth=0.4)
            bottoms += heights
        errs = [psum.wald_err_pct(t / 100.0, n) for t, n in zip(bottoms, ns)]
        ax.errorbar(x + offset, bottoms, yerr=errs,
                    fmt="none", ecolor="black", capsize=3, linewidth=0.9)
        for i, (total, err) in enumerate(zip(bottoms, errs)):
            top_values.append(total + err)
            ax.text(x[i] + offset, total + err + 0.3, f"{total:.1f}%",
                    ha="center", va="bottom", fontsize=8,
                    fontweight="bold" if label == "FT" else "normal")

    pad = max(1.5, 0.2 * max(top_values)) if top_values else 2.0
    y_hi = max(top_values) + pad if top_values else 100.0

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_GROUPS, rotation=20, ha="right")
    task_desc = "main + side task" if require_main_task else "side task"
    ax.set_ylabel(f"Evasion success rate (%) — {task_desc}")
    ax.set_ylim(0.0, y_hi)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9, loc="upper right", ncol=len(psum.STACK_CATS))
    fig.suptitle(
        f"Monitor-QA evasion breakdown (stacked, {task_desc}) — "
        f"{MONITOR} gpt-5.4 instant @ 5% FPR  "
        "(best side-task prompt per group; bars: 80% Wald CI)  "
        "[2026-04-13 ft60]",
        fontsize=12,
    )
    fig.tight_layout()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fname = (
        "mqa_clear_giveaway_grouped_metr_note_v3_main_and_side.png"
        if require_main_task else
        "mqa_clear_giveaway_grouped_metr_note_v3.png"
    )
    out = OUTDIR / fname
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}  (y-axis 0.0-{y_hi:.1f})")
    print("  Best variant per cell:")
    for model, split, var in best_variant_per_cell:
        print(f"    {model:14s} {split:4s} → {var}")


def main() -> None:
    print(f"Monitor: {MONITOR}")
    print(f"Models: {MODEL_GROUPS}")
    print(f"Variants: {[v for v, _ in VARIANTS]}\n")
    print("Plot: mqa_clear_giveaway_grouped (side task only)...")
    plot_grouped(require_main_task=False)
    print("\nPlot: mqa_clear_giveaway_grouped (main + side)...")
    plot_grouped(require_main_task=True)
    print(f"\nDone. Outputs in {OUTDIR}")


if __name__ == "__main__":
    main()
