#!/usr/bin/env python3
"""Generate fine-tuning scaling plots (2026-03-23).

Reads from:
  - results/summaries/tables/summary_by_run.csv

Outputs to:
  - results/summaries/plots/2026_03_23/

Plot: For each model family x eval suite, compliance (%) vs active parameters.
      One line for base model and one line per learning rate (best checkpoint).
      Three model families: gpt-oss, qwen3, qwen3.5.
      Total: 3 families x 2 eval suites = 6 plots.

Usage:
    python scripts/analysis/plot_summary_2026_03_23.py
"""

import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.size"] = 11

SUMMARY_BY_RUN = Path("results/summaries/tables/summary_by_run.csv")
DATE = "2026_03_23"
OUTDIR = Path(f"results/summaries/plots/{DATE}")
OUTDIR.mkdir(parents=True, exist_ok=True)

Z80 = 1.282

# ---------------------------------------------------------------------------
# Model families and active parameter counts
# ---------------------------------------------------------------------------

FAMILIES = {
    "gpt-oss": {
        "models": [
            ("openai/gpt-oss-20b", 20),
            ("openai/gpt-oss-120b", 120),
        ],
        "lrs": ["1e-4", "2.5e-5", "4e-4"],
        "canonical_effort": "medium",
    },
    "qwen3": {
        "models": [
            ("qwen/qwen3-8b", 8),
            ("qwen/qwen3-32b", 32),
        ],
        "lrs": ["1e-4", "2.5e-5", "4e-4"],
        "canonical_effort": "",  # "none" or empty
    },
    "qwen3.5": {
        "models": [
            ("qwen/qwen3.5-35b-a3b", 3),   # 3B active params (MoE)
            ("qwen/qwen3.5-4b", 4),
            ("qwen/qwen3.5-397b-a17b", 17), # 17B active params (MoE)
            ("qwen/qwen3.5-27b", 27),
        ],
        "lrs": ["1e-4", "4e-4"],
        "canonical_effort": "",  # "none" or empty
    },
}

EVAL_DISPLAY = {"cotcontrol": "CoTControl", "reasonif": "ReasonIF"}

# Colors for each learning rate + base
LR_COLORS = {
    "base": "#333333",
    "1e-4": "#2196F3",
    "2.5e-5": "#4CAF50",
    "4e-4": "#E91E63",
}

LR_DISPLAY = {
    "1e-4": "lr=1e-4",
    "2.5e-5": "lr=2.5e-5",
    "4e-4": "lr=4e-4",
}

# Checkpoints to plot scaling curves for
SCALING_CKPTS = [
    ("000060", "step 60"),
    ("final", "final checkpoint"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sf(row, col):
    v = row.get(col, "")
    return float(v) if v else None


def si(row, col):
    v = row.get(col, "")
    return int(v) if v else 0


def ci80(p, n):
    if n <= 0 or p is None:
        return 0.0
    return Z80 * math.sqrt(p * (1 - p) / n)


def extract_lr_ckpt(filename):
    """Extract (lr, checkpoint) from a rollout filename."""
    m = re.search(r"-rif-(?:lr([\d.]+e-\d+)-)?(\d{3,}|final)", filename)
    if m:
        lr = m.group(1) or "4e-4"  # older filenames without lr prefix used 4e-4
        ckpt = m.group(2)
        return lr, ckpt
    return None, None



# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


all_runs = load_csv(SUMMARY_BY_RUN)

# Index: (base_model, eval_type, lr, ckpt) -> best row (latest timestamp)
# For base: lr="" ckpt="" effort=reasoning_effort
# For FT: lr and ckpt from filename, effort=""
indexed = {}
for row in all_runs:
    if si(row, "n_clean") < 10:
        continue
    fn = row.get("file_name", "")
    base_model = row["base_model"]
    eval_type = row["eval_type"]

    if "-rif-" in fn:
        lr, ckpt = extract_lr_ckpt(fn)
        if lr is None:
            continue
        key = (base_model, eval_type, lr, ckpt)
    else:
        effort = row.get("reasoning_effort", "")
        key = (base_model, eval_type, "", "", effort)

    if key not in indexed or row.get("timestamp", "") > indexed[key].get("timestamp", ""):
        indexed[key] = row


def get_base_row(model: str, eval_type: str, family_cfg: dict):
    """Look up the canonical-effort base row for a model."""
    canonical = family_cfg["canonical_effort"]
    if canonical == "medium":
        efforts = ["medium"]
    else:
        efforts = ["none", ""]
    for eff in efforts:
        row = indexed.get((model, eval_type, "", "", eff))
        if row:
            return row
    # Fallback: try any effort
    for eff in ["medium", "high", "low", "none", ""]:
        row = indexed.get((model, eval_type, "", "", eff))
        if row:
            return row
    return None


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

for ckpt_name, ckpt_display in SCALING_CKPTS:
    for family_name, family_cfg in FAMILIES.items():
        models = family_cfg["models"]
        lrs = family_cfg["lrs"]

        for eval_type in ["cotcontrol", "reasonif"]:
            fig, ax = plt.subplots(figsize=(8, 6))

            # --- Base line ---
            bx, by, be = [], [], []
            for model, active_params in models:
                row = get_base_row(model, eval_type, family_cfg)
                if row:
                    p = sf(row, "compliant_rate")
                    n = si(row, "n_clean")
                    if p is not None:
                        bx.append(active_params)
                        by.append(p * 100)
                        be.append(ci80(p, n) * 100)

            if bx:
                ax.errorbar(
                    bx, by, yerr=be, marker="o", linestyle="-", color=LR_COLORS["base"],
                    label="Base", capsize=5, lw=2, markersize=8, zorder=5,
                )

            # --- FT lines (one per LR, using this checkpoint) ---
            for lr in lrs:
                fx, fy, fe = [], [], []
                for model, active_params in models:
                    row = indexed.get((model, eval_type, lr, ckpt_name))
                    if row:
                        p = sf(row, "compliant_rate")
                        n = si(row, "n_clean")
                        if p is not None:
                            fx.append(active_params)
                            fy.append(p * 100)
                            fe.append(ci80(p, n) * 100)

                if fx:
                    ax.errorbar(
                        fx, fy, yerr=fe, marker="s", linestyle="--",
                        color=LR_COLORS.get(lr, "#999999"),
                        label=f"FT ({LR_DISPLAY[lr]})",
                        capsize=5, lw=2, markersize=8, zorder=4,
                    )

            # --- Axis formatting ---
            all_params = sorted(set(p for _, p in models))
            ax.set_xlabel("Active Parameters (B)")
            ax.set_ylabel("Compliance (%)")
            ax.set_title(
                f"{family_name} — {EVAL_DISPLAY[eval_type]} Compliance vs Model Size\n"
                f"Base vs Fine-tuned ({ckpt_display}, 80% CI)"
            )
            ax.set_xscale("log")
            ax.set_xticks(all_params)
            ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
            ax.set_xticklabels([f"{p}B" for p in all_params])
            ax.minorticks_off()
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            # Set y range
            all_vals = by + [v for lr in lrs for v in [
                (sf(indexed.get((m, eval_type, lr, ckpt_name)), "compliant_rate") or 0) * 100
                for m, _ in models
                if indexed.get((m, eval_type, lr, ckpt_name))
            ]]
            if all_vals:
                ymax = max(all_vals)
                ax.set_ylim(0, min(ymax * 1.3, 105))

            plt.tight_layout()
            fname = f"ft_scaling_{family_name}_{eval_type}_{ckpt_name}_{DATE}.png"
            fig.savefig(OUTDIR / fname, dpi=150)
            print(f"Saved {OUTDIR / fname}")
            plt.close()


# ---------------------------------------------------------------------------
# Training progression plots (all families)
# ---------------------------------------------------------------------------
# One figure per (family, LR). Each figure has 2 subplots: cotcontrol (left), reasonif (right).
# Each subplot has one line per model. X = batch number, Y = compliance %.
# Points: base (x=0), checkpoints 000060, 000120, 000180, final (~238).

CHECKPOINTS = ["000060", "000120", "000180", "final"]
CKPT_BATCHES = {"000060": 60, "000120": 120, "000180": 180, "final": 238}

MODEL_COLORS_PROG = {
    # gpt-oss
    "openai/gpt-oss-20b": "#2196F3",
    "openai/gpt-oss-120b": "#E91E63",
    # qwen3
    "qwen/qwen3-8b": "#2196F3",
    "qwen/qwen3-32b": "#E91E63",
    # qwen3.5
    "qwen/qwen3.5-35b-a3b": "#E91E63",
    "qwen/qwen3.5-4b": "#2196F3",
    "qwen/qwen3.5-397b-a17b": "#4CAF50",
    "qwen/qwen3.5-27b": "#FF9800",
}

MODEL_DISPLAY_PROG = {
    # gpt-oss
    "openai/gpt-oss-20b": "GPT-OSS-20B",
    "openai/gpt-oss-120b": "GPT-OSS-120B",
    # qwen3
    "qwen/qwen3-8b": "Qwen3-8B",
    "qwen/qwen3-32b": "Qwen3-32B",
    # qwen3.5
    "qwen/qwen3.5-35b-a3b": "Qwen3.5-35B-A3B (3B active)",
    "qwen/qwen3.5-4b": "Qwen3.5-4B",
    "qwen/qwen3.5-397b-a17b": "Qwen3.5-397B-A17B (17B active)",
    "qwen/qwen3.5-27b": "Qwen3.5-27B",
}

FAMILY_DISPLAY = {
    "gpt-oss": "GPT-OSS",
    "qwen3": "Qwen3",
    "qwen3.5": "Qwen3.5",
}

for family_name, family_cfg in FAMILIES.items():
    family_models = family_cfg["models"]
    family_display = FAMILY_DISPLAY[family_name]

    for lr in family_cfg["lrs"]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

        for ax_idx, eval_type in enumerate(["cotcontrol", "reasonif"]):
            ax = axes[ax_idx]

            for model, active_params in family_models:
                xs, ys, es = [], [], []

                # Base point at x=0
                base_row = get_base_row(model, eval_type, family_cfg)
                if base_row:
                    p = sf(base_row, "compliant_rate")
                    n = si(base_row, "n_clean")
                    if p is not None:
                        xs.append(0)
                        ys.append(p * 100)
                        es.append(ci80(p, n) * 100)

                # Checkpoint points
                for ckpt in CHECKPOINTS:
                    row = indexed.get((model, eval_type, lr, ckpt))
                    if row:
                        p = sf(row, "compliant_rate")
                        n = si(row, "n_clean")
                        if p is not None:
                            xs.append(CKPT_BATCHES[ckpt])
                            ys.append(p * 100)
                            es.append(ci80(p, n) * 100)

                if xs:
                    ax.errorbar(
                        xs, ys, yerr=es, marker="o", linestyle="-",
                        color=MODEL_COLORS_PROG[model],
                        label=MODEL_DISPLAY_PROG[model],
                        capsize=4, lw=2, markersize=7, zorder=4,
                    )

            ax.set_xlabel("Training Batch")
            ax.set_ylabel("Compliance (%)" if ax_idx == 0 else "")
            ax.set_title(EVAL_DISPLAY[eval_type])
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xticks([0, 60, 120, 180, 238])
            ax.set_xticklabels(["base", "60", "120", "180", "final"], fontsize=9)

        fig.suptitle(
            f"{family_display} Training Progression — lr={lr}\n"
            f"Compliance (%) over training checkpoints (80% CI)",
            fontsize=13,
        )
        plt.tight_layout()
        fname = f"ft_progression_{family_name}_lr{lr}_{DATE}.png"
        fig.savefig(OUTDIR / fname, dpi=150)
        print(f"Saved {OUTDIR / fname}")
        plt.close()


# ---------------------------------------------------------------------------
# Best LR selection: per model per checkpoint, pick LR with biggest ReasonIF
# compliance increase vs base
# ---------------------------------------------------------------------------

ALL_MODELS_FLAT = []
for family_cfg in FAMILIES.values():
    for model, active_params in family_cfg["models"]:
        ALL_MODELS_FLAT.append((model, active_params, family_cfg["lrs"], family_cfg))

# best_lr_by_ckpt[ckpt_name][model] = best LR string
best_lr_by_ckpt: dict[str, dict[str, str]] = {}
for ckpt_name, ckpt_display in SCALING_CKPTS:
    best_lr_by_ckpt[ckpt_name] = {}
    for model, _, lrs, _family_cfg in ALL_MODELS_FLAT:
        base_row = get_base_row(model, "reasonif", _family_cfg)
        base_p = sf(base_row, "compliant_rate") if base_row else 0.0
        if base_p is None:
            base_p = 0.0

        best_delta, best = -1.0, lrs[0]
        for lr in lrs:
            ft_row = indexed.get((model, "reasonif", lr, ckpt_name))
            if ft_row:
                ft_p = sf(ft_row, "compliant_rate")
                if ft_p is not None:
                    delta = ft_p - base_p
                    if delta > best_delta:
                        best_delta = delta
                        best = lr
        best_lr_by_ckpt[ckpt_name][model] = best

    print(f"\nBest LR per model (biggest ReasonIF compliance increase at {ckpt_display}):")
    for model, lr in best_lr_by_ckpt[ckpt_name].items():
        print(f"  {model:35s}  best_lr={lr}")

# Alias for step-60 selection (used by FT progression plot).
best_lr_per_model = best_lr_by_ckpt["000060"]


# ---------------------------------------------------------------------------
# Consolidated FT progression (best LR per model, per family)
# ---------------------------------------------------------------------------

for family_name, family_cfg in FAMILIES.items():
    family_models = family_cfg["models"]
    family_display = FAMILY_DISPLAY[family_name]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax_idx, eval_type in enumerate(["cotcontrol", "reasonif"]):
        ax = axes[ax_idx]

        for model, active_params in family_models:
            lr = best_lr_per_model[model]
            xs, ys, es = [], [], []

            # Base point at x=0
            base_row = get_base_row(model, eval_type, family_cfg)
            if base_row:
                p = sf(base_row, "compliant_rate")
                n = si(base_row, "n_clean")
                if p is not None:
                    xs.append(0)
                    ys.append(p * 100)
                    es.append(ci80(p, n) * 100)

            # Checkpoint points
            for ckpt in CHECKPOINTS:
                row = indexed.get((model, eval_type, lr, ckpt))
                if row:
                    p = sf(row, "compliant_rate")
                    n = si(row, "n_clean")
                    if p is not None:
                        xs.append(CKPT_BATCHES[ckpt])
                        ys.append(p * 100)
                        es.append(ci80(p, n) * 100)

            if xs:
                ax.errorbar(
                    xs, ys, yerr=es, marker="o", linestyle="-",
                    color=MODEL_COLORS_PROG[model],
                    label=f"{MODEL_DISPLAY_PROG[model]} (lr={lr})",
                    capsize=4, lw=2, markersize=7, zorder=4,
                )

        ax.set_xlabel("Training Batch")
        ax.set_ylabel("Compliance (%)" if ax_idx == 0 else "")
        ax.set_title(EVAL_DISPLAY[eval_type])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([0, 60, 120, 180, 238])
        ax.set_xticklabels(["base", "60", "120", "180", "final"], fontsize=9)

    fig.suptitle(
        f"{family_display} Training Progression — Best LR per model\n"
        "Compliance (%) over training checkpoints (80% CI)",
        fontsize=13,
    )
    plt.tight_layout()
    fname = f"ft_progression_{family_name}_best_lr_{DATE}.png"
    fig.savefig(OUTDIR / fname, dpi=150)
    print(f"Saved {OUTDIR / fname}")
    plt.close()


# ---------------------------------------------------------------------------
# Consolidated FT scaling (all families, best LR, side-by-side cotcontrol/reasonif)
# ---------------------------------------------------------------------------

CONSOLIDATED_FAMILIES = {
    "GPT-OSS": {
        "models": [
            ("openai/gpt-oss-20b", 20),
            ("openai/gpt-oss-120b", 120),
        ],
        "color": "#E91E63",
        "canonical_effort": "medium",
    },
    "Qwen3": {
        "models": [
            ("qwen/qwen3-8b", 8),
            ("qwen/qwen3-32b", 32),
        ],
        "color": "#2196F3",
        "canonical_effort": "",
    },
    "Qwen3.5 Dense": {
        "models": [
            ("qwen/qwen3.5-4b", 4),
            ("qwen/qwen3.5-27b", 27),
        ],
        "color": "#4CAF50",
        "canonical_effort": "",
    },
    "Qwen3.5 MoE": {
        "models": [
            ("qwen/qwen3.5-35b-a3b", 3),
            ("qwen/qwen3.5-397b-a17b", 17),
        ],
        "color": "#FF9800",
        "canonical_effort": "",
    },
}

for ckpt_name, ckpt_display in SCALING_CKPTS:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    for ax_idx, eval_type in enumerate(["cotcontrol", "reasonif"]):
        ax = axes[ax_idx]

        for fam_name, fam_cfg in CONSOLIDATED_FAMILIES.items():
            color = fam_cfg["color"]
            models = fam_cfg["models"]

            # --- Base line (solid, circle markers) ---
            bx, by, be = [], [], []
            for model, active_params in models:
                row = get_base_row(model, eval_type, fam_cfg)
                if row:
                    p = sf(row, "compliant_rate")
                    n = si(row, "n_clean")
                    if p is not None:
                        bx.append(active_params)
                        by.append(p * 100)
                        be.append(ci80(p, n) * 100)

            if bx:
                ax.errorbar(
                    bx, by, yerr=be, marker="o", linestyle="-", color=color,
                    capsize=4, lw=2, markersize=7, zorder=5,
                )

            # --- FT line (dashed, square markers, best LR per model) ---
            fx, fy, fe = [], [], []
            for model, active_params in models:
                lr = best_lr_by_ckpt[ckpt_name].get(model)
                if not lr:
                    continue
                row = indexed.get((model, eval_type, lr, ckpt_name))
                if row:
                    p = sf(row, "compliant_rate")
                    n = si(row, "n_clean")
                    if p is not None:
                        fx.append(active_params)
                        fy.append(p * 100)
                        fe.append(ci80(p, n) * 100)

            if fx:
                ax.errorbar(
                    fx, fy, yerr=fe, marker="s", linestyle="--", color=color,
                    capsize=4, lw=2, markersize=7, zorder=4,
                )

        # --- Two-part legend: color for family, line style for base/FT ---
        # Laid out as 2 columns: families in a 2×2 block, then base/FT below.
        from matplotlib.lines import Line2D
        fam_items = list(CONSOLIDATED_FAMILIES.items())
        fam_h = {name: Line2D([], [], color=cfg["color"], lw=2, label=name)
                 for name, cfg in fam_items}
        base_h = Line2D([], [], color="gray", lw=2, linestyle="-", marker="o", markersize=6, label="Base")
        ft_h = Line2D([], [], color="gray", lw=2, linestyle=(0, (1.5, 1.5)), marker="s", markersize=6, label="FT")
        # ncol=2 fills column-first, so order as: col1=[row1,row2,row3], col2=[row1,row2,row3]
        all_handles = [
            fam_h["GPT-OSS"], fam_h["Qwen3"], base_h,
            fam_h["Qwen3.5 Dense"], fam_h["Qwen3.5 MoE"], ft_h,
        ]
        ax.legend(handles=all_handles, loc="upper left", fontsize=7, ncol=2, handlelength=4)

        # --- Axis formatting ---
        all_params = sorted(set(
            p for fam_cfg in CONSOLIDATED_FAMILIES.values()
            for _, p in fam_cfg["models"]
        ))
        ax.set_xlabel("Active Parameters (B)")
        ax.set_ylabel("Compliance (%)")
        ax.set_title(EVAL_DISPLAY[eval_type])
        ax.set_xscale("log")
        ax.set_xticks(all_params)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.set_xticklabels([f"{p}B" for p in all_params], fontsize=8, rotation=30)
        ax.minorticks_off()
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"FT Scaling — All Families ({ckpt_display}, best LR per model, 80% CI)",
        fontsize=13,
    )
    plt.tight_layout()
    fname = f"ft_scaling_consolidated_{ckpt_name}_{DATE}.png"
    fig.savefig(OUTDIR / fname, dpi=150)
    print(f"Saved {OUTDIR / fname}")
    plt.close()


# ---------------------------------------------------------------------------
# Data verification
# ---------------------------------------------------------------------------

print(f"\n{'=' * 70}")
print("DATA VERIFICATION")
print(f"{'=' * 70}")

for family_name, family_cfg in FAMILIES.items():
    print(f"\n--- {family_name} ---")
    for model, active_params in family_cfg["models"]:
        # Base
        for et in ["cotcontrol", "reasonif"]:
            row = get_base_row(model, et, family_cfg)
            if row:
                p = sf(row, "compliant_rate")
                print(f"  {model:35s} {et:12s} base       compliance={p:.4f}  n_clean={si(row, 'n_clean')}")
            # FT
            for lr in family_cfg["lrs"]:
                for ckpt_name, _ in SCALING_CKPTS:
                    row = indexed.get((model, et, lr, ckpt_name))
                    if row:
                        p = sf(row, "compliant_rate")
                        print(f"  {model:35s} {et:12s} lr={lr:6s} {ckpt_name:6s}  compliance={p:.4f}  n_clean={si(row, 'n_clean')}")

print(f"\nDone — all plots saved to {OUTDIR}")
