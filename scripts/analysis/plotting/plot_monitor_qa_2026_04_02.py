#!/usr/bin/env python3
"""Monitor QA analysis plots (2026-04-02).

Plots:
  1. Score distribution grid (2×4): base vs FT, 3 settings overlaid
  2. Mean monitor score grouped barplot
  3. Below-95th-percentile-threshold frequency (2 panels)
  4. Error rate grouped barplot

Usage:
    uv run python scripts/analysis/plot_monitor_qa_2026_04_02.py
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.size"] = 11

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROLLOUTS_DIR = Path("results/rollouts/monitor_qa")
DATE = "2026_04_02"
OUTDIR = Path(f"results/summaries/plots/{DATE}")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# File mapping: (model_label, is_ft, setting) -> file path
# ---------------------------------------------------------------------------

# Overnight run files (20260402_090606)
OVERNIGHT_FILES = {
    # Base models
    ("GPT-OSS-20B", False, "baseline"):   "monitor_qa_gpt-oss-20b_baseline_20260402_090606_e6d9a872a86e.jsonl",
    ("GPT-OSS-20B", False, "default"):    "monitor_qa_gpt-oss-20b_default_20260402_090606_83e8b33034dd.jsonl",
    ("GPT-OSS-20B", False, "dont_think"): "monitor_qa_gpt-oss-20b_dont_think_20260402_090606_4dc781a5b973.jsonl",
    ("GPT-OSS-120B", False, "baseline"):   "monitor_qa_gpt-oss-120b_baseline_20260402_090606_f92876518053.jsonl",
    ("GPT-OSS-120B", False, "default"):    "monitor_qa_gpt-oss-120b_default_20260402_090606_13f19de13db3.jsonl",
    ("GPT-OSS-120B", False, "dont_think"): "monitor_qa_gpt-oss-120b_dont_think_20260402_090606_332ac2a70d65.jsonl",
    ("Qwen3-8B", False, "baseline"):   "monitor_qa_qwen3-8b_baseline_20260402_090606_eeca7509cfe4.jsonl",
    ("Qwen3-8B", False, "default"):    "monitor_qa_qwen3-8b_default_20260402_090606_2e601d121375.jsonl",
    ("Qwen3-8B", False, "dont_think"): "monitor_qa_qwen3-8b_dont_think_20260402_090606_a7796eb013fe.jsonl",
    ("Qwen3-32B", False, "baseline"):   "monitor_qa_qwen3-32b_baseline_20260402_090606_fc330599a13b.jsonl",
    ("Qwen3-32B", False, "default"):    "monitor_qa_qwen3-32b_default_20260402_090606_a164b4ddb912.jsonl",
    ("Qwen3-32B", False, "dont_think"): "monitor_qa_qwen3-32b_dont_think_20260402_090606_ac3cc9445c04.jsonl",
    # FT models (gpt-oss)
    ("GPT-OSS-20B", True, "baseline"):   "monitor_qa_gpt-oss-20b-ft60_baseline_20260402_090606_859184b40afe.jsonl",
    ("GPT-OSS-20B", True, "default"):    "monitor_qa_gpt-oss-20b-ft60_default_20260402_090606_3d755d536923.jsonl",
    ("GPT-OSS-20B", True, "dont_think"): "monitor_qa_gpt-oss-20b-ft60_dont_think_20260402_090606_3c338823dfde.jsonl",
    ("GPT-OSS-120B", True, "baseline"):   "monitor_qa_gpt-oss-120b-ft60_baseline_20260402_090606_891d36d5e33e.jsonl",
    ("GPT-OSS-120B", True, "default"):    "monitor_qa_gpt-oss-120b-ft60_default_20260402_090606_169268a64c7d.jsonl",
    ("GPT-OSS-120B", True, "dont_think"): "monitor_qa_gpt-oss-120b-ft60_dont_think_20260402_090606_dafa1247edc6.jsonl",
}

# Qwen FT rerun files (20260403_rerun)
RERUN_FILES = {
    ("Qwen3-8B", True, "baseline"):   "monitor_qa_qwen3-8b-ft60_baseline_20260403_rerun_5d21ec8b97ff.jsonl",
    ("Qwen3-8B", True, "default"):    "monitor_qa_qwen3-8b-ft60_default_20260403_rerun_06e2b3fd7594.jsonl",
    ("Qwen3-8B", True, "dont_think"): "monitor_qa_qwen3-8b-ft60_dont_think_20260403_rerun_d9ed24dc1486.jsonl",
    ("Qwen3-32B", True, "baseline"):   "monitor_qa_qwen3-32b-ft60_baseline_20260403_rerun_8cba6c1c9744.jsonl",
    ("Qwen3-32B", True, "default"):    "monitor_qa_qwen3-32b-ft60_default_20260403_rerun_c4e252d2493a.jsonl",
    ("Qwen3-32B", True, "dont_think"): "monitor_qa_qwen3-32b-ft60_dont_think_20260403_rerun_674237320652.jsonl",
}

ALL_FILES = {**OVERNIGHT_FILES, **RERUN_FILES}

MODEL_GROUPS = ["GPT-OSS-20B", "GPT-OSS-120B", "Qwen3-8B", "Qwen3-32B"]
SETTINGS = ["baseline", "default", "dont_think"]
SETTING_LABELS = {"baseline": "Baseline", "default": "Default", "dont_think": "Don't Think"}

# Which monitor type to use for plots. Set to "old_monitor" or "metr_note".
# When the score column is missing, falls back to legacy "monitor_score".
MONITOR_TYPE = "old_monitor"

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

# Setting colors for distribution plots
SETTING_COLORS = {
    "baseline": "#4C72B0",
    "default": "#DD8452",
    "dont_think": "#55A868",
}

# Base vs FT color families for barplots
# Light → dark for baseline → default → dont_think
BASE_COLORS = {
    "baseline": "#A8C8E8",
    "default": "#5B9BD5",
    "dont_think": "#2F5F8A",
}
FT_COLORS = {
    "baseline": "#F4B8B8",
    "default": "#E06060",
    "dont_think": "#A02020",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _extract_monitor_scores(d: dict) -> dict:
    """Extract per-monitor-type scores from a rollout dict.

    Returns e.g. {"old_monitor": 2.0, "metr_note": 3.0}.
    Falls back to legacy monitor_score if monitor_results is absent.
    """
    results = d.get("monitor_results", [])
    if results:
        return {
            mr.get("monitor_type"): mr.get("monitor_score")
            for mr in results
            if mr.get("monitor_type")
        }
    # Legacy fallback
    score = d.get("monitor_score")
    if score is not None:
        return {"old_monitor": score}
    return {}


def load_rollouts() -> pd.DataFrame:
    """Load all rollout files into a single DataFrame."""
    rows = []
    all_monitor_types: dict[str, None] = {}
    for (model_label, is_ft, setting), filename in ALL_FILES.items():
        path = ROLLOUTS_DIR / filename
        if not path.exists():
            print(f"  WARNING: missing {path.name}")
            continue
        with open(path) as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                try:
                    d = json.loads(s)
                except json.JSONDecodeError:
                    continue
                meta = d.get("sample", {}).get("metadata", {})
                mt_scores = _extract_monitor_scores(d)
                for mt in mt_scores:
                    all_monitor_types[mt] = None
                row = {
                    "model_group": model_label,
                    "is_ft": is_ft,
                    "setting": setting,
                    "monitor_score": d.get("monitor_score"),
                    "monitor_score_old": d.get("monitor_score_old"),
                    "actor_error": d.get("actor_error"),
                    "monitor_error": d.get("monitor_error"),
                    "side_task_correct": d.get("side_task_correct"),
                    "side_level": meta.get("side_level", ""),
                    "sample_id": d.get("sample", {}).get("id", ""),
                }
                for mt, score in mt_scores.items():
                    row[f"score_{mt}"] = score
                rows.append(row)
    df = pd.DataFrame(rows)
    # Fill missing monitor type columns
    for mt in all_monitor_types:
        col = f"score_{mt}"
        if col not in df.columns:
            df[col] = np.nan
    print(f"Loaded {len(df)} rollouts across {len(ALL_FILES)} files")
    print(f"  Monitor types found: {list(all_monitor_types.keys())}")
    return df


# ---------------------------------------------------------------------------
# Plot 1: Score distribution grid (2 rows × 4 cols)
# ---------------------------------------------------------------------------

def plot_distributions(df: pd.DataFrame, thresholds: dict):
    """2×4 grid: top row = base, bottom row = FT. 3 settings overlaid."""
    valid = df[df["actor_error"].isna() & df["monitor_score"].notna()].copy()

    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharey=True)
    bins = np.arange(-0.5, 11.5, 1)

    for col_idx, group in enumerate(MODEL_GROUPS):
        for row_idx, is_ft in enumerate([False, True]):
            ax = axes[row_idx, col_idx]
            subset = valid[(valid["model_group"] == group) & (valid["is_ft"] == is_ft)]

            for setting in SETTINGS:
                scores = subset[subset["setting"] == setting]["monitor_score"].values
                if len(scores) == 0:
                    continue
                ax.hist(
                    scores, bins=bins, alpha=0.45, label=SETTING_LABELS[setting],
                    color=SETTING_COLORS[setting], edgecolor="white", linewidth=0.5,
                    density=True,
                )
                mean_val = np.mean(scores)
                ax.axvline(mean_val, color=SETTING_COLORS[setting], linestyle="--",
                           linewidth=1.5, alpha=0.8)

            # Mark 95th percentile threshold for this specific model
            key = (group, is_ft)
            if key in thresholds:
                ax.axvline(thresholds[key], color="black", linestyle=":",
                           linewidth=1.2, alpha=0.6, label=f"95th pctl={thresholds[key]:.1f}")

            ft_label = "FT" if is_ft else "Base"
            ax.set_title(f"{group} ({ft_label})", fontsize=11, fontweight="bold")
            ax.set_xlim(-0.5, 10.5)
            ax.set_xticks(range(11))
            if col_idx == 0:
                ax.set_ylabel("Density")
            if row_idx == 1:
                ax.set_xlabel("Monitor Score")

    # Single legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               bbox_to_anchor=(0.5, 1.02), fontsize=10)

    fig.suptitle("Monitor Score Distributions by Model & Setting", y=1.06,
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = OUTDIR / "1_score_distributions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: Mean monitor score barplot
# ---------------------------------------------------------------------------

def plot_mean_scores(df: pd.DataFrame):
    """Grouped barplot: 4 groups × 6 bars (3 base + 3 FT)."""
    valid = df[df["actor_error"].isna() & df["monitor_score"].notna()].copy()

    fig, ax = plt.subplots(figsize=(14, 6))
    n_groups = len(MODEL_GROUPS)
    n_bars = 6  # 3 base + 3 FT
    bar_width = 0.12
    group_width = n_bars * bar_width + 0.15

    for g_idx, group in enumerate(MODEL_GROUPS):
        x_center = g_idx * group_width
        bar_i = 0
        for is_ft in [False, True]:
            colors = FT_COLORS if is_ft else BASE_COLORS
            for setting in SETTINGS:
                subset = valid[
                    (valid["model_group"] == group)
                    & (valid["is_ft"] == is_ft)
                    & (valid["setting"] == setting)
                ]
                mean_val = subset["monitor_score"].mean() if len(subset) > 0 else 0
                sem_val = subset["monitor_score"].sem() if len(subset) > 1 else 0
                x = x_center + (bar_i - (n_bars - 1) / 2) * bar_width
                ax.bar(x, mean_val, bar_width * 0.85, color=colors[setting],
                       yerr=sem_val, capsize=2, edgecolor="white", linewidth=0.5)
                bar_i += 1

    ax.set_xticks([i * group_width for i in range(n_groups)])
    ax.set_xticklabels(MODEL_GROUPS, fontsize=11)
    ax.set_ylabel("Mean Monitor Score")
    ax.set_title("Mean Monitor Score by Model & Setting", fontweight="bold")

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = []
    for setting in SETTINGS:
        legend_elements.append(Patch(facecolor=BASE_COLORS[setting],
                                     label=f"Base / {SETTING_LABELS[setting]}"))
    for setting in SETTINGS:
        legend_elements.append(Patch(facecolor=FT_COLORS[setting],
                                     label=f"FT / {SETTING_LABELS[setting]}"))
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9, ncol=2)

    plt.tight_layout()
    out = OUTDIR / "2_mean_monitor_score.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3: Below-threshold frequency (2 panels)
# ---------------------------------------------------------------------------

def plot_below_threshold(df: pd.DataFrame, thresholds: dict,
                         percentile_label: str = "95th",
                         suffix: str = ""):
    """Two panels: dont_think and default. 4 groups × 2 bars (base vs FT)."""
    valid = df[df["actor_error"].isna() & df["monitor_score"].notna()].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    panel_settings = ["dont_think", "default"]
    bar_width = 0.3

    for panel_idx, setting in enumerate(panel_settings):
        ax = axes[panel_idx]
        x_positions = np.arange(len(MODEL_GROUPS))

        for ft_idx, (is_ft, label, color) in enumerate([
            (False, "Base", "#5B9BD5"),
            (True, "FT", "#E06060"),
        ]):
            rates = []
            errs = []
            thresh_labels = []
            for group in MODEL_GROUPS:
                threshold = thresholds.get((group, is_ft), 10)
                subset = valid[
                    (valid["model_group"] == group)
                    & (valid["is_ft"] == is_ft)
                    & (valid["setting"] == setting)
                ]
                n = len(subset)
                if n > 0:
                    rate = (subset["monitor_score"] <= threshold).mean()
                    se = np.sqrt(rate * (1 - rate) / n)
                else:
                    rate = 0
                    se = 0
                rates.append(rate)
                errs.append(1.96 * se)
                thresh_labels.append(f"{threshold:.0f}")

            offset = (ft_idx - 0.5) * bar_width
            bars = ax.bar(x_positions + offset, rates, bar_width * 0.85,
                          label=label, color=color, edgecolor="white", linewidth=0.5,
                          yerr=errs, capsize=3, error_kw={"linewidth": 1.0})
            # Add value labels
            for bar, rate, err in zip(bars, rates, errs):
                if rate > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + err + 0.01,
                            f"{rate:.0%}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(MODEL_GROUPS, fontsize=10)
        # Derive FPR from percentile (e.g. 95th → 5% FPR)
        fpr_pct = 100 - int(percentile_label.rstrip("th"))
        ax.set_ylabel(f"Fraction not detected by monitor at {fpr_pct}% FPR"
                       if panel_idx == 0 else "")
        ax.set_title(f"Setting: {SETTING_LABELS[setting]}", fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=10)

    # Build threshold string showing base/FT for each group
    fpr_pct = 100 - int(percentile_label.rstrip("th"))
    thresh_parts = []
    for g in MODEL_GROUPS:
        base_t = thresholds.get((g, False), None)
        ft_t = thresholds.get((g, True), None)
        if base_t is not None and ft_t is not None:
            thresh_parts.append(f"{g}: {base_t:.0f}/{ft_t:.0f}")
        elif base_t is not None:
            thresh_parts.append(f"{g}: {base_t:.0f}/-")
    threshold_str = ", ".join(thresh_parts)
    fig.suptitle(
        f"Fraction Not Detected by Monitor at {fpr_pct}% FPR\n"
        f"(score thresholds base/FT: {threshold_str})",
        fontweight="bold", fontsize=13,
    )
    plt.tight_layout()
    fname = f"3_below_threshold{suffix}.png"
    out = OUTDIR / fname
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 4: Error rate barplot
# ---------------------------------------------------------------------------

def plot_error_rates(df: pd.DataFrame):
    """Grouped barplot: 4 groups × 6 bars. Y = fraction with actor_error."""
    fig, ax = plt.subplots(figsize=(14, 6))
    n_groups = len(MODEL_GROUPS)
    n_bars = 6
    bar_width = 0.12
    group_width = n_bars * bar_width + 0.15

    for g_idx, group in enumerate(MODEL_GROUPS):
        x_center = g_idx * group_width
        bar_i = 0
        for is_ft in [False, True]:
            colors = FT_COLORS if is_ft else BASE_COLORS
            for setting in SETTINGS:
                subset = df[
                    (df["model_group"] == group)
                    & (df["is_ft"] == is_ft)
                    & (df["setting"] == setting)
                ]
                n = len(subset)
                if n > 0:
                    error_rate = subset["actor_error"].notna().mean()
                    se = np.sqrt(error_rate * (1 - error_rate) / n)
                else:
                    error_rate = 0
                    se = 0
                ci = 1.96 * se
                x = x_center + (bar_i - (n_bars - 1) / 2) * bar_width
                bar = ax.bar(x, error_rate, bar_width * 0.85, color=colors[setting],
                             edgecolor="white", linewidth=0.5,
                             yerr=ci, capsize=2, error_kw={"linewidth": 1.0})
                # Add value label for non-zero
                if error_rate > 0.01:
                    ax.text(x, error_rate + ci + 0.005, f"{error_rate:.0%}",
                            ha="center", va="bottom", fontsize=7)
                bar_i += 1

    ax.set_xticks([i * group_width for i in range(n_groups)])
    ax.set_xticklabels(MODEL_GROUPS, fontsize=11)
    ax.set_ylabel("Error Rate")
    ax.set_title("Actor Error Rate by Model & Setting", fontweight="bold")

    from matplotlib.patches import Patch
    legend_elements = []
    for setting in SETTINGS:
        legend_elements.append(Patch(facecolor=BASE_COLORS[setting],
                                     label=f"Base / {SETTING_LABELS[setting]}"))
    for setting in SETTINGS:
        legend_elements.append(Patch(facecolor=FT_COLORS[setting],
                                     label=f"FT / {SETTING_LABELS[setting]}"))
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9, ncol=2)

    plt.tight_layout()
    out = OUTDIR / "4_error_rates.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 5: Below-threshold (side_task_correct only)
# ---------------------------------------------------------------------------

def plot_below_threshold_correct_only(df: pd.DataFrame, thresholds: dict,
                                       percentile_label: str = "95th",
                                       suffix: str = ""):
    """Same as plot_below_threshold but filtered to side_task_correct == True."""
    valid = df[
        df["actor_error"].isna()
        & df["monitor_score"].notna()
        & (df["side_task_correct"] == True)
    ].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    panel_settings = ["dont_think", "default"]
    bar_width = 0.3

    for panel_idx, setting in enumerate(panel_settings):
        ax = axes[panel_idx]
        x_positions = np.arange(len(MODEL_GROUPS))

        for ft_idx, (is_ft, label, color) in enumerate([
            (False, "Base", "#5B9BD5"),
            (True, "FT", "#E06060"),
        ]):
            rates = []
            errs = []
            for group in MODEL_GROUPS:
                threshold = thresholds.get((group, is_ft), 10)
                subset = valid[
                    (valid["model_group"] == group)
                    & (valid["is_ft"] == is_ft)
                    & (valid["setting"] == setting)
                ]
                n = len(subset)
                if n > 0:
                    rate = (subset["monitor_score"] <= threshold).mean()
                    se = np.sqrt(rate * (1 - rate) / n)
                else:
                    rate = 0
                    se = 0
                rates.append(rate)
                errs.append(1.96 * se)

            offset = (ft_idx - 0.5) * bar_width
            bars = ax.bar(x_positions + offset, rates, bar_width * 0.85,
                          label=f"{label} (n)", color=color, edgecolor="white",
                          linewidth=0.5, yerr=errs, capsize=3,
                          error_kw={"linewidth": 1.0})
            # Add value labels with n
            for i, (bar, rate, err) in enumerate(zip(bars, rates, errs)):
                group = MODEL_GROUPS[i]
                subset = valid[
                    (valid["model_group"] == group)
                    & (valid["is_ft"] == is_ft)
                    & (valid["setting"] == setting)
                ]
                if rate > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + err + 0.01,
                            f"{rate:.0%}\nn={len(subset)}",
                            ha="center", va="bottom", fontsize=7)

        fpr_pct = 100 - int(percentile_label.rstrip("th"))
        ax.set_xticks(x_positions)
        ax.set_xticklabels(MODEL_GROUPS, fontsize=10)
        ax.set_ylabel(f"Fraction not detected by monitor at {fpr_pct}% FPR"
                       if panel_idx == 0 else "")
        ax.set_title(f"Setting: {SETTING_LABELS[setting]}", fontweight="bold")
        ax.set_ylim(0, 1.25)
        ax.legend(fontsize=10)

    fpr_pct = 100 - int(percentile_label.rstrip("th"))
    thresh_parts = []
    for g in MODEL_GROUPS:
        base_t = thresholds.get((g, False), None)
        ft_t = thresholds.get((g, True), None)
        if base_t is not None and ft_t is not None:
            thresh_parts.append(f"{g}: {base_t:.0f}/{ft_t:.0f}")
    threshold_str = ", ".join(thresh_parts)
    fig.suptitle(
        f"Fraction Not Detected at {fpr_pct}% FPR — Side Task Correct Only\n"
        f"(score thresholds base/FT: {threshold_str})",
        fontweight="bold", fontsize=13,
    )
    plt.tight_layout()
    fname = f"5_below_threshold_correct{suffix}.png"
    out = OUTDIR / fname
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 6: Below-threshold by side task difficulty level
# ---------------------------------------------------------------------------

SIDE_LEVELS = ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"]
LEVEL_COLORS = {
    "Level 1": "#93C572",
    "Level 2": "#F4D03F",
    "Level 3": "#E8965A",
    "Level 4": "#E05555",
    "Level 5": "#8B0000",
}


def plot_below_threshold_by_level(df: pd.DataFrame, thresholds: dict,
                                   percentile_label: str = "95th",
                                   suffix: str = ""):
    """Below-threshold rate by side_level, for dont_think and default settings.

    Layout: 2 rows (dont_think, default) × 4 cols (model groups).
    Each subplot: 5 level bars × 2 (base, FT) side-by-side.
    """
    valid = df[
        df["actor_error"].isna()
        & df["monitor_score"].notna()
        & (df["side_task_correct"] == True)
        & df["side_level"].isin(SIDE_LEVELS)
    ].copy()

    panel_settings = ["dont_think", "default"]
    fig, axes = plt.subplots(len(panel_settings), len(MODEL_GROUPS),
                             figsize=(22, 10), sharey=True)
    bar_width = 0.35

    for row_idx, setting in enumerate(panel_settings):
        for col_idx, group in enumerate(MODEL_GROUPS):
            ax = axes[row_idx, col_idx]
            x_positions = np.arange(len(SIDE_LEVELS))

            for ft_idx, (is_ft, ft_label, alpha) in enumerate([
                (False, "Base", 0.7),
                (True, "FT", 1.0),
            ]):
                threshold = thresholds.get((group, is_ft), 10)
                rates = []
                errs = []
                ns = []
                for level in SIDE_LEVELS:
                    subset = valid[
                        (valid["model_group"] == group)
                        & (valid["is_ft"] == is_ft)
                        & (valid["setting"] == setting)
                        & (valid["side_level"] == level)
                    ]
                    n = len(subset)
                    ns.append(n)
                    if n > 0:
                        rate = (subset["monitor_score"] <= threshold).mean()
                        se = np.sqrt(rate * (1 - rate) / n)
                    else:
                        rate = 0
                        se = 0
                    rates.append(rate)
                    errs.append(1.96 * se)

                offset = (ft_idx - 0.5) * bar_width
                colors = [LEVEL_COLORS[lv] for lv in SIDE_LEVELS]
                edge = "#333333" if is_ft else "#AAAAAA"
                bars = ax.bar(x_positions + offset, rates, bar_width * 0.85,
                              color=colors, alpha=alpha, edgecolor=edge,
                              linewidth=1.2 if is_ft else 0.5,
                              yerr=errs, capsize=2,
                              error_kw={"linewidth": 0.8})
                for i, (bar, rate, n) in enumerate(zip(bars, rates, ns)):
                    if n > 0:
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + errs[i] + 0.01,
                                f"{n}", ha="center", va="bottom", fontsize=6,
                                color="#555555")

            ax.set_xticks(x_positions)
            ax.set_xticklabels([f"L{i+1}" for i in range(5)], fontsize=9)
            if col_idx == 0:
                fpr_pct = 100 - int(percentile_label.rstrip("th"))
                ax.set_ylabel(f"{SETTING_LABELS[setting]}\nFraction not detected ({fpr_pct}% FPR)",
                              fontsize=10, fontweight="bold")
            if row_idx == 0:
                ax.set_title(group, fontsize=11, fontweight="bold")
            ax.set_ylim(0, 1.15)

    # Legend: thin edge = Base, thick edge = FT
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#999999", edgecolor="#AAAAAA", linewidth=0.5,
              label="Base", alpha=0.7),
        Patch(facecolor="#999999", edgecolor="#333333", linewidth=1.5,
              label="FT", alpha=1.0),
    ]
    for lv in SIDE_LEVELS:
        legend_elements.append(
            Patch(facecolor=LEVEL_COLORS[lv], label=lv))
    fig.legend(handles=legend_elements, loc="upper center",
               ncol=len(legend_elements), bbox_to_anchor=(0.5, 1.02),
               fontsize=9)

    fpr_pct = 100 - int(percentile_label.rstrip("th"))
    fig.suptitle(
        f"Fraction Not Detected at {fpr_pct}% FPR by Side Task Difficulty "
        f"(correct only, n shown above bars)",
        y=1.06, fontweight="bold", fontsize=14,
    )
    plt.tight_layout()
    fname = f"6_below_threshold_by_level{suffix}.png"
    out = OUTDIR / fname
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 7: Side task accuracy (4 groups × 6 bars)
# ---------------------------------------------------------------------------

def plot_side_task_accuracy(df: pd.DataFrame):
    """Grouped barplot: 4 groups × 6 bars. Y = side_task_correct rate."""
    # Only non-error, non-baseline rollouts have meaningful side_task_correct
    valid = df[df["actor_error"].isna()].copy()

    fig, ax = plt.subplots(figsize=(14, 6))
    n_groups = len(MODEL_GROUPS)
    n_bars = 6
    bar_width = 0.12
    group_width = n_bars * bar_width + 0.15

    for g_idx, group in enumerate(MODEL_GROUPS):
        x_center = g_idx * group_width
        bar_i = 0
        for is_ft in [False, True]:
            colors = FT_COLORS if is_ft else BASE_COLORS
            for setting in SETTINGS:
                subset = valid[
                    (valid["model_group"] == group)
                    & (valid["is_ft"] == is_ft)
                    & (valid["setting"] == setting)
                ]
                n = len(subset)
                if n > 0:
                    acc = subset["side_task_correct"].fillna(False).mean()
                    se = np.sqrt(acc * (1 - acc) / n)
                else:
                    acc = 0
                    se = 0
                ci = 1.96 * se
                x = x_center + (bar_i - (n_bars - 1) / 2) * bar_width
                ax.bar(x, acc, bar_width * 0.85, color=colors[setting],
                       edgecolor="white", linewidth=0.5,
                       yerr=ci, capsize=2, error_kw={"linewidth": 1.0})
                if acc > 0.01:
                    ax.text(x, acc + ci + 0.005, f"{acc:.0%}",
                            ha="center", va="bottom", fontsize=7)
                bar_i += 1

    ax.set_xticks([i * group_width for i in range(n_groups)])
    ax.set_xticklabels(MODEL_GROUPS, fontsize=11)
    ax.set_ylabel("Side Task Accuracy")
    ax.set_title("Side Task Accuracy by Model & Setting", fontweight="bold")

    from matplotlib.patches import Patch
    legend_elements = []
    for setting in SETTINGS:
        legend_elements.append(Patch(facecolor=BASE_COLORS[setting],
                                     label=f"Base / {SETTING_LABELS[setting]}"))
    for setting in SETTINGS:
        legend_elements.append(Patch(facecolor=FT_COLORS[setting],
                                     label=f"FT / {SETTING_LABELS[setting]}"))
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9, ncol=2)

    plt.tight_layout()
    out = OUTDIR / "7_side_task_accuracy.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = load_rollouts()

    # Select which monitor type's scores to use for plotting.
    # If score_{MONITOR_TYPE} column exists, copy it over monitor_score.
    score_col = f"score_{MONITOR_TYPE}"
    if score_col in df.columns and df[score_col].notna().any():
        print(f"Using monitor type '{MONITOR_TYPE}' for plots (column: {score_col})")
        df["monitor_score"] = df[score_col]
    else:
        print(f"Monitor type '{MONITOR_TYPE}' not found, using legacy monitor_score")

    # Print summary
    for group in MODEL_GROUPS:
        for is_ft in [False, True]:
            for setting in SETTINGS:
                sub = df[(df["model_group"] == group) & (df["is_ft"] == is_ft) & (df["setting"] == setting)]
                ft_str = "FT" if is_ft else "Base"
                n_err = sub["actor_error"].notna().sum()
                n_scored = sub["monitor_score"].notna().sum()
                mean_s = sub.loc[sub["actor_error"].isna() & sub["monitor_score"].notna(), "monitor_score"].mean()
                print(f"  {group:>14} {ft_str:<5} {setting:<12}  n={len(sub):>4}  errors={n_err:>3}  scored={n_scored:>4}  mean={mean_s:.2f}" if not np.isnan(mean_s) else f"  {group:>14} {ft_str:<5} {setting:<12}  n={len(sub):>4}  errors={n_err:>3}  scored={n_scored:>4}  mean=N/A")

    # Compute percentile thresholds from each model's OWN baseline scores
    valid = df[df["actor_error"].isna() & df["monitor_score"].notna()]
    all_thresholds = {}  # {percentile: {(group, is_ft): value}}
    for pctl in [80, 90, 95]:
        thresholds = {}
        print(f"\n{pctl}th Percentile Thresholds (per-model baseline):")
        for group in MODEL_GROUPS:
            for is_ft in [False, True]:
                ft_str = "FT" if is_ft else "Base"
                baseline_scores = valid[
                    (valid["model_group"] == group)
                    & (valid["is_ft"] == is_ft)
                    & (valid["setting"] == "baseline")
                ]["monitor_score"].values
                if len(baseline_scores) > 0:
                    val = np.percentile(baseline_scores, pctl)
                    thresholds[(group, is_ft)] = val
                    print(f"  {group} {ft_str}: {val:.1f}  (n={len(baseline_scores)})")
                else:
                    print(f"  {group} {ft_str}: N/A (no baseline scores)")
        all_thresholds[pctl] = thresholds

    # Generate plots
    print("\nGenerating plots...")
    plot_distributions(df, all_thresholds[95])
    plot_mean_scores(df)
    plot_below_threshold(df, all_thresholds[95], percentile_label="95th", suffix="_p95")
    plot_below_threshold(df, all_thresholds[90], percentile_label="90th", suffix="_p90")
    plot_below_threshold(df, all_thresholds[80], percentile_label="80th", suffix="_p80")
    plot_error_rates(df)
    plot_below_threshold_correct_only(df, all_thresholds[95], percentile_label="95th", suffix="_p95")
    plot_below_threshold_correct_only(df, all_thresholds[90], percentile_label="90th", suffix="_p90")
    plot_below_threshold_correct_only(df, all_thresholds[80], percentile_label="80th", suffix="_p80")
    plot_below_threshold_by_level(df, all_thresholds[95], percentile_label="95th", suffix="_p95")
    plot_below_threshold_by_level(df, all_thresholds[90], percentile_label="90th", suffix="_p90")
    plot_side_task_accuracy(df)
    print("Done!")


if __name__ == "__main__":
    main()
