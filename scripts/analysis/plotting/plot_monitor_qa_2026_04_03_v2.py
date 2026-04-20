#!/usr/bin/env python3
"""Monitor QA analysis plots (2026-04-03 v2) — metr_note monitor.

Same plot set as 2026_04_02 but using metr_note monitor scores and the
latest available rollout files.  Y-axes use percent (not fraction) and
below-threshold plots say "bypassing monitor" instead of "not detected".

Plots:
  1. Score distribution grid (2x4): base vs FT, 3 settings overlaid
  2. Mean monitor score grouped barplot
  3. Below-threshold percent at x% FPR (2 panels)
  4. Error rate grouped barplot
  5. Below-threshold (side_task_correct only)
  6. Below-threshold by side task difficulty level
  7. Side task accuracy

Usage:
    uv run python scripts/analysis/plot_monitor_qa_2026_04_03_v2.py
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.size"] = 11

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROLLOUTS_DIR = Path("results/rollouts/monitor_qa")
DATE = "2026_04_03_v2"
OUTDIR = Path(f"results/summaries/plots/{DATE}")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# File mapping: (model_label, is_ft, setting) -> file path
# Uses the latest available files for each combination.
#   Base models: 20260403 run (all Tinker)
#   FT GPT-OSS:  20260402 overnight run
#   FT Qwen:     20260403 rerun
# ---------------------------------------------------------------------------

ALL_FILES = {
    # Base models (20260403)
    ("GPT-OSS-20B", False, "baseline"):   "monitor_qa_gpt-oss-20b_baseline_20260403_082613_e6d9a872a86e.jsonl",
    ("GPT-OSS-20B", False, "default"):    "monitor_qa_gpt-oss-20b_default_20260403_082613_83e8b33034dd.jsonl",
    ("GPT-OSS-20B", False, "dont_think"): "monitor_qa_gpt-oss-20b_dont_think_20260403_082613_4dc781a5b973.jsonl",
    ("GPT-OSS-120B", False, "baseline"):   "monitor_qa_gpt-oss-120b_baseline_20260403_082613_f92876518053.jsonl",
    ("GPT-OSS-120B", False, "default"):    "monitor_qa_gpt-oss-120b_default_20260403_082613_13f19de13db3.jsonl",
    ("GPT-OSS-120B", False, "dont_think"): "monitor_qa_gpt-oss-120b_dont_think_20260403_082613_332ac2a70d65.jsonl",
    ("Qwen3-8B", False, "baseline"):   "monitor_qa_qwen3-8b_baseline_20260403_082613_eeca7509cfe4.jsonl",
    ("Qwen3-8B", False, "default"):    "monitor_qa_qwen3-8b_default_20260403_082613_2e601d121375.jsonl",
    ("Qwen3-8B", False, "dont_think"): "monitor_qa_qwen3-8b_dont_think_20260403_082613_a7796eb013fe.jsonl",
    ("Qwen3-32B", False, "baseline"):   "monitor_qa_qwen3-32b_baseline_20260403_082613_fc330599a13b.jsonl",
    ("Qwen3-32B", False, "default"):    "monitor_qa_qwen3-32b_default_20260403_082613_a164b4ddb912.jsonl",
    ("Qwen3-32B", False, "dont_think"): "monitor_qa_qwen3-32b_dont_think_20260403_082613_ac3cc9445c04.jsonl",
    # FT models — GPT-OSS (20260402 overnight)
    ("GPT-OSS-20B", True, "baseline"):   "monitor_qa_gpt-oss-20b-ft60_baseline_20260402_090606_859184b40afe.jsonl",
    ("GPT-OSS-20B", True, "default"):    "monitor_qa_gpt-oss-20b-ft60_default_20260402_090606_3d755d536923.jsonl",
    ("GPT-OSS-20B", True, "dont_think"): "monitor_qa_gpt-oss-20b-ft60_dont_think_20260402_090606_3c338823dfde.jsonl",
    ("GPT-OSS-120B", True, "baseline"):   "monitor_qa_gpt-oss-120b-ft60_baseline_20260402_090606_891d36d5e33e.jsonl",
    ("GPT-OSS-120B", True, "default"):    "monitor_qa_gpt-oss-120b-ft60_default_20260402_090606_169268a64c7d.jsonl",
    ("GPT-OSS-120B", True, "dont_think"): "monitor_qa_gpt-oss-120b-ft60_dont_think_20260402_090606_dafa1247edc6.jsonl",
    # FT models — Qwen (20260403 rerun)
    ("Qwen3-8B", True, "baseline"):   "monitor_qa_qwen3-8b-ft60_baseline_20260403_rerun_5d21ec8b97ff.jsonl",
    ("Qwen3-8B", True, "default"):    "monitor_qa_qwen3-8b-ft60_default_20260403_rerun_06e2b3fd7594.jsonl",
    ("Qwen3-8B", True, "dont_think"): "monitor_qa_qwen3-8b-ft60_dont_think_20260403_rerun_d9ed24dc1486.jsonl",
    ("Qwen3-32B", True, "baseline"):   "monitor_qa_qwen3-32b-ft60_baseline_20260403_rerun_8cba6c1c9744.jsonl",
    ("Qwen3-32B", True, "default"):    "monitor_qa_qwen3-32b-ft60_default_20260403_rerun_c4e252d2493a.jsonl",
    ("Qwen3-32B", True, "dont_think"): "monitor_qa_qwen3-32b-ft60_dont_think_20260403_rerun_674237320652.jsonl",
}

MODEL_GROUPS = ["GPT-OSS-20B", "GPT-OSS-120B", "Qwen3-8B", "Qwen3-32B"]
SETTINGS = ["baseline", "default", "dont_think"]
SETTING_LABELS = {"baseline": "Baseline", "default": "Default", "dont_think": "Don't Think"}

# Monitor types — primary (gpt-4.1) and secondary (gpt-5.4)
MONITOR_TYPE = "metr_note"
MONITOR_TYPE_GPT54 = "metr_note_gpt54"

MONITOR_LABELS = {
    "metr_note": "GPT-4.1",
    "metr_note_gpt54": "GPT-5.4",
}

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

SETTING_COLORS = {
    "baseline": "#4C72B0",
    "default": "#DD8452",
    "dont_think": "#55A868",
}

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

SIDE_LEVELS = ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"]
LEVEL_COLORS = {
    "Level 1": "#93C572",
    "Level 2": "#F4D03F",
    "Level 3": "#E8965A",
    "Level 4": "#E05555",
    "Level 5": "#8B0000",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _extract_monitor_scores(d: dict) -> dict:
    """Extract per-monitor-type scores from a rollout dict."""
    results = d.get("monitor_results", [])
    if results:
        return {
            mr.get("monitor_type"): mr.get("monitor_score")
            for mr in results
            if mr.get("monitor_type")
        }
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
    for mt in all_monitor_types:
        col = f"score_{mt}"
        if col not in df.columns:
            df[col] = np.nan
    print(f"Loaded {len(df)} rollouts across {len(ALL_FILES)} files")
    print(f"  Monitor types found: {list(all_monitor_types.keys())}")
    return df


# ---------------------------------------------------------------------------
# Plot 1: Score distribution grid (2 rows x 4 cols)
# ---------------------------------------------------------------------------

def plot_distributions(df: pd.DataFrame, thresholds: dict,
                       suffix: str = "", monitor_label: str = ""):
    """2x4 grid: top row = base, bottom row = FT. 3 settings overlaid."""
    monitor_label = monitor_label or MONITOR_LABELS.get(MONITOR_TYPE, MONITOR_TYPE)
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

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               bbox_to_anchor=(0.5, 1.02), fontsize=10)

    fig.suptitle(f"Monitor Score Distributions — {monitor_label}", y=1.06,
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = OUTDIR / f"1_score_distributions{suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: Mean monitor score barplot
# ---------------------------------------------------------------------------

def plot_mean_scores(df: pd.DataFrame, suffix: str = "", monitor_label: str = ""):
    """Grouped barplot: 4 groups x 6 bars (3 base + 3 FT)."""
    monitor_label = monitor_label or MONITOR_LABELS.get(MONITOR_TYPE, MONITOR_TYPE)
    valid = df[df["actor_error"].isna() & df["monitor_score"].notna()].copy()

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
                mean_val = subset["monitor_score"].mean() if len(subset) > 0 else 0
                sem_val = subset["monitor_score"].sem() if len(subset) > 1 else 0
                x = x_center + (bar_i - (n_bars - 1) / 2) * bar_width
                ax.bar(x, mean_val, bar_width * 0.85, color=colors[setting],
                       yerr=sem_val, capsize=2, edgecolor="white", linewidth=0.5)
                bar_i += 1

    ax.set_xticks([i * group_width for i in range(n_groups)])
    ax.set_xticklabels(MODEL_GROUPS, fontsize=11)
    ax.set_ylabel("Mean Monitor Score")
    ax.set_title(f"Mean Monitor Score by Model & Setting — {monitor_label}", fontweight="bold")

    legend_elements = []
    for setting in SETTINGS:
        legend_elements.append(Patch(facecolor=BASE_COLORS[setting],
                                     label=f"Base / {SETTING_LABELS[setting]}"))
    for setting in SETTINGS:
        legend_elements.append(Patch(facecolor=FT_COLORS[setting],
                                     label=f"FT / {SETTING_LABELS[setting]}"))
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9, ncol=2)

    plt.tight_layout()
    out = OUTDIR / f"2_mean_monitor_score{suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3: Below-threshold percent (2 panels)
# ---------------------------------------------------------------------------

def plot_below_threshold(df: pd.DataFrame, thresholds: dict,
                         percentile_label: str = "95th",
                         suffix: str = "",
                         monitor_label: str = ""):
    """Two panels: dont_think and default. 4 groups x 2 bars (base vs FT)."""
    monitor_label = monitor_label or MONITOR_LABELS.get(MONITOR_TYPE, MONITOR_TYPE)
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
            bars = ax.bar(x_positions + offset,
                          [r * 100 for r in rates], bar_width * 0.85,
                          label=label, color=color, edgecolor="white", linewidth=0.5,
                          yerr=[e * 100 for e in errs], capsize=3,
                          error_kw={"linewidth": 1.0})
            for bar, rate, err in zip(bars, rates, errs):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        max(bar.get_height(), 0) + err * 100 + 1,
                        f"{rate * 100:.0f}%", ha="center", va="bottom", fontsize=8)

        fpr_pct = 100 - int(percentile_label.rstrip("th"))
        ax.set_xticks(x_positions)
        ax.set_xticklabels(MODEL_GROUPS, fontsize=10)
        ax.set_ylabel(f"Percent bypassing monitor at {fpr_pct}% FPR"
                       if panel_idx == 0 else "")
        ax.set_title(f"Setting: {SETTING_LABELS[setting]}", fontweight="bold")
        ax.set_ylim(0, 110)
        ax.legend(fontsize=10)

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
        f"Percent Bypassing Monitor at {fpr_pct}% FPR — {monitor_label}\n"
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
    """Grouped barplot: 4 groups x 6 bars. Y = percent with actor_error."""
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
                ax.bar(x, error_rate * 100, bar_width * 0.85, color=colors[setting],
                       edgecolor="white", linewidth=0.5,
                       yerr=ci * 100, capsize=2, error_kw={"linewidth": 1.0})
                if error_rate > 0.01:
                    ax.text(x, error_rate * 100 + ci * 100 + 0.5, f"{error_rate * 100:.0f}%",
                            ha="center", va="bottom", fontsize=7)
                bar_i += 1

    ax.set_xticks([i * group_width for i in range(n_groups)])
    ax.set_xticklabels(MODEL_GROUPS, fontsize=11)
    ax.set_ylabel("Error Rate (%)")
    ax.set_title("Actor Error Rate by Model & Setting", fontweight="bold")

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
                                       suffix: str = "",
                                       monitor_label: str = ""):
    """Same as plot_below_threshold but filtered to side_task_correct == True."""
    monitor_label = monitor_label or MONITOR_LABELS.get(MONITOR_TYPE, MONITOR_TYPE)
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
            bars = ax.bar(x_positions + offset,
                          [r * 100 for r in rates], bar_width * 0.85,
                          label=label, color=color, edgecolor="white",
                          linewidth=0.5,
                          yerr=[e * 100 for e in errs], capsize=3,
                          error_kw={"linewidth": 1.0})
            for i, (bar, rate, err) in enumerate(zip(bars, rates, errs)):
                group = MODEL_GROUPS[i]
                subset = valid[
                    (valid["model_group"] == group)
                    & (valid["is_ft"] == is_ft)
                    & (valid["setting"] == setting)
                ]
                ax.text(bar.get_x() + bar.get_width() / 2,
                        max(bar.get_height(), 0) + err * 100 + 1,
                        f"{rate * 100:.0f}%\nn={len(subset)}",
                        ha="center", va="bottom", fontsize=7)

        fpr_pct = 100 - int(percentile_label.rstrip("th"))
        ax.set_xticks(x_positions)
        ax.set_xticklabels(MODEL_GROUPS, fontsize=10)
        ax.set_ylabel(f"Percent bypassing monitor at {fpr_pct}% FPR"
                       if panel_idx == 0 else "")
        ax.set_title(f"Setting: {SETTING_LABELS[setting]}", fontweight="bold")
        ax.set_ylim(0, 125)
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
        f"Percent Bypassing Monitor at {fpr_pct}% FPR — Side Task Correct Only — {monitor_label}\n"
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

def plot_below_threshold_by_level(df: pd.DataFrame, thresholds: dict,
                                   percentile_label: str = "95th",
                                   suffix: str = "",
                                   monitor_label: str = ""):
    """Below-threshold rate by side_level, for dont_think and default settings.

    Layout: 2 rows (dont_think, default) x 4 cols (model groups).
    Each subplot: 5 level bars x 2 (base, FT) side-by-side.
    """
    monitor_label = monitor_label or MONITOR_LABELS.get(MONITOR_TYPE, MONITOR_TYPE)
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
                bars = ax.bar(x_positions + offset,
                              [r * 100 for r in rates], bar_width * 0.85,
                              color=colors, alpha=alpha, edgecolor=edge,
                              linewidth=1.2 if is_ft else 0.5,
                              yerr=[e * 100 for e in errs], capsize=2,
                              error_kw={"linewidth": 0.8})
                for i, (bar, rate, n) in enumerate(zip(bars, rates, ns)):
                    if n > 0:
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + errs[i] * 100 + 1,
                                f"{n}", ha="center", va="bottom", fontsize=6,
                                color="#555555")

            ax.set_xticks(x_positions)
            ax.set_xticklabels([f"L{i+1}" for i in range(5)], fontsize=9)
            if col_idx == 0:
                fpr_pct = 100 - int(percentile_label.rstrip("th"))
                ax.set_ylabel(f"{SETTING_LABELS[setting]}\nPercent bypassing ({fpr_pct}% FPR)",
                              fontsize=10, fontweight="bold")
            if row_idx == 0:
                ax.set_title(group, fontsize=11, fontweight="bold")
            ax.set_ylim(0, 115)

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
        f"Percent Bypassing Monitor at {fpr_pct}% FPR by Side Task Difficulty — {monitor_label}\n"
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
# Plot 7: Side task accuracy (4 groups x 6 bars)
# ---------------------------------------------------------------------------

def plot_side_task_accuracy(df: pd.DataFrame):
    """Grouped barplot: 4 groups x 6 bars. Y = side_task_correct percent."""
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
                ax.bar(x, acc * 100, bar_width * 0.85, color=colors[setting],
                       edgecolor="white", linewidth=0.5,
                       yerr=ci * 100, capsize=2, error_kw={"linewidth": 1.0})
                if acc > 0.01:
                    ax.text(x, acc * 100 + ci * 100 + 0.5, f"{acc * 100:.0f}%",
                            ha="center", va="bottom", fontsize=7)
                bar_i += 1

    ax.set_xticks([i * group_width for i in range(n_groups)])
    ax.set_xticklabels(MODEL_GROUPS, fontsize=11)
    ax.set_ylabel("Side Task Accuracy (%)")
    ax.set_title("Side Task Accuracy by Model & Setting", fontweight="bold")

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
# Plot 8/9: Task-term mention rate (base vs FT)
# ---------------------------------------------------------------------------

def load_rollouts_with_reasoning() -> pd.DataFrame:
    """Reload rollouts including reasoning text for mention analysis."""
    rows = []
    for (model_label, is_ft, setting), filename in ALL_FILES.items():
        path = ROLLOUTS_DIR / filename
        if not path.exists():
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
                reasoning = (d.get("reasoning") or "").lower()
                rows.append({
                    "model_group": model_label,
                    "is_ft": is_ft,
                    "setting": setting,
                    "actor_error": d.get("actor_error"),
                    "side_task_correct": d.get("side_task_correct"),
                    "mentions_main_task": "main task" in reasoning,
                    "mentions_side_task": "side task" in reasoning,
                })
    df = pd.DataFrame(rows)
    df["mentions_either"] = df["mentions_main_task"] | df["mentions_side_task"]
    df["mentions_neither"] = ~df["mentions_either"]
    df["no_mention_side_task"] = ~df["mentions_side_task"]
    print(f"Loaded {len(df)} rollouts for task-mention analysis")
    return df


def plot_task_mentions(df: pd.DataFrame, metric: str,
                       metric_label: str, plot_num: int):
    """Two panels (dont_think, default) side by side. 4 groups x 2 bars."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    panel_settings = ["dont_think", "default"]
    bar_width = 0.3

    for panel_idx, setting in enumerate(panel_settings):
        ax = axes[panel_idx]
        valid = df[
            (df["actor_error"].isna())
            & (df["setting"] == setting)
        ]
        x_positions = np.arange(len(MODEL_GROUPS))

        for ft_idx, (is_ft, label, color) in enumerate([
            (False, "Base", "#5B9BD5"),
            (True, "FT", "#E06060"),
        ]):
            rates = []
            errs = []
            ns = []
            for group in MODEL_GROUPS:
                subset = valid[
                    (valid["model_group"] == group)
                    & (valid["is_ft"] == is_ft)
                ]
                n = len(subset)
                ns.append(n)
                if n > 0:
                    rate = subset[metric].mean()
                    se = np.sqrt(rate * (1 - rate) / n)
                else:
                    rate = 0
                    se = 0
                rates.append(rate)
                errs.append(1.96 * se)

            offset = (ft_idx - 0.5) * bar_width
            bars = ax.bar(x_positions + offset,
                          [r * 100 for r in rates], bar_width * 0.85,
                          label=label, color=color, edgecolor="white", linewidth=0.5,
                          yerr=[e * 100 for e in errs], capsize=3,
                          error_kw={"linewidth": 1.0})
            for bar, rate, err, n in zip(bars, rates, errs, ns):
                if n > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + err * 100 + 1,
                            f"{rate * 100:.0f}%\nn={n}",
                            ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(MODEL_GROUPS, fontsize=10)
        ax.set_ylabel("Percent of Rollouts" if panel_idx == 0 else "")
        ax.set_title(f"Setting: {SETTING_LABELS[setting]}", fontweight="bold")
        ax.set_ylim(0, 110)
        ax.legend(fontsize=10)

    fig.suptitle(f'{metric_label} in Reasoning', fontweight="bold", fontsize=13)
    plt.tight_layout()
    out = OUTDIR / f"{plot_num}_task_mentions_{metric}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = load_rollouts()

    # Select metr_note scores for plotting.
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
                if not np.isnan(mean_s):
                    print(f"  {group:>14} {ft_str:<5} {setting:<12}  n={len(sub):>4}  errors={n_err:>3}  scored={n_scored:>4}  mean={mean_s:.2f}")
                else:
                    print(f"  {group:>14} {ft_str:<5} {setting:<12}  n={len(sub):>4}  errors={n_err:>3}  scored={n_scored:>4}  mean=N/A")

    # Compute 95th percentile thresholds from each model's OWN baseline scores
    valid = df[df["actor_error"].isna() & df["monitor_score"].notna()]
    thresholds_95 = {}
    print("\n95th Percentile Thresholds (per-model baseline, metr_note / GPT-4.1):")
    for group in MODEL_GROUPS:
        for is_ft in [False, True]:
            ft_str = "FT" if is_ft else "Base"
            baseline_scores = valid[
                (valid["model_group"] == group)
                & (valid["is_ft"] == is_ft)
                & (valid["setting"] == "baseline")
            ]["monitor_score"].values
            if len(baseline_scores) > 0:
                val = np.percentile(baseline_scores, 95)
                thresholds_95[(group, is_ft)] = val
                print(f"  {group} {ft_str}: {val:.1f}  (n={len(baseline_scores)})")
            else:
                print(f"  {group} {ft_str}: N/A (no baseline scores)")

    # Compute 95th percentile thresholds for GPT-5.4 monitor
    gpt54_col = f"score_{MONITOR_TYPE_GPT54}"
    thresholds_95_gpt54 = {}
    if gpt54_col in df.columns and df[gpt54_col].notna().any():
        valid_gpt54 = df[df["actor_error"].isna() & df[gpt54_col].notna()]
        print(f"\n95th Percentile Thresholds (per-model baseline, metr_note / GPT-5.4):")
        for group in MODEL_GROUPS:
            for is_ft in [False, True]:
                ft_str = "FT" if is_ft else "Base"
                baseline_scores = valid_gpt54[
                    (valid_gpt54["model_group"] == group)
                    & (valid_gpt54["is_ft"] == is_ft)
                    & (valid_gpt54["setting"] == "baseline")
                ][gpt54_col].values
                if len(baseline_scores) > 0:
                    val = np.percentile(baseline_scores, 95)
                    thresholds_95_gpt54[(group, is_ft)] = val
                    print(f"  {group} {ft_str}: {val:.1f}  (n={len(baseline_scores)})")
                else:
                    print(f"  {group} {ft_str}: N/A")

    # Generate plots
    print("\nGenerating plots...")

    # Distributions & mean scores — GPT-4.1
    plot_distributions(df, thresholds_95, suffix="_gpt41",
                       monitor_label=MONITOR_LABELS[MONITOR_TYPE])
    plot_mean_scores(df, suffix="_gpt41",
                     monitor_label=MONITOR_LABELS[MONITOR_TYPE])

    # Distributions & mean scores — GPT-5.4
    if gpt54_col in df.columns and df[gpt54_col].notna().any():
        df_gpt54 = df.copy()
        df_gpt54["monitor_score"] = df_gpt54[gpt54_col]
        plot_distributions(df_gpt54, thresholds_95_gpt54, suffix="_gpt54",
                           monitor_label=MONITOR_LABELS[MONITOR_TYPE_GPT54])
        plot_mean_scores(df_gpt54, suffix="_gpt54",
                         monitor_label=MONITOR_LABELS[MONITOR_TYPE_GPT54])

    # Below-threshold — GPT-4.1 monitor
    plot_below_threshold(df, thresholds_95, percentile_label="95th", suffix="_p95_gpt41",
                         monitor_label=MONITOR_LABELS[MONITOR_TYPE])
    plot_below_threshold_correct_only(df, thresholds_95, percentile_label="95th", suffix="_p95_gpt41",
                                       monitor_label=MONITOR_LABELS[MONITOR_TYPE])

    # Below-threshold — GPT-5.4 monitor
    if gpt54_col in df.columns and df[gpt54_col].notna().any():
        plot_below_threshold(df_gpt54, thresholds_95_gpt54, percentile_label="95th",
                             suffix="_p95_gpt54",
                             monitor_label=MONITOR_LABELS[MONITOR_TYPE_GPT54])
        plot_below_threshold_correct_only(df_gpt54, thresholds_95_gpt54, percentile_label="95th",
                                           suffix="_p95_gpt54",
                                           monitor_label=MONITOR_LABELS[MONITOR_TYPE_GPT54])

    plot_error_rates(df)
    plot_below_threshold_by_level(df, thresholds_95, percentile_label="95th", suffix="_p95",
                                   monitor_label=MONITOR_LABELS[MONITOR_TYPE])
    plot_side_task_accuracy(df)

    # Task-mention analysis (base vs FT)
    bvf = load_rollouts_with_reasoning()
    plot_task_mentions(bvf, "mentions_neither",
                       'Doesn\'t mention either "main task" or "side task"', plot_num=8)
    plot_task_mentions(bvf, "no_mention_side_task",
                       'Doesn\'t mention "side task"', plot_num=9)

    # Task-mention analysis — filtered to side_task_correct only
    bvf_correct = bvf[bvf["side_task_correct"] == True].copy()
    print(f"Task-mention analysis (correct only): {len(bvf_correct)} rollouts")
    plot_task_mentions(bvf_correct, "mentions_neither",
                       'Doesn\'t mention either "main task" or "side task" (side task correct only)',
                       plot_num=10)
    plot_task_mentions(bvf_correct, "no_mention_side_task",
                       'Doesn\'t mention "side task" (side task correct only)',
                       plot_num=11)

    print("Done!")


if __name__ == "__main__":
    main()
