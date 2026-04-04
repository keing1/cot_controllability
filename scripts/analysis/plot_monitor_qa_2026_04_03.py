#!/usr/bin/env python3
"""Monitor QA base-model reproducibility plots (2026-04-03).

Compares two runs of the 4 base models × 3 prompt settings:
  - Run 1 (20260402): Qwen on OpenRouter, GPT-OSS on Tinker
  - Run 2 (20260403): All on Tinker

Plots:
  1. Mean monitor score grouped barplot (4 groups × 6 bars: 3 run1 + 3 run2)
  2. Below-95th-percentile-threshold frequency (4 groups × 6 bars)

Usage:
    uv run python scripts/analysis/plot_monitor_qa_2026_04_03.py
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
DATE = "2026_04_03"
OUTDIR = Path(f"results/summaries/plots/{DATE}")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# File mapping: (model_label, run, setting) -> filename
#   run = 1 (20260402, mixed backend) or 2 (20260403, all tinker)
# ---------------------------------------------------------------------------

RUN1_FILES = {
    ("GPT-OSS-20B", 1, "baseline"):   "monitor_qa_gpt-oss-20b_baseline_20260402_090606_e6d9a872a86e.jsonl",
    ("GPT-OSS-20B", 1, "default"):    "monitor_qa_gpt-oss-20b_default_20260402_090606_83e8b33034dd.jsonl",
    ("GPT-OSS-20B", 1, "dont_think"): "monitor_qa_gpt-oss-20b_dont_think_20260402_090606_4dc781a5b973.jsonl",
    ("GPT-OSS-120B", 1, "baseline"):   "monitor_qa_gpt-oss-120b_baseline_20260402_090606_f92876518053.jsonl",
    ("GPT-OSS-120B", 1, "default"):    "monitor_qa_gpt-oss-120b_default_20260402_090606_13f19de13db3.jsonl",
    ("GPT-OSS-120B", 1, "dont_think"): "monitor_qa_gpt-oss-120b_dont_think_20260402_090606_332ac2a70d65.jsonl",
    ("Qwen3-8B", 1, "baseline"):   "monitor_qa_qwen3-8b_baseline_20260402_090606_eeca7509cfe4.jsonl",
    ("Qwen3-8B", 1, "default"):    "monitor_qa_qwen3-8b_default_20260402_090606_2e601d121375.jsonl",
    ("Qwen3-8B", 1, "dont_think"): "monitor_qa_qwen3-8b_dont_think_20260402_090606_a7796eb013fe.jsonl",
    ("Qwen3-32B", 1, "baseline"):   "monitor_qa_qwen3-32b_baseline_20260402_090606_fc330599a13b.jsonl",
    ("Qwen3-32B", 1, "default"):    "monitor_qa_qwen3-32b_default_20260402_090606_a164b4ddb912.jsonl",
    ("Qwen3-32B", 1, "dont_think"): "monitor_qa_qwen3-32b_dont_think_20260402_090606_ac3cc9445c04.jsonl",
}

RUN2_FILES = {
    ("GPT-OSS-20B", 2, "baseline"):   "monitor_qa_gpt-oss-20b_baseline_20260403_082613_e6d9a872a86e.jsonl",
    ("GPT-OSS-20B", 2, "default"):    "monitor_qa_gpt-oss-20b_default_20260403_082613_83e8b33034dd.jsonl",
    ("GPT-OSS-20B", 2, "dont_think"): "monitor_qa_gpt-oss-20b_dont_think_20260403_082613_4dc781a5b973.jsonl",
    ("GPT-OSS-120B", 2, "baseline"):   "monitor_qa_gpt-oss-120b_baseline_20260403_082613_f92876518053.jsonl",
    ("GPT-OSS-120B", 2, "default"):    "monitor_qa_gpt-oss-120b_default_20260403_082613_13f19de13db3.jsonl",
    ("GPT-OSS-120B", 2, "dont_think"): "monitor_qa_gpt-oss-120b_dont_think_20260403_082613_332ac2a70d65.jsonl",
    ("Qwen3-8B", 2, "baseline"):   "monitor_qa_qwen3-8b_baseline_20260403_082613_eeca7509cfe4.jsonl",
    ("Qwen3-8B", 2, "default"):    "monitor_qa_qwen3-8b_default_20260403_082613_2e601d121375.jsonl",
    ("Qwen3-8B", 2, "dont_think"): "monitor_qa_qwen3-8b_dont_think_20260403_082613_a7796eb013fe.jsonl",
    ("Qwen3-32B", 2, "baseline"):   "monitor_qa_qwen3-32b_baseline_20260403_082613_fc330599a13b.jsonl",
    ("Qwen3-32B", 2, "default"):    "monitor_qa_qwen3-32b_default_20260403_082613_a164b4ddb912.jsonl",
    ("Qwen3-32B", 2, "dont_think"): "monitor_qa_qwen3-32b_dont_think_20260403_082613_ac3cc9445c04.jsonl",
}

ALL_FILES = {**RUN1_FILES, **RUN2_FILES}

# Base vs FT file mapping for task-mention analysis.
# Uses the newest available run for each (model, is_ft) pair:
#   Base models: run 2 (20260403, all Tinker)
#   FT models: GPT-OSS from overnight (20260402), Qwen from rerun (20260403)
BASE_VS_FT_FILES = {
    # Base models (run 2)
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
    # FT models
    ("GPT-OSS-20B", True, "baseline"):   "monitor_qa_gpt-oss-20b-ft60_baseline_20260402_090606_859184b40afe.jsonl",
    ("GPT-OSS-20B", True, "default"):    "monitor_qa_gpt-oss-20b-ft60_default_20260402_090606_3d755d536923.jsonl",
    ("GPT-OSS-20B", True, "dont_think"): "monitor_qa_gpt-oss-20b-ft60_dont_think_20260402_090606_3c338823dfde.jsonl",
    ("GPT-OSS-120B", True, "baseline"):   "monitor_qa_gpt-oss-120b-ft60_baseline_20260402_090606_891d36d5e33e.jsonl",
    ("GPT-OSS-120B", True, "default"):    "monitor_qa_gpt-oss-120b-ft60_default_20260402_090606_169268a64c7d.jsonl",
    ("GPT-OSS-120B", True, "dont_think"): "monitor_qa_gpt-oss-120b-ft60_dont_think_20260402_090606_dafa1247edc6.jsonl",
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

# Which monitor type to use for plots. Set to "old_monitor" or "metr_note".
MONITOR_TYPE = "old_monitor"

# ---------------------------------------------------------------------------
# Colors: Run 1 (blues) vs Run 2 (reds)
# Light → dark for baseline → default → dont_think
# ---------------------------------------------------------------------------

RUN1_COLORS = {
    "baseline": "#A8C8E8",
    "default": "#5B9BD5",
    "dont_think": "#2F5F8A",
}
RUN2_COLORS = {
    "baseline": "#F4B8B8",
    "default": "#E06060",
    "dont_think": "#A02020",
}

RUN_LABELS = {1: "Run 1 (mixed backend)", 2: "Run 2 (all Tinker)"}


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
    for (model_label, run, setting), filename in ALL_FILES.items():
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
                mt_scores = _extract_monitor_scores(d)
                for mt in mt_scores:
                    all_monitor_types[mt] = None
                row = {
                    "model_group": model_label,
                    "run": run,
                    "setting": setting,
                    "monitor_score": d.get("monitor_score"),
                    "actor_error": d.get("actor_error"),
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
# Plot 1: Mean monitor score barplot
# ---------------------------------------------------------------------------

def plot_mean_scores(df: pd.DataFrame):
    """Grouped barplot: 4 groups × 6 bars (3 run1 + 3 run2)."""
    valid = df[df["actor_error"].isna() & df["monitor_score"].notna()].copy()

    fig, ax = plt.subplots(figsize=(14, 6))
    n_groups = len(MODEL_GROUPS)
    n_bars = 6  # 3 run1 + 3 run2
    bar_width = 0.12
    group_width = n_bars * bar_width + 0.15

    for g_idx, group in enumerate(MODEL_GROUPS):
        x_center = g_idx * group_width
        bar_i = 0
        for run in [1, 2]:
            colors = RUN1_COLORS if run == 1 else RUN2_COLORS
            for setting in SETTINGS:
                subset = valid[
                    (valid["model_group"] == group)
                    & (valid["run"] == run)
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
    ax.set_title("Mean Monitor Score by Model & Setting (Run 1 vs Run 2)", fontweight="bold")

    legend_elements = []
    for setting in SETTINGS:
        legend_elements.append(Patch(facecolor=RUN1_COLORS[setting],
                                     label=f"Run 1 / {SETTING_LABELS[setting]}"))
    for setting in SETTINGS:
        legend_elements.append(Patch(facecolor=RUN2_COLORS[setting],
                                     label=f"Run 2 / {SETTING_LABELS[setting]}"))
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9, ncol=2)

    plt.tight_layout()
    out = OUTDIR / "1_mean_monitor_score.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: Below-threshold frequency at 5% FPR (two panels: dont_think, default)
# ---------------------------------------------------------------------------

def plot_below_threshold(df: pd.DataFrame, thresholds: dict):
    """Two panels (dont_think, default). 4 groups × 2 bars (run1 vs run2)."""
    valid = df[df["actor_error"].isna() & df["monitor_score"].notna()].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    panel_settings = ["dont_think", "default"]
    bar_width = 0.3

    for panel_idx, setting in enumerate(panel_settings):
        ax = axes[panel_idx]
        x_positions = np.arange(len(MODEL_GROUPS))

        for run_idx, (run, label, color) in enumerate([
            (1, "Run 1 (mixed)", "#5B9BD5"),
            (2, "Run 2 (Tinker)", "#E06060"),
        ]):
            rates = []
            errs = []
            for group in MODEL_GROUPS:
                threshold = thresholds.get((group, run), 10)
                subset = valid[
                    (valid["model_group"] == group)
                    & (valid["run"] == run)
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

            offset = (run_idx - 0.5) * bar_width
            bars = ax.bar(x_positions + offset, rates, bar_width * 0.85,
                          label=label, color=color, edgecolor="white", linewidth=0.5,
                          yerr=errs, capsize=3, error_kw={"linewidth": 1.0})
            for bar, rate, err in zip(bars, rates, errs):
                if rate > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + err + 0.01,
                            f"{rate:.0%}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(MODEL_GROUPS, fontsize=10)
        ax.set_ylabel("Fraction not detected by monitor at 5% FPR"
                       if panel_idx == 0 else "")
        ax.set_title(f"Setting: {SETTING_LABELS[setting]}", fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=10)

    thresh_parts = []
    for g in MODEL_GROUPS:
        t1 = thresholds.get((g, 1), None)
        t2 = thresholds.get((g, 2), None)
        if t1 is not None and t2 is not None:
            thresh_parts.append(f"{g}: {t1:.0f}/{t2:.0f}")
    threshold_str = ", ".join(thresh_parts)
    fig.suptitle(
        f"Fraction Not Detected by Monitor at 5% FPR\n"
        f"(score thresholds run1/run2: {threshold_str})",
        fontweight="bold", fontsize=13,
    )
    plt.tight_layout()
    out = OUTDIR / "2_below_threshold_5pct_fpr.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3: Error rate barplot
# ---------------------------------------------------------------------------

def plot_error_rates(df: pd.DataFrame):
    """Grouped barplot: 4 groups × 6 bars (3 run1 + 3 run2). Y = actor error rate."""
    fig, ax = plt.subplots(figsize=(14, 6))
    n_groups = len(MODEL_GROUPS)
    n_bars = 6
    bar_width = 0.12
    group_width = n_bars * bar_width + 0.15

    for g_idx, group in enumerate(MODEL_GROUPS):
        x_center = g_idx * group_width
        bar_i = 0
        for run in [1, 2]:
            colors = RUN1_COLORS if run == 1 else RUN2_COLORS
            for setting in SETTINGS:
                subset = df[
                    (df["model_group"] == group)
                    & (df["run"] == run)
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
                if error_rate > 0.01:
                    ax.text(x, error_rate + ci + 0.005, f"{error_rate:.0%}",
                            ha="center", va="bottom", fontsize=7)
                bar_i += 1

    ax.set_xticks([i * group_width for i in range(n_groups)])
    ax.set_xticklabels(MODEL_GROUPS, fontsize=11)
    ax.set_ylabel("Error Rate")
    ax.set_title("Actor Error Rate by Model & Setting (Run 1 vs Run 2)", fontweight="bold")

    legend_elements = []
    for setting in SETTINGS:
        legend_elements.append(Patch(facecolor=RUN1_COLORS[setting],
                                     label=f"Run 1 / {SETTING_LABELS[setting]}"))
    for setting in SETTINGS:
        legend_elements.append(Patch(facecolor=RUN2_COLORS[setting],
                                     label=f"Run 2 / {SETTING_LABELS[setting]}"))
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9, ncol=2)

    plt.tight_layout()
    out = OUTDIR / "3_error_rates.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Data loading: base vs FT (for task-mention analysis)
# ---------------------------------------------------------------------------

def load_base_vs_ft_rollouts() -> pd.DataFrame:
    """Load base & FT rollout files with reasoning mention flags."""
    rows = []
    for (model_label, is_ft, setting), filename in BASE_VS_FT_FILES.items():
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
                reasoning = (d.get("reasoning") or "").lower()
                rows.append({
                    "model_group": model_label,
                    "is_ft": is_ft,
                    "setting": setting,
                    "actor_error": d.get("actor_error"),
                    "mentions_main_task": "main task" in reasoning,
                    "mentions_side_task": "side task" in reasoning,
                })
    df = pd.DataFrame(rows)
    df["mentions_either"] = df["mentions_main_task"] | df["mentions_side_task"]
    print(f"Loaded {len(df)} rollouts for base-vs-FT mention analysis")
    return df


# ---------------------------------------------------------------------------
# Plot 4 & 5: Task-term mention rate (base vs FT, one plot per setting)
# ---------------------------------------------------------------------------

def plot_task_mentions(df: pd.DataFrame, setting: str, metric: str,
                       metric_label: str, plot_num: int):
    """Bar plot: 4 model groups × 2 bars (base vs FT)."""
    valid = df[
        (df["actor_error"].isna())
        & (df["setting"] == setting)
    ].copy()

    fig, ax = plt.subplots(figsize=(10, 5))
    bar_width = 0.3
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
        bars = ax.bar(x_positions + offset, rates, bar_width * 0.85,
                      label=label, color=color, edgecolor="white", linewidth=0.5,
                      yerr=errs, capsize=3, error_kw={"linewidth": 1.0})
        for bar, rate, err, n in zip(bars, rates, errs, ns):
            if n > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + err + 0.01,
                        f"{rate:.0%}\nn={n}",
                        ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(MODEL_GROUPS, fontsize=11)
    ax.set_ylabel("Fraction of Rollouts")
    ax.set_ylim(0, 1.0)
    ax.set_title(
        f'{metric_label} in Reasoning — {SETTING_LABELS[setting]}',
        fontweight="bold",
    )
    ax.legend(fontsize=10)

    plt.tight_layout()
    out = OUTDIR / f"{plot_num}_task_mentions_{setting}_{metric}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = load_rollouts()

    # Select which monitor type's scores to use for plotting.
    score_col = f"score_{MONITOR_TYPE}"
    if score_col in df.columns and df[score_col].notna().any():
        print(f"Using monitor type '{MONITOR_TYPE}' for plots (column: {score_col})")
        df["monitor_score"] = df[score_col]
    else:
        print(f"Monitor type '{MONITOR_TYPE}' not found, using legacy monitor_score")

    # Print summary
    for group in MODEL_GROUPS:
        for run in [1, 2]:
            for setting in SETTINGS:
                sub = df[(df["model_group"] == group) & (df["run"] == run) & (df["setting"] == setting)]
                n_err = sub["actor_error"].notna().sum()
                n_scored = sub["monitor_score"].notna().sum()
                valid_scores = sub.loc[sub["actor_error"].isna() & sub["monitor_score"].notna(), "monitor_score"]
                mean_s = valid_scores.mean()
                mean_str = f"{mean_s:.2f}" if not np.isnan(mean_s) else "N/A"
                print(f"  {group:>14} run{run} {setting:<12}  n={len(sub):>4}  errors={n_err:>3}  scored={n_scored:>4}  mean={mean_str}")

    # Compute 95th percentile thresholds from each (model, run)'s baseline scores
    valid = df[df["actor_error"].isna() & df["monitor_score"].notna()]
    thresholds = {}
    print("\n95th Percentile Thresholds (per-model, per-run baseline):")
    for group in MODEL_GROUPS:
        for run in [1, 2]:
            baseline_scores = valid[
                (valid["model_group"] == group)
                & (valid["run"] == run)
                & (valid["setting"] == "baseline")
            ]["monitor_score"].values
            if len(baseline_scores) > 0:
                val = np.percentile(baseline_scores, 95)
                thresholds[(group, run)] = val
                print(f"  {group} run{run}: {val:.1f}  (n={len(baseline_scores)})")
            else:
                print(f"  {group} run{run}: N/A")

    # Generate plots
    print("\nGenerating plots...")
    plot_mean_scores(df)
    plot_below_threshold(df, thresholds)
    plot_error_rates(df)

    # Task-mention analysis (base vs FT)
    bvf = load_base_vs_ft_rollouts()
    plot_task_mentions(bvf, "default", "mentions_either",
                       'Mentions "main task" or "side task"', plot_num=4)
    plot_task_mentions(bvf, "dont_think", "mentions_either",
                       'Mentions "main task" or "side task"', plot_num=4)
    plot_task_mentions(bvf, "default", "mentions_side_task",
                       'Mentions "side task"', plot_num=5)
    plot_task_mentions(bvf, "dont_think", "mentions_side_task",
                       'Mentions "side task"', plot_num=5)
    print("Done!")


if __name__ == "__main__":
    main()
