#!/usr/bin/env python3
"""Grouped bar plot: Base vs FT controllability (2026-03-24).

Two subplots (CoTControl, ReasonIF), each with grouped bars for
4 models: gpt-oss-20b, gpt-oss-120b, qwen3-8b, qwen3-32b.
Each group has a Base bar and a FT bar (lr=1e-4, step 60).

Reads from:
  - results/summaries/tables/summary_by_run.csv

Outputs to:
  - results/summaries/plots/2026_03_24/

Usage:
    python scripts/analysis/plot_summary_2026_03_24.py
"""

import csv
import math
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.size"] = 11

SUMMARY_BY_RUN = Path("results/summaries/tables/summary_by_run.csv")
SUMMARY_BY_INSTRUCTION = Path("results/summaries/tables/summary_by_instruction.csv")
DATE = "2026_03_24"
OUTDIR = Path(f"results/summaries/plots/{DATE}")
OUTDIR.mkdir(parents=True, exist_ok=True)

Z80 = 1.282

# ---------------------------------------------------------------------------
# Models to plot (in display order)
# ---------------------------------------------------------------------------

MODELS = [
    {
        "model": "openai/gpt-oss-20b",
        "label": "GPT-OSS-20B",
        "base_effort": "medium",  # gpt-oss uses medium reasoning effort
    },
    {
        "model": "openai/gpt-oss-120b",
        "label": "GPT-OSS-120B",
        "base_effort": "medium",
    },
    {
        "model": "qwen/qwen3-8b",
        "label": "Qwen3-8B",
        "base_effort": "none",
    },
    {
        "model": "qwen/qwen3-32b",
        "label": "Qwen3-32B",
        "base_effort": "none",
    },
]

FT_LR = "1e-4"
FT_CKPT = "000060"
FT_CKPTS = ["000060", "000120", "000180", "final"]
FT_CKPT_DISPLAY = {"000060": "60", "000120": "120", "000180": "180", "final": "final"}

EVAL_TYPES = ["cotcontrol", "reasonif"]
EVAL_DISPLAY = {"cotcontrol": "CoTControl", "reasonif": "ReasonIF"}

# All models for compliance_by plots (same order as March 18/20)
ALL_MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3-8b",
    "qwen/qwen3-32b",
    "qwen/qwen3-235b-a22b",
    "qwen/qwen3.5-4b",
    "qwen/qwen3.5-27b",
    "qwen/qwen3.5-35b-a3b",
    "qwen/qwen3.5-397b-a17b",
    "moonshotai/kimi-k2-thinking",
    "moonshotai/kimi-k2.5",
]
GROUP_STARTS = [2, 5, 9]  # qwen3, qwen3.5, kimi boundaries

COTCONTROL_TYPES = sorted([
    "alternating_case", "end_of_sentence", "ignore_question",
    "lowercase_thinking", "meow_between_words", "multiple_word_suppression",
    "repeat_sentences", "uppercase_thinking", "word_suppression",
])
REASONIF_TYPES = sorted([
    "end_checker", "english_capital", "json_format",
    "no_comma", "number_words", "reasoning_language",
])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sf(row, col):
    if row is None:
        return None
    v = row.get(col, "")
    return float(v) if v else None


def si(row, col):
    if row is None:
        return 0
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
        lr = m.group(1) or "1e-4"
        return lr, m.group(2)
    return None, None


def is_ft_row(row):
    """Check if row is a fine-tuned model (has -rif- in filename)."""
    return "-rif-" in row.get("file_name", "")


def is_base_row(row):
    """Check if row is a base model (no checkpoint, no -rif-)."""
    return not row.get("checkpoint", "") and not is_ft_row(row)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_data():
    """Load summary CSV and extract base + FT rows for target models."""
    model_names = {m["model"] for m in MODELS}

    with open(SUMMARY_BY_RUN) as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r["base_model"] in model_names]

    # For each (model, eval_type), find base and FT rows
    data = {}  # (model, eval_type) -> {"base": row, "ft": row}

    for m_cfg in MODELS:
        model = m_cfg["model"]
        effort = m_cfg["base_effort"]

        for eval_type in EVAL_TYPES:
            key = (model, eval_type)
            data[key] = {"base": None, "ft": None}

            # Find base row: no checkpoint, correct effort, max_tokens=28000
            base_candidates = [
                r for r in rows
                if r["base_model"] == model
                and r["eval_type"] == eval_type
                and is_base_row(r)
                and r.get("reasoning_effort", "") == effort
                and r.get("max_tokens", "") == "28000"
            ]
            if base_candidates:
                # Pick latest by timestamp
                base_candidates.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
                data[key]["base"] = base_candidates[0]

            # Find FT row: lr=1e-4, checkpoint=000060, latest run
            ft_candidates = []
            for r in rows:
                if r["base_model"] != model or r["eval_type"] != eval_type:
                    continue
                if not is_ft_row(r):
                    continue
                lr, ckpt = extract_lr_ckpt(r.get("file_name", ""))
                if lr == FT_LR and ckpt == FT_CKPT:
                    ft_candidates.append(r)

            if ft_candidates:
                # Pick latest by timestamp
                ft_candidates.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
                data[key]["ft"] = ft_candidates[0]

    return data


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_plot(data):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    bar_width = 0.32
    x = np.arange(len(MODELS))

    colors = {"base": "#78909C", "ft": "#1E88E5"}

    for ax_idx, eval_type in enumerate(EVAL_TYPES):
        ax = axes[ax_idx]

        base_vals = []
        base_errs = []
        ft_vals = []
        ft_errs = []

        for m_cfg in MODELS:
            model = m_cfg["model"]
            key = (model, eval_type)

            # Base
            base_row = data[key]["base"]
            if base_row:
                p = sf(base_row, "compliant_rate")
                n = si(base_row, "n_clean")
                base_vals.append((p or 0) * 100)
                base_errs.append(ci80(p, n) * 100)
            else:
                base_vals.append(0)
                base_errs.append(0)

            # FT
            ft_row = data[key]["ft"]
            if ft_row:
                p = sf(ft_row, "compliant_rate")
                n = si(ft_row, "n_clean")
                ft_vals.append((p or 0) * 100)
                ft_errs.append(ci80(p, n) * 100)
            else:
                ft_vals.append(0)
                ft_errs.append(0)

        bars_base = ax.bar(
            x - bar_width / 2, base_vals, bar_width,
            yerr=base_errs, capsize=3,
            color=colors["base"], alpha=0.85,
            label="Base",
            error_kw={"linewidth": 1},
        )
        bars_ft = ax.bar(
            x + bar_width / 2, ft_vals, bar_width,
            yerr=ft_errs, capsize=3,
            color=colors["ft"], alpha=0.85,
            label="FT (step 60)",
            error_kw={"linewidth": 1},
        )

        # Value labels on top of bars
        for bars, errs in [(bars_base, base_errs), (bars_ft, ft_errs)]:
            for bar, err in zip(bars, errs):
                height = bar.get_height()
                if height > 0:
                    ax.annotate(
                        f"{height:.1f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height + err),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom",
                        fontsize=8, fontweight="bold",
                    )

        ax.set_title(EVAL_DISPLAY[eval_type], fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([m["label"] for m in MODELS], rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Compliance (%)" if ax_idx == 0 else "")
        ax.legend(fontsize=9, loc="upper left")
        ax.set_ylim(0, None)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "CoT Controllability: Base vs Fine-Tuned (80% CI)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()

    out_path = OUTDIR / "base_vs_ft_lr1e-4_step60_compliance.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def make_plot_meta_discussion(data):
    """Base vs FT (step 60) grouped bar plot for meta discussion rate."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    bar_width = 0.32
    x = np.arange(len(MODELS))
    colors = {"base": "#78909C", "ft": "#1E88E5"}

    for ax_idx, eval_type in enumerate(EVAL_TYPES):
        ax = axes[ax_idx]

        base_vals, base_errs = [], []
        ft_vals, ft_errs = [], []

        for m_cfg in MODELS:
            model = m_cfg["model"]
            key = (model, eval_type)

            base_row = data[key]["base"]
            if base_row:
                p = sf(base_row, "meta_discussion_rate")
                n = si(base_row, "n_clean")
                base_vals.append((p or 0) * 100)
                base_errs.append(ci80(p, n) * 100)
            else:
                base_vals.append(0)
                base_errs.append(0)

            ft_row = data[key]["ft"]
            if ft_row:
                p = sf(ft_row, "meta_discussion_rate")
                n = si(ft_row, "n_clean")
                ft_vals.append((p or 0) * 100)
                ft_errs.append(ci80(p, n) * 100)
            else:
                ft_vals.append(0)
                ft_errs.append(0)

        bars_base = ax.bar(
            x - bar_width / 2, base_vals, bar_width,
            yerr=base_errs, capsize=3,
            color=colors["base"], alpha=0.85, label="Base",
            error_kw={"linewidth": 1},
        )
        bars_ft = ax.bar(
            x + bar_width / 2, ft_vals, bar_width,
            yerr=ft_errs, capsize=3,
            color=colors["ft"], alpha=0.85, label="FT (step 60)",
            error_kw={"linewidth": 1},
        )

        for bars, errs in [(bars_base, base_errs), (bars_ft, ft_errs)]:
            for bar, err in zip(bars, errs):
                height = bar.get_height()
                if height > 0:
                    ax.annotate(
                        f"{height:.1f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height + err),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom",
                        fontsize=8, fontweight="bold",
                    )

        ax.set_title(EVAL_DISPLAY[eval_type], fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([m["label"] for m in MODELS], rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Meta Discussion Rate (%)" if ax_idx == 0 else "")
        ax.legend(fontsize=9, loc="upper right")
        ax.set_ylim(0, None)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Meta Discussion Rate: Base vs Fine-Tuned (80% CI)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()

    out_path = OUTDIR / "base_vs_ft_lr1e-4_step60_meta_discussion.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Training curve data + plot
# ---------------------------------------------------------------------------

MODEL_COLORS = {
    "openai/gpt-oss-20b": "#FF2A00",
    "openai/gpt-oss-120b": "#9933FF",
    "qwen/qwen3-8b": "#4895EF",         # light blue
    "qwen/qwen3-32b": "#1D3557",        # navy
}


def load_training_curve_data():
    """Load base + all lr=1e-4 checkpoints for the 4 target models."""
    model_names = {m["model"] for m in MODELS}

    with open(SUMMARY_BY_RUN) as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r["base_model"] in model_names]

    # result: (model, eval_type) -> {"base": row, "ckpts": {ckpt: row}}
    data = {}

    for m_cfg in MODELS:
        model = m_cfg["model"]
        effort = m_cfg["base_effort"]

        for eval_type in EVAL_TYPES:
            key = (model, eval_type)
            data[key] = {"base": None, "ckpts": {}}

            # Base row
            base_candidates = [
                r for r in rows
                if r["base_model"] == model
                and r["eval_type"] == eval_type
                and is_base_row(r)
                and r.get("reasoning_effort", "") == effort
                and r.get("max_tokens", "") == "28000"
            ]
            if base_candidates:
                base_candidates.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
                data[key]["base"] = base_candidates[0]

            # FT rows per checkpoint
            for ckpt in FT_CKPTS:
                candidates = []
                for r in rows:
                    if r["base_model"] != model or r["eval_type"] != eval_type:
                        continue
                    if not is_ft_row(r):
                        continue
                    lr, c = extract_lr_ckpt(r.get("file_name", ""))
                    if lr == FT_LR and c == ckpt:
                        candidates.append(r)
                if candidates:
                    candidates.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
                    data[key]["ckpts"][ckpt] = candidates[0]

    return data


def make_training_curve_plot(data):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax_idx, eval_type in enumerate(EVAL_TYPES):
        ax = axes[ax_idx]

        for m_cfg in MODELS:
            model = m_cfg["model"]
            key = (model, eval_type)
            color = MODEL_COLORS[model]

            xs = []  # numeric x positions
            vals = []
            errs = []
            xlabels = []

            # Base at x=0
            base_row = data[key]["base"]
            if base_row:
                p = sf(base_row, "compliant_rate")
                n = si(base_row, "n_clean")
                xs.append(0)
                vals.append((p or 0) * 100)
                errs.append(ci80(p, n) * 100)
                xlabels.append("base")

            # Checkpoints
            for i, ckpt in enumerate(FT_CKPTS):
                row = data[key]["ckpts"].get(ckpt)
                if row:
                    p = sf(row, "compliant_rate")
                    n = si(row, "n_clean")
                    xs.append(i + 1)
                    vals.append((p or 0) * 100)
                    errs.append(ci80(p, n) * 100)

            ax.errorbar(
                xs, vals, yerr=errs,
                fmt="o-", color=color, capsize=3,
                label=m_cfg["label"],
                linewidth=1.5, markersize=5,
                elinewidth=1,
            )

        tick_positions = [0] + list(range(1, len(FT_CKPTS) + 1))
        tick_labels = ["base"] + [FT_CKPT_DISPLAY[c] for c in FT_CKPTS]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=9)
        ax.set_xlabel("Batch", fontsize=10)
        ax.set_ylabel("Compliance (%)" if ax_idx == 0 else "")
        ax.set_title(EVAL_DISPLAY[eval_type], fontsize=13, fontweight="bold")
        ax.set_ylim(0, None)
        ax.grid(axis="y", alpha=0.3)

    # Single shared legend at bottom center
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(
        "Controllability over Fine-Tuning (80% CI)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()

    out_path = OUTDIR / "ft_training_curve_lr1e-4_compliance.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def make_training_curve_meta_discussion_plot(data):
    """Meta discussion rate over fine-tuning checkpoints for all 4 models."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax_idx, eval_type in enumerate(EVAL_TYPES):
        ax = axes[ax_idx]

        for m_cfg in MODELS:
            model = m_cfg["model"]
            key = (model, eval_type)
            color = MODEL_COLORS[model]

            xs = []
            vals = []
            errs = []

            # Base at x=0
            base_row = data[key]["base"]
            if base_row:
                p = sf(base_row, "meta_discussion_rate")
                n = si(base_row, "n_clean")
                xs.append(0)
                vals.append((p or 0) * 100)
                errs.append(ci80(p, n) * 100)

            # Checkpoints
            for i, ckpt in enumerate(FT_CKPTS):
                row = data[key]["ckpts"].get(ckpt)
                if row:
                    p = sf(row, "meta_discussion_rate")
                    n = si(row, "n_clean")
                    xs.append(i + 1)
                    vals.append((p or 0) * 100)
                    errs.append(ci80(p, n) * 100)

            ax.errorbar(
                xs, vals, yerr=errs,
                fmt="o-", color=color, capsize=3,
                label=m_cfg["label"],
                linewidth=1.5, markersize=5,
                elinewidth=1,
            )

        tick_positions = [0] + list(range(1, len(FT_CKPTS) + 1))
        tick_labels = ["base"] + [FT_CKPT_DISPLAY[c] for c in FT_CKPTS]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=9)
        ax.set_xlabel("Batch", fontsize=10)
        ax.set_ylabel("Meta Discussion Rate (%)" if ax_idx == 0 else "")
        ax.set_title(EVAL_DISPLAY[eval_type], fontsize=13, fontweight="bold")
        ax.set_ylim(0, None)
        ax.grid(axis="y", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(
        "Meta Discussion Rate over Fine-Tuning (80% CI)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()

    out_path = OUTDIR / "ft_training_curve_lr1e-4_meta_discussion.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Effort sweep: combined gpt-oss-20b + 120b (base vs FT across efforts)
# ---------------------------------------------------------------------------

EFFORT_ORDER = ["low", "medium", "high"]

# Tinker session IDs for the lr=1e-4 step-60 FT checkpoints
STEP60_SESSION_IDS = {
    "openai/gpt-oss-120b": "1cbfc28d",
    "openai/gpt-oss-20b": "6adcf41e",
}

EFFORT_MODELS = [
    {"model": "openai/gpt-oss-20b", "label": "GPT-OSS-20B", "color": "#FF2A00"},
    {"model": "openai/gpt-oss-120b", "label": "GPT-OSS-120B", "color": "#9933FF"},
]


def load_effort_sweep_data():
    """Load base and FT-step60 rows at each effort level for gpt-oss models."""
    with open(SUMMARY_BY_RUN) as f:
        all_rows = list(csv.DictReader(f))

    # Base: gpt-oss, no checkpoint, max_tokens=28000, keyed by (model, eval, effort)
    base_effort = {}
    for r in all_rows:
        if "gpt-oss" not in r["base_model"]:
            continue
        if (r.get("checkpoint") or "").strip():
            continue
        if r.get("max_tokens") != "28000":
            continue
        if si(r, "n_clean") < 10:
            continue
        k = (r["base_model"], r["eval_type"], r["reasoning_effort"])
        if k not in base_effort or r["timestamp"] > base_effort[k]["timestamp"]:
            base_effort[k] = r

    # FT step60: match by tinker session ID + 000060 in checkpoint
    ft_effort = {}
    for r in all_rows:
        cp = (r.get("checkpoint") or "").strip()
        if not cp or "000060" not in cp:
            continue
        model = r["base_model"]
        sid = STEP60_SESSION_IDS.get(model)
        if sid and sid in cp:
            k = (model, r["eval_type"], r["reasoning_effort"])
            if k not in ft_effort or r["timestamp"] > ft_effort[k]["timestamp"]:
                ft_effort[k] = r

    return base_effort, ft_effort


def make_effort_sweep_plot(base_effort, ft_effort):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    x = np.arange(len(EFFORT_ORDER))

    for ax_idx, eval_type in enumerate(EVAL_TYPES):
        ax = axes[ax_idx]

        for m_cfg in EFFORT_MODELS:
            model = m_cfg["model"]
            color = m_cfg["color"]

            # Base line (solid)
            by, be = [], []
            bx_valid = []
            for i, effort in enumerate(EFFORT_ORDER):
                row = base_effort.get((model, eval_type, effort))
                if row:
                    p = sf(row, "compliant_rate")
                    n = si(row, "n_clean")
                    bx_valid.append(i)
                    by.append((p or 0) * 100)
                    be.append(ci80(p, n) * 100)
            if bx_valid:
                ax.errorbar(
                    bx_valid, by, yerr=be,
                    fmt="o-", color=color, capsize=4,
                    label=f"{m_cfg['label']} Base",
                    linewidth=1.5, markersize=6, elinewidth=1,
                )

            # FT line (dotted)
            fy, fe = [], []
            fx_valid = []
            for i, effort in enumerate(EFFORT_ORDER):
                row = ft_effort.get((model, eval_type, effort))
                if row:
                    p = sf(row, "compliant_rate")
                    n = si(row, "n_clean")
                    fx_valid.append(i)
                    fy.append((p or 0) * 100)
                    fe.append(ci80(p, n) * 100)
            if fx_valid:
                ax.errorbar(
                    fx_valid, fy, yerr=fe,
                    fmt="s:", color=color, capsize=4,
                    label=f"{m_cfg['label']} FT",
                    linewidth=1.5, markersize=6, elinewidth=1,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(EFFORT_ORDER, fontsize=10)
        ax.set_xlabel("Reasoning Effort", fontsize=10)
        ax.set_ylabel("Compliance (%)" if ax_idx == 0 else "")
        ax.set_title(EVAL_DISPLAY[eval_type], fontsize=13, fontweight="bold")
        ax.set_ylim(0, None)
        ax.grid(axis="y", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(
        "Controllability vs Reasoning Effort: Base vs FT (80% CI)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()

    out_path = OUTDIR / "effort_sweep_combined_lr1e-4.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Param scaling: combined (base vs FT step60 by model size)
# ---------------------------------------------------------------------------

import matplotlib.ticker as ticker

PARAM_FAMILIES = {
    "qwen3": {
        "models": [("qwen/qwen3-8b", 8), ("qwen/qwen3-32b", 32)],
        "color": "#1976D2",
    },
    "gpt-oss": {
        "models": [("openai/gpt-oss-20b", 20), ("openai/gpt-oss-120b", 120)],
        "color": "#B53A3A",
    },
}


def load_param_scaling_data():
    """Load base and FT-step60 (lr=1e-4, 20260319 run) for param scaling."""
    with open(SUMMARY_BY_RUN) as f:
        all_rows = list(csv.DictReader(f))

    # Base: canonical effort, prefer 28K for gpt-oss
    base = {}
    for r in all_rows:
        if (r.get("checkpoint") or "").strip():
            continue
        model = r["base_model"]
        et = r["eval_type"]
        effort = r.get("reasoning_effort", "")
        # Check canonical effort
        if "gpt-oss" in model:
            if effort != "medium":
                continue
            if r.get("max_tokens") != "28000":
                continue
        else:
            if effort not in ("none", ""):
                continue
        if si(r, "n_clean") < 10:
            continue
        k = (model, et)
        if k not in base or r["timestamp"] > base[k]["timestamp"]:
            base[k] = r

    # FT step60 from 20260319 run
    ft = {}
    for r in all_rows:
        fn = r.get("file_name", "")
        if "rif-" not in fn:
            continue
        if "000060" not in fn:
            continue
        if "20260319" not in fn:
            continue
        if si(r, "n_clean") < 10:
            continue
        k = (r["base_model"], r["eval_type"])
        if k not in ft or r["timestamp"] > ft[k]["timestamp"]:
            ft[k] = r

    return base, ft


def make_param_scaling_plot(base, ft):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax_idx, eval_type in enumerate(EVAL_TYPES):
        ax = axes[ax_idx]

        for family, cfg in PARAM_FAMILIES.items():
            color = cfg["color"]
            model_params = cfg["models"]

            # Base (solid, circle)
            bx, by, be = [], [], []
            for model, params in model_params:
                row = base.get((model, eval_type))
                if row:
                    p = sf(row, "compliant_rate") or 0
                    n = si(row, "n_clean")
                    bx.append(params)
                    by.append(p * 100)
                    be.append(ci80(p, n) * 100)
            if bx:
                ax.errorbar(
                    bx, by, yerr=be, marker="o", linestyle="-", color=color,
                    label=f"{family} base", capsize=4, lw=1.5, markersize=6,
                    elinewidth=1,
                )

            # FT (dashed, square)
            fx, fy, fe = [], [], []
            for model, params in model_params:
                row = ft.get((model, eval_type))
                if row:
                    p = sf(row, "compliant_rate") or 0
                    n = si(row, "n_clean")
                    fx.append(params)
                    fy.append(p * 100)
                    fe.append(ci80(p, n) * 100)
            if fx:
                ax.errorbar(
                    fx, fy, yerr=fe, marker="s", linestyle="--", color=color,
                    label=f"{family} FT", capsize=4, lw=1.5, markersize=6,
                    elinewidth=1,
                )

        ax.set_xlabel("Parameters (B)", fontsize=10)
        ax.set_ylabel("Compliance (%)" if ax_idx == 0 else "")
        ax.set_title(EVAL_DISPLAY[eval_type], fontsize=13, fontweight="bold")
        ax.set_xscale("log")
        ax.set_xticks([8, 20, 32, 120])
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.set_xticklabels(["8B", "20B", "32B", "120B"])
        ax.set_ylim(0, None)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(
        "Controllability vs Model Size: Base vs FT (80% CI)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()

    out_path = OUTDIR / "param_scaling_combined.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Compliance by model / instruction / meta discussion (all models)
# ---------------------------------------------------------------------------


def short_name(m: str) -> str:
    return m.split("/")[-1]


def include_row(row: dict) -> bool:
    """max_tokens > 20K, except gpt-oss-20b base."""
    mt = int(row["max_tokens"]) if row["max_tokens"] else 0
    cp = (row.get("checkpoint") or "").strip()
    if "gpt-oss-20b" in row["base_model"] and not cp:
        return True
    return mt > 20000


def is_canonical_effort(row: dict) -> bool:
    effort = row.get("reasoning_effort", "")
    if "gpt-oss" in row["base_model"]:
        return effort == "medium"
    return effort in ("none", "")


def add_group_seps(ax, models_list):
    for gs in GROUP_STARTS:
        if gs < len(models_list):
            ax.axvline(x=gs - 0.5, color="gray", linewidth=0.5, alpha=0.3)


def load_all_models_data():
    """Load base runs and instruction-level data for all models."""
    all_runs = load_csv(SUMMARY_BY_RUN)
    all_instr = load_csv(SUMMARY_BY_INSTRUCTION)

    def best_by_key(rows, key_fn, filter_fn=None, min_clean=10):
        best = {}
        for r in rows:
            if filter_fn and not filter_fn(r):
                continue
            if si(r, "n_clean") < min_clean:
                continue
            k = key_fn(r)
            if k not in best or r["timestamp"] > best[k]["timestamp"]:
                best[k] = r
        return best

    base_runs = best_by_key(
        all_runs,
        key_fn=lambda r: (r["base_model"], r["eval_type"]),
        filter_fn=lambda r: (
            include_row(r)
            and not (r.get("checkpoint") or "").strip()
            and is_canonical_effort(r)
        ),
    )
    base_instr = best_by_key(
        all_instr,
        key_fn=lambda r: (r["base_model"], r["eval_type"], r["instruction_type"]),
        filter_fn=lambda r: (
            include_row(r)
            and not (r.get("checkpoint") or "").strip()
            and is_canonical_effort(r)
        ),
    )

    model_key = {m: i for i, m in enumerate(ALL_MODELS)}
    models = sorted(
        {k[0] for k in base_runs},
        key=lambda m: model_key.get(m, len(ALL_MODELS)),
    )
    return base_runs, base_instr, models


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def make_compliance_by_model_plot(base_runs, models):
    """1a/1b: single bar chart per eval_type showing all models."""
    cmap = matplotlib.colormaps.get_cmap("tab20")
    colors = {m: cmap(i / max(len(models) - 1, 1)) for i, m in enumerate(models)}

    for eval_type in EVAL_TYPES:
        fig, ax = plt.subplots(figsize=(13, 5.5))
        ms = [m for m in models if (m, eval_type) in base_runs]
        vals, errs, names = [], [], []
        for m in ms:
            row = base_runs[(m, eval_type)]
            p = sf(row, "compliant_rate") or 0
            n = si(row, "n_clean")
            vals.append(p * 100)
            errs.append(ci80(p, n) * 100)
            names.append(short_name(m))

        bars = ax.bar(
            range(len(ms)), vals, yerr=errs, capsize=4,
            color=[colors[m] for m in ms], edgecolor="white",
            error_kw=dict(lw=1.5, capthick=1.2),
        )
        for bar, v, e in zip(bars, vals, errs):
            label_y = v + e + 0.5
            ax.text(
                bar.get_x() + bar.get_width() / 2, label_y,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=13, fontweight="bold",
            )
        ax.set_ylabel("Compliance (%)", fontsize=16)
        ax.set_title(
            f"{EVAL_DISPLAY[eval_type]} Controllability by Model (80% CI)",
            fontsize=18, fontweight="bold",
        )
        ax.set_xticks(range(len(ms)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=14)
        ax.tick_params(axis="y", labelsize=13)
        top = max(v + e for v, e in zip(vals, errs)) if vals else 10
        ax.set_ylim(0, top * 1.25)
        add_group_seps(ax, ms)
        fig.tight_layout()
        fname = f"controllability_by_model_{eval_type}_{DATE}.png"
        fig.savefig(OUTDIR / fname, dpi=180, bbox_inches="tight")
        print(f"Saved {OUTDIR / fname}")
        plt.close(fig)


def make_compliance_by_instruction_plot(base_instr, models):
    """1c/1d: grouped bars by instruction type per model."""
    itype_cmap = matplotlib.colormaps.get_cmap("tab20")

    for eval_type, itypes in [("cotcontrol", COTCONTROL_TYPES), ("reasonif", REASONIF_TYPES)]:
        ms = [m for m in models if any(
            (m, eval_type, t) in base_instr for t in itypes
        )]
        n_models = len(ms)
        n_itypes = len(itypes)
        bar_w = 0.8 / n_itypes
        itype_colors = {t: itype_cmap(i / max(n_itypes - 1, 1)) for i, t in enumerate(itypes)}

        fig, ax = plt.subplots(figsize=(max(15, n_models * 1.3), 6.5))
        x = np.arange(n_models)

        for i, itype in enumerate(itypes):
            offset = (i - n_itypes / 2 + 0.5) * bar_w
            vals, errs_list = [], []
            for m in ms:
                row = base_instr.get((m, eval_type, itype))
                p = sf(row, "compliant_rate") or 0
                n = si(row, "n_clean")
                vals.append(p * 100)
                errs_list.append(ci80(p, n) * 100)
            ax.bar(
                x + offset, vals, bar_w, yerr=errs_list, capsize=2,
                label=itype, color=itype_colors[itype], edgecolor="white",
                linewidth=0.3, error_kw=dict(lw=0.8, capthick=0.6),
            )

        ax.set_ylabel("Compliance (%)", fontsize=16)
        ax.set_title(
            f"{EVAL_DISPLAY[eval_type]} Controllability by Instruction Type (80% CI)",
            fontsize=18, fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels([short_name(m) for m in ms], rotation=30, ha="right", fontsize=14)
        ax.tick_params(axis="y", labelsize=13)
        ax.legend(loc="upper right", fontsize=12, ncol=2, framealpha=0.9)
        ymax = ax.get_ylim()[1]
        ax.set_ylim(0, ymax * 1.1 if ymax > 0 else 10)
        add_group_seps(ax, ms)
        fig.tight_layout()
        fname = f"controllability_by_instruction_{eval_type}_{DATE}.png"
        fig.savefig(OUTDIR / fname, dpi=180, bbox_inches="tight")
        print(f"Saved {OUTDIR / fname}")
        plt.close(fig)


def make_meta_discussion_plot(base_runs, models):
    """Meta discussion rate by model and eval suite."""
    fig, ax = plt.subplots(figsize=(13, 5.5))
    et_colors = {"cotcontrol": "#4C72B0", "reasonif": "#DD8452"}
    width = 0.35
    x = np.arange(len(models))

    for i, et in enumerate(EVAL_TYPES):
        vals, errs_list = [], []
        for m in models:
            row = base_runs.get((m, et))
            p = sf(row, "meta_discussion_rate") or 0
            n = si(row, "n_clean")
            vals.append(p * 100)
            errs_list.append(ci80(p, n) * 100)
        offset = (i - 0.5) * width
        ax.bar(
            x + offset, vals, width, yerr=errs_list, capsize=3,
            label=EVAL_DISPLAY[et], color=et_colors[et], edgecolor="white",
            error_kw=dict(lw=1, capthick=0.8),
        )

    ax.set_ylabel("Meta Discussion Rate (%)", fontsize=16)
    ax.set_title(
        "Meta Discussion Rate by Model and Eval Suite (80% CI)",
        fontsize=18, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([short_name(m) for m in models], rotation=30, ha="right", fontsize=14)
    ax.tick_params(axis="y", labelsize=13)
    ax.legend(fontsize=14)
    ymax = max(
        max((sf(base_runs.get((m, et)), "meta_discussion_rate") or 0) for m in models)
        for et in EVAL_TYPES
    ) * 100
    ax.set_ylim(0, min(ymax * 1.35, 100) if ymax > 0 else 10)
    add_group_seps(ax, models)
    fig.tight_layout()
    fname = f"meta_discussion_by_model_{DATE}.png"
    fig.savefig(OUTDIR / fname, dpi=180, bbox_inches="tight")
    print(f"Saved {OUTDIR / fname}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Base vs FT excluding uppercase_thinking from CoTControl
# ---------------------------------------------------------------------------

EXCLUDE_COTCONTROL = {"uppercase_thinking"}


def load_data_excl_uppercase():
    """Re-aggregate from instruction-level CSV, excluding uppercase_thinking for cotcontrol."""
    model_names = {m["model"] for m in MODELS}
    all_instr = load_csv(SUMMARY_BY_INSTRUCTION)

    # Group instruction rows by (base_model, eval_type, file_name) to identify runs,
    # then pick the best run per (model, eval_type) for base and FT.

    # Collect per-run instruction rows
    runs = {}  # (base_model, eval_type, file_name) -> [rows]
    for r in all_instr:
        if r["base_model"] not in model_names:
            continue
        k = (r["base_model"], r["eval_type"], r["file_name"])
        runs.setdefault(k, []).append(r)

    def aggregate_run(instr_rows, exclude_types):
        """Re-aggregate compliance from instruction rows, excluding certain types."""
        total_compliant = 0
        total_clean = 0
        for r in instr_rows:
            if r["instruction_type"] in exclude_types:
                continue
            if r["instruction_type"] == "baseline":
                continue
            n = si(r, "n_clean")
            p = sf(r, "compliant_rate")
            if n > 0 and p is not None:
                total_compliant += p * n
                total_clean += n
        if total_clean == 0:
            return None, 0
        return total_compliant / total_clean, total_clean

    data = {}
    for m_cfg in MODELS:
        model = m_cfg["model"]
        effort = m_cfg["base_effort"]

        for eval_type in EVAL_TYPES:
            key = (model, eval_type)
            data[key] = {"base": None, "ft": None}

            exclude = EXCLUDE_COTCONTROL if eval_type == "cotcontrol" else set()

            # Find best base run
            best_base = None
            for (bm, et, fn), instr_rows in runs.items():
                if bm != model or et != eval_type:
                    continue
                sample_row = instr_rows[0]
                if not is_base_row(sample_row):
                    continue
                if sample_row.get("reasoning_effort", "") != effort:
                    continue
                if sample_row.get("max_tokens", "") != "28000":
                    continue
                ts = sample_row.get("timestamp", "")
                if best_base is None or ts > best_base[0]:
                    best_base = (ts, instr_rows)
            if best_base:
                p, n = aggregate_run(best_base[1], exclude)
                if p is not None:
                    data[key]["base"] = {"compliant_rate": p, "n_clean": n}

            # Find best FT run (lr=1e-4, step 60)
            best_ft = None
            for (bm, et, fn), instr_rows in runs.items():
                if bm != model or et != eval_type:
                    continue
                sample_row = instr_rows[0]
                if not is_ft_row(sample_row):
                    continue
                lr, ckpt = extract_lr_ckpt(sample_row.get("file_name", ""))
                if lr != FT_LR or ckpt != FT_CKPT:
                    continue
                ts = sample_row.get("timestamp", "")
                if best_ft is None or ts > best_ft[0]:
                    best_ft = (ts, instr_rows)
            if best_ft:
                p, n = aggregate_run(best_ft[1], exclude)
                if p is not None:
                    data[key]["ft"] = {"compliant_rate": p, "n_clean": n}

    return data


def make_plot_excl_uppercase(data):
    """Same as make_plot but with data that excludes uppercase_thinking."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    bar_width = 0.32
    x = np.arange(len(MODELS))
    colors = {"base": "#78909C", "ft": "#1E88E5"}

    for ax_idx, eval_type in enumerate(EVAL_TYPES):
        ax = axes[ax_idx]

        base_vals, base_errs = [], []
        ft_vals, ft_errs = [], []

        for m_cfg in MODELS:
            model = m_cfg["model"]
            key = (model, eval_type)

            for vals, errs, condition in [
                (base_vals, base_errs, "base"),
                (ft_vals, ft_errs, "ft"),
            ]:
                row = data[key][condition]
                if row:
                    p = row["compliant_rate"]
                    n = row["n_clean"]
                    vals.append(p * 100)
                    errs.append(ci80(p, n) * 100)
                else:
                    vals.append(0)
                    errs.append(0)

        bars_base = ax.bar(
            x - bar_width / 2, base_vals, bar_width,
            yerr=base_errs, capsize=3,
            color=colors["base"], alpha=0.85, label="Base",
            error_kw={"linewidth": 1},
        )
        bars_ft = ax.bar(
            x + bar_width / 2, ft_vals, bar_width,
            yerr=ft_errs, capsize=3,
            color=colors["ft"], alpha=0.85, label="FT (step 60)",
            error_kw={"linewidth": 1},
        )

        for bars, errs in [(bars_base, base_errs), (bars_ft, ft_errs)]:
            for bar, err in zip(bars, errs):
                height = bar.get_height()
                if height > 0:
                    ax.annotate(
                        f"{height:.1f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height + err),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, fontweight="bold",
                    )

        subtitle = EVAL_DISPLAY[eval_type]
        if eval_type == "cotcontrol":
            subtitle += " (excl. uppercase)"
        ax.set_title(subtitle, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([m["label"] for m in MODELS], rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Compliance (%)" if ax_idx == 0 else "")
        ax.legend(fontsize=9, loc="upper left")
        ax.set_ylim(0, None)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "CoT Controllability: Base vs Fine-Tuned (80% CI, excl. uppercase)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()

    out_path = OUTDIR / "base_vs_ft_lr1e-4_step60_excl_uppercase.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Reference comparison plots (our results vs paper numbers)
# ---------------------------------------------------------------------------

# Paper reference: CoTControl aggregate compliance
REF_COTCONTROL = {
    "openai/gpt-oss-20b": 1.2,
    "openai/gpt-oss-120b": 4.6,
    "qwen/qwen3-32b": 2.8,
}

# Paper reference: ReasonIF aggregate compliance
REF_REASONIF = {
    "openai/gpt-oss-20b": 11.0,
    "openai/gpt-oss-120b": 9.0,
    "qwen/qwen3-235b-a22b": 25.0,
}

# Paper reference: ReasonIF by instruction type (as fractions)
# Order: gpt-oss-20b, gpt-oss-120b, qwen3-235b
REF_REASONIF_INSTR = {
    "reasoning_language": {"openai/gpt-oss-20b": 0.27, "openai/gpt-oss-120b": 0.27, "qwen/qwen3-235b-a22b": 0.42},
    "number_words":       {"openai/gpt-oss-20b": 0.34, "openai/gpt-oss-120b": 0.25, "qwen/qwen3-235b-a22b": 0.36},
    "end_checker":        {"openai/gpt-oss-20b": 0.00, "openai/gpt-oss-120b": 0.00, "qwen/qwen3-235b-a22b": 0.59},
    "json_format":        {"openai/gpt-oss-20b": 0.00, "openai/gpt-oss-120b": 0.00, "qwen/qwen3-235b-a22b": 0.00},
    "english_capital":    {"openai/gpt-oss-20b": 0.00, "openai/gpt-oss-120b": 0.00, "qwen/qwen3-235b-a22b": 0.00},
    "no_comma":           {"openai/gpt-oss-20b": 0.04, "openai/gpt-oss-120b": 0.00, "qwen/qwen3-235b-a22b": 0.09},
}

REF_MODELS_COT = [
    ("openai/gpt-oss-20b", "GPT-OSS-20B"),
    ("openai/gpt-oss-120b", "GPT-OSS-120B"),
    ("qwen/qwen3-32b", "Qwen3-32B"),
]

REF_MODELS_RIF = [
    ("openai/gpt-oss-20b", "GPT-OSS-20B"),
    ("openai/gpt-oss-120b", "GPT-OSS-120B"),
    ("qwen/qwen3-235b-a22b", "Qwen3-235B"),
]


def make_ref_comparison_aggregate(base_runs):
    """Two subplots: CoTControl and ReasonIF, grouped bars (Paper vs Ours)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    bar_width = 0.32
    colors = {"paper": "#AAAAAA", "ours": "#1E88E5"}

    for ax_idx, (eval_type, ref_data, ref_models) in enumerate([
        ("cotcontrol", REF_COTCONTROL, REF_MODELS_COT),
        ("reasonif", REF_REASONIF, REF_MODELS_RIF),
    ]):
        ax = axes[ax_idx]
        x = np.arange(len(ref_models))

        paper_vals = []
        our_vals = []
        our_errs = []

        for model, label in ref_models:
            paper_vals.append(ref_data[model])

            row = base_runs.get((model, eval_type))
            p = sf(row, "compliant_rate")
            n = si(row, "n_clean")
            our_vals.append((p or 0) * 100)
            our_errs.append(ci80(p, n) * 100)

        bars_paper = ax.bar(
            x - bar_width / 2, paper_vals, bar_width,
            color=colors["paper"], alpha=0.85, label="Paper",
            edgecolor="white",
        )
        bars_ours = ax.bar(
            x + bar_width / 2, our_vals, bar_width,
            yerr=our_errs, capsize=3,
            color=colors["ours"], alpha=0.85, label="Ours",
            error_kw={"linewidth": 1}, edgecolor="white",
        )

        # Value labels
        for bars, vals in [(bars_paper, paper_vals), (bars_ours, our_vals)]:
            for bar, v in zip(bars, vals):
                ax.annotate(
                    f"{v:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=12, fontweight="bold",
                )

        ax.set_title(EVAL_DISPLAY[eval_type], fontsize=18, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in ref_models], fontsize=14)
        ax.set_ylabel("Compliance (%)", fontsize=16)
        ax.tick_params(axis="y", labelsize=13)
        ax.legend(fontsize=14)
        ax.set_ylim(0, None)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Controllability: Paper vs Ours (80% CI)",
        fontsize=18, fontweight="bold",
    )
    fig.tight_layout()
    out_path = OUTDIR / f"ref_comparison_aggregate_{DATE}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def make_ref_comparison_reasonif_instr(base_instr):
    """ReasonIF by instruction type: grouped bars (Paper vs Ours) per model."""
    ref_models = REF_MODELS_RIF
    itypes = ["reasoning_language", "number_words", "end_checker",
              "json_format", "english_capital", "no_comma"]
    itype_labels = ["multilingual", "word limit", "disclaimer",
                    "json", "uppercase", "no comma"]

    fig, axes = plt.subplots(1, len(ref_models), figsize=(6 * len(ref_models), 6))
    bar_width = 0.35
    colors = {"paper": "#AAAAAA", "ours": "#1E88E5"}

    for ax_idx, (model, label) in enumerate(ref_models):
        ax = axes[ax_idx]
        x = np.arange(len(itypes))

        paper_vals = []
        our_vals = []
        our_errs = []

        for itype in itypes:
            paper_vals.append(REF_REASONIF_INSTR[itype].get(model, 0) * 100)

            row = base_instr.get((model, "reasonif", itype))
            p = sf(row, "compliant_rate")
            n = si(row, "n_clean")
            our_vals.append((p or 0) * 100)
            our_errs.append(ci80(p, n) * 100)

        ax.bar(
            x - bar_width / 2, paper_vals, bar_width,
            color=colors["paper"], alpha=0.85, label="Paper",
            edgecolor="white",
        )
        ax.bar(
            x + bar_width / 2, our_vals, bar_width,
            yerr=our_errs, capsize=3,
            color=colors["ours"], alpha=0.85, label="Ours",
            error_kw={"linewidth": 1}, edgecolor="white",
        )

        ax.set_title(label, fontsize=18, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(itype_labels, rotation=35, ha="right", fontsize=13)
        ax.tick_params(axis="y", labelsize=13)
        if ax_idx == 0:
            ax.set_ylabel("Compliance (%)", fontsize=16)
        ax.legend(fontsize=13)
        ax.set_ylim(0, None)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "ReasonIF by Instruction Type: Paper vs Ours (80% CI)",
        fontsize=18, fontweight="bold",
    )
    fig.tight_layout()
    out_path = OUTDIR / f"ref_comparison_reasonif_instr_{DATE}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data = load_data()

    # Print summary
    print(f"{'Model':<25s} {'Eval':<12s} {'Base comp%':>10s} {'FT comp%':>10s}")
    print("-" * 60)
    for m_cfg in MODELS:
        for eval_type in EVAL_TYPES:
            key = (m_cfg["model"], eval_type)
            base = data[key]["base"]
            ft = data[key]["ft"]
            base_str = f"{sf(base, 'compliant_rate') * 100:.1f}" if base and sf(base, "compliant_rate") is not None else "N/A"
            ft_str = f"{sf(ft, 'compliant_rate') * 100:.1f}" if ft and sf(ft, "compliant_rate") is not None else "N/A"
            print(f"{m_cfg['label']:<25s} {eval_type:<12s} {base_str:>10s} {ft_str:>10s}")

    make_plot(data)
    make_plot_meta_discussion(data)

    # Training curve plots
    tc_data = load_training_curve_data()
    make_training_curve_plot(tc_data)
    make_training_curve_meta_discussion_plot(tc_data)

    # Effort sweep plot
    base_effort, ft_effort = load_effort_sweep_data()
    make_effort_sweep_plot(base_effort, ft_effort)

    # Param scaling plot
    ps_base, ps_ft = load_param_scaling_data()
    make_param_scaling_plot(ps_base, ps_ft)

    # All-models plots
    base_runs, base_instr, all_models = load_all_models_data()
    make_compliance_by_model_plot(base_runs, all_models)
    make_compliance_by_instruction_plot(base_instr, all_models)
    make_meta_discussion_plot(base_runs, all_models)

    # Base vs FT excluding uppercase_thinking
    excl_data = load_data_excl_uppercase()
    make_plot_excl_uppercase(excl_data)

    # Reference comparison plots
    make_ref_comparison_aggregate(base_runs)
    make_ref_comparison_reasonif_instr(base_instr)


if __name__ == "__main__":
    main()
