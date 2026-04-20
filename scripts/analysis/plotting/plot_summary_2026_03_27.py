#!/usr/bin/env python3
"""FT plots with stripped SFT data (2026-03-27).

Re-does all FT plots from 2026-03-24 using:
  - Stripped SFT fine-tunes (run_id 20260328_000734)
  - ReasonIF with analysis_channel for gpt-oss (base + FT)
  - Base gpt-oss medium effort, qwen none
  - New base effort sweep data (low/medium/high with analysis_channel)

Plots:
  1. base_vs_ft_lr1e-4_step60_compliance
  2. base_vs_ft_lr1e-4_step60_excl_uppercase
  3. base_vs_ft_lr1e-4_step60_meta_discussion
  4. ft_training_curve_lr1e-4_compliance
  5. ft_training_curve_lr1e-4_meta_discussion
  6. effort_sweep_combined_lr1e-4
  7. param_scaling_combined

Usage:
    python scripts/analysis/plot_summary_2026_03_27.py
"""

import csv
import math
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.size"] = 11

SUMMARY_BY_RUN = Path("results/summaries/tables/summary_by_run.csv")
SUMMARY_BY_INSTRUCTION = Path("results/summaries/tables/summary_by_instruction.csv")
DATE = "2026_03_27"
OUTDIR = Path(f"results/summaries/plots/{DATE}")
OUTDIR.mkdir(parents=True, exist_ok=True)

Z80 = 1.282

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

MODELS = [
    {"model": "openai/gpt-oss-20b", "label": "GPT-OSS-20B", "base_effort": "medium"},
    {"model": "openai/gpt-oss-120b", "label": "GPT-OSS-120B", "base_effort": "medium"},
    {"model": "qwen/qwen3-8b", "label": "Qwen3-8B", "base_effort": "none"},
    {"model": "qwen/qwen3-32b", "label": "Qwen3-32B", "base_effort": "none"},
]

MODEL_COLORS = {
    "openai/gpt-oss-20b": "#FF2A00",
    "openai/gpt-oss-120b": "#9933FF",
    "qwen/qwen3-8b": "#4895EF",
    "qwen/qwen3-32b": "#1D3557",
}

FT_LR = "1e-4"
FT_CKPT = "000060"
FT_CKPTS = ["000060", "000120", "000180", "final"]
FT_CKPT_DISPLAY = {"000060": "60", "000120": "120", "000180": "180", "final": "final"}
FT_RUN_ID = "20260328_000734"  # The stripped SFT run

EVAL_TYPES = ["cotcontrol", "reasonif"]
EVAL_DISPLAY = {"cotcontrol": "CoTControl", "reasonif": "ReasonIF"}

EXCLUDE_COTCONTROL = {"uppercase_thinking"}

EFFORT_ORDER = ["low", "medium", "high"]
EFFORT_MODELS = [
    {"model": "openai/gpt-oss-20b", "label": "GPT-OSS-20B", "color": "#FF2A00"},
    {"model": "openai/gpt-oss-120b", "label": "GPT-OSS-120B", "color": "#9933FF"},
]

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


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def is_stripped_ft(row):
    """Check if row is from the stripped FT run."""
    fn = row.get("file_name", "")
    return "-rif-stripped-" in fn and FT_RUN_ID in fn


def is_ft_row(row):
    """Check if row is any fine-tuned model."""
    return "-rif-" in row.get("file_name", "")


def is_base_row(row):
    """Check if row is a base model (no checkpoint)."""
    return not row.get("checkpoint", "") and not is_ft_row(row)


def extract_lr_ckpt(filename):
    """Extract (lr, checkpoint) from a stripped FT rollout filename."""
    m = re.search(r"-rif-stripped-(?:lr([\d.]+e-\d+)-)?(\d{3,}|final)", filename)
    if m:
        lr = m.group(1) or "1e-4"
        return lr, m.group(2)
    return None, None


def is_ac_row(row):
    """Check if row has analysis_channel=True."""
    return row.get("analysis_channel", "").strip().lower() == "true"


def best_base_row(rows, model, eval_type, effort):
    """Find best base row for a model/eval, preferring AC for gpt-oss reasonif."""
    candidates = [
        r for r in rows
        if r["base_model"] == model
        and r["eval_type"] == eval_type
        and is_base_row(r)
        and r.get("reasoning_effort", "") == effort
        and r.get("max_tokens", "") == "28000"
    ]
    # For gpt-oss reasonif, require analysis_channel=True
    if "gpt-oss" in model and eval_type == "reasonif":
        ac_candidates = [r for r in candidates if is_ac_row(r)]
        if ac_candidates:
            candidates = ac_candidates
    if not candidates:
        return None
    candidates.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return candidates[0]


def best_ft_row(rows, model, eval_type, ckpt):
    """Find best stripped FT row for a model/eval/checkpoint."""
    candidates = []
    for r in rows:
        if r["base_model"] != model or r["eval_type"] != eval_type:
            continue
        if not is_stripped_ft(r):
            continue
        lr, c = extract_lr_ckpt(r.get("file_name", ""))
        if lr == FT_LR and c == ckpt:
            candidates.append(r)
    if not candidates:
        return None
    # For gpt-oss reasonif, prefer analysis_channel=True
    if "gpt-oss" in model and eval_type == "reasonif":
        ac_candidates = [r for r in candidates if is_ac_row(r)]
        if ac_candidates:
            candidates = ac_candidates
    candidates.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    if len(candidates) > 1:
        print(f"  WARNING: multiple FT rows for {model} {eval_type} {ckpt}: "
              f"{[c['file_name'] for c in candidates]}")
    return candidates[0]


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_data():
    """Load base + FT step 60 data for all 4 models."""
    with open(SUMMARY_BY_RUN) as f:
        rows = list(csv.DictReader(f))

    data = {}
    for m_cfg in MODELS:
        model = m_cfg["model"]
        effort = m_cfg["base_effort"]
        for eval_type in EVAL_TYPES:
            key = (model, eval_type)
            data[key] = {
                "base": best_base_row(rows, model, eval_type, effort),
                "ft": best_ft_row(rows, model, eval_type, FT_CKPT),
            }
    return data


def load_training_curve_data():
    """Load base + all checkpoints for 4 models."""
    with open(SUMMARY_BY_RUN) as f:
        rows = list(csv.DictReader(f))

    data = {}
    for m_cfg in MODELS:
        model = m_cfg["model"]
        effort = m_cfg["base_effort"]
        for eval_type in EVAL_TYPES:
            key = (model, eval_type)
            data[key] = {
                "base": best_base_row(rows, model, eval_type, effort),
                "ckpts": {},
            }
            for ckpt in FT_CKPTS:
                row = best_ft_row(rows, model, eval_type, ckpt)
                if row:
                    data[key]["ckpts"][ckpt] = row
    return data


# ---------------------------------------------------------------------------
# 1. Base vs FT step 60 compliance
# ---------------------------------------------------------------------------

def make_plot_compliance(data):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    bar_width = 0.32
    x = np.arange(len(MODELS))
    colors = {"base": "#78909C", "ft": "#1E88E5"}

    for ax_idx, eval_type in enumerate(EVAL_TYPES):
        ax = axes[ax_idx]
        base_vals, base_errs, ft_vals, ft_errs = [], [], [], []

        for m_cfg in MODELS:
            key = (m_cfg["model"], eval_type)
            for vals, errs, cond in [(base_vals, base_errs, "base"), (ft_vals, ft_errs, "ft")]:
                row = data[key][cond]
                p = sf(row, "compliant_rate")
                n = si(row, "n_clean")
                vals.append((p or 0) * 100)
                errs.append(ci80(p, n) * 100)

        bars_base = ax.bar(x - bar_width / 2, base_vals, bar_width, yerr=base_errs, capsize=3,
               color=colors["base"], alpha=0.85, label="Base", error_kw={"linewidth": 1})
        bars_ft = ax.bar(x + bar_width / 2, ft_vals, bar_width, yerr=ft_errs, capsize=3,
               color=colors["ft"], alpha=0.85, label="FT (step 60)", error_kw={"linewidth": 1})

        for bars, errs_l in [(bars_base.patches, base_errs), (bars_ft.patches, ft_errs)]:
            for bar, err in zip(bars, errs_l):
                h = bar.get_height()
                if h > 0:
                    ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h + err),
                                xytext=(0, 3), textcoords="offset points",
                                ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_title(EVAL_DISPLAY[eval_type], fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([m["label"] for m in MODELS], rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Compliance (%)" if ax_idx == 0 else "")
        ax.legend(fontsize=9, loc="upper left")
        ax.set_ylim(0, None)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("CoT Controllability: Base vs Fine-Tuned (80% CI)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUTDIR / "base_vs_ft_lr1e-4_step60_compliance.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# 2. Base vs FT step 60 excluding uppercase_thinking
# ---------------------------------------------------------------------------

def load_data_excl_uppercase():
    """Re-aggregate from instruction-level CSV, excluding uppercase_thinking for cotcontrol."""
    model_names = {m["model"] for m in MODELS}
    all_instr = load_csv(SUMMARY_BY_INSTRUCTION)

    runs = {}
    for r in all_instr:
        if r["base_model"] not in model_names:
            continue
        k = (r["base_model"], r["eval_type"], r["file_name"])
        runs.setdefault(k, []).append(r)

    def aggregate_run(instr_rows, exclude_types):
        total_compliant = 0
        total_clean = 0
        for r in instr_rows:
            if r["instruction_type"] in exclude_types or r["instruction_type"] == "baseline":
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

            # Base
            best = None
            for (bm, et, fn), instr_rows in runs.items():
                if bm != model or et != eval_type:
                    continue
                sr = instr_rows[0]
                if not is_base_row(sr):
                    continue
                if sr.get("reasoning_effort", "") != effort:
                    continue
                if sr.get("max_tokens", "") != "28000":
                    continue
                if "gpt-oss" in model and eval_type == "reasonif" and not is_ac_row(sr):
                    continue
                ts = sr.get("timestamp", "")
                if best is None or ts > best[0]:
                    best = (ts, instr_rows)
            if best:
                p, n = aggregate_run(best[1], exclude)
                if p is not None:
                    data[key]["base"] = {"compliant_rate": p, "n_clean": n}

            # FT step 60
            best = None
            for (bm, et, fn), instr_rows in runs.items():
                if bm != model or et != eval_type:
                    continue
                sr = instr_rows[0]
                if not is_stripped_ft(sr):
                    continue
                lr, ckpt = extract_lr_ckpt(sr.get("file_name", ""))
                if lr != FT_LR or ckpt != FT_CKPT:
                    continue
                ts = sr.get("timestamp", "")
                if best is None or ts > best[0]:
                    best = (ts, instr_rows)
            if best:
                p, n = aggregate_run(best[1], exclude)
                if p is not None:
                    data[key]["ft"] = {"compliant_rate": p, "n_clean": n}

    return data


def make_plot_excl_uppercase(data):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    bar_width = 0.32
    x = np.arange(len(MODELS))
    colors = {"base": "#78909C", "ft": "#1E88E5"}

    for ax_idx, eval_type in enumerate(EVAL_TYPES):
        ax = axes[ax_idx]
        base_vals, base_errs, ft_vals, ft_errs = [], [], [], []

        for m_cfg in MODELS:
            key = (m_cfg["model"], eval_type)
            for vals, errs, cond in [(base_vals, base_errs, "base"), (ft_vals, ft_errs, "ft")]:
                row = data[key][cond]
                if row:
                    p = row["compliant_rate"]
                    n = row["n_clean"]
                    vals.append(p * 100)
                    errs.append(ci80(p, n) * 100)
                else:
                    vals.append(0)
                    errs.append(0)

        bars_base = ax.bar(x - bar_width / 2, base_vals, bar_width, yerr=base_errs, capsize=3,
               color=colors["base"], alpha=0.85, label="Base", error_kw={"linewidth": 1})
        bars_ft = ax.bar(x + bar_width / 2, ft_vals, bar_width, yerr=ft_errs, capsize=3,
               color=colors["ft"], alpha=0.85, label="FT (step 60)", error_kw={"linewidth": 1})

        for bars, errs_l in [(bars_base.patches, base_errs), (bars_ft.patches, ft_errs)]:
            for bar, err in zip(bars, errs_l):
                h = bar.get_height()
                if h > 0:
                    ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h + err),
                                xytext=(0, 3), textcoords="offset points",
                                ha="center", va="bottom", fontsize=8, fontweight="bold")

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

    fig.suptitle("CoT Controllability: Base vs FT (80% CI, excl. uppercase)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUTDIR / "base_vs_ft_lr1e-4_step60_excl_uppercase.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# 3. Base vs FT step 60 meta discussion
# ---------------------------------------------------------------------------

def make_plot_meta_discussion(data):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    bar_width = 0.32
    x = np.arange(len(MODELS))
    colors = {"base": "#78909C", "ft": "#1E88E5"}

    for ax_idx, eval_type in enumerate(EVAL_TYPES):
        ax = axes[ax_idx]
        base_vals, base_errs, ft_vals, ft_errs = [], [], [], []

        for m_cfg in MODELS:
            key = (m_cfg["model"], eval_type)
            for vals, errs, cond in [(base_vals, base_errs, "base"), (ft_vals, ft_errs, "ft")]:
                row = data[key][cond]
                p = sf(row, "meta_discussion_rate")
                n = si(row, "n_clean")
                vals.append((p or 0) * 100)
                errs.append(ci80(p, n) * 100)

        bars_base = ax.bar(x - bar_width / 2, base_vals, bar_width, yerr=base_errs, capsize=3,
               color=colors["base"], alpha=0.85, label="Base", error_kw={"linewidth": 1})
        bars_ft = ax.bar(x + bar_width / 2, ft_vals, bar_width, yerr=ft_errs, capsize=3,
               color=colors["ft"], alpha=0.85, label="FT (step 60)", error_kw={"linewidth": 1})

        for bars, errs_l in [(bars_base.patches, base_errs), (bars_ft.patches, ft_errs)]:
            for bar, err in zip(bars, errs_l):
                h = bar.get_height()
                if h > 0:
                    ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h + err),
                                xytext=(0, 3), textcoords="offset points",
                                ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_title(EVAL_DISPLAY[eval_type], fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([m["label"] for m in MODELS], rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Meta Discussion Rate (%)" if ax_idx == 0 else "")
        ax.legend(fontsize=9, loc="upper right")
        ax.set_ylim(0, None)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Meta Discussion Rate: Base vs Fine-Tuned (80% CI)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUTDIR / "base_vs_ft_lr1e-4_step60_meta_discussion.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# 4/5. Training curve plots (compliance + meta discussion)
# ---------------------------------------------------------------------------

def _make_training_curve(data, metric, ylabel, title, filename):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax_idx, eval_type in enumerate(EVAL_TYPES):
        ax = axes[ax_idx]
        for m_cfg in MODELS:
            model = m_cfg["model"]
            key = (model, eval_type)
            color = MODEL_COLORS[model]
            xs, vals, errs = [], [], []

            base_row = data[key]["base"]
            if base_row:
                p = sf(base_row, metric)
                n = si(base_row, "n_clean")
                xs.append(0)
                vals.append((p or 0) * 100)
                errs.append(ci80(p, n) * 100)

            for i, ckpt in enumerate(FT_CKPTS):
                row = data[key]["ckpts"].get(ckpt)
                if row:
                    p = sf(row, metric)
                    n = si(row, "n_clean")
                    xs.append(i + 1)
                    vals.append((p or 0) * 100)
                    errs.append(ci80(p, n) * 100)

            ax.errorbar(xs, vals, yerr=errs, fmt="o-", color=color, capsize=3,
                        label=m_cfg["label"], linewidth=1.5, markersize=5, elinewidth=1)

        tick_positions = [0] + list(range(1, len(FT_CKPTS) + 1))
        tick_labels = ["base"] + [FT_CKPT_DISPLAY[c] for c in FT_CKPTS]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=9)
        ax.set_xlabel("Batch", fontsize=10)
        ax.set_ylabel(ylabel if ax_idx == 0 else "")
        ax.set_title(EVAL_DISPLAY[eval_type], fontsize=13, fontweight="bold")
        ax.set_ylim(0, None)
        ax.grid(axis="y", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUTDIR / filename
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# 6. Effort sweep (base vs FT, gpt-oss only, reasonif only with AC)
# ---------------------------------------------------------------------------

def load_effort_sweep_data():
    """Load base and FT-step60 rows at each effort level for gpt-oss models."""
    with open(SUMMARY_BY_RUN) as f:
        all_rows = list(csv.DictReader(f))

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
        et = r["eval_type"]
        # For reasonif, require analysis_channel=True
        if et == "reasonif" and not is_ac_row(r):
            continue
        k = (r["base_model"], et, r["reasoning_effort"])
        if k not in base_effort or r["timestamp"] > base_effort[k]["timestamp"]:
            base_effort[k] = r

    # FT step60: match any gpt-oss row whose checkpoint contains 000060
    # and whose filename has -rif-stripped- (covers any run ID)
    ft_effort = {}
    for r in all_rows:
        if "gpt-oss" not in r["base_model"]:
            continue
        fn = r.get("file_name", "")
        if "-rif-stripped-" not in fn:
            continue
        cp = (r.get("checkpoint") or "").strip()
        if not cp or "000060" not in cp:
            continue
        if si(r, "n_clean") < 10:
            continue
        et = r["eval_type"]
        # For reasonif, require analysis_channel=True
        if et == "reasonif" and not is_ac_row(r):
            continue
        k = (r["base_model"], et, r["reasoning_effort"])
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

            # Base (solid)
            by, be, bx_valid = [], [], []
            for i, effort in enumerate(EFFORT_ORDER):
                row = base_effort.get((model, eval_type, effort))
                if row:
                    p = sf(row, "compliant_rate")
                    n = si(row, "n_clean")
                    bx_valid.append(i)
                    by.append((p or 0) * 100)
                    be.append(ci80(p, n) * 100)
            if bx_valid:
                ax.errorbar(bx_valid, by, yerr=be, fmt="o-", color=color, capsize=4,
                            label=f"{m_cfg['label']} Base", linewidth=1.5, markersize=6, elinewidth=1)

            # FT (single point at medium — no multi-effort FT data)
            fy, fe, fx_valid = [], [], []
            for i, effort in enumerate(EFFORT_ORDER):
                row = ft_effort.get((model, eval_type, effort))
                if row:
                    p = sf(row, "compliant_rate")
                    n = si(row, "n_clean")
                    fx_valid.append(i)
                    fy.append((p or 0) * 100)
                    fe.append(ci80(p, n) * 100)
            if fx_valid:
                line_fmt = "s:" if len(fx_valid) > 1 else "s"
                ax.errorbar(fx_valid, fy, yerr=fe, fmt=line_fmt, color=color, capsize=4,
                            label=f"{m_cfg['label']} FT (med.)", linewidth=1.5,
                            markersize=8, elinewidth=1, markeredgewidth=1.5)

        ax.set_xticks(x)
        ax.set_xticklabels(EFFORT_ORDER, fontsize=10)
        ax.set_xlabel("Reasoning Effort", fontsize=10)
        ax.set_ylabel("Compliance (%)" if ax_idx == 0 else "")
        ax.set_title(EVAL_DISPLAY[eval_type], fontsize=13, fontweight="bold")
        ax.set_ylim(0, None)
        ax.grid(axis="y", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Controllability vs Reasoning Effort: Base vs FT (80% CI)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUTDIR / "effort_sweep_combined_lr1e-4.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# 7. Param scaling
# ---------------------------------------------------------------------------

def load_param_scaling_data():
    with open(SUMMARY_BY_RUN) as f:
        all_rows = list(csv.DictReader(f))

    base = {}
    for r in all_rows:
        if (r.get("checkpoint") or "").strip():
            continue
        model = r["base_model"]
        et = r["eval_type"]
        effort = r.get("reasoning_effort", "")
        if "gpt-oss" in model:
            if effort != "medium" or r.get("max_tokens") != "28000":
                continue
            if et == "reasonif" and not is_ac_row(r):
                continue
        elif "qwen" in model:
            if effort not in ("none", ""):
                continue
        if si(r, "n_clean") < 10:
            continue
        k = (model, et)
        if k not in base or r["timestamp"] > base[k]["timestamp"]:
            base[k] = r

    ft = {}
    for r in all_rows:
        if not is_stripped_ft(r):
            continue
        lr, ckpt = extract_lr_ckpt(r.get("file_name", ""))
        if lr != FT_LR or ckpt != FT_CKPT:
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
            for cond, marker, ls, suffix in [("base", "o", "-", "base"), ("ft", "s", "--", "FT")]:
                src = base if cond == "base" else ft
                px, py, pe = [], [], []
                for model, params in cfg["models"]:
                    row = src.get((model, eval_type))
                    if row:
                        p = sf(row, "compliant_rate") or 0
                        n = si(row, "n_clean")
                        px.append(params)
                        py.append(p * 100)
                        pe.append(ci80(p, n) * 100)
                if px:
                    ax.errorbar(px, py, yerr=pe, marker=marker, linestyle=ls, color=color,
                                label=f"{family} {suffix}", capsize=4, lw=1.5, markersize=6, elinewidth=1)

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
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Controllability vs Model Size: Base vs FT (80% CI)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUTDIR / "param_scaling_combined.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# 8. Compliance by instruction type over training (one figure per model)
# ---------------------------------------------------------------------------

INSTR_COLORS_COTCONTROL = {
    "word_suppression": "#E53935",
    "multiple_word_suppression": "#D81B60",
    "uppercase_thinking": "#8E24AA",
    "lowercase_thinking": "#5E35B1",
    "alternating_case": "#3949AB",
    "repeat_sentences": "#1E88E5",
    "end_of_sentence": "#00ACC1",
    "meow_between_words": "#43A047",
    "ignore_question": "#F4511E",
}

INSTR_COLORS_REASONIF = {
    "reasoning_language": "#E53935",
    "number_words": "#8E24AA",
    "english_capital": "#1E88E5",
    "end_checker": "#00ACC1",
    "json_format": "#43A047",
    "no_comma": "#F4511E",
}

INSTR_DISPLAY = {
    "word_suppression": "word suppression",
    "multiple_word_suppression": "multi-word suppr.",
    "uppercase_thinking": "uppercase",
    "lowercase_thinking": "lowercase",
    "alternating_case": "alternating case",
    "repeat_sentences": "repeat sentences",
    "end_of_sentence": "end of sentence",
    "meow_between_words": "meow between words",
    "ignore_question": "ignore question",
    "reasoning_language": "language",
    "number_words": "number words",
    "english_capital": "english capital",
    "end_checker": "end checker",
    "json_format": "json format",
    "no_comma": "no comma",
}


def load_instruction_training_data():
    """Load instruction-level data for base + all checkpoints, keyed by
    (model, eval_type) -> {instr_type -> {"base": row, "ckpts": {ckpt: row}}}
    """
    all_instr = load_csv(SUMMARY_BY_INSTRUCTION)
    model_names = {m["model"] for m in MODELS}

    # Group rows by file_name
    by_file: dict[str, list[dict]] = {}
    for r in all_instr:
        if r["base_model"] not in model_names:
            continue
        by_file.setdefault(r["file_name"], []).append(r)

    data: dict[tuple, dict[str, dict]] = {}
    for m_cfg in MODELS:
        model = m_cfg["model"]
        effort = m_cfg["base_effort"]
        for eval_type in EVAL_TYPES:
            key = (model, eval_type)
            data[key] = {}

            # Find best base file
            best_base_ts = ""
            best_base_rows = None
            for fn, rows in by_file.items():
                r0 = rows[0]
                if r0["base_model"] != model or r0["eval_type"] != eval_type:
                    continue
                if not is_base_row(r0):
                    continue
                if r0.get("reasoning_effort", "") != effort:
                    continue
                if r0.get("max_tokens", "") != "28000":
                    continue
                if "gpt-oss" in model and eval_type == "reasonif" and not is_ac_row(r0):
                    continue
                ts = r0.get("timestamp", "")
                if ts > best_base_ts:
                    best_base_ts = ts
                    best_base_rows = rows

            if best_base_rows:
                for r in best_base_rows:
                    it = r["instruction_type"]
                    if it == "baseline":
                        continue
                    data[key].setdefault(it, {"base": None, "ckpts": {}})
                    data[key][it]["base"] = r

            # Find best FT file per checkpoint
            for ckpt in FT_CKPTS:
                best_ft_ts = ""
                best_ft_rows = None
                for fn, rows in by_file.items():
                    r0 = rows[0]
                    if r0["base_model"] != model or r0["eval_type"] != eval_type:
                        continue
                    if not is_stripped_ft(r0):
                        continue
                    lr, c = extract_lr_ckpt(r0.get("file_name", ""))
                    if lr != FT_LR or c != ckpt:
                        continue
                    if "gpt-oss" in model and eval_type == "reasonif" and not is_ac_row(r0):
                        continue
                    ts = r0.get("timestamp", "")
                    if ts > best_ft_ts:
                        best_ft_ts = ts
                        best_ft_rows = rows

                if best_ft_rows:
                    for r in best_ft_rows:
                        it = r["instruction_type"]
                        if it == "baseline":
                            continue
                        data[key].setdefault(it, {"base": None, "ckpts": {}})
                        data[key][it]["ckpts"][ckpt] = r

    return data


def make_instruction_training_curves(instr_data):
    """One figure per model: 2 subplots (cotcontrol, reasonif), one line per instruction type."""
    instr_colors = {
        "cotcontrol": INSTR_COLORS_COTCONTROL,
        "reasonif": INSTR_COLORS_REASONIF,
    }

    for m_cfg in MODELS:
        model = m_cfg["model"]
        model_label = m_cfg["label"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for ax_idx, eval_type in enumerate(EVAL_TYPES):
            ax = axes[ax_idx]
            key = (model, eval_type)
            colors = instr_colors[eval_type]

            instr_types = sorted(
                instr_data[key].keys(),
                key=lambda it: INSTR_DISPLAY.get(it, it),
            )

            for it in instr_types:
                it_data = instr_data[key][it]
                color = colors.get(it, "#888888")
                label = INSTR_DISPLAY.get(it, it)

                xs, vals, errs = [], [], []

                base_row = it_data.get("base")
                if base_row:
                    p = sf(base_row, "compliant_rate")
                    n = si(base_row, "n_clean")
                    xs.append(0)
                    vals.append((p or 0) * 100)
                    errs.append(ci80(p, n) * 100)

                for i, ckpt in enumerate(FT_CKPTS):
                    row = it_data["ckpts"].get(ckpt)
                    if row:
                        p = sf(row, "compliant_rate")
                        n = si(row, "n_clean")
                        xs.append(i + 1)
                        vals.append((p or 0) * 100)
                        errs.append(ci80(p, n) * 100)

                if xs:
                    ax.errorbar(xs, vals, yerr=errs, fmt="o-", color=color, capsize=3,
                                label=label, linewidth=1.5, markersize=4, elinewidth=1,
                                alpha=0.85)

            tick_positions = [0] + list(range(1, len(FT_CKPTS) + 1))
            tick_labels = ["base"] + [FT_CKPT_DISPLAY[c] for c in FT_CKPTS]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, fontsize=9)
            ax.set_xlabel("Batch", fontsize=10)
            ax.set_ylabel("Compliance (%)" if ax_idx == 0 else "")
            ax.set_title(EVAL_DISPLAY[eval_type], fontsize=13, fontweight="bold")
            ax.set_ylim(0, 105)
            ax.grid(axis="y", alpha=0.3)
            ax.legend(fontsize=7, loc="upper left", ncol=1)

        fig.suptitle(f"{model_label}: Compliance by Instruction over Training (80% CI)",
                     fontsize=14, fontweight="bold")
        fig.tight_layout()
        model_short = model.split("/")[-1]
        out = OUTDIR / f"ft_training_curve_by_instruction_{model_short}.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# 9. Grouped bar: compliance by instruction, base vs FT step 60
# ---------------------------------------------------------------------------

def make_instruction_bars(instr_data, eval_type):
    """One figure per eval type. Groups = instruction types, 8 bars per group
    (base + FT for each of 4 models)."""
    # Collect instruction types from any model that has data
    all_instr = set()
    for m_cfg in MODELS:
        key = (m_cfg["model"], eval_type)
        all_instr |= {it for it in instr_data.get(key, {}) if it != "baseline"}
    instr_types = sorted(all_instr, key=lambda it: INSTR_DISPLAY.get(it, it))

    n_groups = len(instr_types)
    n_models = len(MODELS)
    n_bars = n_models * 2  # base + FT per model
    bar_width = 0.8 / n_bars
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(max(14, n_groups * 1.8), 6))

    for m_idx, m_cfg in enumerate(MODELS):
        model = m_cfg["model"]
        key = (model, eval_type)
        color = MODEL_COLORS[model]
        # Lighter shade for base, full for FT
        base_alpha = 0.45
        ft_alpha = 0.90

        base_vals, base_errs, ft_vals, ft_errs = [], [], [], []
        for it in instr_types:
            it_data = instr_data.get(key, {}).get(it, {"base": None, "ckpts": {}})

            base_row = it_data.get("base")
            p = sf(base_row, "compliant_rate") if base_row else None
            n = si(base_row, "n_clean") if base_row else 0
            base_vals.append((p or 0) * 100)
            base_errs.append(ci80(p, n) * 100)

            ft_row = it_data.get("ckpts", {}).get(FT_CKPT)
            p = sf(ft_row, "compliant_rate") if ft_row else None
            n = si(ft_row, "n_clean") if ft_row else 0
            ft_vals.append((p or 0) * 100)
            ft_errs.append(ci80(p, n) * 100)

        # Position: pair base+FT bars next to each other within the group
        offset_base = (m_idx * 2 - n_bars / 2 + 0.5) * bar_width
        offset_ft = (m_idx * 2 + 1 - n_bars / 2 + 0.5) * bar_width

        ax.bar(x + offset_base, base_vals, bar_width, yerr=base_errs, capsize=2,
               color=color, alpha=base_alpha, label=f"{m_cfg['label']} Base",
               error_kw={"linewidth": 0.8}, edgecolor="white", linewidth=0.3)
        ax.bar(x + offset_ft, ft_vals, bar_width, yerr=ft_errs, capsize=2,
               color=color, alpha=ft_alpha, label=f"{m_cfg['label']} FT",
               error_kw={"linewidth": 0.8}, edgecolor="white", linewidth=0.3)

    labels = [INSTR_DISPLAY.get(it, it) for it in instr_types]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=20)
    ax.set_ylabel("Compliance (%)", fontsize=21)
    ax.tick_params(axis="y", labelsize=18)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    ax.legend(fontsize=15, loc="upper right", ncol=4)

    fig.suptitle(f"{EVAL_DISPLAY[eval_type]}: Compliance by Instruction — Base vs FT Step 60 (80% CI)",
                 fontsize=24, fontweight="bold")
    fig.tight_layout()
    out = OUTDIR / f"instruction_bars_{eval_type}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# 10-13. All-models plots (compliance by model/instruction, meta discussion,
#        instruction correlation heatmap)
# ---------------------------------------------------------------------------

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

INSTRUCTION_CLASS = {
    "alternating_case": "style", "lowercase_thinking": "style",
    "uppercase_thinking": "style", "english_capital": "style",
    "json_format": "style", "reasoning_language": "style",
    "word_suppression": "suppression", "multiple_word_suppression": "suppression",
    "ignore_question": "suppression", "no_comma": "suppression",
    "number_words": "suppression",
    "meow_between_words": "addition", "end_of_sentence": "addition",
    "repeat_sentences": "addition", "end_checker": "addition",
}


def short_name(m: str) -> str:
    return m.split("/")[-1]


def _is_canonical_effort(row: dict) -> bool:
    effort = row.get("reasoning_effort", "")
    if "gpt-oss" in row["base_model"]:
        return effort == "medium"
    return effort in ("none", "")


def _add_group_seps(ax, models_list):
    for gs in GROUP_STARTS:
        if gs < len(models_list):
            ax.axvline(x=gs - 0.5, color="gray", linewidth=0.5, alpha=0.3)


def load_all_models_data():
    """Load base runs and instruction-level data for all models.

    Uses max_tokens=28000 and prefers analysis_channel=True for gpt-oss reasonif.
    """
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
            # Prefer analysis_channel=True for gpt-oss reasonif
            if "gpt-oss" in r.get("base_model", "") and r.get("eval_type") == "reasonif":
                curr_ac = is_ac_row(r)
                if k in best:
                    prev_ac = is_ac_row(best[k])
                    if curr_ac and not prev_ac:
                        best[k] = r
                        continue
                    if not curr_ac and prev_ac:
                        continue
            if k not in best or r["timestamp"] > best[k]["timestamp"]:
                best[k] = r
        return best

    def _base_filter(r):
        if (r.get("checkpoint") or "").strip():
            return False
        if not _is_canonical_effort(r):
            return False
        mt = r.get("max_tokens", "")
        if mt not in ("28000", "32768"):
            return False
        return True

    base_runs = best_by_key(
        all_runs,
        key_fn=lambda r: (r["base_model"], r["eval_type"]),
        filter_fn=_base_filter,
    )
    base_instr = best_by_key(
        all_instr,
        key_fn=lambda r: (r["base_model"], r["eval_type"], r["instruction_type"]),
        filter_fn=_base_filter,
    )

    model_key = {m: i for i, m in enumerate(ALL_MODELS)}
    models = sorted(
        {k[0] for k in base_runs},
        key=lambda m: model_key.get(m, len(ALL_MODELS)),
    )
    return base_runs, base_instr, models


def make_compliance_by_model_plot(base_runs, models):
    """Bar chart per eval_type showing compliance for all models."""
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
            ax.text(
                bar.get_x() + bar.get_width() / 2, v + e + 0.5,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=13,
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
        _add_group_seps(ax, ms)
        fig.tight_layout()
        fname = f"controllability_by_model_{eval_type}.png"
        fig.savefig(OUTDIR / fname, dpi=180, bbox_inches="tight")
        print(f"Saved: {OUTDIR / fname}")
        plt.close(fig)


def make_compliance_by_instruction_plot(base_instr, models):
    """Grouped bars by instruction type per model."""
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
        _add_group_seps(ax, ms)
        fig.tight_layout()
        fname = f"controllability_by_instruction_{eval_type}.png"
        fig.savefig(OUTDIR / fname, dpi=180, bbox_inches="tight")
        print(f"Saved: {OUTDIR / fname}")
        plt.close(fig)


def make_meta_discussion_by_model_plot(base_runs, models):
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
    _add_group_seps(ax, models)
    fig.tight_layout()
    fname = "meta_discussion_by_model.png"
    fig.savefig(OUTDIR / fname, dpi=180, bbox_inches="tight")
    print(f"Saved: {OUTDIR / fname}")
    plt.close(fig)


def make_instruction_correlation_plot(base_instr, models):
    """Instruction-type compliance correlation heatmap grouped by class."""
    # Build compliance vectors: instruction_type -> {model: rate}
    compliance_by_type: dict[str, dict[str, float]] = {}
    for (base_model, eval_type, itype), row in base_instr.items():
        if itype == "baseline":
            continue
        val = sf(row, "compliant_rate")
        if val is None:
            continue
        compliance_by_type.setdefault(itype, {})[base_model] = val

    # Filter: keep only instruction types with >= 2 models showing non-zero compliance
    nonzero_cc = [t for t in COTCONTROL_TYPES
                  if sum(v > 0 for v in compliance_by_type.get(t, {}).values()) >= 2]
    nonzero_rif = [t for t in REASONIF_TYPES
                   if sum(v > 0 for v in compliance_by_type.get(t, {}).values()) >= 2]
    valid_itypes = nonzero_cc + nonzero_rif
    n = len(valid_itypes)

    # Compute pairwise Pearson correlation (require >= 3 common models)
    corr = np.full((n, n), np.nan)
    for i, ti in enumerate(valid_itypes):
        for j, tj in enumerate(valid_itypes):
            vi = compliance_by_type.get(ti, {})
            vj = compliance_by_type.get(tj, {})
            common = sorted(set(vi) & set(vj))
            if len(common) < 3:
                continue
            a = np.array([vi[m] for m in common])
            b = np.array([vj[m] for m in common])
            if np.std(a) < 1e-9 or np.std(b) < 1e-9:
                corr[i, j] = 0.0
            else:
                corr[i, j] = np.corrcoef(a, b)[0, 1]

    type_to_idx = {t: i for i, t in enumerate(valid_itypes)}

    # Group by class
    class_types = {"style": [], "suppression": [], "addition": []}
    for t in valid_itypes:
        cls = INSTRUCTION_CLASS.get(t, "style")
        class_types[cls].append(t)

    class_ordered = class_types["style"] + class_types["suppression"] + class_types["addition"]
    n_style = len(class_types["style"])
    n_suppression = len(class_types["suppression"])
    n_total = len(class_ordered)

    corr_by_class = np.full((n_total, n_total), np.nan)
    for i, ti in enumerate(class_ordered):
        for j, tj in enumerate(class_ordered):
            if ti in type_to_idx and tj in type_to_idx:
                corr_by_class[i, j] = corr[type_to_idx[ti], type_to_idx[tj]]

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(corr_by_class, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")

    for boundary in [n_style - 0.5, n_style + n_suppression - 0.5]:
        ax.axhline(y=boundary, color="black", linewidth=2)
        ax.axvline(x=boundary, color="black", linewidth=2)

    ax.set_xticks(range(n_total))
    ax.set_xticklabels(class_ordered, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_total))
    ax.set_yticklabels(class_ordered, fontsize=8)

    ax.text(n_style / 2, -1.5, "style", ha="center", fontsize=10, fontweight="bold")
    ax.text(n_style + n_suppression / 2, -1.5, "suppression", ha="center", fontsize=10, fontweight="bold")
    n_addition = len(class_types["addition"])
    ax.text(n_style + n_suppression + n_addition / 2, -1.5, "addition",
            ha="center", fontsize=10, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Pearson r")
    ax.set_title("Instruction-Type Compliance Correlation (Grouped by Class)", pad=30)
    plt.tight_layout()
    fname = "instruction_type_correlation_by_class.png"
    fig.savefig(OUTDIR / fname, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTDIR / fname}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load and print summary
    data = load_data()
    print(f"\n{'Model':<25s} {'Eval':<12s} {'Base comp%':>10s} {'FT comp%':>10s}  {'Base file'}")
    print("-" * 100)
    for m_cfg in MODELS:
        for eval_type in EVAL_TYPES:
            key = (m_cfg["model"], eval_type)
            base = data[key]["base"]
            ft = data[key]["ft"]
            base_str = f"{sf(base, 'compliant_rate') * 100:.1f}" if base and sf(base, "compliant_rate") is not None else "N/A"
            ft_str = f"{sf(ft, 'compliant_rate') * 100:.1f}" if ft and sf(ft, "compliant_rate") is not None else "N/A"
            base_fn = base.get("file_name", "") if base else ""
            print(f"{m_cfg['label']:<25s} {eval_type:<12s} {base_str:>10s} {ft_str:>10s}  {base_fn}")

    # 1. Compliance
    make_plot_compliance(data)

    # 2. Excl uppercase
    excl_data = load_data_excl_uppercase()
    make_plot_excl_uppercase(excl_data)

    # 3. Meta discussion
    make_plot_meta_discussion(data)

    # 4-5. Training curves
    tc_data = load_training_curve_data()
    _make_training_curve(tc_data, "compliant_rate", "Compliance (%)",
                         "Controllability over Fine-Tuning (80% CI)",
                         "ft_training_curve_lr1e-4_compliance.png")
    _make_training_curve(tc_data, "meta_discussion_rate", "Meta Discussion Rate (%)",
                         "Meta Discussion Rate over Fine-Tuning (80% CI)",
                         "ft_training_curve_lr1e-4_meta_discussion.png")
    _make_training_curve(tc_data, "correct_rate", "Accuracy (%)",
                         "Accuracy over Fine-Tuning (80% CI)",
                         "ft_training_curve_lr1e-4_accuracy.png")

    # 6. Effort sweep
    base_effort, ft_effort = load_effort_sweep_data()
    make_effort_sweep_plot(base_effort, ft_effort)

    # 7. Param scaling
    ps_base, ps_ft = load_param_scaling_data()
    make_param_scaling_plot(ps_base, ps_ft)

    # 8. Compliance by instruction type over training
    instr_data = load_instruction_training_data()
    make_instruction_training_curves(instr_data)

    # 9. Grouped bars: compliance by instruction, base vs FT step 60
    make_instruction_bars(instr_data, "cotcontrol")
    make_instruction_bars(instr_data, "reasonif")

    # 10-13. All-models plots (compliance by model, by instruction, meta discussion, correlation)
    all_base_runs, all_base_instr, all_models = load_all_models_data()
    make_compliance_by_model_plot(all_base_runs, all_models)
    make_compliance_by_instruction_plot(all_base_instr, all_models)
    make_meta_discussion_by_model_plot(all_base_runs, all_models)
    make_instruction_correlation_plot(all_base_instr, all_models)

    print(f"\nAll plots saved to {OUTDIR}")


if __name__ == "__main__":
    main()
