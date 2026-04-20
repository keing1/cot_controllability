#!/usr/bin/env python3
"""Word count distribution plots: base vs FT (2026-03-31).

Shows the distribution of total completion word counts (reasoning + response)
for base vs FT models. One subplot per model, two distributions each.

Usage:
    python scripts/analysis/plot_summary_2026_03_31.py
"""

import csv
import json
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.size"] = 11

SUMMARY_BY_RUN = Path("results/summaries/tables/summary_by_run.csv")
ROLLOUTS_DIR = Path("results/rollouts")
DATE = "2026_03_31"
OUTDIR = Path(f"results/summaries/plots/{DATE}")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

MODELS = [
    {"model": "openai/gpt-oss-20b", "label": "GPT-OSS-20B", "base_effort": "medium"},
    {"model": "openai/gpt-oss-120b", "label": "GPT-OSS-120B", "base_effort": "medium"},
    {"model": "qwen/qwen3-8b", "label": "Qwen3-8B", "base_effort": "none"},
    {"model": "qwen/qwen3-32b", "label": "Qwen3-32B", "base_effort": "none"},
]

FT_LR = "1e-4"
FT_CKPT = "000060"
FT_RUN_ID = "20260328_000734"

EVAL_TYPES = ["cotcontrol", "reasonif"]
EVAL_DISPLAY = {"cotcontrol": "CoTControl", "reasonif": "ReasonIF"}

SFT_DIR = Path("results/sft")
SFT_PARQUETS = {
    "openai/gpt-oss-20b": "gpt-oss-20b-reasonif-sft-stripped-ac.parquet",
    "openai/gpt-oss-120b": "gpt-oss-120b-reasonif-sft-stripped-ac.parquet",
    "qwen/qwen3-8b": "qwen3-8b-reasonif-sft-stripped.parquet",
    "qwen/qwen3-32b": "qwen3-32b-reasonif-sft-stripped.parquet",
}


# ---------------------------------------------------------------------------
# Helpers (same as 0327 script)
# ---------------------------------------------------------------------------

def is_stripped_ft(row):
    fn = row.get("file_name", "")
    return "-rif-stripped-" in fn and FT_RUN_ID in fn


def is_ft_row(row):
    return "-rif-" in row.get("file_name", "")


def is_base_row(row):
    return not row.get("checkpoint", "") and not is_ft_row(row)


def is_ac_row(row):
    return row.get("analysis_channel", "").strip().lower() == "true"


def extract_lr_ckpt(filename):
    m = re.search(r"-rif-stripped-(?:lr([\d.]+e-\d+)-)?(\d{3,}|final)", filename)
    if m:
        lr = m.group(1) or "1e-4"
        return lr, m.group(2)
    return None, None


def best_base_row(rows, model, eval_type, effort):
    candidates = [
        r for r in rows
        if r["base_model"] == model
        and r["eval_type"] == eval_type
        and is_base_row(r)
        and r.get("reasoning_effort", "") == effort
        and r.get("max_tokens", "") == "28000"
    ]
    if "gpt-oss" in model and eval_type == "reasonif":
        ac_candidates = [r for r in candidates if is_ac_row(r)]
        if ac_candidates:
            candidates = ac_candidates
    if not candidates:
        return None
    candidates.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return candidates[0]


def best_ft_row(rows, model, eval_type, ckpt):
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
    if "gpt-oss" in model and eval_type == "reasonif":
        ac_candidates = [r for r in candidates if is_ac_row(r)]
        if ac_candidates:
            candidates = ac_candidates
    candidates.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return candidates[0]


def load_rollouts(file_name):
    path = ROLLOUTS_DIR / file_name
    rollouts = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        try:
            rollouts.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rollouts


def word_count(rollout):
    """Total word count of reasoning + response."""
    text = (rollout.get("reasoning") or "") + " " + (rollout.get("response") or "")
    return len(text.split())


# ---------------------------------------------------------------------------
# Load data: collect word counts per model per condition
# ---------------------------------------------------------------------------

def load_word_counts():
    """Returns {model: {"base": [word_counts], "ft": [word_counts]}} across both eval types."""
    with open(SUMMARY_BY_RUN) as f:
        all_runs = list(csv.DictReader(f))

    counts = {}
    files_used = {}

    for m_cfg in MODELS:
        model = m_cfg["model"]
        effort = m_cfg["base_effort"]
        label = m_cfg["label"]
        counts[model] = {"base": [], "ft": []}

        for eval_type in EVAL_TYPES:
            for cond in ["base", "ft"]:
                if cond == "base":
                    row = best_base_row(all_runs, model, eval_type, effort)
                else:
                    row = best_ft_row(all_runs, model, eval_type, FT_CKPT)

                if not row:
                    print(f"  WARNING: no {cond} row for {label} {eval_type}")
                    continue

                fn = row["file_name"]
                key = (model, eval_type, cond)
                files_used[key] = fn

                rollouts = load_rollouts(fn)
                # Exclude baseline control_mode
                rollouts = [r for r in rollouts if r.get("control_mode") != "baseline"]
                wc = [word_count(r) for r in rollouts]
                counts[model][cond].extend(wc)

    # Print files used
    print("\nFiles used:")
    for (model, eval_type, cond), fn in sorted(files_used.items()):
        short = model.split("/")[-1]
        print(f"  {short:<20s} {eval_type:<12s} {cond:<5s} {fn}")

    return counts


# ---------------------------------------------------------------------------
# Plot: word count distributions
# ---------------------------------------------------------------------------

def make_word_count_distribution_plot(counts):
    """2x2 grid: one subplot per model, overlaid histograms for base vs FT."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    colors = {"base": "#78909C", "ft": "#1E88E5"}

    for idx, m_cfg in enumerate(MODELS):
        model = m_cfg["model"]
        label = m_cfg["label"]
        ax = axes[idx]

        base_wc = np.array(counts[model]["base"])
        ft_wc = np.array(counts[model]["ft"])

        if len(base_wc) == 0 and len(ft_wc) == 0:
            ax.set_title(f"{label} (no data)")
            continue

        # Log-spaced bins
        all_wc = np.concatenate([w for w in [base_wc, ft_wc] if len(w) > 0])
        all_wc_pos = all_wc[all_wc > 0]
        if len(all_wc_pos) == 0:
            ax.set_title(f"{label} (no data)")
            continue
        lo = max(all_wc_pos.min(), 1)
        hi = np.percentile(all_wc_pos, 99.5)
        bins = np.logspace(np.log10(lo), np.log10(hi), 50)

        if len(base_wc) > 0:
            base_pos = base_wc[base_wc > 0]
            ax.hist(base_pos, bins=bins, alpha=0.55, color=colors["base"],
                    label=f"Base (n={len(base_wc)}, med={np.median(base_wc):.0f})",
                    density=False, edgecolor="white", linewidth=0.3)
        if len(ft_wc) > 0:
            ft_pos = ft_wc[ft_wc > 0]
            ax.hist(ft_pos, bins=bins, alpha=0.55, color=colors["ft"],
                    label=f"FT step 60 (n={len(ft_wc)}, med={np.median(ft_wc):.0f})",
                    density=False, edgecolor="white", linewidth=0.3)

        # Add median lines
        if len(base_wc) > 0:
            ax.axvline(np.median(base_wc), color=colors["base"], linestyle="--",
                       linewidth=1.5, alpha=0.8)
        if len(ft_wc) > 0:
            ax.axvline(np.median(ft_wc), color=colors["ft"], linestyle="--",
                       linewidth=1.5, alpha=0.8)

        ax.set_xscale("log")
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.set_xlabel("Word Count (reasoning + response)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="both", alpha=0.3)

    fig.suptitle("Completion Word Count Distribution: Base vs Fine-Tuned\n"
                 "(CoTControl + ReasonIF, excl. baseline mode)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUTDIR / "word_count_distribution_base_vs_ft.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


def make_number_words_plot(all_runs):
    """number_words instruction only: show reasoning word count distributions
    and how they relate to per-sample thresholds."""
    colors = {"base": "#78909C", "ft": "#1E88E5"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for idx, m_cfg in enumerate(MODELS):
        model = m_cfg["model"]
        label = m_cfg["label"]
        effort = m_cfg["base_effort"]
        ax = axes[idx]

        data = {}  # cond -> list of (reasoning_words, threshold, compliant)
        for cond in ["base", "ft"]:
            if cond == "base":
                row = best_base_row(all_runs, model, "reasonif", effort)
            else:
                row = best_ft_row(all_runs, model, "reasonif", FT_CKPT)
            if not row:
                data[cond] = []
                continue

            rollouts = load_rollouts(row["file_name"])
            nw = [r for r in rollouts if r.get("control_mode") == "number_words"]
            entries = []
            for r in nw:
                reasoning = r.get("reasoning") or ""
                rw = len(reasoning.split())
                threshold = (r.get("sample", {}).get("metadata", {})
                              .get("constraint_args", {}).get("num_words"))
                compliant = r.get("compliant")
                entries.append((rw, threshold, compliant))
            data[cond] = entries

        # Plot 1: reasoning word count distributions
        base_rw = np.array([e[0] for e in data.get("base", [])])
        ft_rw = np.array([e[0] for e in data.get("ft", [])])

        if len(base_rw) == 0 and len(ft_rw) == 0:
            ax.set_title(f"{label} (no data)")
            continue

        all_rw = np.concatenate([w for w in [base_rw, ft_rw] if len(w) > 0])
        all_rw_pos = all_rw[all_rw > 0]
        if len(all_rw_pos) == 0:
            ax.set_title(f"{label} (no data)")
            continue
        lo = max(all_rw_pos.min(), 1)
        hi = np.percentile(all_rw_pos, 99.5)
        bins = np.logspace(np.log10(lo), np.log10(hi), 40)

        if len(base_rw) > 0:
            base_pos = base_rw[base_rw > 0]
            ax.hist(base_pos, bins=bins, alpha=0.55, color=colors["base"],
                    label=f"Base (n={len(base_rw)}, med={np.median(base_rw):.0f})",
                    density=False, edgecolor="white", linewidth=0.3)
        if len(ft_rw) > 0:
            ft_pos = ft_rw[ft_rw > 0]
            ax.hist(ft_pos, bins=bins, alpha=0.55, color=colors["ft"],
                    label=f"FT (n={len(ft_rw)}, med={np.median(ft_rw):.0f})",
                    density=False, edgecolor="white", linewidth=0.3)

        if len(base_rw) > 0:
            ax.axvline(np.median(base_rw), color=colors["base"], linestyle="--",
                       linewidth=1.5, alpha=0.8)
        if len(ft_rw) > 0:
            ax.axvline(np.median(ft_rw), color=colors["ft"], linestyle="--",
                       linewidth=1.5, alpha=0.8)

        # Show median threshold as a reference
        thresholds = [e[1] for e in data.get("base", []) if isinstance(e[1], (int, float))]
        if thresholds:
            med_thresh = np.median(thresholds)
            ax.axvline(med_thresh, color="#E53935", linestyle=":", linewidth=2, alpha=0.7,
                       label=f"Median threshold ({med_thresh:.0f})")

        # Add compliance rates
        for cond, cond_label in [("base", "Base"), ("ft", "FT")]:
            entries = data.get(cond, [])
            if entries:
                n_compliant = sum(1 for e in entries if e[2] is True)
                rate = n_compliant / len(entries) * 100
                ax.text(0.98, 0.75 - (0.08 if cond == "ft" else 0),
                        f"{cond_label} compliance: {rate:.0f}%",
                        transform=ax.transAxes, fontsize=8, ha="right",
                        color=colors[cond], fontweight="bold")

        ax.set_xscale("log")
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.set_xlabel("Reasoning Word Count")
        ax.set_ylabel("Count")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(axis="both", alpha=0.3)

    fig.suptitle("number_words: Reasoning Word Count Distribution — Base vs FT\n"
                 "(ReasonIF, with median threshold shown)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUTDIR / "word_count_distribution_number_words.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def make_number_words_scatter(all_runs):
    """Scatter: reasoning word count vs threshold, colored by compliance."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    for idx, m_cfg in enumerate(MODELS):
        model = m_cfg["model"]
        label = m_cfg["label"]
        effort = m_cfg["base_effort"]

        for cond_idx, (cond, cond_label) in enumerate([("base", "Base"), ("ft", "FT")]):
            ax = axes[cond_idx, idx]

            if cond == "base":
                row = best_base_row(all_runs, model, "reasonif", effort)
            else:
                row = best_ft_row(all_runs, model, "reasonif", FT_CKPT)

            if not row:
                ax.set_title(f"{label} {cond_label} (no data)")
                continue

            rollouts = load_rollouts(row["file_name"])
            nw = [r for r in rollouts if r.get("control_mode") == "number_words"]

            thresholds, rw_list, compliant_list = [], [], []
            for r in nw:
                reasoning = r.get("reasoning") or ""
                rw = len(reasoning.split())
                threshold = (r.get("sample", {}).get("metadata", {})
                              .get("constraint_args", {}).get("num_words"))
                if not isinstance(threshold, (int, float)):
                    continue
                thresholds.append(threshold)
                rw_list.append(rw)
                compliant_list.append(r.get("compliant") is True)

            thresholds = np.array(thresholds)
            rw_arr = np.array(rw_list)
            comp = np.array(compliant_list)

            if len(thresholds) == 0:
                ax.set_title(f"{label} {cond_label} (no data)")
                continue

            # Plot compliant and non-compliant
            ax.scatter(thresholds[comp], rw_arr[comp], s=18, alpha=0.6,
                       color="#43A047", label=f"Compliant ({comp.sum()})", zorder=3)
            ax.scatter(thresholds[~comp], rw_arr[~comp], s=18, alpha=0.6,
                       color="#E53935", label=f"Non-compliant ({(~comp).sum()})", zorder=3)

            # Diagonal: reasoning_words == threshold
            all_vals = np.concatenate([thresholds, rw_arr])
            lo = max(all_vals[all_vals > 0].min() * 0.8, 1)
            hi = all_vals.max() * 1.2
            ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, linewidth=1)

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal")
            ax.set_xlabel("Threshold (words)" if cond_idx == 1 else "")
            ax.set_ylabel("Reasoning Words" if idx == 0 else "")
            rate = comp.sum() / len(comp) * 100
            ax.set_title(f"{label} {cond_label} ({rate:.0f}%)", fontsize=11, fontweight="bold")
            ax.legend(fontsize=7, loc="upper left")
            ax.grid(alpha=0.2)

    fig.suptitle("number_words: Reasoning Words vs Threshold\n"
                 "(points below diagonal = compliant)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUTDIR / "number_words_scatter.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def make_number_words_compliance_by_threshold(all_runs):
    """Compliance rate vs eval threshold, base vs FT, one subplot per model."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    colors = {"base": "#78909C", "ft": "#1E88E5"}

    for idx, m_cfg in enumerate(MODELS):
        model = m_cfg["model"]
        label = m_cfg["label"]
        effort = m_cfg["base_effort"]
        ax = axes[idx]

        for cond, cond_label, color, marker in [
            ("base", "Base", colors["base"], "o"),
            ("ft", "FT step 60", colors["ft"], "s"),
        ]:
            if cond == "base":
                row = best_base_row(all_runs, model, "reasonif", effort)
            else:
                row = best_ft_row(all_runs, model, "reasonif", FT_CKPT)

            if not row:
                continue

            rollouts = load_rollouts(row["file_name"])
            nw = [r for r in rollouts if r.get("control_mode") == "number_words"]

            # Group by threshold
            by_thresh = {}
            for r in nw:
                t = (r.get("sample", {}).get("metadata", {})
                      .get("constraint_args", {}).get("num_words"))
                if t is None:
                    continue
                t = int(t)
                by_thresh.setdefault(t, []).append(r.get("compliant") is True)

            thresholds = sorted(by_thresh.keys())
            rates = []
            sizes = []
            for t in thresholds:
                compliant_list = by_thresh[t]
                rates.append(sum(compliant_list) / len(compliant_list) * 100)
                sizes.append(len(compliant_list))

            ax.plot(thresholds, rates, marker=marker, color=color,
                    linewidth=1.5, markersize=6, alpha=0.85, label=cond_label)

            # Annotate sample counts
            for t, rate, n in zip(thresholds, rates, sizes):
                ax.annotate(f"n={n}", (t, rate), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=6, color=color)

        ax.set_xscale("log")
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.set_xlabel("Threshold (words)")
        ax.set_ylabel("Compliance Rate (%)")
        ax.set_ylim(-5, 105)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(axis="both", alpha=0.3)

    fig.suptitle("number_words: Compliance Rate by Eval Threshold — Base vs FT",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUTDIR / "number_words_compliance_by_threshold.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def make_number_words_eval_vs_sft(all_runs):
    """Compare number_words thresholds and reasoning word counts:
    eval-time thresholds vs SFT thresholds vs SFT reasoning word counts."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    colors = {
        "eval_thresh": "#E53935",
        "sft_thresh": "#1E88E5",
        "sft_reasoning": "#43A047",
    }

    for idx, m_cfg in enumerate(MODELS):
        model = m_cfg["model"]
        label = m_cfg["label"]
        effort = m_cfg["base_effort"]
        ax = axes[idx]

        # 1. Eval-time thresholds (from base rollouts)
        row = best_base_row(all_runs, model, "reasonif", effort)
        eval_thresholds = []
        if row:
            rollouts = load_rollouts(row["file_name"])
            for r in rollouts:
                if r.get("control_mode") != "number_words":
                    continue
                t = (r.get("sample", {}).get("metadata", {})
                      .get("constraint_args", {}).get("num_words"))
                if t is not None:
                    eval_thresholds.append(float(t))
        eval_thresholds = np.array(eval_thresholds)

        # 2. SFT thresholds and reasoning word counts
        pq_fn = SFT_PARQUETS.get(model)
        sft_thresholds = []
        sft_reasoning_wc = []
        if pq_fn:
            df = pd.read_parquet(SFT_DIR / pq_fn)
            nw = df[df["constraint_name"].apply(lambda x: "number_words" in str(x))]
            for _, sft_row in nw.iterrows():
                args = sft_row.constraint_args
                for a in args:
                    if isinstance(a, dict) and a.get("num_words") is not None:
                        sft_thresholds.append(float(a["num_words"]))
                for m in sft_row.messages:
                    if m.get("role") == "assistant":
                        thinking = m.get("thinking") or ""
                        wc = len(re.findall(r"\w+", thinking))
                        sft_reasoning_wc.append(wc)
        sft_thresholds = np.array(sft_thresholds)
        sft_reasoning_wc = np.array(sft_reasoning_wc)

        # Combine all for bin range
        all_vals = np.concatenate([
            v for v in [eval_thresholds, sft_thresholds, sft_reasoning_wc]
            if len(v) > 0
        ])
        all_pos = all_vals[all_vals > 0]
        if len(all_pos) == 0:
            ax.set_title(f"{label} (no data)")
            continue

        lo = max(all_pos.min() * 0.8, 1)
        hi = all_pos.max() * 1.2
        bins = np.logspace(np.log10(lo), np.log10(hi), 35)

        if len(eval_thresholds) > 0:
            et_pos = eval_thresholds[eval_thresholds > 0]
            ax.hist(et_pos, bins=bins, alpha=0.5, color=colors["eval_thresh"],
                    label=f"Eval thresholds (n={len(eval_thresholds)}, "
                          f"med={np.median(eval_thresholds):.0f})",
                    density=False, edgecolor="white", linewidth=0.3)

        if len(sft_thresholds) > 0:
            st_pos = sft_thresholds[sft_thresholds > 0]
            ax.hist(st_pos, bins=bins, alpha=0.5, color=colors["sft_thresh"],
                    label=f"SFT thresholds (n={len(sft_thresholds)}, "
                          f"med={np.median(sft_thresholds):.0f})",
                    density=False, edgecolor="white", linewidth=0.3)

        if len(sft_reasoning_wc) > 0:
            sr_pos = sft_reasoning_wc[sft_reasoning_wc > 0]
            ax.hist(sr_pos, bins=bins, alpha=0.5, color=colors["sft_reasoning"],
                    label=f"SFT reasoning wc (n={len(sft_reasoning_wc)}, "
                          f"med={np.median(sft_reasoning_wc):.0f})",
                    density=False, edgecolor="white", linewidth=0.3)

        # Median lines
        for arr, color, ls in [
            (eval_thresholds, colors["eval_thresh"], "--"),
            (sft_thresholds, colors["sft_thresh"], "--"),
            (sft_reasoning_wc, colors["sft_reasoning"], ":"),
        ]:
            if len(arr) > 0:
                ax.axvline(np.median(arr), color=color, linestyle=ls,
                           linewidth=1.5, alpha=0.8)

        ax.set_xscale("log")
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.set_xlabel("Word Count")
        ax.set_ylabel("Count")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(axis="both", alpha=0.3)

    fig.suptitle("number_words: Eval Thresholds vs SFT Training Data\n"
                 "(threshold distributions and SFT reasoning word counts)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUTDIR / "number_words_eval_vs_sft.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def make_number_words_delta_distribution(all_runs):
    """Distribution of (actual_reasoning_words - threshold) per sample,
    base vs FT overlaid, one subplot per model."""
    colors = {"base": "#78909C", "ft": "#1E88E5"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for idx, m_cfg in enumerate(MODELS):
        model = m_cfg["model"]
        label = m_cfg["label"]
        effort = m_cfg["base_effort"]
        ax = axes[idx]

        # Collect deltas for both conditions first to compute shared bins
        all_deltas = {}  # cond -> np.array
        for cond in ["base", "ft"]:
            if cond == "base":
                row = best_base_row(all_runs, model, "reasonif", effort)
            else:
                row = best_ft_row(all_runs, model, "reasonif", FT_CKPT)

            if not row:
                all_deltas[cond] = np.array([])
                continue

            rollouts = load_rollouts(row["file_name"])
            nw = [r for r in rollouts if r.get("control_mode") == "number_words"]

            deltas = []
            for r in nw:
                reasoning = r.get("reasoning") or ""
                rw = len(reasoning.split())
                threshold = (r.get("sample", {}).get("metadata", {})
                              .get("constraint_args", {}).get("num_words"))
                if not isinstance(threshold, (int, float)):
                    continue
                deltas.append(rw - float(threshold))
            all_deltas[cond] = np.array(deltas)

        # Compute shared bins across both conditions
        combined = np.concatenate([d for d in all_deltas.values() if len(d) > 0])
        if len(combined) == 0:
            ax.set_title(f"{label} (no data)")
            continue
        bins = np.linspace(combined.min(), combined.max(), 50)

        for cond, cond_label, color in [
            ("base", "Base", colors["base"]),
            ("ft", "FT step 60", colors["ft"]),
        ]:
            deltas = all_deltas[cond]
            if len(deltas) == 0:
                continue

            n_compliant = int((deltas < 0).sum())
            rate = n_compliant / len(deltas) * 100
            ax.hist(deltas, bins=bins, alpha=0.55, color=color,
                    label=f"{cond_label} (n={len(deltas)}, med={np.median(deltas):.0f}, "
                          f"comply={rate:.0f}%)",
                    density=False, edgecolor="white", linewidth=0.3)
            ax.axvline(np.median(deltas), color=color, linestyle="--",
                       linewidth=1.5, alpha=0.8)

        # Vertical line at 0 (threshold boundary)
        ax.axvline(0, color="#E53935", linestyle="-", linewidth=1.5, alpha=0.7,
                   label="Threshold (delta=0)")

        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.set_xlabel("Reasoning Words − Threshold")
        ax.set_ylabel("Count")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(axis="both", alpha=0.3)

    fig.suptitle("number_words: Distribution of (Actual − Threshold) — Base vs FT\n"
                 "(negative = compliant)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUTDIR / "number_words_delta_distribution.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def make_word_count_distribution_by_eval(counts_by_eval):
    """Separate plots per eval type: 2x2 grid each."""
    colors = {"base": "#78909C", "ft": "#1E88E5"}

    for eval_type in EVAL_TYPES:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        axes = axes.flatten()

        for idx, m_cfg in enumerate(MODELS):
            model = m_cfg["model"]
            label = m_cfg["label"]
            ax = axes[idx]

            base_wc = np.array(counts_by_eval.get((model, eval_type, "base"), []))
            ft_wc = np.array(counts_by_eval.get((model, eval_type, "ft"), []))

            if len(base_wc) == 0 and len(ft_wc) == 0:
                ax.set_title(f"{label} (no data)")
                continue

            all_wc = np.concatenate([w for w in [base_wc, ft_wc] if len(w) > 0])
            all_wc_pos = all_wc[all_wc > 0]
            if len(all_wc_pos) == 0:
                ax.set_title(f"{label} (no data)")
                continue
            lo = max(all_wc_pos.min(), 1)
            hi = np.percentile(all_wc_pos, 99.5)
            bins = np.logspace(np.log10(lo), np.log10(hi), 50)

            if len(base_wc) > 0:
                base_pos = base_wc[base_wc > 0]
                ax.hist(base_pos, bins=bins, alpha=0.55, color=colors["base"],
                        label=f"Base (n={len(base_wc)}, med={np.median(base_wc):.0f})",
                        density=False, edgecolor="white", linewidth=0.3)
            if len(ft_wc) > 0:
                ft_pos = ft_wc[ft_wc > 0]
                ax.hist(ft_pos, bins=bins, alpha=0.55, color=colors["ft"],
                        label=f"FT step 60 (n={len(ft_wc)}, med={np.median(ft_wc):.0f})",
                        density=False, edgecolor="white", linewidth=0.3)

            if len(base_wc) > 0:
                ax.axvline(np.median(base_wc), color=colors["base"], linestyle="--",
                           linewidth=1.5, alpha=0.8)
            if len(ft_wc) > 0:
                ax.axvline(np.median(ft_wc), color=colors["ft"], linestyle="--",
                           linewidth=1.5, alpha=0.8)

            ax.set_xscale("log")
            ax.set_title(label, fontsize=13, fontweight="bold")
            ax.set_xlabel("Word Count (reasoning + response)")
            ax.set_ylabel("Count")
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(axis="both", alpha=0.3)

        fig.suptitle(f"{EVAL_DISPLAY[eval_type]}: Word Count Distribution — Base vs FT\n"
                     "(excl. baseline mode)",
                     fontsize=14, fontweight="bold")
        fig.tight_layout()
        out = OUTDIR / f"word_count_distribution_{eval_type}.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    with open(SUMMARY_BY_RUN) as f:
        all_runs = list(csv.DictReader(f))

    counts = {}
    counts_by_eval = {}

    print("Loading rollouts...")
    for m_cfg in MODELS:
        model = m_cfg["model"]
        effort = m_cfg["base_effort"]
        label = m_cfg["label"]
        counts[model] = {"base": [], "ft": []}

        for eval_type in EVAL_TYPES:
            for cond in ["base", "ft"]:
                if cond == "base":
                    row = best_base_row(all_runs, model, eval_type, effort)
                else:
                    row = best_ft_row(all_runs, model, eval_type, FT_CKPT)

                if not row:
                    print(f"  WARNING: no {cond} row for {label} {eval_type}")
                    continue

                fn = row["file_name"]
                rollouts = load_rollouts(fn)
                rollouts = [r for r in rollouts if r.get("control_mode") != "baseline"]
                wc = [word_count(r) for r in rollouts]

                counts[model][cond].extend(wc)
                counts_by_eval[(model, eval_type, cond)] = wc

                print(f"  {label:<16s} {eval_type:<12s} {cond:<5s} "
                      f"n={len(wc):>4d}  med={np.median(wc):>6.0f}  "
                      f"mean={np.mean(wc):>6.0f}  {fn}")

    # Combined plot (both eval types)
    make_word_count_distribution_plot(counts)

    # Per-eval-type plots
    make_word_count_distribution_by_eval(counts_by_eval)

    # number_words specific plots
    make_number_words_plot(all_runs)
    make_number_words_scatter(all_runs)
    make_number_words_compliance_by_threshold(all_runs)
    make_number_words_delta_distribution(all_runs)
    make_number_words_eval_vs_sft(all_runs)

    print(f"\nAll plots saved to {OUTDIR}")


if __name__ == "__main__":
    main()
