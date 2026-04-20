#!/usr/bin/env python3
"""FT plots with LoRA-1 fine-tunes (2026-04-07).

Re-does the three main FT comparison plots from 2026-03-27 using:
  - Same base model eval results
  - New LoRA rank=1, lr=1e-4 FT eval results (run_id 20260407_194548)

Plots:
  1. base_vs_ft_lr1e-4_step60_compliance
  2. base_vs_ft_lr1e-4_step60_excl_uppercase
  3. base_vs_ft_lr1e-4_step60_meta_discussion

FT data is read directly from rollout JSONL files (not summary CSV).
Base data is read from summary CSV (same rows as 2026-03-27).

Usage:
    python scripts/analysis/plot_summary_2026_04_07.py
"""

import csv
import json
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
DATE = "2026_04_07"
OUTDIR = Path(f"results/summaries/plots/{DATE}")
OUTDIR.mkdir(parents=True, exist_ok=True)

ROLLOUT_DIR = Path("results/rollouts")

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

FT_RUN_ID = "20260407_194548"

EVAL_TYPES = ["cotcontrol", "reasonif"]
EVAL_DISPLAY = {"cotcontrol": "CoTControl", "reasonif": "ReasonIF"}

EXCLUDE_COTCONTROL = {"uppercase_thinking"}

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


def is_ac_row(row):
    return row.get("analysis_channel", "").strip().lower() == "true"


def is_base_row(row):
    fn = row.get("file_name", "")
    return not row.get("checkpoint", "") and "-rif-" not in fn


# ---------------------------------------------------------------------------
# Base data from summary CSV (same logic as 2026-03-27)
# ---------------------------------------------------------------------------

def best_base_row(rows, model, eval_type, effort):
    """Find best base row: max_tokens=28000, prefer AC for gpt-oss reasonif."""
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


# ---------------------------------------------------------------------------
# FT data from rollout JSONL files
# ---------------------------------------------------------------------------

def find_ft_rollout_file(model, eval_type):
    """Find the lora1 rollout file for a model/eval_type.

    If multiple files exist (e.g. with/without analysis_channel), prefer the one
    with analysis_channel=True for gpt-oss reasonif.
    """
    model_slug = model.replace("/", "_")
    pattern = f"{model_slug}-rif-lora1-lr1e-4-final_{eval_type}_all_{FT_RUN_ID}_*.jsonl"
    matches = list(ROLLOUT_DIR.glob(pattern))
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]

    # Multiple files — pick the right one based on header
    need_ac = "gpt-oss" in model and eval_type == "reasonif"
    for m in matches:
        with open(m) as f:
            first_line = f.readline().strip()
        if first_line.startswith("#"):
            header = json.loads(first_line.lstrip("#").strip())
            ac = header.get("analysis_channel", False)
            if need_ac and ac:
                return m
            if not need_ac and not ac:
                return m
    # Fallback: newest by mtime
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    print(f"  WARNING: multiple FT rollout files for {model} {eval_type}, using newest: {matches[0].name}")
    return matches[0]


def load_ft_rollouts(filepath):
    """Load rollouts from a JSONL file, returning (header, rollouts)."""
    header = None
    rollouts = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                header = json.loads(line.lstrip("#").strip())
                continue
            rollouts.append(json.loads(line))
    return header, rollouts


def compute_metrics_from_rollouts(rollouts, exclude_modes=None):
    """Compute compliance and meta_discussion rates from rollout list.

    Filters out rollouts with reasoning errors (empty reasoning).
    Returns dict with compliant_rate, meta_discussion_rate, n_clean.
    """
    exclude_modes = exclude_modes or set()
    clean = []
    for r in rollouts:
        mode = r.get("control_mode", "")
        if mode == "baseline" or mode in exclude_modes:
            continue
        # Skip reasoning errors (no reasoning produced)
        reasoning = r.get("reasoning") or ""
        if not reasoning.strip():
            continue
        clean.append(r)

    if not clean:
        return {"compliant_rate": None, "meta_discussion_rate": None, "n_clean": 0}

    n = len(clean)
    n_compliant = sum(1 for r in clean if r.get("compliant"))
    n_meta = sum(1 for r in clean if r.get("meta_discussion"))

    return {
        "compliant_rate": n_compliant / n,
        "meta_discussion_rate": n_meta / n,
        "n_clean": n,
    }


def compute_metrics_by_instruction(rollouts, exclude_modes=None):
    """Compute per-instruction-type metrics, then aggregate excluding specified modes.

    Returns dict with compliant_rate, n_clean (aggregated over non-excluded modes).
    """
    exclude_modes = exclude_modes or set()

    # Group by instruction type
    by_type = {}
    for r in rollouts:
        mode = r.get("control_mode", "")
        if mode == "baseline":
            continue
        reasoning = r.get("reasoning") or ""
        if not reasoning.strip():
            continue
        by_type.setdefault(mode, []).append(r)

    total_compliant = 0
    total_clean = 0
    for mode, items in by_type.items():
        if mode in exclude_modes:
            continue
        n = len(items)
        nc = sum(1 for r in items if r.get("compliant"))
        total_compliant += nc
        total_clean += n

    if total_clean == 0:
        return {"compliant_rate": None, "n_clean": 0}

    return {"compliant_rate": total_compliant / total_clean, "n_clean": total_clean}


# ---------------------------------------------------------------------------
# Load all data
# ---------------------------------------------------------------------------

def load_data():
    """Load base (from CSV) + FT (from JSONL) for all 4 models.

    Returns data dict with keys (model, eval_type) -> {base: row, ft: metrics_dict, ft_file: path}
    """
    with open(SUMMARY_BY_RUN) as f:
        csv_rows = list(csv.DictReader(f))

    data = {}
    for m_cfg in MODELS:
        model = m_cfg["model"]
        effort = m_cfg["base_effort"]
        for eval_type in EVAL_TYPES:
            key = (model, eval_type)
            base = best_base_row(csv_rows, model, eval_type, effort)

            ft_file = find_ft_rollout_file(model, eval_type)
            ft_metrics = None
            ft_header = None
            if ft_file:
                ft_header, ft_rollouts = load_ft_rollouts(ft_file)
                ft_metrics = compute_metrics_from_rollouts(ft_rollouts)
                ft_metrics["file"] = ft_file.name

                # Validate constraints
                if ft_header:
                    mt = ft_header.get("max_tokens")
                    ac = ft_header.get("analysis_channel", False)
                    if mt != 28000:
                        print(f"  WARNING: {ft_file.name} has max_tokens={mt}, expected 28000")
                    if "gpt-oss" in model and eval_type == "reasonif" and not ac:
                        print(f"  WARNING: {ft_file.name} has analysis_channel={ac}, "
                              f"expected True for gpt-oss reasonif")

            data[key] = {
                "base": base,
                "ft": ft_metrics,
                "ft_file": ft_file,
                "ft_header": ft_header,
            }
    return data


def load_data_excl_uppercase():
    """Load data with cotcontrol excluding uppercase_thinking.

    Base: re-aggregate from instruction-level CSV.
    FT: re-aggregate from JSONL rollouts.
    """
    # --- Base from instruction-level CSV ---
    model_names = {m["model"] for m in MODELS}
    with open(SUMMARY_BY_INSTRUCTION) as f:
        all_instr = list(csv.DictReader(f))

    runs = {}
    for r in all_instr:
        if r["base_model"] not in model_names:
            continue
        k = (r["base_model"], r["eval_type"], r["file_name"])
        runs.setdefault(k, []).append(r)

    def aggregate_csv_run(instr_rows, exclude_types):
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

            # Base from CSV
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
                p, n = aggregate_csv_run(best[1], exclude)
                base_fn = best[1][0].get("file_name", "")
                if p is not None:
                    data[key]["base"] = {"compliant_rate": p, "n_clean": n, "file": base_fn}

            # FT from JSONL
            ft_file = find_ft_rollout_file(model, eval_type)
            if ft_file:
                _, ft_rollouts = load_ft_rollouts(ft_file)
                metrics = compute_metrics_by_instruction(ft_rollouts, exclude_modes=exclude)
                if metrics["compliant_rate"] is not None:
                    metrics["file"] = ft_file.name
                    data[key]["ft"] = metrics

    return data


# ---------------------------------------------------------------------------
# Plot 1: Base vs FT compliance
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
            # Base
            row = data[key]["base"]
            p = sf(row, "compliant_rate")
            n = si(row, "n_clean")
            base_vals.append((p or 0) * 100)
            base_errs.append(ci80(p, n) * 100)

            # FT
            ft = data[key]["ft"]
            if ft and ft.get("compliant_rate") is not None:
                p_ft = ft["compliant_rate"]
                n_ft = ft["n_clean"]
                ft_vals.append(p_ft * 100)
                ft_errs.append(ci80(p_ft, n_ft) * 100)
            else:
                ft_vals.append(0)
                ft_errs.append(0)

        bars_base = ax.bar(x - bar_width / 2, base_vals, bar_width, yerr=base_errs, capsize=3,
               color=colors["base"], alpha=0.85, label="Base", error_kw={"linewidth": 1})
        bars_ft = ax.bar(x + bar_width / 2, ft_vals, bar_width, yerr=ft_errs, capsize=3,
               color=colors["ft"], alpha=0.85, label="FT (LoRA-1)", error_kw={"linewidth": 1})

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

    fig.suptitle("CoT Controllability: Base vs LoRA-1 Fine-Tuned (80% CI)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUTDIR / "base_vs_ft_lr1e-4_step60_compliance.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 2: Base vs FT excluding uppercase_thinking
# ---------------------------------------------------------------------------

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
               color=colors["ft"], alpha=0.85, label="FT (LoRA-1)", error_kw={"linewidth": 1})

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
# Plot 3: Base vs FT meta discussion
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
            # Base
            row = data[key]["base"]
            p = sf(row, "meta_discussion_rate")
            n = si(row, "n_clean")
            base_vals.append((p or 0) * 100)
            base_errs.append(ci80(p, n) * 100)

            # FT
            ft = data[key]["ft"]
            if ft and ft.get("meta_discussion_rate") is not None:
                p_ft = ft["meta_discussion_rate"]
                n_ft = ft["n_clean"]
                ft_vals.append(p_ft * 100)
                ft_errs.append(ci80(p_ft, n_ft) * 100)
            else:
                ft_vals.append(0)
                ft_errs.append(0)

        bars_base = ax.bar(x - bar_width / 2, base_vals, bar_width, yerr=base_errs, capsize=3,
               color=colors["base"], alpha=0.85, label="Base", error_kw={"linewidth": 1})
        bars_ft = ax.bar(x + bar_width / 2, ft_vals, bar_width, yerr=ft_errs, capsize=3,
               color=colors["ft"], alpha=0.85, label="FT (LoRA-1)", error_kw={"linewidth": 1})

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

    fig.suptitle("Meta Discussion Rate: Base vs LoRA-1 Fine-Tuned (80% CI)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUTDIR / "base_vs_ft_lr1e-4_step60_meta_discussion.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 4/5: Instruction bars (base vs FT, grouped by instruction type)
# ---------------------------------------------------------------------------

MODEL_COLORS = {
    "openai/gpt-oss-20b": "#FF2A00",
    "openai/gpt-oss-120b": "#9933FF",
    "qwen/qwen3-8b": "#4895EF",
    "qwen/qwen3-32b": "#1D3557",
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


def load_instruction_data():
    """Load per-instruction base (CSV) + FT (JSONL) data.

    Returns dict: (model, eval_type) -> {instr_type -> {"base": csv_row, "ft": metrics_dict}}
    Each model/eval_type traces to exactly one base file and one FT file.
    """
    # --- Base from instruction-level CSV ---
    all_instr = []
    with open(SUMMARY_BY_INSTRUCTION) as f:
        all_instr = list(csv.DictReader(f))

    model_names = {m["model"] for m in MODELS}
    by_file = {}
    for r in all_instr:
        if r["base_model"] not in model_names:
            continue
        by_file.setdefault(r["file_name"], []).append(r)

    data = {}
    for m_cfg in MODELS:
        model = m_cfg["model"]
        effort = m_cfg["base_effort"]
        for eval_type in EVAL_TYPES:
            key = (model, eval_type)
            data[key] = {}

            # Best base file
            best_ts = ""
            best_rows = None
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
                if ts > best_ts:
                    best_ts = ts
                    best_rows = rows

            if best_rows:
                for r in best_rows:
                    it = r["instruction_type"]
                    if it == "baseline":
                        continue
                    data[key].setdefault(it, {"base": None, "ft": None})
                    data[key][it]["base"] = r

            # FT from JSONL - compute per-instruction metrics
            ft_file = find_ft_rollout_file(model, eval_type)
            if ft_file:
                _, ft_rollouts = load_ft_rollouts(ft_file)
                by_mode = {}
                for r in ft_rollouts:
                    mode = r.get("control_mode", "")
                    if mode == "baseline":
                        continue
                    reasoning = r.get("reasoning") or ""
                    if not reasoning.strip():
                        continue
                    by_mode.setdefault(mode, []).append(r)

                for mode, items in by_mode.items():
                    n = len(items)
                    nc = sum(1 for r in items if r.get("compliant"))
                    data[key].setdefault(mode, {"base": None, "ft": None})
                    data[key][mode]["ft"] = {
                        "compliant_rate": nc / n if n > 0 else 0,
                        "n_clean": n,
                    }

    return data


def make_instruction_bars(instr_data, eval_type):
    """One figure per eval type. Groups = instruction types, 8 bars per group
    (base + FT for each of 4 models)."""
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
        base_alpha = 0.45
        ft_alpha = 0.90

        base_vals, base_errs, ft_vals, ft_errs = [], [], [], []
        for it in instr_types:
            it_data = instr_data.get(key, {}).get(it, {"base": None, "ft": None})

            base_row = it_data.get("base")
            p = sf(base_row, "compliant_rate") if base_row else None
            n = si(base_row, "n_clean") if base_row else 0
            base_vals.append((p or 0) * 100)
            base_errs.append(ci80(p, n) * 100)

            ft_row = it_data.get("ft")
            if ft_row:
                p_ft = ft_row["compliant_rate"]
                n_ft = ft_row["n_clean"]
                ft_vals.append(p_ft * 100)
                ft_errs.append(ci80(p_ft, n_ft) * 100)
            else:
                ft_vals.append(0)
                ft_errs.append(0)

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

    fig.suptitle(f"{EVAL_DISPLAY[eval_type]}: Compliance by Instruction — Base vs FT LoRA-1 (80% CI)",
                 fontsize=24, fontweight="bold")
    fig.tight_layout()
    out = OUTDIR / f"instruction_bars_{eval_type}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading data (FT run: {FT_RUN_ID})...\n")

    data = load_data()

    # Print summary table with file sources
    print(f"\n{'Model':<25s} {'Eval':<12s} {'Base comp%':>10s} {'FT comp%':>10s}  {'FT file'}")
    print("-" * 110)
    for m_cfg in MODELS:
        for eval_type in EVAL_TYPES:
            key = (m_cfg["model"], eval_type)
            base = data[key]["base"]
            ft = data[key]["ft"]
            base_str = f"{sf(base, 'compliant_rate') * 100:.1f}" if base and sf(base, "compliant_rate") is not None else "N/A"
            ft_str = f"{ft['compliant_rate'] * 100:.1f}" if ft and ft.get("compliant_rate") is not None else "N/A"
            ft_fn = data[key]["ft_file"].name if data[key]["ft_file"] else "MISSING"
            base_fn = base.get("file_name", "") if base else "MISSING"
            print(f"{m_cfg['label']:<25s} {eval_type:<12s} {base_str:>10s} {ft_str:>10s}  FT: {ft_fn}")
            print(f"{'':>49s}  Base: {base_fn}")

    # Verify: every data point traces to exactly one file
    print("\n--- File provenance check (plots 1 & 3) ---")
    all_ok = True
    for m_cfg in MODELS:
        for eval_type in EVAL_TYPES:
            key = (m_cfg["model"], eval_type)
            base = data[key]["base"]
            ft_file = data[key]["ft_file"]
            base_fn = base.get("file_name", "") if base else None
            ft_fn = ft_file.name if ft_file else None
            if not base_fn:
                print(f"  MISSING base: {m_cfg['label']} {eval_type}")
                all_ok = False
            if not ft_fn:
                print(f"  MISSING ft:   {m_cfg['label']} {eval_type}")
                all_ok = False
    if all_ok:
        print("  All 16 bars map to exactly one rollout file each.")

    # 1. Compliance
    print("\nPlot 1: Compliance...")
    make_plot_compliance(data)

    # 2. Excl uppercase
    print("\nPlot 2: Excl uppercase...")
    excl_data = load_data_excl_uppercase()

    print("\n--- File provenance check (plot 2) ---")
    all_ok = True
    for m_cfg in MODELS:
        for eval_type in EVAL_TYPES:
            key = (m_cfg["model"], eval_type)
            base = excl_data[key]["base"]
            ft = excl_data[key]["ft"]
            base_fn = base.get("file") if base else None
            ft_fn = ft.get("file") if ft else None
            if not base_fn:
                print(f"  MISSING base: {m_cfg['label']} {eval_type}")
                all_ok = False
            if not ft_fn:
                print(f"  MISSING ft:   {m_cfg['label']} {eval_type}")
                all_ok = False
    if all_ok:
        print("  All 16 bars map to exactly one rollout file each.")

    make_plot_excl_uppercase(excl_data)

    # 3. Meta discussion
    print("\nPlot 3: Meta discussion...")
    make_plot_meta_discussion(data)

    # 4-5. Instruction bars
    print("\nPlot 4-5: Instruction bars...")
    instr_data = load_instruction_data()
    make_instruction_bars(instr_data, "cotcontrol")
    make_instruction_bars(instr_data, "reasonif")


if __name__ == "__main__":
    main()
