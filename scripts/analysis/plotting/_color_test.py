#!/usr/bin/env python3
"""Generate 10 color palette candidates for review."""

import csv
import math
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.size"] = 9

SUMMARY_BY_RUN = Path("results/summaries/tables/summary_by_run.csv")
OUTDIR = Path("results/summaries/plots/2026_03_24")
Z80 = 1.282

PALETTES = {
    "A: Teal + Coral": {
        "openai/gpt-oss-20b": "#F4976C",
        "openai/gpt-oss-120b": "#C0392B",
        "qwen/qwen3-8b": "#76D7C4",
        "qwen/qwen3-32b": "#148F77",
    },
    "B: Purple + Gold": {
        "openai/gpt-oss-20b": "#F0C05A",
        "openai/gpt-oss-120b": "#C8860C",
        "qwen/qwen3-8b": "#B39DDB",
        "qwen/qwen3-32b": "#5E35B1",
    },
    "C: Blue + Orange": {
        "openai/gpt-oss-20b": "#FF9E4A",
        "openai/gpt-oss-120b": "#D65F02",
        "qwen/qwen3-8b": "#67A9CF",
        "qwen/qwen3-32b": "#1B6497",
    },
    "D: Olive + Rose": {
        "openai/gpt-oss-20b": "#F1948A",
        "openai/gpt-oss-120b": "#922B21",
        "qwen/qwen3-8b": "#A8C97F",
        "qwen/qwen3-32b": "#4A7C2E",
    },
    "E: Navy + Amber": {
        "openai/gpt-oss-20b": "#FFCA28",
        "openai/gpt-oss-120b": "#F57F17",
        "qwen/qwen3-8b": "#7FA8D1",
        "qwen/qwen3-32b": "#1A3A5C",
    },
    "F: Slate + Terracotta": {
        "openai/gpt-oss-20b": "#E8836B",
        "openai/gpt-oss-120b": "#A63D28",
        "qwen/qwen3-8b": "#8FAABE",
        "qwen/qwen3-32b": "#34495E",
    },
    "G: Emerald + Plum": {
        "openai/gpt-oss-20b": "#CE93D8",
        "openai/gpt-oss-120b": "#7B1FA2",
        "qwen/qwen3-8b": "#81C784",
        "qwen/qwen3-32b": "#2E7D32",
    },
    "H: Indigo + Peach": {
        "openai/gpt-oss-20b": "#FFAB91",
        "openai/gpt-oss-120b": "#D84315",
        "qwen/qwen3-8b": "#9FA8DA",
        "qwen/qwen3-32b": "#283593",
    },
    "I: Sky + Brick": {
        "openai/gpt-oss-20b": "#EF8D6E",
        "openai/gpt-oss-120b": "#8E2E1E",
        "qwen/qwen3-8b": "#64B5F6",
        "qwen/qwen3-32b": "#0D47A1",
    },
    "J: Warm + Cool Gray": {
        "openai/gpt-oss-20b": "#F4A582",
        "openai/gpt-oss-120b": "#B2182B",
        "qwen/qwen3-8b": "#92C5DE",
        "qwen/qwen3-32b": "#2166AC",
    },
}

MODELS = [
    {"model": "openai/gpt-oss-20b", "label": "GPT-OSS-20B", "base_effort": "medium"},
    {"model": "openai/gpt-oss-120b", "label": "GPT-OSS-120B", "base_effort": "medium"},
    {"model": "qwen/qwen3-8b", "label": "Qwen3-8B", "base_effort": "none"},
    {"model": "qwen/qwen3-32b", "label": "Qwen3-32B", "base_effort": "none"},
]
EVAL_TYPES = ["cotcontrol", "reasonif"]
FT_CKPTS = ["000060", "000120", "000180", "final"]
CKPT_DISPLAY = {"000060": "60", "000120": "120", "000180": "180", "final": "final"}
EFFORT_ORDER = ["low", "medium", "high"]


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
    m = re.search(r"-rif-(?:lr([\d.]+e-\d+)-)?(\d{3,}|final)", filename)
    if m:
        return m.group(1) or "1e-4", m.group(2)
    return None, None

def is_ft_row(row):
    return "-rif-" in row.get("file_name", "")

def is_base_row(row):
    return not row.get("checkpoint", "") and not is_ft_row(row)


# Load data once
with open(SUMMARY_BY_RUN) as f:
    all_rows = list(csv.DictReader(f))

model_names = {m["model"] for m in MODELS}
rows = [r for r in all_rows if r["base_model"] in model_names]

# Training curve data
tc_data = {}
for m_cfg in MODELS:
    model, effort = m_cfg["model"], m_cfg["base_effort"]
    for et in EVAL_TYPES:
        key = (model, et)
        tc_data[key] = {"base": None, "ckpts": {}}
        base_cands = [r for r in rows if r["base_model"] == model and r["eval_type"] == et
                      and is_base_row(r) and r.get("reasoning_effort", "") == effort
                      and r.get("max_tokens", "") == "28000"]
        if base_cands:
            base_cands.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
            tc_data[key]["base"] = base_cands[0]
        for ckpt in FT_CKPTS:
            cands = [r for r in rows if r["base_model"] == model and r["eval_type"] == et
                     and is_ft_row(r) and extract_lr_ckpt(r.get("file_name", "")) == ("1e-4", ckpt)]
            if cands:
                cands.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
                tc_data[key]["ckpts"][ckpt] = cands[0]

# Effort sweep data
STEP60_SESSION_IDS = {"openai/gpt-oss-120b": "1cbfc28d", "openai/gpt-oss-20b": "6adcf41e"}
base_effort = {}
for r in all_rows:
    if "gpt-oss" not in r["base_model"]: continue
    if (r.get("checkpoint") or "").strip(): continue
    if r.get("max_tokens") != "28000": continue
    if si(r, "n_clean") < 10: continue
    k = (r["base_model"], r["eval_type"], r["reasoning_effort"])
    if k not in base_effort or r["timestamp"] > base_effort[k]["timestamp"]:
        base_effort[k] = r

ft_effort = {}
for r in all_rows:
    cp = (r.get("checkpoint") or "").strip()
    if not cp or "000060" not in cp: continue
    model = r["base_model"]
    sid = STEP60_SESSION_IDS.get(model)
    if sid and sid in cp:
        k = (model, r["eval_type"], r["reasoning_effort"])
        if k not in ft_effort or r["timestamp"] > ft_effort[k]["timestamp"]:
            ft_effort[k] = r

EFFORT_MODELS = [
    {"model": "openai/gpt-oss-20b", "label": "GPT-OSS-20B"},
    {"model": "openai/gpt-oss-120b", "label": "GPT-OSS-120B"},
]

# Generate one big figure: 10 rows x 4 cols (training_curve_cot, training_curve_rif, effort_cot, effort_rif)
fig, all_axes = plt.subplots(10, 4, figsize=(20, 40))

for row_idx, (name, colors) in enumerate(PALETTES.items()):
    # Training curve plots (cols 0-1)
    for col_idx, et in enumerate(EVAL_TYPES):
        ax = all_axes[row_idx][col_idx]
        for m_cfg in MODELS:
            model = m_cfg["model"]
            key = (model, et)
            color = colors[model]
            xs, vals, errs = [], [], []
            base_row = tc_data[key]["base"]
            if base_row:
                p = sf(base_row, "compliant_rate")
                n = si(base_row, "n_clean")
                xs.append(0); vals.append((p or 0)*100); errs.append(ci80(p,n)*100)
            for i, ckpt in enumerate(FT_CKPTS):
                r = tc_data[key]["ckpts"].get(ckpt)
                if r:
                    p = sf(r, "compliant_rate"); n = si(r, "n_clean")
                    xs.append(i+1); vals.append((p or 0)*100); errs.append(ci80(p,n)*100)
            ax.errorbar(xs, vals, yerr=errs, fmt="o-", color=color, capsize=2,
                       label=m_cfg["label"], linewidth=1.5, markersize=4, elinewidth=0.8)

        ticks = [0] + list(range(1, len(FT_CKPTS)+1))
        ax.set_xticks(ticks)
        ax.set_xticklabels(["base"] + [CKPT_DISPLAY[c] for c in FT_CKPTS], fontsize=7)
        ax.set_ylim(0, None)
        ax.grid(axis="y", alpha=0.3)
        if col_idx == 0:
            ax.set_ylabel(name, fontsize=8, fontweight="bold")
        if row_idx == 0:
            et_name = "CoTControl" if et == "cotcontrol" else "ReasonIF"
            ax.set_title(f"Training Curve — {et_name}", fontsize=9, fontweight="bold")
        if row_idx == 0 and col_idx == 0:
            ax.legend(fontsize=6, loc="upper left")

    # Effort sweep plots (cols 2-3)
    for col_idx, et in enumerate(EVAL_TYPES):
        ax = all_axes[row_idx][col_idx + 2]
        x = np.arange(len(EFFORT_ORDER))
        for m_cfg in EFFORT_MODELS:
            model = m_cfg["model"]
            color = colors[model]
            # Base
            bx, by, be = [], [], []
            for i, eff in enumerate(EFFORT_ORDER):
                r = base_effort.get((model, et, eff))
                if r:
                    p = sf(r, "compliant_rate"); n = si(r, "n_clean")
                    bx.append(i); by.append((p or 0)*100); be.append(ci80(p,n)*100)
            if bx:
                ax.errorbar(bx, by, yerr=be, fmt="o-", color=color, capsize=3,
                           label=f"{m_cfg['label']} Base", linewidth=1.5, markersize=5, elinewidth=0.8)
            # FT
            fx, fy, fe = [], [], []
            for i, eff in enumerate(EFFORT_ORDER):
                r = ft_effort.get((model, et, eff))
                if r:
                    p = sf(r, "compliant_rate"); n = si(r, "n_clean")
                    fx.append(i); fy.append((p or 0)*100); fe.append(ci80(p,n)*100)
            if fx:
                ax.errorbar(fx, fy, yerr=fe, fmt="s:", color=color, capsize=3,
                           label=f"{m_cfg['label']} FT", linewidth=1.5, markersize=5, elinewidth=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(EFFORT_ORDER, fontsize=7)
        ax.set_ylim(0, None)
        ax.grid(axis="y", alpha=0.3)
        if row_idx == 0:
            et_name = "CoTControl" if et == "cotcontrol" else "ReasonIF"
            ax.set_title(f"Effort Sweep — {et_name}", fontsize=9, fontweight="bold")
        if row_idx == 0 and col_idx == 0:
            ax.legend(fontsize=5, loc="upper right")

fig.tight_layout()
out_path = OUTDIR / "_color_palette_candidates.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")
