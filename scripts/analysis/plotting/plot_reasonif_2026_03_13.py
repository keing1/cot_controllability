#!/usr/bin/env python3
"""ReasonIF evaluation plots - 2026-03-13.

Three grouped bar plots:
1. Average metrics across models (compliance, accuracy, meta-discussion, compliance-less-meta)
2. GPT-OSS 20B compliance by mode vs paper baselines
3. Compliance by mode for all three models
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

ROLLOUT_DIR = Path(__file__).resolve().parents[1] / "results" / "rollouts"

# Use OLD GPT-OSS 20B results (matches paper settings)
FILES = {
    "GPT-OSS 20B": ROLLOUT_DIR / "old" / "openai_gpt-oss-20b_reasonif_all_e1a4cfc07786.jsonl",
    "Qwen3-8B": ROLLOUT_DIR / "qwen_qwen3-8b_reasonif_all_4edd2f6c604c.jsonl",
    "Qwen3-32B": ROLLOUT_DIR / "qwen_qwen3-32b_reasonif_all_83d870255ccd.jsonl",
}

# ReasonIF mode name -> paper name mapping
MODE_LABELS = {
    "reasoning_language": "Multilinguality",
    "number_words": "Word Limit",
    "english_capital": "Uppercase",
    "end_checker": "Disclaimer",
    "json_format": "JSON",
    "no_comma": "Remove Commas",
}

MODE_ORDER = ["reasoning_language", "number_words", "english_capital", "end_checker", "json_format", "no_comma"]

# Paper baselines for GPT-OSS 20B (from ReasonIF paper)
PAPER_BASELINES = {
    "reasoning_language": 0.27,
    "number_words": 0.34,
    "english_capital": 0.0,
    "end_checker": 0.0,
    "json_format": 0.0,
    "no_comma": 0.04,
}


def load_rollouts(path: Path) -> list[dict]:
    rollouts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            try:
                rollouts.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rollouts


def compute_mode_metrics(rollouts: list[dict]) -> dict:
    """Compute per-mode metrics from rollouts."""
    by_mode = defaultdict(list)
    for r in rollouts:
        by_mode[r["control_mode"]].append(r)

    metrics = {}
    for mode, rs in by_mode.items():
        n = len(rs)
        correct = sum(1 for r in rs if r.get("correct"))
        has_compliance = [r for r in rs if r.get("compliant") is not None]
        compliant = sum(1 for r in has_compliance if r["compliant"])
        has_meta = [r for r in rs if r.get("meta_discussion") is not None]
        meta = sum(1 for r in has_meta if r["meta_discussion"])

        # Sample-level: compliant AND no meta-discussion
        has_both = [r for r in rs if r.get("compliant") is not None and r.get("meta_discussion") is not None]
        compliant_no_meta = sum(1 for r in has_both if r["compliant"] and not r["meta_discussion"])

        metrics[mode] = {
            "n": n,
            "accuracy": correct / n if n else 0,
            "compliance": compliant / len(has_compliance) if has_compliance else 0,
            "meta_discussion": meta / len(has_meta) if has_meta else 0,
            "compliant_no_meta": compliant_no_meta / len(has_both) if has_both else 0,
        }
    return metrics


# ---------------------------------------------------------------------------
# Load all data
# ---------------------------------------------------------------------------

all_data = {}
for model_name, path in FILES.items():
    if not path.exists():
        print(f"WARNING: {path} not found, skipping {model_name}")
        continue
    rollouts = load_rollouts(path)
    all_data[model_name] = compute_mode_metrics(rollouts)
    print(f"Loaded {model_name}: {len(rollouts)} rollouts")

models = list(all_data.keys())

# ---------------------------------------------------------------------------
# Plot 1: Average metrics across models
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

metric_names = ["Accuracy", "Compliance", "Meta-Discussion", "Compliant\nw/o Meta"]
metric_keys = ["accuracy", "compliance", "meta_discussion", "compliant_no_meta"]
x = np.arange(len(metric_names))
width = 0.22
offsets = np.arange(len(models)) - (len(models) - 1) / 2

colors_models = ["#4C72B0", "#55A868", "#C44E52"]

for i, (model, color) in enumerate(zip(models, colors_models)):
    mode_metrics = all_data[model]
    vals = [np.mean([m[key] for m in mode_metrics.values()]) for key in metric_keys]
    bars = ax.bar(x + offsets[i] * width, vals, width, label=model, color=color)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.1%}", ha="center", va="bottom", fontsize=8)

ax.set_ylabel("Rate")
ax.set_title("ReasonIF: Average Metrics by Model")
ax.set_xticks(x)
ax.set_xticklabels(metric_names)
ax.legend(loc="upper right")
ax.set_ylim(0, ax.get_ylim()[1] + 0.05)
plt.tight_layout()
plt.savefig(ROLLOUT_DIR.parent / "summaries" / "reasonif_avg_metrics_2026_03_13.png", dpi=150)
print("Saved plot 1: reasonif_avg_metrics_2026_03_13.png")

# ---------------------------------------------------------------------------
# Plot 2: GPT-OSS 20B compliance by mode vs paper baselines
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

mode_labels_ordered = [MODE_LABELS[m] for m in MODE_ORDER]
x = np.arange(len(MODE_ORDER))
width = 0.35

ours = []
paper = []
for mode in MODE_ORDER:
    ours.append(all_data["GPT-OSS 20B"][mode]["compliance"])
    paper.append(PAPER_BASELINES[mode])

bars1 = ax.bar(x - width / 2, paper, width, label="Paper Baseline", color="#4C72B0")
bars2 = ax.bar(x + width / 2, ours, width, label="Our Reproduction", color="#55A868")

for bars, vals in [(bars1, paper), (bars2, ours)]:
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, max(bar.get_height(), 0) + 0.01,
                f"{val:.0%}", ha="center", va="bottom", fontsize=9)

# Add light gridlines so 0% bars are distinguishable from empty space
ax.yaxis.grid(True, alpha=0.3, linestyle="--")
ax.set_axisbelow(True)

ax.set_ylabel("Compliance Rate")
ax.set_title("GPT-OSS 20B: Compliance by Mode — Paper vs Our Reproduction")
ax.set_xticks(x)
ax.set_xticklabels(mode_labels_ordered, rotation=15, ha="right")
ax.legend()
ax.set_ylim(0, max(max(ours), max(paper)) + 0.15)
plt.tight_layout()
plt.savefig(ROLLOUT_DIR.parent / "summaries" / "reasonif_gpt_oss_20b_vs_paper_2026_03_13.png", dpi=150)
print("Saved plot 2: reasonif_gpt_oss_20b_vs_paper_2026_03_13.png")

# ---------------------------------------------------------------------------
# Plot 3: Compliance by mode for all three models
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(MODE_ORDER))
width = 0.25
offsets = np.arange(len(models)) - (len(models) - 1) / 2

colors_models = ["#4C72B0", "#55A868", "#C44E52"]

for i, (model, color) in enumerate(zip(models, colors_models)):
    vals = [all_data[model][mode]["compliance"] for mode in MODE_ORDER]
    bars = ax.bar(x + offsets[i] * width, vals, width, label=model, color=color)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, max(bar.get_height(), 0) + 0.01,
                f"{val:.0%}", ha="center", va="bottom", fontsize=8)

ax.set_ylabel("Compliance Rate")
ax.set_title("ReasonIF: Compliance by Mode and Model")
ax.set_xticks(x)
ax.set_xticklabels([MODE_LABELS[m] for m in MODE_ORDER], rotation=15, ha="right")
ax.legend()
ax.set_ylim(0, max(0.5, ax.get_ylim()[1] + 0.1))
plt.tight_layout()
plt.savefig(ROLLOUT_DIR.parent / "summaries" / "reasonif_compliance_by_mode_2026_03_13.png", dpi=150)
print("Saved plot 3: reasonif_compliance_by_mode_2026_03_13.png")

plt.show()
print("Done.")
