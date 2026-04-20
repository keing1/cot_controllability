"""Generate summary charts for compliance and reasoning limit rates."""

import json
import re
from datetime import date
from pathlib import Path
from collections import defaultdict
import glob

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.size"] = 11

EXCLUDE = {"deepseek/deepseek-r1", "openai/gpt-oss-120b"}
OUTDIR = Path("results/summaries/plots")
OUTDIR.mkdir(parents=True, exist_ok=True)
DATE = date.today().strftime("%Y_%m_%d")


def load_rollouts(f):
    rollouts = []
    for line in Path(f).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        try:
            rollouts.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rollouts


def get_config(f):
    header = Path(f).read_text().splitlines()[0]
    if header.startswith("#"):
        return json.loads(header[2:])
    return {}


def short_name(model: str) -> str:
    return model.split("/")[-1]


# --- Load and deduplicate runs ---
runs = {}  # (model, dataset) -> rollouts
for f in sorted(glob.glob("results/rollouts/*.jsonl")):
    if "debug" in f:
        continue
    config = get_config(f)
    model = config.get("model", "?")
    dataset = config.get("dataset", "?")
    if model in EXCLUDE:
        continue
    # Skip fine-tuned checkpoints: filename should start with model slug + dataset
    fname = Path(f).stem
    model_slug = model.replace("/", "_")
    expected_prefix = f"{model_slug}_{dataset}_"
    if not fname.startswith(expected_prefix):
        continue
    rollouts = load_rollouts(f)
    key = (model, dataset)
    if key not in runs or len(rollouts) > len(runs[key]):
        runs[key] = rollouts


def compute_compliance(rollouts, exclude_baseline=True):
    """Compliance rate excluding compliant=None."""
    filtered = rollouts
    if exclude_baseline:
        filtered = [r for r in filtered if r["control_mode"] != "baseline"]
    has_c = [r for r in filtered if r.get("compliant") is not None]
    if not has_c:
        return None
    return sum(1 for r in has_c if r["compliant"]) / len(has_c)



def compute_no_reasoning_rate(rollouts):
    """Fraction of rollouts with no reasoning trace."""
    if not rollouts:
        return None
    no_r = sum(1 for r in rollouts if not r.get("reasoning"))
    return no_r / len(rollouts)


# --- Gather metrics ---
models_seen = set()
compliance = defaultdict(dict)  # dataset -> model -> rate
no_reasoning = defaultdict(dict)  # dataset -> model -> rate

for (model, dataset), rollouts in runs.items():
    models_seen.add(model)
    c = compute_compliance(rollouts, exclude_baseline=(dataset == "cotcontrol"))
    if c is not None:
        compliance[dataset][model] = c
    nr = compute_no_reasoning_rate(rollouts)
    if nr is not None:
        no_reasoning[dataset][model] = nr


# Sort models: gpt-oss first, then qwen3, then qwen3.5, by size
def model_sort_key(m):
    name = m.lower()
    if "gpt-oss" in name:
        family = 0
    elif "qwen3.5" in name:
        family = 2
    elif "qwen3" in name:
        family = 1
    else:
        family = 3
    size_match = re.search(r"(\d+)b", name)
    size = int(size_match.group(1)) if size_match else 0
    return (family, size)


all_models = sorted(models_seen, key=model_sort_key)

# Color scheme
colors = {}
palette = plt.cm.tab10
for i, m in enumerate(all_models):
    colors[m] = palette(i)




# ============================================================
# Chart 1a: Compliance by model - CoTControl
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
models_cc = [m for m in all_models if m in compliance.get("cotcontrol", {})]
names_cc = [short_name(m) for m in models_cc]
vals_cc = [compliance["cotcontrol"][m] * 100 for m in models_cc]
bars = ax.bar(names_cc, vals_cc, color=[colors[m] for m in models_cc], edgecolor="white")
ax.set_ylabel("Compliance (%)")
ax.set_title("CoTControl Compliance by Model")
ax.set_ylim(0, max(vals_cc) * 1.3 if vals_cc else 10)
for bar, val in zip(bars, vals_cc):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
fname = OUTDIR / f"compliance_by_model_cotcontrol_{DATE}.png"
fig.savefig(fname, dpi=150)
print(f"Saved {fname}")
plt.close()


# ============================================================
# Chart 1b: Compliance by model - ReasonIF
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
models_rif = [m for m in all_models if m in compliance.get("reasonif", {})]
names_rif = [short_name(m) for m in models_rif]
vals_rif = [compliance["reasonif"][m] * 100 for m in models_rif]
bars = ax.bar(names_rif, vals_rif, color=[colors[m] for m in models_rif], edgecolor="white")
ax.set_ylabel("Compliance (%)")
ax.set_title("ReasonIF Compliance by Model")
ax.set_ylim(0, max(vals_rif) * 1.3 if vals_rif else 10)
for bar, val in zip(bars, vals_rif):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
fname = OUTDIR / f"compliance_by_model_reasonif_{DATE}.png"
fig.savefig(fname, dpi=150)
print(f"Saved {fname}")
plt.close()


# ============================================================
# Chart 2: Dot plot - ReasonIF compliance (x) vs CoTControl compliance (y)
# ============================================================
fig, ax = plt.subplots(figsize=(7, 7))
both_models = [m for m in all_models
               if m in compliance.get("reasonif", {}) and m in compliance.get("cotcontrol", {})]
for m in both_models:
    x = compliance["reasonif"][m] * 100
    y = compliance["cotcontrol"][m] * 100
    ax.scatter(x, y, color=colors[m], s=100, zorder=5, edgecolors="black", linewidths=0.5)
    ax.annotate(short_name(m), (x, y), textcoords="offset points",
                xytext=(8, 4), fontsize=8)

ax.set_xlabel("ReasonIF Compliance (%)")
ax.set_ylabel("CoTControl Compliance (%)")
ax.set_title("Compliance: ReasonIF vs CoTControl")
lim = max(
    max(compliance["reasonif"][m] for m in both_models) * 100,
    max(compliance["cotcontrol"][m] for m in both_models) * 100,
) * 1.2
ax.plot([0, lim], [0, lim], "k--", alpha=0.2, linewidth=1)
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.set_aspect("equal")
plt.tight_layout()
fname = OUTDIR / f"compliance_reasonif_vs_cotcontrol_{DATE}.png"
fig.savefig(fname, dpi=150)
print(f"Saved {fname}")
plt.close()


# ============================================================
# Chart 3: Grouped bar plot - reasoning limit rate by model
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
both_nr = [m for m in all_models
           if m in no_reasoning.get("cotcontrol", {}) or m in no_reasoning.get("reasonif", {})]
x_pos = np.arange(len(both_nr))
width = 0.35

vals_cc_nr = [no_reasoning.get("cotcontrol", {}).get(m, 0) * 100 for m in both_nr]
vals_rif_nr = [no_reasoning.get("reasonif", {}).get(m, 0) * 100 for m in both_nr]

bars1 = ax.bar(x_pos - width / 2, vals_cc_nr, width, label="CoTControl", color="#4C72B0", edgecolor="white")
bars2 = ax.bar(x_pos + width / 2, vals_rif_nr, width, label="ReasonIF", color="#DD8452", edgecolor="white")

for bar, val in zip(bars1, vals_cc_nr):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=8)
for bar, val in zip(bars2, vals_rif_nr):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=8)

ax.set_ylabel("No Reasoning Rate (%)")
ax.set_title("Reasoning Limit Rate by Model and Dataset")
ax.set_xticks(x_pos)
ax.set_xticklabels([short_name(m) for m in both_nr], rotation=30, ha="right")
ax.legend()
ax.set_ylim(0, max(max(vals_cc_nr), max(vals_rif_nr)) * 1.25)
plt.tight_layout()
fname = OUTDIR / f"reasoning_limit_rate_by_model_{DATE}.png"
fig.savefig(fname, dpi=150)
print(f"Saved {fname}")
plt.close()
