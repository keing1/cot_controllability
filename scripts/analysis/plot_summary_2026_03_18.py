"""Generate summary plots for controllability evaluation (2026-03-18).

Reads from:
  - results/summaries/tables/summary_by_run.csv
  - results/summaries/tables/summary_by_instruction.csv

Outputs to:
  - results/summaries/plots/{DATE}/

Filtering: max_tokens > 20K, except gpt-oss-20b base (no checkpoint).
Model order: gpt-oss, qwen3, qwen3.5, kimi — increasing total params within group.
"""

import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.size"] = 11

SUMMARY_BY_RUN = Path("results/summaries/tables/summary_by_run.csv")
SUMMARY_BY_INSTRUCTION = Path("results/summaries/tables/summary_by_instruction.csv")
DATE = "2026_03_18"
OUTDIR = Path(f"results/summaries/plots/{DATE}")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Canonical model order: gpt-oss, qwen3, qwen3.5, kimi (increasing total params).
MODEL_ORDER = [
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

# Indices where a new model family begins (for visual separators).
GROUP_STARTS = [2, 5, 9]  # qwen3 starts at 2, qwen3.5 at 5, kimi at 9


# ── Helpers ──────────────────────────────────────────────────────────────


def short_name(m: str) -> str:
    return m.split("/")[-1]


def model_key(m: str) -> int:
    try:
        return MODEL_ORDER.index(m)
    except ValueError:
        return len(MODEL_ORDER)


def include_row(row: dict) -> bool:
    mt = int(row["max_tokens"]) if row["max_tokens"] else 0
    cp = (row.get("checkpoint") or "").strip()
    if "gpt-oss-20b" in row["base_model"] and not cp:
        return True
    return mt > 20000


def safe_float(row: dict | None, col: str):
    if row is None:
        return None
    v = row.get(col, "")
    return float(v) if v else None


def add_group_seps(ax, models_list):
    """Add faint vertical lines between model family groups."""
    model_indices = {m: i for i, m in enumerate(models_list)}
    for gs in GROUP_STARTS:
        if gs < len(models_list) and models_list[gs] in model_indices:
            ax.axvline(x=gs - 0.5, color="gray", linewidth=0.5, alpha=0.3)


# ── Data loading ─────────────────────────────────────────────────────────


def load_runs():
    """One row per (model, eval_type), latest run wins."""
    with open(SUMMARY_BY_RUN) as f:
        rows = [r for r in csv.DictReader(f) if include_row(r)]
    best: dict[tuple, dict] = {}
    for r in rows:
        k = (r["base_model"], r["eval_type"])
        if k not in best or r["timestamp"] > best[k]["timestamp"]:
            best[k] = r
    return best


def load_instructions():
    """One row per (model, eval_type, instruction_type), latest run wins."""
    with open(SUMMARY_BY_INSTRUCTION) as f:
        rows = [r for r in csv.DictReader(f) if include_row(r)]
    best: dict[tuple, dict] = {}
    for r in rows:
        k = (r["base_model"], r["eval_type"], r["instruction_type"])
        if k not in best or r["timestamp"] > best[k]["timestamp"]:
            best[k] = r
    return best


runs = load_runs()
instr = load_instructions()
models = sorted({k[0] for k in runs}, key=model_key)

# Fixed color palette per model.
cmap = matplotlib.colormaps.get_cmap("tab20")
COLORS = {m: cmap(i / max(len(models) - 1, 1)) for i, m in enumerate(models)}


# ── Helper for single-metric bar chart ───────────────────────────────────


def bar_chart(metric_col: str, eval_type: str, ylabel: str, title: str, fname: str):
    fig, ax = plt.subplots(figsize=(12, 5))
    ms = [m for m in models if (m, eval_type) in runs]
    vals = [(safe_float(runs[(m, eval_type)], metric_col) or 0) * 100 for m in ms]
    names = [short_name(m) for m in ms]

    bars = ax.bar(range(len(ms)), vals, color=[COLORS[m] for m in ms], edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{v:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(range(len(ms)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0, max(vals) * 1.3 if vals and max(vals) > 0 else 10)
    add_group_seps(ax, ms)
    plt.tight_layout()
    fig.savefig(OUTDIR / fname, dpi=150)
    print(f"Saved {OUTDIR / fname}")
    plt.close()


# ── Helper for grouped bar (models on x-axis, bars for each eval_type) ──


def grouped_bar_by_eval(metric_col: str, ylabel: str, title: str, fname: str):
    fig, ax = plt.subplots(figsize=(12, 5))
    eval_types = ["cotcontrol", "reasonif"]
    et_colors = {"cotcontrol": "#4C72B0", "reasonif": "#DD8452"}
    width = 0.35

    x = np.arange(len(models))
    for i, et in enumerate(eval_types):
        vals = [(safe_float(runs.get((m, et)), metric_col) or 0) * 100 for m in models]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=et, color=et_colors[et], edgecolor="white")
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{v:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([short_name(m) for m in models], rotation=30, ha="right")
    ax.legend()
    ymax = max(
        max((safe_float(runs.get((m, et)), metric_col) or 0) for m in models)
        for et in eval_types
    ) * 100
    ax.set_ylim(0, ymax * 1.25 if ymax > 0 else 10)
    add_group_seps(ax, models)
    plt.tight_layout()
    fig.savefig(OUTDIR / fname, dpi=150)
    print(f"Saved {OUTDIR / fname}")
    plt.close()


# ════════════════════════════════════════════════════════════════════════
# 1a: Compliance by model — CoTControl
# ════════════════════════════════════════════════════════════════════════

bar_chart(
    "compliant_rate",
    "cotcontrol",
    "Compliance (%)",
    "CoTControl Compliance by Model",
    f"compliance_by_model_cotcontrol_{DATE}.png",
)

# ════════════════════════════════════════════════════════════════════════
# 1b: Compliance by model — ReasonIF
# ════════════════════════════════════════════════════════════════════════

bar_chart(
    "compliant_rate",
    "reasonif",
    "Compliance (%)",
    "ReasonIF Compliance by Model",
    f"compliance_by_model_reasonif_{DATE}.png",
)

# ════════════════════════════════════════════════════════════════════════
# 1c: 2D scatter — CoTControl vs ReasonIF compliance
# ════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(7, 7))
both = [m for m in models if (m, "cotcontrol") in runs and (m, "reasonif") in runs]
for m in both:
    x = (safe_float(runs[(m, "reasonif")], "compliant_rate") or 0) * 100
    y = (safe_float(runs[(m, "cotcontrol")], "compliant_rate") or 0) * 100
    ax.scatter(x, y, color=COLORS[m], s=100, zorder=5, edgecolors="black", linewidths=0.5)
    ax.annotate(short_name(m), (x, y), textcoords="offset points", xytext=(8, 4), fontsize=8)

ax.set_xlabel("ReasonIF Compliance (%)")
ax.set_ylabel("CoTControl Compliance (%)")
ax.set_title("Compliance: ReasonIF vs CoTControl")
all_vals = (
    [(safe_float(runs[(m, "reasonif")], "compliant_rate") or 0) * 100 for m in both]
    + [(safe_float(runs[(m, "cotcontrol")], "compliant_rate") or 0) * 100 for m in both]
)
lim = max(all_vals) * 1.2 if all_vals else 10
ax.plot([0, lim], [0, lim], "k--", alpha=0.2, linewidth=1)
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.set_aspect("equal")
plt.tight_layout()
fname = OUTDIR / f"compliance_reasonif_vs_cotcontrol_{DATE}.png"
fig.savefig(fname, dpi=150)
print(f"Saved {fname}")
plt.close()

# ════════════════════════════════════════════════════════════════════════
# 1d: Reasoning error rate — grouped bar by eval type
# ════════════════════════════════════════════════════════════════════════

grouped_bar_by_eval(
    "reasoning_error_rate",
    "Reasoning Error Rate (%)",
    "Reasoning Error Rate by Model and Eval Type",
    f"reasoning_error_rate_by_model_{DATE}.png",
)

# ════════════════════════════════════════════════════════════════════════
# 2a: Accuracy — grouped bar (eval type groups, model bars)
# ════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(14, 6))
eval_types = ["cotcontrol", "reasonif"]
n = len(models)
bar_width = 0.8 / n
x = np.arange(len(eval_types))

for i, m in enumerate(models):
    offset = (i - n / 2 + 0.5) * bar_width
    vals = [(safe_float(runs.get((m, et)), "correct_rate") or 0) * 100 for et in eval_types]
    ax.bar(x + offset, vals, bar_width, label=short_name(m), color=COLORS[m], edgecolor="white", linewidth=0.5)

ax.set_ylabel("Accuracy (%)")
ax.set_title("Accuracy by Eval Type and Model")
ax.set_xticks(x)
ax.set_xticklabels(eval_types, fontsize=12)
ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
plt.tight_layout()
fname = OUTDIR / f"accuracy_by_eval_type_{DATE}.png"
fig.savefig(fname, dpi=150)
print(f"Saved {fname}")
plt.close()

# ════════════════════════════════════════════════════════════════════════
# 2b: Meta discussion rate — grouped bar (eval type groups, model bars)
# ════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(14, 6))

for i, m in enumerate(models):
    offset = (i - n / 2 + 0.5) * bar_width
    vals = [(safe_float(runs.get((m, et)), "meta_discussion_rate") or 0) * 100 for et in eval_types]
    ax.bar(x + offset, vals, bar_width, label=short_name(m), color=COLORS[m], edgecolor="white", linewidth=0.5)

ax.set_ylabel("Meta Discussion Rate (%)")
ax.set_title("Meta Discussion Rate by Eval Type and Model")
ax.set_xticks(x)
ax.set_xticklabels(eval_types, fontsize=12)
ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
plt.tight_layout()
fname = OUTDIR / f"meta_discussion_by_eval_type_{DATE}.png"
fig.savefig(fname, dpi=150)
print(f"Saved {fname}")
plt.close()

# ════════════════════════════════════════════════════════════════════════
# 3a: Compliance by instruction type — CoTControl (per model)
# ════════════════════════════════════════════════════════════════════════

COTCONTROL_TYPES = sorted([
    "alternating_case", "end_of_sentence", "ignore_question",
    "lowercase_thinking", "meow_between_words", "multiple_word_suppression",
    "repeat_sentences", "uppercase_thinking", "word_suppression",
])
REASONIF_TYPES = sorted([
    "end_checker", "english_capital", "json_format",
    "no_comma", "number_words", "reasoning_language",
])

itype_cmap = matplotlib.colormaps.get_cmap("tab20")


def compliance_by_instruction(eval_type: str, instruction_types: list[str], title: str, fname: str):
    ms = [m for m in models if (m, eval_type) in runs]
    n_models = len(ms)
    n_itypes = len(instruction_types)
    bar_w = 0.8 / n_itypes
    itype_colors = {t: itype_cmap(i / max(n_itypes - 1, 1)) for i, t in enumerate(instruction_types)}

    fig, ax = plt.subplots(figsize=(max(14, n_models * 1.2), 6))
    x = np.arange(n_models)

    for i, itype in enumerate(instruction_types):
        offset = (i - n_itypes / 2 + 0.5) * bar_w
        vals = []
        for m in ms:
            row = instr.get((m, eval_type, itype))
            vals.append((safe_float(row, "compliant_rate") or 0) * 100)
        ax.bar(x + offset, vals, bar_w, label=itype, color=itype_colors[itype], edgecolor="white", linewidth=0.3)

    ax.set_ylabel("Compliance (%)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([short_name(m) for m in ms], rotation=30, ha="right")
    ax.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.9)
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.1 if ymax > 0 else 10)
    add_group_seps(ax, ms)
    plt.tight_layout()
    fig.savefig(OUTDIR / fname, dpi=150)
    print(f"Saved {OUTDIR / fname}")
    plt.close()


compliance_by_instruction(
    "cotcontrol",
    COTCONTROL_TYPES,
    "CoTControl Compliance by Instruction Type and Model",
    f"compliance_by_instruction_cotcontrol_{DATE}.png",
)

# ════════════════════════════════════════════════════════════════════════
# 3b: Compliance by instruction type — ReasonIF (per model)
# ════════════════════════════════════════════════════════════════════════

compliance_by_instruction(
    "reasonif",
    REASONIF_TYPES,
    "ReasonIF Compliance by Instruction Type and Model",
    f"compliance_by_instruction_reasonif_{DATE}.png",
)

# ════════════════════════════════════════════════════════════════════════
# 4: Instruction-type compliance correlation across models
# ════════════════════════════════════════════════════════════════════════

# Build compliance vectors: instruction_type -> {model: rate}
compliance_by_type: dict[str, dict[str, float]] = {}
for (base_model, eval_type, itype), row in instr.items():
    if itype == "baseline":
        continue
    val = safe_float(row, "compliant_rate")
    if val is None:
        continue
    compliance_by_type.setdefault(itype, {})[base_model] = val

# Filter out instruction types with zero compliance across all models,
# or where only one model has a non-zero compliance score.
nonzero_cc = [t for t in COTCONTROL_TYPES if sum(v > 0 for v in compliance_by_type.get(t, {}).values()) >= 2]
nonzero_rif = [t for t in REASONIF_TYPES if sum(v > 0 for v in compliance_by_type.get(t, {}).values()) >= 2]
all_types = nonzero_cc + nonzero_rif
n_cc = len(nonzero_cc)
n_types = len(all_types)

# Compute pairwise Pearson correlation.
corr = np.full((n_types, n_types), np.nan)
for i, ti in enumerate(all_types):
    for j, tj in enumerate(all_types):
        vi = compliance_by_type.get(ti, {})
        vj = compliance_by_type.get(tj, {})
        common = sorted(set(vi) & set(vj))
        if len(common) < 3:
            continue
        a = np.array([vi[m] for m in common])
        b = np.array([vj[m] for m in common])
        # Avoid NaN from constant arrays.
        if np.std(a) < 1e-9 or np.std(b) < 1e-9:
            corr[i, j] = 0.0
        else:
            corr[i, j] = np.corrcoef(a, b)[0, 1]

fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")

# Thick separator line between cotcontrol and reasonif blocks.
ax.axhline(y=n_cc - 0.5, color="black", linewidth=2)
ax.axvline(x=n_cc - 0.5, color="black", linewidth=2)

# Labels.
ax.set_xticks(range(n_types))
ax.set_xticklabels(all_types, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(n_types))
ax.set_yticklabels(all_types, fontsize=8)

# Bracket labels for the two eval groups.
ax.text(n_cc / 2, -1.5, "CoTControl", ha="center", fontsize=10, fontweight="bold")
ax.text(n_cc + len(nonzero_rif) / 2, -1.5, "ReasonIF", ha="center", fontsize=10, fontweight="bold")

cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Pearson r")
ax.set_title("Instruction-Type Compliance Correlation Across Models", pad=30)
plt.tight_layout()
fname = OUTDIR / f"instruction_type_compliance_correlation_{DATE}.png"
fig.savefig(fname, dpi=150, bbox_inches="tight")
print(f"Saved {fname}")
plt.close()

# ════════════════════════════════════════════════════════════════════════
# 5: 3×3 class-level correlation grid (style / suppression / addition)
# ════════════════════════════════════════════════════════════════════════

from controllability.constants import INSTRUCTION_CLASS

CLASS_ORDER = ["style", "suppression", "addition"]

# Build a lookup from instruction type index in all_types to its class.
type_to_idx = {t: i for i, t in enumerate(all_types)}

# Group the filtered instruction types by class.
class_types: dict[str, list[str]] = {c: [] for c in CLASS_ORDER}
for t in all_types:
    cls = INSTRUCTION_CLASS.get(t)
    if cls in class_types:
        class_types[cls].append(t)

# Compute mean pairwise correlation for each class pair, excluding self-correlations.
class_corr = np.full((3, 3), np.nan)
for ci, c1 in enumerate(CLASS_ORDER):
    for cj, c2 in enumerate(CLASS_ORDER):
        pairs = []
        for ti in class_types[c1]:
            for tj in class_types[c2]:
                if ti == tj:
                    continue
                i, j = type_to_idx[ti], type_to_idx[tj]
                if not np.isnan(corr[i, j]):
                    pairs.append(corr[i, j])
        if pairs:
            class_corr[ci, cj] = np.mean(pairs)

fig, ax = plt.subplots(figsize=(5, 5))
im = ax.imshow(class_corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")

# Annotate cells with values.
for i in range(3):
    for j in range(3):
        val = class_corr[i, j]
        if not np.isnan(val):
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=14, color=color)

ax.set_xticks(range(3))
ax.set_xticklabels(CLASS_ORDER, fontsize=11)
ax.set_yticks(range(3))
ax.set_yticklabels(CLASS_ORDER, fontsize=11)
cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Mean Pearson r")
ax.set_title("Compliance Correlation by Instruction Class", pad=12)
plt.tight_layout()
fname = OUTDIR / f"instruction_class_correlation_{DATE}.png"
fig.savefig(fname, dpi=150, bbox_inches="tight")
print(f"Saved {fname}")
plt.close()

# ════════════════════════════════════════════════════════════════════════
# 6: Full correlation heatmap grouped by instruction class
# ════════════════════════════════════════════════════════════════════════

# Order instruction types by class instead of by eval type.
class_ordered_types = class_types["style"] + class_types["suppression"] + class_types["addition"]
n_style = len(class_types["style"])
n_suppression = len(class_types["suppression"])
n_class_types = len(class_ordered_types)

# Build reordered correlation matrix.
corr_by_class = np.full((n_class_types, n_class_types), np.nan)
for i, ti in enumerate(class_ordered_types):
    for j, tj in enumerate(class_ordered_types):
        oi, oj = type_to_idx[ti], type_to_idx[tj]
        corr_by_class[i, j] = corr[oi, oj]

fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(corr_by_class, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")

# Separator lines between class groups.
for boundary in [n_style - 0.5, n_style + n_suppression - 0.5]:
    ax.axhline(y=boundary, color="black", linewidth=2)
    ax.axvline(x=boundary, color="black", linewidth=2)

# Labels.
ax.set_xticks(range(n_class_types))
ax.set_xticklabels(class_ordered_types, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(n_class_types))
ax.set_yticklabels(class_ordered_types, fontsize=8)

# Bracket labels for the three class groups.
ax.text(n_style / 2, -1.5, "style", ha="center", fontsize=10, fontweight="bold")
ax.text(n_style + n_suppression / 2, -1.5, "suppression", ha="center", fontsize=10, fontweight="bold")
ax.text(n_style + n_suppression + len(class_types["addition"]) / 2, -1.5, "addition", ha="center", fontsize=10, fontweight="bold")

cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Pearson r")
ax.set_title("Instruction-Type Compliance Correlation (Grouped by Class)", pad=30)
plt.tight_layout()
fname = OUTDIR / f"instruction_type_correlation_by_class_{DATE}.png"
fig.savefig(fname, dpi=150, bbox_inches="tight")
print(f"Saved {fname}")
plt.close()

print("\nDone — all plots saved.")
