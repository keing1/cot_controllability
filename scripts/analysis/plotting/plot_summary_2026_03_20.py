#!/usr/bin/env python3
"""Generate summary plots for controllability evaluation (2026-03-20).

Reads from:
  - results/summaries/tables/summary_by_run.csv
  - results/summaries/tables/summary_by_instruction.csv

Outputs to:
  - results/summaries/plots/2026_03_20/

Plot Sets:
  1. Reproduced March 18 plots with 80% CI error bars:
     - 1a/1b: Compliance by model (one per eval suite)
     - 1c/1d: Compliance by instruction type (one per eval suite)
     - 1e: Meta discussion rate by eval type (grouped bar)
     - 1f: Reasoning error rate by eval type (grouped bar)

  2. Parameter scaling — compliance vs model size (base vs FT step60):
     - 2a/2b: One per eval suite, qwen3 and gpt-oss families

  3. Effort sweep — compliance vs reasoning effort (base vs FT step60):
     - 3: 2x2 grid (model x eval_type)

Data selection:
  - Set 1: base models at canonical effort (medium for gpt-oss, none for others),
    max_tokens > 20K except gpt-oss-20b base
  - FT: lr=1e-4 step-60 checkpoints from 20260319 run
  - Error bars: 80% CI (z=1.282) via binomial SE
"""

import csv
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.size"] = 11

SUMMARY_BY_RUN = Path("results/summaries/tables/summary_by_run.csv")
SUMMARY_BY_INSTRUCTION = Path("results/summaries/tables/summary_by_instruction.csv")
DATE = "2026_03_20"
OUTDIR = Path(f"results/summaries/plots/{DATE}")
OUTDIR.mkdir(parents=True, exist_ok=True)

# 80% CI z-score
Z80 = 1.282

# Same model order as March 18.
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

# Display names for eval types (not all-caps).
EVAL_DISPLAY = {"cotcontrol": "CoTControl", "reasonif": "ReasonIF"}


# ── Helpers ──────────────────────────────────────────────────────────────


def short_name(m: str) -> str:
    return m.split("/")[-1]


def model_key(m: str) -> int:
    try:
        return MODEL_ORDER.index(m)
    except ValueError:
        return len(MODEL_ORDER)


def include_row(row: dict) -> bool:
    """March 18 filter: max_tokens > 20K, except gpt-oss-20b base."""
    mt = int(row["max_tokens"]) if row["max_tokens"] else 0
    cp = (row.get("checkpoint") or "").strip()
    if "gpt-oss-20b" in row["base_model"] and not cp:
        return True
    return mt > 20000


def is_canonical_effort(row: dict) -> bool:
    """Check if the row uses the model's canonical reasoning effort."""
    effort = row.get("reasoning_effort", "")
    if "gpt-oss" in row["base_model"]:
        return effort == "medium"
    return effort in ("none", "")


def sf(row: dict | None, col: str) -> float | None:
    """Safe float extraction."""
    if row is None:
        return None
    v = row.get(col, "")
    return float(v) if v else None


def si(row: dict | None, col: str) -> int:
    """Safe int extraction."""
    if row is None:
        return 0
    v = row.get(col, "")
    return int(v) if v else 0


def ci80(p: float | None, n: int) -> float:
    """80% CI half-width for binomial proportion."""
    if n <= 0 or p is None:
        return 0.0
    return Z80 * math.sqrt(p * (1 - p) / n)


def add_group_seps(ax, models_list):
    """Add faint vertical lines between model family groups."""
    for gs in GROUP_STARTS:
        if gs < len(models_list):
            ax.axvline(x=gs - 0.5, color="gray", linewidth=0.5, alpha=0.3)


# ── Data loading ─────────────────────────────────────────────────────────


def load_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def best_by_key(rows, key_fn, filter_fn=None, min_clean=10):
    """Dict of key -> row, keeping latest timestamp per key.

    Skips rows with n_clean < min_clean to avoid partial / debug runs.
    """
    best: dict[tuple, dict] = {}
    for r in rows:
        if filter_fn and not filter_fn(r):
            continue
        if si(r, "n_clean") < min_clean:
            continue
        k = key_fn(r)
        if k not in best or r["timestamp"] > best[k]["timestamp"]:
            best[k] = r
    return best


all_runs = load_csv(SUMMARY_BY_RUN)
all_instr = load_csv(SUMMARY_BY_INSTRUCTION)

# ── Set 1 data: base models at canonical effort ──

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

models = sorted({k[0] for k in base_runs}, key=model_key)

# Fixed color palette.
cmap = matplotlib.colormaps.get_cmap("tab20")
COLORS = {m: cmap(i / max(len(models) - 1, 1)) for i, m in enumerate(models)}

# ── Set 2 data: FT step60 from lr=1e-4 run (20260319) ──

ft_step60_runs = best_by_key(
    all_runs,
    key_fn=lambda r: (r["base_model"], r["eval_type"]),
    filter_fn=lambda r: (
        "rif-000060" in r.get("file_name", "")
        and "20260319" in r.get("file_name", "")
    ),
)

# Base models for Set 2 — relax max_tokens filter for qwen3-8b
base_any = best_by_key(
    all_runs,
    key_fn=lambda r: (r["base_model"], r["eval_type"]),
    filter_fn=lambda r: (
        not (r.get("checkpoint") or "").strip()
        and is_canonical_effort(r)
    ),
)

# Prefer 28K runs for gpt-oss base (effort sweep results)
gpt_base_28k = best_by_key(
    all_runs,
    key_fn=lambda r: (r["base_model"], r["eval_type"]),
    filter_fn=lambda r: (
        not (r.get("checkpoint") or "").strip()
        and "gpt-oss" in r["base_model"]
        and r.get("max_tokens") == "28000"
        and r.get("reasoning_effort") == "medium"
        and si(r, "n_clean") > 0
    ),
)

# ── Set 3 data: effort sweep (base vs FT) ──

# Base gpt-oss at different efforts (28K, no checkpoint)
base_effort = best_by_key(
    all_runs,
    key_fn=lambda r: (r["base_model"], r["eval_type"], r["reasoning_effort"]),
    filter_fn=lambda r: (
        not (r.get("checkpoint") or "").strip()
        and "gpt-oss" in r["base_model"]
        and r.get("max_tokens") == "28000"
        and si(r, "n_clean") > 0
    ),
)

# FT step60 at different efforts — match by tinker checkpoint path
STEP60_SESSION_IDS = {
    "openai/gpt-oss-120b": "1cbfc28d",
    "openai/gpt-oss-20b": "6adcf41e",
}
ft_effort: dict[tuple, dict] = {}
for r in all_runs:
    cp = (r.get("checkpoint") or "").strip()
    if not cp or "000060" not in cp:
        continue
    model = r["base_model"]
    sid = STEP60_SESSION_IDS.get(model)
    if sid and sid in cp:
        k = (model, r["eval_type"], r["reasoning_effort"])
        if k not in ft_effort or r["timestamp"] > ft_effort[k]["timestamp"]:
            ft_effort[k] = r


# ── Verify data selection ────────────────────────────────────────────────

print("=" * 70)
print("DATA VERIFICATION")
print("=" * 70)

print(f"\nSet 1 — Base models: {len(models)}")
for m in models:
    for et in ["cotcontrol", "reasonif"]:
        row = base_runs.get((m, et))
        if row:
            print(f"  {m:35s} {et:12s} compliance={sf(row, 'compliant_rate'):.4f}"
                  f"  n_clean={si(row, 'n_clean'):5d}  effort={row.get('reasoning_effort','')}"
                  f"  max_tokens={row.get('max_tokens','')}")

print(f"\nSet 2 — FT step60 (lr=1e-4, 20260319):")
for k, row in sorted(ft_step60_runs.items()):
    print(f"  {k[0]:35s} {k[1]:12s} compliance={sf(row, 'compliant_rate'):.4f}"
          f"  n_clean={si(row, 'n_clean'):5d}")

print(f"\nSet 3 — Base effort sweep:")
for k in sorted(base_effort):
    row = base_effort[k]
    print(f"  {k[0]:25s} {k[2]:8s} {k[1]:12s} compliance={sf(row, 'compliant_rate'):.4f}"
          f"  n_clean={si(row, 'n_clean'):5d}")

print(f"\nSet 3 — FT effort sweep:")
for k in sorted(ft_effort):
    row = ft_effort[k]
    print(f"  {k[0]:25s} {k[2]:8s} {k[1]:12s} compliance={sf(row, 'compliant_rate'):.4f}"
          f"  n_clean={si(row, 'n_clean'):5d}")

print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════
# SET 1: Reproduced March 18 plots with 80% CI error bars
# ══════════════════════════════════════════════════════════════════════════


# ── 1a/1b: Compliance by model ───────────────────────────────────────────

for eval_type in ["cotcontrol", "reasonif"]:
    fig, ax = plt.subplots(figsize=(12, 5))
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
        color=[COLORS[m] for m in ms], edgecolor="white",
        error_kw=dict(lw=1.5, capthick=1.2),
    )
    for bar, v, e in zip(bars, vals, errs):
        label_y = v + e + 0.5  # always above the error bar
        ax.text(
            bar.get_x() + bar.get_width() / 2, label_y,
            f"{v:.1f}%", ha="center", va="bottom", fontsize=9,
        )
    ax.set_ylabel("Compliance (%)")
    ax.set_title(f"{EVAL_DISPLAY[eval_type]} Compliance by Model (80% CI)")
    ax.set_xticks(range(len(ms)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    top = max(v + e for v, e in zip(vals, errs)) if vals else 10
    ax.set_ylim(0, top * 1.25)
    add_group_seps(ax, ms)
    plt.tight_layout()
    fname = f"compliance_by_model_{eval_type}_{DATE}.png"
    fig.savefig(OUTDIR / fname, dpi=150)
    print(f"Saved {OUTDIR / fname}")
    plt.close()


# ── 1c/1d: Compliance by instruction type ────────────────────────────────

itype_cmap = matplotlib.colormaps.get_cmap("tab20")

for eval_type, itypes in [("cotcontrol", COTCONTROL_TYPES), ("reasonif", REASONIF_TYPES)]:
    ms = [m for m in models if (m, eval_type) in base_runs]
    n_models = len(ms)
    n_itypes = len(itypes)
    bar_w = 0.8 / n_itypes
    itype_colors = {t: itype_cmap(i / max(n_itypes - 1, 1)) for i, t in enumerate(itypes)}

    fig, ax = plt.subplots(figsize=(max(14, n_models * 1.2), 6))
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

    ax.set_ylabel("Compliance (%)")
    ax.set_title(f"{EVAL_DISPLAY[eval_type]} Compliance by Instruction Type (80% CI)")
    ax.set_xticks(x)
    ax.set_xticklabels([short_name(m) for m in ms], rotation=30, ha="right")
    ax.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.9)
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.1 if ymax > 0 else 10)
    add_group_seps(ax, ms)
    plt.tight_layout()
    fname = f"compliance_by_instruction_{eval_type}_{DATE}.png"
    fig.savefig(OUTDIR / fname, dpi=150)
    print(f"Saved {OUTDIR / fname}")
    plt.close()


# ── 1e/1f: Meta discussion and reasoning error rate ──────────────────────

for metric, ylabel, title_label, fname_pfx in [
    ("meta_discussion_rate", "Meta Discussion Rate (%)", "Meta Discussion Rate", "meta_discussion"),
    ("reasoning_error_rate", "Reasoning Error Rate (%)", "Reasoning Error Rate", "reasoning_error_rate"),
]:
    fig, ax = plt.subplots(figsize=(12, 5))
    eval_types = ["cotcontrol", "reasonif"]
    et_colors = {"cotcontrol": "#4C72B0", "reasonif": "#DD8452"}
    width = 0.35
    x = np.arange(len(models))

    for i, et in enumerate(eval_types):
        vals, errs_list = [], []
        for m in models:
            row = base_runs.get((m, et))
            p = sf(row, metric) or 0
            # reasoning_error_rate denominator is n_rollouts_actual; others use n_clean
            n = si(row, "n_rollouts_actual") if metric == "reasoning_error_rate" else si(row, "n_clean")
            vals.append(p * 100)
            errs_list.append(ci80(p, n) * 100)
        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset, vals, width, yerr=errs_list, capsize=3,
            label=EVAL_DISPLAY[et], color=et_colors[et], edgecolor="white",
            error_kw=dict(lw=1, capthick=0.8),
        )
        if metric != "meta_discussion_rate":
            for bar, v, e in zip(bars, vals, errs_list):
                if v > 0:
                    label_y = v + e + 0.5
                    ax.text(
                        bar.get_x() + bar.get_width() / 2, label_y,
                        f"{v:.0f}%", ha="center", va="bottom", fontsize=8,
                    )

    ax.set_ylabel(ylabel)
    ax.set_title(f"{title_label} by Model and Eval Suite (80% CI)")
    ax.set_xticks(x)
    ax.set_xticklabels([short_name(m) for m in models], rotation=30, ha="right")
    ax.legend()
    ymax = max(
        max((sf(base_runs.get((m, et)), metric) or 0) for m in models)
        for et in eval_types
    ) * 100
    ax.set_ylim(0, min(ymax * 1.35, 100) if ymax > 0 else 10)
    add_group_seps(ax, models)
    plt.tight_layout()
    fname = f"{fname_pfx}_by_model_{DATE}.png"
    fig.savefig(OUTDIR / fname, dpi=150)
    print(f"Saved {OUTDIR / fname}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# SET 2: Parameter scaling — compliance vs model size (base vs FT)
# ══════════════════════════════════════════════════════════════════════════

PARAM_FAMILIES = {
    "qwen3": [("qwen/qwen3-8b", 8), ("qwen/qwen3-32b", 32)],
    "gpt-oss": [("openai/gpt-oss-20b", 20), ("openai/gpt-oss-120b", 120)],
}
FAMILY_COLORS = {"qwen3": "#2196F3", "gpt-oss": "#E91E63"}

for eval_type in ["cotcontrol", "reasonif"]:
    fig, ax = plt.subplots(figsize=(8, 6))

    for family, model_params in PARAM_FAMILIES.items():
        color = FAMILY_COLORS[family]

        # Base line (solid)
        bx, by, be = [], [], []
        for model, params in model_params:
            if "gpt-oss" in model:
                row = gpt_base_28k.get((model, eval_type))
            else:
                row = base_runs.get((model, eval_type)) or base_any.get((model, eval_type))
            if row:
                p = sf(row, "compliant_rate") or 0
                n = si(row, "n_clean")
                bx.append(params)
                by.append(p * 100)
                be.append(ci80(p, n) * 100)
        if bx:
            ax.errorbar(
                bx, by, yerr=be, marker="o", linestyle="-", color=color,
                label=f"{family} base", capsize=5, lw=2, markersize=8,
            )

        # FT step60 line (dashed)
        fx, fy, fe = [], [], []
        for model, params in model_params:
            row = ft_step60_runs.get((model, eval_type))
            if row:
                p = sf(row, "compliant_rate") or 0
                n = si(row, "n_clean")
                fx.append(params)
                fy.append(p * 100)
                fe.append(ci80(p, n) * 100)
        if fx:
            ax.errorbar(
                fx, fy, yerr=fe, marker="s", linestyle="--", color=color,
                label=f"{family} FT", capsize=5, lw=2, markersize=8,
            )

    ax.set_xlabel("Parameters (B)")
    ax.set_ylabel("Compliance (%)")
    ax.set_title(f"{EVAL_DISPLAY[eval_type]} — Compliance vs Model Size (Base vs FT)")
    ax.set_xscale("log")
    ax.set_xticks([8, 20, 32, 120])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xticklabels(["8B", "20B", "32B", "120B"])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f"param_scaling_{eval_type}_{DATE}.png"
    fig.savefig(OUTDIR / fname, dpi=150)
    print(f"Saved {OUTDIR / fname}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# SET 3: Effort sweep — base vs FT at different reasoning efforts
# ══════════════════════════════════════════════════════════════════════════

EFFORT_ORDER = ["low", "medium", "high"]
EFFORT_MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]

for model in EFFORT_MODELS:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for col_idx, eval_type in enumerate(["cotcontrol", "reasonif"]):
        ax = axes[col_idx]
        x = np.arange(len(EFFORT_ORDER))

        # Base line (solid)
        by_vals, be_vals = [], []
        for effort in EFFORT_ORDER:
            row = base_effort.get((model, eval_type, effort))
            p = sf(row, "compliant_rate") if row else None
            n = si(row, "n_clean") if row else 0
            by_vals.append((p or 0) * 100 if p is not None else None)
            be_vals.append(ci80(p, n) * 100 if p is not None else 0)

        valid = [(xi, y, e) for xi, y, e in zip(x, by_vals, be_vals) if y is not None]
        if valid:
            vx, vy, ve = zip(*valid)
            ax.errorbar(
                vx, vy, yerr=ve, marker="o", linestyle="-", color="#4C72B0",
                label="Base", capsize=5, lw=2, markersize=8,
            )

        # FT line (dashed)
        fy_vals, fe_vals = [], []
        for effort in EFFORT_ORDER:
            row = ft_effort.get((model, eval_type, effort))
            p = sf(row, "compliant_rate") if row else None
            n = si(row, "n_clean") if row else 0
            fy_vals.append((p or 0) * 100 if p is not None else None)
            fe_vals.append(ci80(p, n) * 100 if p is not None else 0)

        valid = [(xi, y, e) for xi, y, e in zip(x, fy_vals, fe_vals) if y is not None]
        if valid:
            vx, vy, ve = zip(*valid)
            ax.errorbar(
                vx, vy, yerr=ve, marker="s", linestyle="--", color="#DD8452",
                label="FT", capsize=5, lw=2, markersize=8,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(EFFORT_ORDER)
        ax.set_xlabel("Reasoning Effort")
        ax.set_ylabel("Compliance (%)")
        ax.set_title(f"{EVAL_DISPLAY[eval_type]}")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"{short_name(model)} — Compliance vs Reasoning Effort: Base vs FT (80% CI)",
        fontsize=13,
    )
    plt.tight_layout()
    model_slug = short_name(model)
    fname = f"effort_sweep_{model_slug}_{DATE}.png"
    fig.savefig(OUTDIR / fname, dpi=150)
    print(f"Saved {OUTDIR / fname}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# SET 4: Correlation plots
# ══════════════════════════════════════════════════════════════════════════


# ── 4a: 2D scatter — CoTControl vs ReasonIF compliance ─────────────────

fig, ax = plt.subplots(figsize=(7, 7))
both = [m for m in models if (m, "cotcontrol") in base_runs and (m, "reasonif") in base_runs]
for m in both:
    xv = (sf(base_runs[(m, "reasonif")], "compliant_rate") or 0) * 100
    yv = (sf(base_runs[(m, "cotcontrol")], "compliant_rate") or 0) * 100
    ax.scatter(xv, yv, color=COLORS[m], s=100, zorder=5, edgecolors="black", linewidths=0.5)
    ax.annotate(short_name(m), (xv, yv), textcoords="offset points", xytext=(8, 4), fontsize=8)

ax.set_xlabel("ReasonIF Compliance (%)")
ax.set_ylabel("CoTControl Compliance (%)")
ax.set_title("Compliance: ReasonIF vs CoTControl")
all_vals = (
    [(sf(base_runs[(m, "reasonif")], "compliant_rate") or 0) * 100 for m in both]
    + [(sf(base_runs[(m, "cotcontrol")], "compliant_rate") or 0) * 100 for m in both]
)
lim = max(all_vals) * 1.2 if all_vals else 10
ax.plot([0, lim], [0, lim], "k--", alpha=0.2, linewidth=1)
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.set_aspect("equal")
plt.tight_layout()
fname = f"compliance_reasonif_vs_cotcontrol_{DATE}.png"
fig.savefig(OUTDIR / fname, dpi=150)
print(f"Saved {OUTDIR / fname}")
plt.close()


# ── 4b: Instruction-type compliance correlation heatmap ────────────────

# Build compliance vectors: instruction_type -> {model: rate}
compliance_by_type: dict[str, dict[str, float]] = {}
for (base_model, eval_type, itype), row in base_instr.items():
    if itype == "baseline":
        continue
    val = sf(row, "compliant_rate")
    if val is None:
        continue
    compliance_by_type.setdefault(itype, {})[base_model] = val

# Filter out types with fewer than 2 non-zero models.
nonzero_cc = [t for t in COTCONTROL_TYPES if sum(v > 0 for v in compliance_by_type.get(t, {}).values()) >= 2]
nonzero_rif = [t for t in REASONIF_TYPES if sum(v > 0 for v in compliance_by_type.get(t, {}).values()) >= 2]
all_corr_types = nonzero_cc + nonzero_rif
n_cc = len(nonzero_cc)
n_corr_types = len(all_corr_types)

# Compute pairwise Pearson correlation.
corr = np.full((n_corr_types, n_corr_types), np.nan)
for i, ti in enumerate(all_corr_types):
    for j, tj in enumerate(all_corr_types):
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

fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")

ax.axhline(y=n_cc - 0.5, color="black", linewidth=2)
ax.axvline(x=n_cc - 0.5, color="black", linewidth=2)

ax.set_xticks(range(n_corr_types))
ax.set_xticklabels(all_corr_types, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(n_corr_types))
ax.set_yticklabels(all_corr_types, fontsize=8)

ax.text(n_cc / 2, -1.5, "CoTControl", ha="center", fontsize=10, fontweight="bold")
ax.text(n_cc + len(nonzero_rif) / 2, -1.5, "ReasonIF", ha="center", fontsize=10, fontweight="bold")

cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Pearson r")
ax.set_title("Instruction-Type Compliance Correlation Across Models", pad=30)
plt.tight_layout()
fname = f"instruction_type_compliance_correlation_{DATE}.png"
fig.savefig(OUTDIR / fname, dpi=150, bbox_inches="tight")
print(f"Saved {OUTDIR / fname}")
plt.close()


# ── 4c: 3×3 class-level correlation grid ──────────────────────────────

from controllability.constants import INSTRUCTION_CLASS

CLASS_ORDER = ["style", "suppression", "addition"]
type_to_idx = {t: i for i, t in enumerate(all_corr_types)}

class_types: dict[str, list[str]] = {c: [] for c in CLASS_ORDER}
for t in all_corr_types:
    cls = INSTRUCTION_CLASS.get(t)
    if cls in class_types:
        class_types[cls].append(t)

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
fname = f"instruction_class_correlation_{DATE}.png"
fig.savefig(OUTDIR / fname, dpi=150, bbox_inches="tight")
print(f"Saved {OUTDIR / fname}")
plt.close()


# ── 4d: Full correlation heatmap grouped by instruction class ──────────

class_ordered_types = class_types["style"] + class_types["suppression"] + class_types["addition"]
n_style = len(class_types["style"])
n_suppression = len(class_types["suppression"])
n_class_types = len(class_ordered_types)

corr_by_class = np.full((n_class_types, n_class_types), np.nan)
for i, ti in enumerate(class_ordered_types):
    for j, tj in enumerate(class_ordered_types):
        oi, oj = type_to_idx[ti], type_to_idx[tj]
        corr_by_class[i, j] = corr[oi, oj]

fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(corr_by_class, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")

for boundary in [n_style - 0.5, n_style + n_suppression - 0.5]:
    ax.axhline(y=boundary, color="black", linewidth=2)
    ax.axvline(x=boundary, color="black", linewidth=2)

ax.set_xticks(range(n_class_types))
ax.set_xticklabels(class_ordered_types, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(n_class_types))
ax.set_yticklabels(class_ordered_types, fontsize=8)

ax.text(n_style / 2, -1.5, "style", ha="center", fontsize=10, fontweight="bold")
ax.text(n_style + n_suppression / 2, -1.5, "suppression", ha="center", fontsize=10, fontweight="bold")
ax.text(n_style + n_suppression + len(class_types["addition"]) / 2, -1.5, "addition", ha="center", fontsize=10, fontweight="bold")

cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Pearson r")
ax.set_title("Instruction-Type Compliance Correlation (Grouped by Class)", pad=30)
plt.tight_layout()
fname = f"instruction_type_correlation_by_class_{DATE}.png"
fig.savefig(OUTDIR / fname, dpi=150, bbox_inches="tight")
print(f"Saved {OUTDIR / fname}")
plt.close()


print(f"\nDone — all plots saved to {OUTDIR}")
