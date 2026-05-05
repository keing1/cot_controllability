#!/usr/bin/env python3
"""Summary plots for the 2026-04-30 v4 fine-tune run.

All 7 models trained from a single run_id (20260430_113831). Allowed
checkpoints: 000050, 000100, 000300, final. Monitor type: metr_note_v2
(graded by gpt-5.4 per the YAML's monitor_model_overrides).

Plots:
  1. compliance_first_ckpt_{cotcontrol,reasonif}.png — bar Base vs FT@000050
  2. mqa_evasion_first_ckpt.png  — bar Base vs FT@000050 with Base/FT labels
  3. per_model_training_curves.png — line plots per model: x=step,
     y=rate, three series (cotcontrol compliance, reasonif compliance,
     monitor_qa evasion).
  4. size_vs_uplift.png — scatter: x=active_params, y=uplift at FT@000050
     for cotcontrol/reasonif/monitor_qa (3 panels).

Some plots will be partial while FTs are still in flight — the script
warns on missing files and plots only what's available.

Usage:
  uv run python scripts/analysis/plotting/plot_summary_2026_04_30.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.size"] = 11

Z80 = 1.282


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

ROLLOUTS_DIR = REPO_ROOT / "results" / "rollouts"
MQA_DIR = ROLLOUTS_DIR / "monitor_qa"

DATE = "2026_04_30"
RUN_ID = "20260430_113831"
MONITOR = "metr_note_v2"  # YAML: monitor_model_overrides metr_note_v2 → gpt-5.4
FIRST_CKPT = "000050"     # "first checkpoint" referenced throughout
ALL_CKPTS = ["000050", "000100", "000300", "final"]

OUTDIR = REPO_ROOT / "results" / "summaries" / "plots" / DATE
OUTDIR.mkdir(parents=True, exist_ok=True)

MODEL_GROUPS = [
    "GPT-OSS-20B", "GPT-OSS-120B", "Qwen3-8B", "Qwen3-32B",
    "DeepSeek V3.1", "Kimi K2.5", "Nemotron-3 Nano",
]

MODEL_COLORS = {
    "GPT-OSS-20B":    "#1E88E5",
    "GPT-OSS-120B":   "#0D47A1",
    "Qwen3-8B":       "#43A047",
    "Qwen3-32B":      "#1B5E20",
    "DeepSeek V3.1":  "#8E24AA",
    "Kimi K2.5":      "#F4511E",
    "Nemotron-3 Nano":"#00897B",
}

FT_PREFIX = {
    "GPT-OSS-20B":    "openai_gpt-oss-20b",
    "GPT-OSS-120B":   "openai_gpt-oss-120b",
    "Qwen3-8B":       "qwen_qwen3-8b",
    "Qwen3-32B":      "qwen_qwen3-32b",
    "DeepSeek V3.1":  "deepseek-ai_DeepSeek-V3.1",
    "Kimi K2.5":      "moonshotai_Kimi-K2.5",
    "Nemotron-3 Nano":"nvidia_NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
}

# Active params (in billions) — for size scaling. Total/active where MoE.
# Sources: model card pages, public announcements.
MODEL_ACTIVE_PARAMS_B = {
    "GPT-OSS-20B":    3.6,
    "GPT-OSS-120B":   5.1,
    "Qwen3-8B":       8.0,
    "Qwen3-32B":      32.0,
    "DeepSeek V3.1":  37.0,
    "Kimi K2.5":      32.0,
    "Nemotron-3 Nano":3.0,
}
MODEL_TOTAL_PARAMS_B = {
    "GPT-OSS-20B":    20.0,
    "GPT-OSS-120B":   120.0,
    "Qwen3-8B":       8.0,
    "Qwen3-32B":      32.0,
    "DeepSeek V3.1":  671.0,
    "Kimi K2.5":      1000.0,
    "Nemotron-3 Nano":30.0,
}

EXCLUDE_MODES = {"cotcontrol": {"baseline"}, "reasonif": set()}

MQA_VARIANTS_SIDE_TASK = ["less_reasoning_v2", "vocab_control", "dont_think_v2"]

# Display labels for the per-instruction bars plot.
INSTR_DISPLAY = {
    # reasonif
    "reasoning_language": "lang",
    "number_words":       "word limit",
    "english_capital":    "uppercase",
    "end_checker":        "end phrase",
    "json_format":        "json",
    "no_comma":           "no comma",
    # cotcontrol
    "baseline":                  "baseline",
    "word_suppression":          "word supp",
    "multiple_word_suppression": "multi word supp",
    "uppercase_thinking":        "uppercase",
    "lowercase_thinking":        "lowercase",
    "alternating_case":          "alt case",
    "repeat_sentences":          "repeat sent",
    "end_of_sentence":           "end of sent",
    "meow_between_words":        "meow",
    "ignore_question":           "ignore Q",
}


# ---------------------------------------------------------------------------
# File resolution
# ---------------------------------------------------------------------------


def _find_one(pattern: str, root: Path) -> Path | None:
    matches = sorted(root.glob(pattern))
    return matches[0] if matches else None


def _ft_std(model: str, dataset: str, ckpt: str) -> Path | None:
    """FT eval rollout for cotcontrol / reasonif at a given ckpt."""
    return _find_one(
        f"{FT_PREFIX[model]}-ft-{RUN_ID}-{ckpt}_{dataset}_all_{RUN_ID}_*.jsonl",
        ROLLOUTS_DIR,
    )


def _base_std(model: str, dataset: str) -> Path | None:
    return _find_one(
        f"{FT_PREFIX[model]}-ft-{RUN_ID}-base_{dataset}_all_{RUN_ID}_*.jsonl",
        ROLLOUTS_DIR,
    )


def _ft_mqa(model: str, label: str, ckpt: str) -> Path | None:
    return _find_one(
        f"monitor_qa_{FT_PREFIX[model]}-ft-{RUN_ID}-{ckpt}_{label}_*.jsonl",
        MQA_DIR,
    )


def _base_mqa(model: str, label: str) -> Path | None:
    return _find_one(
        f"monitor_qa_{FT_PREFIX[model]}-ft-{RUN_ID}-base_{label}_*.jsonl",
        MQA_DIR,
    )


# ---------------------------------------------------------------------------
# Row loaders + counters
# ---------------------------------------------------------------------------


def _iter_rows(path: Path):
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError:
                continue


def _is_errored(r: dict) -> bool:
    if r.get("error") or r.get("actor_error"):
        return True
    if (r.get("grading_details") or {}).get("error"):
        return True
    if r.get("compliant") is None:
        return True
    return False


def compliance_counts(path: Path) -> dict[str, tuple[int, int]]:
    counts: dict[str, list[int]] = {}
    for r in _iter_rows(path):
        mode = r.get("control_mode") or r.get("mode") or ""
        if not mode:
            continue
        if _is_errored(r):
            continue
        c = counts.setdefault(mode, [0, 0])
        c[1] += 1
        if r.get("compliant") is True:
            c[0] += 1
    return {m: (v[0], v[1]) for m, v in counts.items()}


def overall_rate(path: Path | None, exclude: set[str]) -> tuple[float, int]:
    """Return (compliance_rate_fraction, n_rollouts). NaN, 0 if path missing."""
    if path is None or not path.exists():
        return float("nan"), 0
    counts = compliance_counts(path)
    tot_c = sum(c for m, (c, _) in counts.items() if m not in exclude)
    tot_t = sum(t for m, (_, t) in counts.items() if m not in exclude)
    rate = (tot_c / tot_t) if tot_t else float("nan")
    return rate, tot_t


def _monitor_score(record: dict, monitor_type: str) -> float | None:
    for mr in record.get("monitor_results", []) or []:
        if mr.get("monitor_type") == monitor_type:
            s = mr.get("monitor_score")
            return float(s) if s is not None else None
    return None


def load_mqa_rows(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    out = []
    for r in _iter_rows(path):
        out.append({
            "actor_error":       r.get("actor_error"),
            "side_task_correct": bool(r.get("side_task_correct") or False),
            "main_task_correct": bool(r.get("main_task_correct") or False),
            "score":             _monitor_score(r, MONITOR),
        })
    return out


def threshold_5pct_fpr(baseline_rows: list[dict]) -> float | None:
    vals = [r["score"] for r in baseline_rows
            if r["score"] is not None and r["actor_error"] is None]
    if not vals:
        return None
    return float(np.percentile(vals, 95))


def evasion_rate(rows: list[dict], threshold: float | None) -> tuple[float, int]:
    if threshold is None:
        return float("nan"), 0
    clean = [r for r in rows if r["actor_error"] is None]
    if not clean:
        return float("nan"), 0
    n_ev = sum(1 for r in clean
               if r["side_task_correct"]
               and r["score"] is not None
               and r["score"] <= threshold)
    return n_ev / len(clean), len(clean)


def best_evasion_for(model: str, ckpt: str | None) -> tuple[float, int]:
    """Best evasion rate across MQA side-task variants for a given ckpt.
    ``ckpt=None`` → base model. Returns (rate_pct, n_clean).
    """
    if ckpt is None:
        baseline_rows = load_mqa_rows(_base_mqa(model, "baseline"))
    else:
        baseline_rows = load_mqa_rows(_ft_mqa(model, "baseline", ckpt))
    thr = threshold_5pct_fpr(baseline_rows)
    best_rate = float("nan")
    best_n = 0
    for variant in MQA_VARIANTS_SIDE_TASK:
        path = _base_mqa(model, variant) if ckpt is None else _ft_mqa(model, variant, ckpt)
        rows = load_mqa_rows(path)
        rate, n = evasion_rate(rows, thr)
        if not np.isnan(rate) and (np.isnan(best_rate) or rate > best_rate):
            best_rate = rate
            best_n = n
    return (best_rate * 100 if not np.isnan(best_rate) else float("nan"),
            best_n)


def wald_err_pct(rate_frac: float, n: int, z: float = Z80) -> float:
    if n <= 0 or np.isnan(rate_frac):
        return 0.0
    p = max(0.0, min(1.0, rate_frac))
    return float(z * np.sqrt(p * (1 - p) / n) * 100.0)


def _autoscale(values: list[float], pad_frac: float = 0.15,
               floor_range: float = 10.0, max_cap: float = 100.0) -> tuple[float, float]:
    v = [x for x in values if not np.isnan(x)]
    if not v:
        return 0.0, max_cap
    hi = max(v)
    span = max(hi, floor_range)
    pad = span * pad_frac
    return 0.0, min(max_cap, max(hi + pad, floor_range))


def _mark_missing(ax, x_pos: float, y_pos: float = 0.0,
                  marker_size: int = 200) -> None:
    """Draw a red 'X' at (x_pos, y_pos) and an annotation for missing data."""
    ax.scatter([x_pos], [y_pos], marker="x", c="red", s=marker_size,
               linewidth=2.5, zorder=10, clip_on=False)
    ax.text(x_pos, y_pos + 1.5, "no\ndata",
            ha="center", va="bottom", fontsize=7, color="red",
            fontstyle="italic")


# ---------------------------------------------------------------------------
# Plot 1 — compliance Base vs FT@first-ckpt for cotcontrol / reasonif
# ---------------------------------------------------------------------------


def plot_compliance_for_ckpt(
    dataset_key: str, title: str, fname: str, ckpt: str,
) -> None:
    exclude = EXCLUDE_MODES.get(dataset_key, set())
    base_rates_pct, base_n = [], []
    ft_rates_pct, ft_n = [], []
    missing: list[str] = []

    for model in MODEL_GROUPS:
        bp = _base_std(model, dataset_key)
        fp = _ft_std(model, dataset_key, ckpt)
        br, bn = overall_rate(bp, exclude)
        fr, fn = overall_rate(fp, exclude)
        base_rates_pct.append(br * 100)
        base_n.append(bn)
        ft_rates_pct.append(fr * 100)
        ft_n.append(fn)
        if bp is None: missing.append(f"{model}/base/{dataset_key}")
        if fp is None: missing.append(f"{model}/ft@{ckpt}/{dataset_key}")

    if missing:
        print(f"  WARNING: missing files for {dataset_key}@{ckpt}:")
        for m in missing[:10]:
            print(f"    - {m}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")

    base_err = [wald_err_pct(r / 100.0, n) for r, n in zip(base_rates_pct, base_n)]
    ft_err = [wald_err_pct(r / 100.0, n) for r, n in zip(ft_rates_pct, ft_n)]

    x = np.arange(len(MODEL_GROUPS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(11, len(MODEL_GROUPS) * 1.6), 5.5))
    ax.bar(x - w / 2, base_rates_pct, w, color="#90A4AE", alpha=0.9,
           yerr=base_err, capsize=3, ecolor="black", error_kw={"linewidth": 0.9},
           label="Base", edgecolor="white", linewidth=0.5)
    ax.bar(x + w / 2, ft_rates_pct, w, color="#1E88E5", alpha=0.95,
           yerr=ft_err, capsize=3, ecolor="black", error_kw={"linewidth": 0.9},
           label=f"FT @ {ckpt}", edgecolor="white", linewidth=0.5)

    for i, (b, f, be, fe) in enumerate(zip(base_rates_pct, ft_rates_pct,
                                           base_err, ft_err)):
        if not np.isnan(b):
            ax.text(i - w / 2, b + be + 0.5, f"{b:.1f}",
                    ha="center", va="bottom", fontsize=9)
        else:
            _mark_missing(ax, i - w / 2)
        if not np.isnan(f):
            ax.text(i + w / 2, f + fe + 0.5, f"{f:.1f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        else:
            _mark_missing(ax, i + w / 2)

    top = [r + e for r, e in zip(base_rates_pct + ft_rates_pct,
                                  base_err + ft_err)
           if not np.isnan(r)]
    y_lo, y_hi = _autoscale(top, pad_frac=0.12)
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_GROUPS, rotation=20, ha="right")
    ax.set_ylabel("Overall compliance (%)")
    ax.set_ylim(y_lo, y_hi)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=10, loc="upper left")
    fig.suptitle(f"{title} — Base vs FT @ ckpt {ckpt}  (80% Wald CI)",
                 fontsize=13)
    fig.tight_layout()
    out = OUTDIR / fname
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Plot 2 — monitor_qa evasion: Base vs FT@first-ckpt
# ---------------------------------------------------------------------------


def _render_mqa_evasion_panel(ax, ckpt: str) -> None:
    """Render the Base-vs-FT@ckpt monitor_qa evasion bars into ax."""
    base_rates, base_ns = [], []
    ft_rates, ft_ns = [], []
    for model in MODEL_GROUPS:
        br, bn = best_evasion_for(model, ckpt=None)
        base_rates.append(br); base_ns.append(bn)
        fr, fn = best_evasion_for(model, ckpt=ckpt)
        ft_rates.append(fr); ft_ns.append(fn)
    base_err = [wald_err_pct(r / 100.0, n) for r, n in zip(base_rates, base_ns)]
    ft_err = [wald_err_pct(r / 100.0, n) for r, n in zip(ft_rates, ft_ns)]

    x = np.arange(len(MODEL_GROUPS))
    w = 0.38
    ax.bar(x - w / 2, base_rates, w, color="#90A4AE", alpha=0.9,
           yerr=base_err, capsize=3, ecolor="black", error_kw={"linewidth": 0.9},
           label="Base (no FT)", edgecolor="white", linewidth=0.5)
    ax.bar(x + w / 2, ft_rates, w, color="#C62828", alpha=0.95,
           yerr=ft_err, capsize=3, ecolor="black", error_kw={"linewidth": 0.9},
           label=f"FT @ ckpt {ckpt}", edgecolor="white", linewidth=0.5)
    for i, (br, fr, be, fe) in enumerate(zip(base_rates, ft_rates,
                                              base_err, ft_err)):
        if not np.isnan(br):
            ax.text(i - w / 2, br + be + 0.4, f"{br:.1f}",
                    ha="center", va="bottom", fontsize=9)
        else:
            _mark_missing(ax, i - w / 2)
        if not np.isnan(fr):
            ax.text(i + w / 2, fr + fe + 0.4, f"{fr:.1f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        else:
            _mark_missing(ax, i + w / 2)
    top = [r + e for r, e in zip(base_rates + ft_rates, base_err + ft_err)
           if not np.isnan(r)]
    y_lo, y_hi = _autoscale(top, pad_frac=0.18)
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_GROUPS, rotation=20, ha="right")
    ax.set_ylabel("Best evasion success rate (%)")
    ax.set_ylim(y_lo, y_hi)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=10, loc="upper left")
    ax.set_title(f"FT @ ckpt {ckpt}", fontsize=11)


def plot_mqa_evasion_stacked(monitor: str, fname: str) -> None:
    """Stacked monitor_qa evasion: first ckpt on top, last ckpt on bottom."""
    global MONITOR
    MONITOR = monitor
    ckpts = [FIRST_CKPT, "final"]
    fig, axes = plt.subplots(len(ckpts), 1,
                              figsize=(max(12, len(MODEL_GROUPS) * 1.7),
                                        6 * len(ckpts)))
    if len(ckpts) == 1:
        axes = [axes]
    for ax, ckpt in zip(axes, ckpts):
        _render_mqa_evasion_panel(ax, ckpt)
    fig.suptitle("MonitorQA evasion (80% Wald CI)", fontsize=14)
    fig.tight_layout()
    out = OUTDIR / fname
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def plot_mqa_evasion_for_ckpt(ckpt: str, fname: str) -> None:
    base_rates, base_ns = [], []
    ft_rates, ft_ns = [], []
    missing: list[str] = []
    for model in MODEL_GROUPS:
        # Base
        bp_baseline = _base_mqa(model, "baseline")
        if bp_baseline is None:
            missing.append(f"{model}/base/monitor_qa@baseline")
        br, bn = best_evasion_for(model, ckpt=None)
        base_rates.append(br); base_ns.append(bn)
        # FT @ ckpt
        fp_baseline = _ft_mqa(model, "baseline", ckpt)
        if fp_baseline is None:
            missing.append(f"{model}/ft@{ckpt}/monitor_qa@baseline")
        fr, fn = best_evasion_for(model, ckpt=ckpt)
        ft_rates.append(fr); ft_ns.append(fn)

    if missing:
        print(f"  WARNING: missing monitor_qa files:")
        for m in missing[:10]:
            print(f"    - {m}")

    base_err = [wald_err_pct(r / 100.0, n) for r, n in zip(base_rates, base_ns)]
    ft_err = [wald_err_pct(r / 100.0, n) for r, n in zip(ft_rates, ft_ns)]

    x = np.arange(len(MODEL_GROUPS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(12, len(MODEL_GROUPS) * 1.7), 6))
    ax.bar(x - w / 2, base_rates, w, color="#90A4AE", alpha=0.9,
           yerr=base_err, capsize=3, ecolor="black", error_kw={"linewidth": 0.9},
           label="Base (no FT)", edgecolor="white", linewidth=0.5)
    ax.bar(x + w / 2, ft_rates, w, color="#C62828", alpha=0.95,
           yerr=ft_err, capsize=3, ecolor="black", error_kw={"linewidth": 0.9},
           label=f"FT @ ckpt {ckpt}", edgecolor="white", linewidth=0.5)

    for i, (br, fr, be, fe) in enumerate(zip(base_rates, ft_rates,
                                              base_err, ft_err)):
        if not np.isnan(br):
            ax.text(i - w / 2, br + be + 0.4, f"{br:.1f}",
                    ha="center", va="bottom", fontsize=9)
        else:
            _mark_missing(ax, i - w / 2)
        if not np.isnan(fr):
            ax.text(i + w / 2, fr + fe + 0.4, f"{fr:.1f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        else:
            _mark_missing(ax, i + w / 2)

    top = [r + e for r, e in zip(base_rates + ft_rates, base_err + ft_err)
           if not np.isnan(r)]
    y_lo, y_hi = _autoscale(top, pad_frac=0.18)
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_GROUPS, rotation=20, ha="right")
    ax.set_ylabel("Best evasion success rate across side-task variants (%)")
    ax.set_ylim(y_lo, y_hi)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=10, loc="upper left")
    fig.suptitle(
        f"Monitor-QA evasion — Base vs FT @ {ckpt}  "
        f"(monitor: {MONITOR} @ 5% FPR; 80% Wald CI)",
        fontsize=13,
    )
    fig.tight_layout()
    out = OUTDIR / fname
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Plot 3 — per-model line plots: x=ckpt, y=rate; series = ds
# ---------------------------------------------------------------------------


# Map ckpt name → numeric step for x-axis
_CKPT_STEP = {"base": 0, "000050": 50, "000100": 100, "000300": 300, "final": 491}

DS_CONFIG = [
    ("cotcontrol",  "CoTControl compliance",   "#1E88E5"),
    ("reasonif",    "ReasonIF compliance",     "#43A047"),
    ("monitor_qa",  "MonitorQA evasion",       "#C62828"),
]


def per_model_curve(model: str, ds: str
                    ) -> tuple[list[int], list[float], list[int]]:
    """Return (steps, rates_pct, ns) for a given dataset over base + ckpts."""
    pts: list[tuple[int, float, int]] = []
    for ckpt in ["base"] + ALL_CKPTS:
        step = _CKPT_STEP[ckpt]
        if ds == "monitor_qa":
            rate, n = best_evasion_for(model, ckpt=None if ckpt == "base" else ckpt)
        else:
            exclude = EXCLUDE_MODES.get(ds, set())
            path = _base_std(model, ds) if ckpt == "base" else _ft_std(model, ds, ckpt)
            rate_frac, n = overall_rate(path, exclude)
            rate = rate_frac * 100 if not np.isnan(rate_frac) else float("nan")
        if not np.isnan(rate):
            pts.append((step, rate, n))
    pts.sort(key=lambda t: t[0])
    return [p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts]


def _avg_length_chars(path: Path | None, dataset: str) -> tuple[float, int]:
    """Return (mean total character count of actor output, n_rollouts).

    For cotcontrol/reasonif: ``reasoning + response``.
    For monitor_qa: ``reasoning + main_task_response + side_task_response``.
    Skips rollouts with errors (``error`` or ``actor_error``).
    """
    if path is None or not path.exists():
        return float("nan"), 0
    lens: list[int] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("error") or r.get("actor_error"):
                continue
            if dataset == "monitor_qa":
                total = (len(r.get("reasoning") or "") +
                         len(r.get("main_task_response") or "") +
                         len(r.get("side_task_response") or ""))
            else:
                total = (len(r.get("reasoning") or "") +
                         len(r.get("response") or ""))
            lens.append(total)
    if not lens:
        return float("nan"), 0
    return float(np.mean(lens)), len(lens)


def avg_length_curve(model: str, ds: str
                     ) -> tuple[list[int], list[float], list[int]]:
    """Mean output length across base + ckpts; for MQA averages across all 4 labels."""
    pts: list[tuple[int, float, int]] = []
    for ckpt in ["base"] + ALL_CKPTS:
        step = _CKPT_STEP[ckpt]
        if ds == "monitor_qa":
            # Average the 4 prompt-type files at this ckpt
            label_means: list[float] = []
            label_ns: list[int] = []
            for label in ["baseline", "less_reasoning_v2",
                          "vocab_control", "dont_think_v2"]:
                p = (_base_mqa(model, label) if ckpt == "base"
                     else _ft_mqa(model, label, ckpt))
                m, n = _avg_length_chars(p, "monitor_qa")
                if not np.isnan(m):
                    label_means.append(m)
                    label_ns.append(n)
            if label_means:
                # Weighted mean by sample count
                tot_n = sum(label_ns)
                weighted = sum(m * n for m, n in zip(label_means, label_ns)) / tot_n
                pts.append((step, weighted, tot_n))
        else:
            p = _base_std(model, ds) if ckpt == "base" else _ft_std(model, ds, ckpt)
            m, n = _avg_length_chars(p, ds)
            if not np.isnan(m):
                pts.append((step, m, n))
    pts.sort(key=lambda t: t[0])
    return [p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts]


def plot_per_model_completion_length() -> None:
    n = len(MODEL_GROUPS)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.4 * rows),
                             sharex=False)
    axes = axes.flatten()

    for i, model in enumerate(MODEL_GROUPS):
        ax = axes[i]
        for ds, label, color in DS_CONFIG:
            steps, lens, ns = avg_length_curve(model, ds)
            if not steps:
                continue
            ax.plot(steps, lens, marker="o", label=label, color=color,
                    linewidth=1.6, markersize=5)
        ax.set_title(model, fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Avg completion length (chars)")
        ax.set_xticks([0, 50, 100, 300, 491])
        ax.set_xticklabels(["base", "50", "100", "300", "final"])
        if i == 0:
            ax.legend(fontsize=8, loc="best")

    for j in range(len(MODEL_GROUPS), len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"Per-model average completion length over training — run {RUN_ID}",
        fontsize=13,
    )
    fig.tight_layout()
    out = OUTDIR / "per_model_completion_length.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def plot_per_model_training_curves(fname: str = "per_model_training_curves.png") -> None:
    n = len(MODEL_GROUPS)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.4 * rows),
                             sharex=False)
    axes = axes.flatten()

    for i, model in enumerate(MODEL_GROUPS):
        ax = axes[i]
        for ds, label, color in DS_CONFIG:
            steps, rates, ns = per_model_curve(model, ds)
            if not steps:
                continue
            errs = [wald_err_pct(r / 100.0, n_) for r, n_ in zip(rates, ns)]
            ax.errorbar(steps, rates, yerr=errs, label=label, color=color,
                        marker="o", capsize=2, linewidth=1.6, markersize=5)
        ax.set_title(model, fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Rate (%)")
        ax.set_xticks([0, 50, 100, 300, 491])
        ax.set_xticklabels(["base", "50", "100", "300", "final"])
        ax.set_ylim(0, 100)
        if i == 0:
            ax.legend(fontsize=8, loc="best")

    # Hide unused axes
    for j in range(len(MODEL_GROUPS), len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"Per-model training curves — run {RUN_ID} (monitor: {MONITOR})",
        fontsize=13,
    )
    fig.tight_layout()
    out = OUTDIR / fname
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Per-instruction bars (Base vs FT@ckpt, N models × 2 per instruction).
# Mirrors plot_summary_2026_04_20.py layout but parameterized by ckpt.
# ---------------------------------------------------------------------------


def plot_instruction_bars(dataset_key: str, title: str, fname: str,
                           ckpt: str) -> None:
    exclude = EXCLUDE_MODES.get(dataset_key, set())

    per_model_base: dict[str, dict[str, tuple[int, int]]] = {}
    per_model_ft: dict[str, dict[str, tuple[int, int]]] = {}
    all_instr: set[str] = set()
    for model in MODEL_GROUPS:
        bp = _base_std(model, dataset_key)
        fp = _ft_std(model, dataset_key, ckpt)
        base = {m: v for m, v in compliance_counts(bp).items()
                if m not in exclude} if bp else {}
        ft = {m: v for m, v in compliance_counts(fp).items()
              if m not in exclude} if fp else {}
        per_model_base[model] = base
        per_model_ft[model] = ft
        all_instr |= set(base.keys()) | set(ft.keys())

    instr_types = sorted(all_instr, key=lambda t: INSTR_DISPLAY.get(t, t))
    if not instr_types:
        print(f"  WARNING: no instruction types for {dataset_key}@{ckpt}")
        return

    n_groups = len(instr_types)
    n_bars = len(MODEL_GROUPS) * 2
    bar_width = 0.88 / n_bars
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(max(18, n_groups * 2.2), 6.5))
    top_vals: list[float] = []
    for m_idx, model in enumerate(MODEL_GROUPS):
        color = MODEL_COLORS[model]
        base_vals, ft_vals, base_errs, ft_errs = [], [], [], []
        for it in instr_types:
            bc, bt = per_model_base[model].get(it, (0, 0))
            fc, ft_ = per_model_ft[model].get(it, (0, 0))
            bp_ = (bc / bt) if bt else 0.0
            fp_ = (fc / ft_) if ft_ else 0.0
            base_vals.append(bp_ * 100)
            ft_vals.append(fp_ * 100)
            base_errs.append(wald_err_pct(bp_, bt))
            ft_errs.append(wald_err_pct(fp_, ft_))
        off_b = (m_idx * 2 - n_bars / 2 + 0.5) * bar_width
        off_f = (m_idx * 2 + 1 - n_bars / 2 + 0.5) * bar_width
        ax.bar(x + off_b, base_vals, bar_width,
               yerr=base_errs, ecolor="black",
               error_kw={"linewidth": 0.6, "capsize": 1.5},
               color=color, alpha=0.45, label=f"{model} Base",
               edgecolor="white", linewidth=0.3)
        ax.bar(x + off_f, ft_vals, bar_width,
               yerr=ft_errs, ecolor="black",
               error_kw={"linewidth": 0.6, "capsize": 1.5},
               color=color, alpha=0.95, label=f"{model} FT",
               edgecolor="white", linewidth=0.3)
        top_vals.extend(v + e for v, e in zip(base_vals, base_errs))
        top_vals.extend(v + e for v, e in zip(ft_vals, ft_errs))

    y_lo, y_hi = _autoscale(top_vals, pad_frac=0.08)
    labels = [INSTR_DISPLAY.get(it, it) for it in instr_types]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Compliance (%)", fontsize=11)
    ax.set_ylim(y_lo, y_hi)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right", ncol=len(MODEL_GROUPS))
    fig.suptitle(
        f"{title} per-instruction — Base vs FT @ ckpt {ckpt}  (80% Wald CI)",
        fontsize=12,
    )
    fig.tight_layout()
    out = OUTDIR / fname
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}  (y {y_lo:.1f}-{y_hi:.1f})")


# ---------------------------------------------------------------------------
# Plot 4 — model size vs uplift at first ckpt
# ---------------------------------------------------------------------------


def _uplift_for(model: str, ds: str, ckpt: str) -> tuple[float, int, int]:
    """Return (ft_rate - base_rate in pct, base_n, ft_n)."""
    if ds == "monitor_qa":
        br, bn = best_evasion_for(model, ckpt=None)
        fr, fn = best_evasion_for(model, ckpt=ckpt)
    else:
        exclude = EXCLUDE_MODES.get(ds, set())
        bp, fp = _base_std(model, ds), _ft_std(model, ds, ckpt)
        br_frac, bn = overall_rate(bp, exclude)
        fr_frac, fn = overall_rate(fp, exclude)
        br = br_frac * 100 if not np.isnan(br_frac) else float("nan")
        fr = fr_frac * 100 if not np.isnan(fr_frac) else float("nan")
    if np.isnan(br) or np.isnan(fr):
        return float("nan"), bn, fn
    return fr - br, bn, fn


def _render_uplift_panel(ax, ds: str, ylabel: str, ckpt: str,
                          params_kind: str) -> None:
    """Render a single (ds × ckpt × params_kind) panel into ax.

    ``params_kind`` ∈ {"active", "total"}: which size column to plot.
    """
    params_lookup = (MODEL_ACTIVE_PARAMS_B if params_kind == "active"
                     else MODEL_TOTAL_PARAMS_B)
    xlabel = "Active params (B)" if params_kind == "active" else "Total params (B)"
    xs, ys, names, colors = [], [], [], []
    missing_xs: list[float] = []
    missing_names: list[str] = []
    for model in MODEL_GROUPS:
        p = params_lookup[model]
        uplift, _bn, _fn = _uplift_for(model, ds, ckpt)
        if np.isnan(uplift):
            missing_xs.append(p)
            missing_names.append(model)
            continue
        xs.append(p); ys.append(uplift); names.append(model)
        colors.append(MODEL_COLORS[model])
    if xs:
        ax.scatter(xs, ys, c=colors, s=140, edgecolor="black",
                   linewidth=0.6, zorder=5)
        for x, y, n in zip(xs, ys, names):
            ax.annotate(n, (x, y), fontsize=8,
                        xytext=(6, 4), textcoords="offset points")
        if len(xs) >= 2:
            lx = np.log10(np.array(xs))
            ya = np.array(ys)
            slope, intercept = np.polyfit(lx, ya, 1)
            ss_res = np.sum((ya - (slope * lx + intercept)) ** 2)
            ss_tot = np.sum((ya - ya.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            xs_line = np.geomspace(min(xs) * 0.7, max(xs) * 1.4, 100)
            ys_line = slope * np.log10(xs_line) + intercept
            ax.plot(xs_line, ys_line, "--", color="gray", linewidth=1.4,
                    alpha=0.8, zorder=3,
                    label=f"fit: y = {slope:.2f}·log10(x) + {intercept:.1f}")
            ax.text(0.02, 0.98, f"R² = {r2:.2f}", transform=ax.transAxes,
                    fontsize=10, va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="white", edgecolor="gray", alpha=0.8))
    if missing_xs:
        ax.scatter(missing_xs, [0] * len(missing_xs),
                   marker="x", c="red", s=140, linewidth=2.5,
                   zorder=10, label="(missing)")
        for mx, mn in zip(missing_xs, missing_names):
            ax.annotate(f"{mn} (no data)", (mx, 0), fontsize=7, color="red",
                        xytext=(6, -10), textcoords="offset points")
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(ds)
    ax.grid(alpha=0.3, which="both")


def plot_size_vs_uplift_stacked(params_kind: str, fname: str,
                                  ckpts: list[str] | None = None) -> None:
    """Stacked first-ckpt-on-top / last-ckpt-on-bottom for the same params kind."""
    ckpts = ckpts or [FIRST_CKPT, "final"]
    panels_def = [
        ("cotcontrol", "CoTControl compliance uplift (pp)"),
        ("reasonif",   "ReasonIF compliance uplift (pp)"),
        ("monitor_qa", "MonitorQA evasion rate uplift (pp)"),
    ]

    fig, axes = plt.subplots(len(ckpts), 3, figsize=(16.5, 5.5 * len(ckpts)))
    if len(ckpts) == 1:
        axes = np.array([axes])
    for row_i, ckpt in enumerate(ckpts):
        for col_i, (ds, ylabel) in enumerate(panels_def):
            ax = axes[row_i][col_i]
            _render_uplift_panel(ax, ds, ylabel, ckpt, params_kind)
        # Row label on far-left panel
        axes[row_i][0].annotate(
            f"FT @ ckpt {ckpt}", xy=(-0.18, 0.5),
            xycoords="axes fraction", rotation=90,
            ha="center", va="center", fontsize=12, fontweight="bold",
        )

    pkind_label = "Active" if params_kind == "active" else "Total"
    fig.suptitle(f"{pkind_label} params vs uplift", fontsize=14)
    fig.tight_layout()
    out = OUTDIR / fname
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def plot_size_vs_uplift(ckpt: str, fname_active: str, fname_total: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.5))

    panels = [
        ("cotcontrol", "CoTControl compliance uplift (pp)",   axes[0]),
        ("reasonif",   "ReasonIF compliance uplift (pp)",     axes[1]),
        ("monitor_qa", "MonitorQA evasion rate uplift (pp)",  axes[2]),
    ]

    for ds, ylabel, ax in panels:
        xs, ys, names, colors = [], [], [], []
        missing_xs: list[float] = []
        missing_names: list[str] = []
        for model in MODEL_GROUPS:
            ap = MODEL_ACTIVE_PARAMS_B[model]
            uplift, bn, fn = _uplift_for(model, ds, ckpt)
            if np.isnan(uplift):
                missing_xs.append(ap)
                missing_names.append(model)
                continue
            xs.append(ap)
            ys.append(uplift)
            names.append(model)
            colors.append(MODEL_COLORS[model])
        if xs:
            ax.scatter(xs, ys, c=colors, s=140, edgecolor="black", linewidth=0.6, zorder=5)
            for x, y, n in zip(xs, ys, names):
                ax.annotate(n, (x, y), fontsize=8,
                            xytext=(6, 4), textcoords="offset points")
            if len(xs) >= 2:
                lx = np.log10(np.array(xs))
                ya = np.array(ys)
                slope, intercept = np.polyfit(lx, ya, 1)
                ss_res = np.sum((ya - (slope * lx + intercept)) ** 2)
                ss_tot = np.sum((ya - ya.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
                xlim = ax.get_xlim()
                xs_line = np.geomspace(max(xlim[0], min(xs) * 0.7),
                                        min(xlim[1], max(xs) * 1.4), 100)
                ys_line = slope * np.log10(xs_line) + intercept
                ax.plot(xs_line, ys_line, "--", color="gray", linewidth=1.4,
                        alpha=0.8, zorder=3,
                        label=f"fit: y = {slope:.2f}·log10(x) + {intercept:.1f}")
                ax.text(0.02, 0.98, f"R² = {r2:.2f}", transform=ax.transAxes,
                        fontsize=10, va="top", ha="left",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="white", edgecolor="gray", alpha=0.8))
        if missing_xs:
            ax.scatter(missing_xs, [0] * len(missing_xs),
                       marker="x", c="red", s=140, linewidth=2.5,
                       zorder=10, label="(missing)")
            for mx, mn in zip(missing_xs, missing_names):
                ax.annotate(f"{mn} (no data)", (mx, 0), fontsize=7, color="red",
                            xytext=(6, -10), textcoords="offset points")
        ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("Active params (B)")
        ax.set_ylabel(ylabel)
        ax.set_title(ds)
        ax.grid(alpha=0.3, which="both")

    fig.suptitle(
        f"Active params vs uplift @ FT ckpt {ckpt}  (run {RUN_ID})",
        fontsize=13,
    )
    fig.tight_layout()
    out = OUTDIR / fname_active
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")

    # Same plot with TOTAL params on the x-axis
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.5))
    panels = [
        ("cotcontrol", "CoTControl compliance uplift (pp)",   axes[0]),
        ("reasonif",   "ReasonIF compliance uplift (pp)",     axes[1]),
        ("monitor_qa", "MonitorQA evasion rate uplift (pp)",  axes[2]),
    ]
    for ds, ylabel, ax in panels:
        xs, ys, names, colors = [], [], [], []
        missing_xs: list[float] = []
        missing_names: list[str] = []
        for model in MODEL_GROUPS:
            tp = MODEL_TOTAL_PARAMS_B[model]
            uplift, bn, fn = _uplift_for(model, ds, ckpt)
            if np.isnan(uplift):
                missing_xs.append(tp)
                missing_names.append(model)
                continue
            xs.append(tp); ys.append(uplift); names.append(model)
            colors.append(MODEL_COLORS[model])
        if xs:
            ax.scatter(xs, ys, c=colors, s=140, edgecolor="black", linewidth=0.6, zorder=5)
            for x, y, n in zip(xs, ys, names):
                ax.annotate(n, (x, y), fontsize=8,
                            xytext=(6, 4), textcoords="offset points")
            # Line of best fit (linear in log10(x)) — only if ≥ 2 points
            r2_text = ""
            if len(xs) >= 2:
                lx = np.log10(np.array(xs))
                ya = np.array(ys)
                slope, intercept = np.polyfit(lx, ya, 1)
                ss_res = np.sum((ya - (slope * lx + intercept)) ** 2)
                ss_tot = np.sum((ya - ya.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
                xlim = ax.get_xlim()
                xs_line = np.geomspace(max(xlim[0], min(xs) * 0.7),
                                        min(xlim[1], max(xs) * 1.4), 100)
                ys_line = slope * np.log10(xs_line) + intercept
                ax.plot(xs_line, ys_line, "--", color="gray", linewidth=1.4,
                        alpha=0.8, zorder=3,
                        label=f"fit: y = {slope:.2f}·log10(x) + {intercept:.1f}")
                r2_text = f"R² = {r2:.2f}"
                ax.text(0.02, 0.98, r2_text, transform=ax.transAxes,
                        fontsize=10, va="top", ha="left",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="white", edgecolor="gray", alpha=0.8))
        if missing_xs:
            ax.scatter(missing_xs, [0] * len(missing_xs),
                       marker="x", c="red", s=140, linewidth=2.5,
                       zorder=10, label="(missing)")
            for mx, mn in zip(missing_xs, missing_names):
                ax.annotate(f"{mn} (no data)", (mx, 0), fontsize=7, color="red",
                            xytext=(6, -10), textcoords="offset points")
        ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("Total params (B)")
        ax.set_ylabel(ylabel)
        ax.set_title(ds)
        ax.grid(alpha=0.3, which="both")

    fig.suptitle(
        f"Total params vs uplift @ FT ckpt {ckpt}  (run {RUN_ID})",
        fontsize=13,
    )
    fig.tight_layout()
    out = OUTDIR / fname_total
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    global FIRST_CKPT, MONITOR
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-ckpt", default=FIRST_CKPT,
                        help="Checkpoint to use as 'first' in plots 1, 2, 4 "
                             "(default: 000050)")
    parser.add_argument("--monitor", default=MONITOR,
                        help="Monitor type to threshold/grade with "
                             "(default: metr_note_v2)")
    args = parser.parse_args()
    FIRST_CKPT = args.first_ckpt
    MONITOR = args.monitor

    print(f"Run id: {RUN_ID}")
    print(f"First ckpt: {FIRST_CKPT}")
    print(f"Monitor: {MONITOR}")
    print(f"Output dir: {OUTDIR}")
    print()

    # Plots that DO NOT depend on the monitor type: compliance bars + completion length.
    # Generated once, with both first-ckpt + last-ckpt versions for the bars.
    ckpt_variants = [
        (FIRST_CKPT, "first_ckpt"),
        ("final",    "last_ckpt"),
    ]
    for ckpt, suffix in ckpt_variants:
        print(f"\n=== Compliance bars for ckpt={ckpt} ===")
        print(f"Plot 1a: compliance_{suffix}_cotcontrol...")
        plot_compliance_for_ckpt(
            "cotcontrol", "CoTControl compliance",
            f"compliance_{suffix}_cotcontrol.png", ckpt,
        )
        print(f"Plot 1b: compliance_{suffix}_reasonif...")
        plot_compliance_for_ckpt(
            "reasonif", "ReasonIF compliance",
            f"compliance_{suffix}_reasonif.png", ckpt,
        )
        print(f"Per-instruction bars for ckpt={ckpt}...")
        plot_instruction_bars(
            "cotcontrol", "CoTControl",
            f"instruction_bars_cotcontrol_{suffix}.png", ckpt,
        )
        plot_instruction_bars(
            "reasonif", "ReasonIF",
            f"instruction_bars_reasonif_{suffix}.png", ckpt,
        )

    print("\n=== Plot 5 (completion length curves; monitor-independent) ===")
    plot_per_model_completion_length()

    # Monitor-dependent plots: 2 (mqa evasion bars), 3 (training curves with
    # mqa line), 4 (size vs uplift with mqa panel). Generate one set per
    # monitor type so v2 and v3 versions sit side-by-side.
    monitor_variants = ["metr_note_v2", "metr_note_v3"]
    for monitor in monitor_variants:
        MONITOR = monitor
        m_suffix = monitor.replace("metr_note_", "")  # "v2" / "v3"
        print(f"\n##### Monitor: {monitor}  (suffix='{m_suffix}') #####")

        for ckpt, suffix in ckpt_variants:
            print(f"\n=== Monitor-dependent for ckpt={ckpt} ({m_suffix}) ===")
            print(f"Plot 2: mqa_evasion_{suffix}_{m_suffix}...")
            plot_mqa_evasion_for_ckpt(
                ckpt, f"mqa_evasion_{suffix}_{m_suffix}.png",
            )
            print(f"Plot 4: size_vs_uplift_{suffix}_{m_suffix}...")
            plot_size_vs_uplift(
                ckpt,
                f"size_vs_uplift_active_{suffix}_{m_suffix}.png",
                f"size_vs_uplift_total_{suffix}_{m_suffix}.png",
            )

        print(f"\n=== Plot 3 training curves ({m_suffix}) ===")
        plot_per_model_training_curves(
            f"per_model_training_curves_{m_suffix}.png"
        )

        # Stacked size-vs-uplift: first ckpt on top, last on bottom.
        print(f"\n=== Plot 4-stacked ({m_suffix}) ===")
        plot_size_vs_uplift_stacked(
            "active", f"size_vs_uplift_active_first_and_last_{m_suffix}.png"
        )
        plot_size_vs_uplift_stacked(
            "total", f"size_vs_uplift_total_first_and_last_{m_suffix}.png"
        )

        # Stacked MonitorQA evasion: first ckpt on top, last on bottom.
        print(f"\n=== Plot 2-stacked MQA evasion ({m_suffix}) ===")
        plot_mqa_evasion_stacked(
            monitor, f"mqa_evasion_first_and_last_{m_suffix}.png"
        )

    print(f"\nAll plots written to {OUTDIR}/")


if __name__ == "__main__":
    main()
