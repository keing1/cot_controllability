"""Aggregate metric computation from rollouts."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from controllability.types import MonitorResult, MonitorRollout, Rollout


def compute_metrics(rollouts: list[Rollout]) -> dict:
    """Compute aggregate metrics from a list of rollouts.

    Returns a nested dict structured by mode, with accuracy, compliance,
    and meta-discussion rates.
    """
    if not rollouts:
        return {"total": 0, "by_mode": {}}

    by_mode: dict[str, list[Rollout]] = defaultdict(list)
    for r in rollouts:
        by_mode[r.control_mode].append(r)

    results = {"total": len(rollouts), "by_mode": {}}

    for mode, mode_rollouts in sorted(by_mode.items()):
        n = len(mode_rollouts)
        errored = [r for r in mode_rollouts if r.grading_details.get("error")]
        valid = [r for r in mode_rollouts if not r.grading_details.get("error")]

        correct = sum(1 for r in valid if r.correct is True)
        compliant = sum(1 for r in valid if r.compliant is True)
        meta = sum(1 for r in valid if r.meta_discussion is True)

        # Compliance-only among non-None (some modes like ignore_question may be None)
        compliance_graded = [r for r in valid if r.compliant is not None]
        n_compliance = len(compliance_graded)

        results["by_mode"][mode] = {
            "n": n,
            "n_valid": len(valid),
            "n_errored": len(errored),
            "accuracy": correct / len(valid) if valid else 0.0,
            "compliance_rate": (
                sum(1 for r in compliance_graded if r.compliant) / n_compliance
                if n_compliance > 0
                else None
            ),
            "meta_discussion_rate": meta / len(valid) if valid else 0.0,
            "correct": correct,
            "compliant": compliant,
            "meta_discussion": meta,
        }

    # Overall accuracy
    all_valid = [r for r in rollouts if not r.grading_details.get("error")]
    results["overall_accuracy"] = (
        sum(1 for r in all_valid if r.correct is True) / len(all_valid) if all_valid else 0.0
    )

    return results


def format_metrics(metrics: dict) -> str:
    """Format metrics dict into a readable table string."""
    lines = []
    lines.append(f"Total rollouts: {metrics['total']}")
    lines.append(f"Overall accuracy: {metrics.get('overall_accuracy', 0):.1%}")
    lines.append("")
    lines.append(f"{'Mode':<30} {'N':>5} {'Acc':>7} {'Comply':>7} {'Meta':>7}")
    lines.append("-" * 60)

    for mode, m in sorted(metrics.get("by_mode", {}).items()):
        acc = f"{m['accuracy']:.1%}"
        comply = f"{m['compliance_rate']:.1%}" if m['compliance_rate'] is not None else "N/A"
        meta = f"{m['meta_discussion_rate']:.1%}"
        lines.append(f"{mode:<30} {m['n']:>5} {acc:>7} {comply:>7} {meta:>7}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Monitor QA metrics
# ---------------------------------------------------------------------------


def _get_monitor_types(rollouts: list[MonitorRollout]) -> list[str]:
    """Collect all monitor types present across rollouts, in stable order."""
    seen: dict[str, None] = {}
    for r in rollouts:
        for mr in r.monitor_results:
            if mr.monitor_type not in seen:
                seen[mr.monitor_type] = None
    return list(seen.keys())


def compute_monitor_metrics(rollouts: list[MonitorRollout]) -> dict:
    """Compute aggregate metrics for monitor_qa rollouts, grouped by mode.

    When rollouts have ``monitor_results``, per-monitor-type scores are
    included under ``by_monitor_type`` within each mode dict.
    """
    if not rollouts:
        return {"total": 0, "by_mode": {}}

    by_mode: dict[str, list[MonitorRollout]] = defaultdict(list)
    for r in rollouts:
        by_mode[r.mode].append(r)

    monitor_types = _get_monitor_types(rollouts)
    results: dict = {"total": len(rollouts), "by_mode": {}, "monitor_types": monitor_types}

    for mode, mode_rollouts in sorted(by_mode.items()):
        n = len(mode_rollouts)
        errored = [r for r in mode_rollouts if r.actor_error or r.monitor_error]
        valid = [r for r in mode_rollouts if not r.actor_error and not r.monitor_error]

        # Side task accuracy (only meaningful for side_task mode)
        st_graded = [r for r in valid if r.side_task_correct is not None]
        st_correct = sum(1 for r in st_graded if r.side_task_correct is True)
        st_accuracy = st_correct / len(st_graded) if st_graded else None

        # Legacy monitor scores (from flat fields)
        scored = [r for r in valid if r.monitor_score is not None]
        avg_score = sum(r.monitor_score for r in scored) / len(scored) if scored else None

        # Per-monitor-type scores
        by_monitor_type: dict[str, dict] = {}
        for mt in monitor_types:
            mt_scores = []
            mt_errors = 0
            for r in valid:
                for mr in r.monitor_results:
                    if mr.monitor_type == mt:
                        if mr.monitor_error:
                            mt_errors += 1
                        elif mr.monitor_score is not None:
                            mt_scores.append(mr.monitor_score)
            by_monitor_type[mt] = {
                "avg_score": sum(mt_scores) / len(mt_scores) if mt_scores else None,
                "n_scored": len(mt_scores),
                "n_errors": mt_errors,
            }

        results["by_mode"][mode] = {
            "n": n,
            "n_valid": len(valid),
            "n_errored": len(errored),
            "side_task_accuracy": st_accuracy,
            "n_side_task_graded": len(st_graded),
            "avg_monitor_score": avg_score,
            "n_scored": len(scored),
            "by_monitor_type": by_monitor_type,
        }

    return results


def format_monitor_metrics(metrics: dict) -> str:
    """Format monitor_qa metrics into a readable table string."""
    monitor_types = metrics.get("monitor_types", [])
    lines = []
    lines.append(f"Total rollouts: {metrics['total']}")
    lines.append("")

    # Build header
    header = f"{'Mode':<15} {'N':>5} {'Valid':>6} {'ST Acc':>8}"
    if monitor_types:
        for mt in monitor_types:
            header += f" {mt:>14}"
    else:
        header += f" {'Avg Score':>10}"
    lines.append(header)
    lines.append("-" * len(header))

    for mode, m in sorted(metrics.get("by_mode", {}).items()):
        st_acc = f"{m['side_task_accuracy']:.1%}" if m["side_task_accuracy"] is not None else "N/A"
        row = f"{mode:<15} {m['n']:>5} {m['n_valid']:>6} {st_acc:>8}"

        by_mt = m.get("by_monitor_type", {})
        if monitor_types and by_mt:
            for mt in monitor_types:
                mt_data = by_mt.get(mt, {})
                avg = mt_data.get("avg_score")
                row += f" {avg:>14.2f}" if avg is not None else f" {'N/A':>14}"
        else:
            avg_sc = f"{m['avg_monitor_score']:.2f}" if m["avg_monitor_score"] is not None else "N/A"
            row += f" {avg_sc:>10}"

        lines.append(row)

    return "\n".join(lines)


def write_monitor_metrics_csv(
    rollouts: list[MonitorRollout], path: str | Path,
) -> None:
    """Write per-rollout monitor_qa metrics to a CSV file.

    Includes a column for each monitor type's score.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    monitor_types = _get_monitor_types(rollouts)

    fieldnames = [
        "id", "sample_id", "mode", "actor_model", "monitor_model",
        "side_task", "side_task_response", "side_task_correct",
        "monitor_score",
    ]
    # Add per-monitor-type score columns
    for mt in monitor_types:
        fieldnames.append(f"score_{mt}")
    fieldnames.extend(["actor_error", "monitor_error"])

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rollouts:
            row = {
                "id": r.id,
                "sample_id": r.sample.id,
                "mode": r.mode,
                "actor_model": r.model,
                "monitor_model": r.monitor_model,
                "side_task": r.sample.metadata.get("side_task", ""),
                "side_task_response": r.side_task_response,
                "side_task_correct": r.side_task_correct,
                "monitor_score": r.monitor_score,
                "actor_error": r.actor_error or "",
                "monitor_error": r.monitor_error or "",
            }
            # Per-monitor-type scores
            mt_scores = {mr.monitor_type: mr.monitor_score for mr in r.monitor_results}
            for mt in monitor_types:
                row[f"score_{mt}"] = mt_scores.get(mt, "")
            writer.writerow(row)
