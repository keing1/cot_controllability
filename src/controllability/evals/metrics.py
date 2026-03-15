"""Aggregate metric computation from rollouts."""

from __future__ import annotations

from collections import defaultdict

from controllability.types import Rollout


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
