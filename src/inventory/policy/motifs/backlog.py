"""Backlog-level and age-profile motifs for operational policies."""

from __future__ import annotations

from inventory.core.config import BacklogConfig, BacklogMode
from inventory.policy.ast import AST, add, const, dist, mul, safe_ratio, state, sub


def _sum(nodes: list[AST]) -> AST:
    return nodes[0] if len(nodes) == 1 else add(*nodes)


def backlog_motifs(backlog: BacklogConfig) -> dict[str, AST]:
    """Return motifs valid for exactly the supplied backlog representation."""

    if not isinstance(backlog, BacklogConfig):
        raise TypeError("backlog must be a BacklogConfig")
    if not backlog.enabled:
        return {}

    if backlog.uses_scalar_total:
        total = state("B")
        motifs: dict[str, AST] = {
            "total_backlog": total,
            "backlog_to_mean_demand": safe_ratio(total, dist("mu")),
            "demand_mean": dist("mu"),
            "demand_cv": dist("cv"),
        }
        if backlog.mode is BacklogMode.GEOMETRIC:
            survival = float(backlog.survival_probability or 0.0)
            motifs["expected_retained_backlog_if_unserved"] = mul(
                const(survival), total
            )
            motifs["expected_abandoned_backlog_if_unserved"] = mul(
                const(1.0 - survival), total
            )
        else:
            motifs["expected_retained_backlog_if_unserved"] = total
            motifs["expected_abandoned_backlog_if_unserved"] = const(0.0)
        return motifs

    patience = int(backlog.patience or 0)
    cohorts = [state(f"b{index}") for index in range(1, patience + 1)]
    total = _sum(cohorts)
    age_moment = _sum(
        [
            mul(const(float(index)), cohort)
            for index, cohort in enumerate(cohorts, start=1)
        ]
    )
    oldest = cohorts[-1]
    return {
        "total_backlog": total,
        "youngest_backlog": cohorts[0],
        "oldest_backlog": oldest,
        "imminent_abandonment_if_unserved": oldest,
        "retained_backlog_if_unserved": sub(total, oldest),
        "backlog_age_moment": age_moment,
        "mean_backlog_age": safe_ratio(age_moment, total),
        "backlog_to_mean_demand": safe_ratio(total, dist("mu")),
        "demand_mean": dist("mu"),
        "demand_cv": dist("cv"),
    }


__all__ = ["backlog_motifs"]
