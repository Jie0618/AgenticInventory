from __future__ import annotations

from inventory.policy.ast.core import AST, add, const, cost, dist, mul, safe_ratio, state, sub


def default_motifs(lead_time: int = 1) -> dict[str, AST]:
    return {} if lead_time < 1 else lead_time_motifs(lead_time)


def pipeline_motifs(lead_time: int) -> dict[str, AST]:
    """Return state summaries that depend only on pipeline coordinates."""
    if lead_time < 1:
        raise ValueError("lead_time must be >= 1")
    pipeline = [state(f"p{index}") for index in range(1, lead_time + 1)]
    total_pipeline = add(*pipeline) if len(pipeline) > 1 else pipeline[0]
    weighted_terms = [
        mul(const(float(index)), state(f"p{index}"))
        for index in range(1, lead_time + 1)
    ]
    weighted_lag = add(*weighted_terms) if len(weighted_terms) > 1 else weighted_terms[0]
    tail_width = max(1, lead_time // 2)
    tail_denominator = tail_width * (tail_width + 1) / 2.0
    tail_terms = [
        mul(const(float(tail_width - position) / tail_denominator), state(f"p{lead_time - position}"))
        for position in range(tail_width)
    ]
    init_tail_linear_half = add(*tail_terms) if len(tail_terms) > 1 else tail_terms[0]
    return {
        "total_pipeline": total_pipeline,
        # Preserve the unnormalised first pipeline moment as a reusable O(L)
        # state summary.  It retains both pipeline quantity and timing and is
        # well defined when the pipeline is empty.
        "pipeline_lag_moment": weighted_lag,
        "near_pipeline": state("p1"),
        "far_pipeline": state(f"p{lead_time}"),
        "pipeline_imbalance": sub(state("p1"), state(f"p{lead_time}")),
        "pipeline_mean_arrival_lag": safe_ratio(weighted_lag, total_pipeline),
        "init_tail_linear_half": init_tail_linear_half,
    }


def lead_time_motifs(lead_time: int) -> dict[str, AST]:
    """Return O(L), expectation-free summaries for nonperishable lead time."""

    pipeline = [state(f"p{index}") for index in range(1, lead_time + 1)]
    motifs = pipeline_motifs(lead_time)
    motifs.update({
        "available_inventory": add(state("x"), state("p1")),
        "inventory_position": add(state("x"), *pipeline),
        "demand_mean": dist("mu"),
        "demand_cv": dist("cv"),
        "cost_ratio": safe_ratio(cost("b"), cost("h")),
    })
    cumulative: AST = state("x")
    for index, term in enumerate(pipeline, start=1):
        cumulative = add(cumulative, term)
        motifs[f"cumulative_inventory_{index}"] = cumulative
    return motifs


__all__ = ["default_motifs", "lead_time_motifs", "pipeline_motifs"]
