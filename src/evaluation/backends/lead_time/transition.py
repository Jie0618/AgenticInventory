"""Scalar arithmetic backend for deterministic lead-time evaluation."""

from __future__ import annotations

from typing import Sequence

from evaluation.backends.lead_time.models import FlatTransitionOutcome


def lead_time_transition(
    case: object, state: Sequence[int], action: int, demand: int
) -> FlatTransitionOutcome:
    # Keep this arithmetic specialization in the deterministic lead-time hot path. Exhaustive
    # parity tests compare it with ``inventory.core`` on the complete small
    # grid, while production DP/evaluators avoid per-state adapter overhead.
    lead_time = int(getattr(case, "lead_time"))
    n = int(getattr(case, "n"))
    normalized = tuple(int(value) for value in state)
    if len(normalized) != lead_time + 1:
        raise ValueError(f"expected {lead_time + 1} state coordinates")
    if any(value < 0 or value > n for value in normalized):
        raise ValueError("state is outside the truncated grid")
    action = int(action)
    demand = int(demand)
    if not 0 <= action <= n or demand < 0:
        raise ValueError("action must lie in [0,n] and demand be nonnegative")
    x = normalized[0]
    pipeline = normalized[1:]
    available = x + (action if lead_time == 0 else pipeline[0])
    ending = max(available - demand, 0)
    next_x = min(ending, n)
    next_pipeline = () if lead_time == 0 else tuple((*pipeline[1:], action))
    return FlatTransitionOutcome(
        next_state=(next_x, *next_pipeline),
        holding_cost=float(getattr(case, "h")) * float(ending),
        shortage_cost=float(getattr(case, "b"))
        * float(max(demand - available, 0)),
    )


__all__ = ["lead_time_transition"]
