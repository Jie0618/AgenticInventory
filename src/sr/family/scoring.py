"""The paper's equal-weight mean--CVaR80 score at both aggregation levels."""

from __future__ import annotations

import math
from typing import Sequence

from sr.family.models import CaseMetric, ScenarioScore
from inventory.common.fitness import SCORE_PROTOCOL, metrics as fitness_metrics

def validate_score_protocol(protocol: str) -> str:
    if protocol != SCORE_PROTOCOL:
        raise ValueError(f"unsupported score protocol: {protocol!r}; expected {SCORE_PROTOCOL}")
    return protocol


def scenario_score(
    metrics: Sequence[CaseMetric],
    context_id: str,
    *, protocol: str = SCORE_PROTOCOL,
) -> ScenarioScore:
    validate_score_protocol(protocol)
    rows = [row for row in metrics if row.context_id == str(context_id) and row.status == "complete"]
    if not rows:
        raise ValueError(f"no complete metrics for context={context_id}")
    # Preserve the sign: a negative gap means that the candidate beats the
    # declared reference.  Clipping here makes all improvements tie at zero
    # and destroys the ordering needed by Stage III search.
    gaps = [float(row.gap_pct) for row in rows]
    values = fitness_metrics(gaps)
    return ScenarioScore(str(context_id), values["mean_gap_pct"], values["cvar80_gap_pct"],
                         values["fitness"])


def cross_context_score(scores: Sequence[ScenarioScore], *, protocol: str = SCORE_PROTOCOL) -> float:
    validate_score_protocol(protocol)
    values = [float(score.robust_score) for score in scores]
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("all context scores require finite robust scores")
    return fitness_metrics(values)["fitness"]


__all__ = [
    "cross_context_score",
    "scenario_score",
    "validate_score_protocol",
]
