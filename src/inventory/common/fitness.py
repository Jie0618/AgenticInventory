"""Shared equal-weight mean/CVaR fitness for search and reporting."""
from __future__ import annotations

import math
from statistics import fmean

ALPHA = 0.80
TAIL_WEIGHT = 0.40
SCORE_PROTOCOL = "mean-cvar80-60-40"


def empirical_cvar(values, alpha=ALPHA):
    """Upper-tail CVaR, assigning fractional mass at the boundary."""
    ordered = sorted(map(float, values), reverse=True)
    if not ordered or not 0 <= alpha < 1 or not all(map(math.isfinite, ordered)):
        raise ValueError("CVaR requires finite nonempty losses and 0 <= alpha < 1")
    mass = (1 - alpha) * len(ordered)
    return sum(value * max(0.0, min(1.0, mass - index))
               for index, value in enumerate(ordered)) / mass


def metrics(values):
    values = list(map(float, values))
    tail = empirical_cvar(values)
    mean = fmean(values)
    return {"count": len(values), "mean_gap_pct": mean, "cvar80_gap_pct": tail,
            "fitness": (1 - TAIL_WEIGHT) * mean + TAIL_WEIGHT * tail}
