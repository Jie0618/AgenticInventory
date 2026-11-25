"""Backlog survival and finite-state projection helpers."""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Sequence

from inventory.common.numeric import integer_scalar
from inventory.core.config import BacklogMode, InventoryModelConfig


InventoryConsumer = Callable[
    [tuple[int, ...], int], tuple[tuple[int, ...], int, int]
]


def project_backlog(
    values: Sequence[int], capacity: Optional[int]
) -> tuple[tuple[int, ...], int]:
    """Discard youngest obligations first while preserving older demand."""

    result = [int(value) for value in values]
    if capacity is None:
        return tuple(result), 0
    excess = max(0, sum(result) - int(capacity))
    discarded = 0
    for index in range(len(result)):
        removed = min(result[index], excess)
        result[index] -= removed
        excess -= removed
        discarded += removed
        if excess == 0:
            break
    if excess:
        raise RuntimeError("backlog capacity projection failed")
    return tuple(result), discarded


def serve_backlog(
    inventory: tuple[int, ...],
    backlog: Sequence[int],
    consume: InventoryConsumer,
    *,
    scalar_total: bool,
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    """Serve backlog FIFO, consuming the oldest cohort before younger ones."""

    remaining_backlog = [int(value) for value in backlog]
    served_total = 0
    if not remaining_backlog:
        return inventory, (), 0
    if len(remaining_backlog) == 1 and scalar_total:
        inventory, served_total, _ = consume(inventory, remaining_backlog[0])
        remaining_backlog[0] -= served_total
        return inventory, tuple(remaining_backlog), served_total
    for index in range(len(remaining_backlog) - 1, -1, -1):
        inventory, served, _ = consume(inventory, remaining_backlog[index])
        remaining_backlog[index] -= served
        served_total += served
    return inventory, tuple(remaining_backlog), served_total


def classify_unmet(mode: BacklogMode, unmet: int) -> tuple[int, int]:
    """Return ``(new_backlog, lost_sales)`` for current unmet demand."""

    if mode in (BacklogMode.DISABLED, BacklogMode.LOST_SALES):
        return 0, int(unmet)
    return int(unmet), 0


def advance_backlog(
    config: InventoryModelConfig,
    backlog: Sequence[int],
    new_backlog: int,
    abandoned: int,
) -> tuple[tuple[int, ...], int, int, int]:
    """Apply survival, patience aging, and finite-grid projection.

    Returns ``(next_backlog, eligible_geometric, abandoned, discarded)``.
    """

    values = tuple(int(value) for value in backlog)
    mode = config.backlog.mode
    if mode in (BacklogMode.DISABLED, BacklogMode.LOST_SALES):
        return (), 0, 0, 0
    if len(values) == 1 and config.backlog.uses_scalar_total:
        if mode is BacklogMode.GEOMETRIC:
            eligible = values[0] + int(new_backlog)
            abandoned_i = integer_scalar(abandoned, name="event.abandoned")
            if abandoned_i > eligible:
                raise ValueError("event.abandoned exceeds eligible geometric backlog")
            total = eligible - abandoned_i
        else:
            eligible = values[0]
            abandoned_i = 0
            total = eligible + int(new_backlog)
        projected, discarded = project_backlog((total,), config.capacity)
        return projected, eligible, abandoned_i, discarded

    # Finite deterministic patience drops the oldest cohort after service.
    abandoned_i = values[-1] if values else 0
    shifted = (int(new_backlog), *values[:-1]) if values else ()
    projected, discarded = project_backlog(shifted, config.capacity)
    return projected, 0, abandoned_i, discarded


def binomial_branches(total: int, survival: float) -> tuple[tuple[int, float], ...]:
    """Return ``(abandoned, probability)`` branches for geometric patience."""

    total = int(total)
    if total < 0:
        raise ValueError("eligible backlog must be nonnegative")
    rho = float(survival)
    abandon_probability = 1.0 - rho
    if abandon_probability == 0.0:
        return ((0, 1.0),)
    if abandon_probability == 1.0:
        return ((total, 1.0),)
    branches: list[tuple[int, float]] = []
    for abandoned in range(total + 1):
        log_probability = (
            math.lgamma(total + 1)
            - math.lgamma(abandoned + 1)
            - math.lgamma(total - abandoned + 1)
            + abandoned * math.log(abandon_probability)
            + (total - abandoned) * math.log(rho)
        )
        probability = math.exp(log_probability)
        if probability > 0.0:
            branches.append((abandoned, float(probability)))
    mass = math.fsum(probability for _, probability in branches)
    if mass <= 0.0:
        raise RuntimeError("binomial branch probabilities underflowed")
    return tuple((value, probability / mass) for value, probability in branches)


def binomial_sample(rng: Any, total: int, probability: float) -> int:
    if hasattr(rng, "binomial"):
        return int(rng.binomial(int(total), float(probability)))
    return sum(
        float(rng.random()) < float(probability) for _ in range(int(total))
    )


__all__ = [
    "InventoryConsumer",
    "advance_backlog",
    "binomial_branches",
    "binomial_sample",
    "classify_unmet",
    "project_backlog",
    "serve_backlog",
]
