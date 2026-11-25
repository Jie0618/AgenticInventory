"""Lead-time support selection and known-ETA pipeline evolution."""

from __future__ import annotations

from typing import Any, Optional

from inventory.common.numeric import integer_scalar
from inventory.core.config import InventoryModelConfig, LeadTimeMode
from inventory.core.state import InventoryState


def clip_coordinate(value: int, capacity: Optional[int]) -> tuple[int, int]:
    if capacity is None:
        return int(value), 0
    clipped = min(int(value), int(capacity))
    return clipped, int(value) - clipped


def lead_time_choices(
    config: InventoryModelConfig,
    action: int,
    realized_lead_time: Any = None,
) -> tuple[tuple[int, float], ...]:
    """Return finite ETA choices; stochastic ETA is known when ordering."""

    lead = config.lead_time
    if not lead.enabled or action == 0:
        return ((0, 1.0),)
    if lead.mode is LeadTimeMode.DETERMINISTIC:
        return ((int(lead.periods or 0), 1.0),)
    if realized_lead_time is not None:
        eta = integer_scalar(realized_lead_time, name="event.lead_time")
        if eta not in lead.support:
            raise ValueError(f"event lead_time {eta} is outside configured support")
        return ((eta, 1.0),)
    return tuple(zip(lead.support, lead.probabilities))


def pipeline_and_arrival(
    config: InventoryModelConfig,
    state: InventoryState,
    action: int,
    eta: int,
) -> tuple[tuple[int, ...], int, int]:
    """Shift known-ETA pipeline buckets and return due receipts."""

    pipeline = list(state.pipeline)
    if not config.lead_time.enabled:
        return (), action, 0
    if not pipeline:
        if eta != 0:
            raise ValueError("a zero-dimensional pipeline only accepts ETA=0")
        return (), action, 0
    if eta < 0 or eta > len(pipeline):
        raise ValueError(
            f"realized lead time {eta} is incompatible with pipeline schema"
        )
    arrivals = int(pipeline[0])
    shifted = pipeline[1:] + [0]
    clipped = 0
    if eta == 0:
        arrivals += action
    else:
        index = eta - 1
        shifted[index], clipped = clip_coordinate(
            shifted[index] + action, config.capacity
        )
    return tuple(shifted), arrivals, clipped


__all__ = ["clip_coordinate", "lead_time_choices", "pipeline_and_arrival"]
