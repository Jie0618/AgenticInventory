"""Inventory consumption, lifetime selection, aging, and projection helpers."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from inventory.common.numeric import integer_scalar
from inventory.core.config import InventoryModelConfig, PerishabilityMode


def consume_nonperishable(inventory: int, quantity: int) -> tuple[int, int]:
    used = min(int(inventory), int(quantity))
    return int(inventory) - used, used


def consume_fefo(
    inventory: Sequence[int], quantity: int
) -> tuple[tuple[int, ...], int, int]:
    remaining = [int(value) for value in inventory]
    need = int(quantity)
    served = 0
    for index in range(len(remaining)):
        used = min(remaining[index], need)
        remaining[index] -= used
        need -= used
        served += used
        if need == 0:
            break
    return tuple(remaining), served, need


def project_inventory(
    values: Sequence[int], capacity: Optional[int]
) -> tuple[tuple[int, ...], int]:
    """Project by discarding freshest units first."""

    result = [int(value) for value in values]
    if capacity is None:
        return tuple(result), 0
    excess = max(0, sum(result) - int(capacity))
    discarded = 0
    for index in range(len(result) - 1, -1, -1):
        removed = min(result[index], excess)
        result[index] -= removed
        excess -= removed
        discarded += removed
        if excess == 0:
            break
    if excess:
        raise RuntimeError("inventory capacity projection failed")
    return tuple(result), discarded


def lifetime_choices(
    config: InventoryModelConfig,
    incoming: int,
    realized_lifetime: Any = None,
) -> tuple[tuple[Optional[int], float], ...]:
    perishability = config.perishability
    if not perishability.enabled or incoming == 0:
        return ((None, 1.0),)
    if perishability.mode is PerishabilityMode.DETERMINISTIC:
        return ((int(perishability.shelf_life or 0), 1.0),)
    if realized_lifetime is not None:
        lifetime = integer_scalar(
            realized_lifetime, name="event.lifetime", minimum=1
        )
        if lifetime not in perishability.lifetime_support:
            raise ValueError(
                f"event lifetime {lifetime} is outside configured support"
            )
        return ((lifetime, 1.0),)
    return tuple(
        zip(
            perishability.lifetime_support,
            perishability.lifetime_probabilities,
        )
    )


def receive_inventory(
    config: InventoryModelConfig,
    inventory: Sequence[int],
    incoming: int,
    lifetime: Optional[int],
) -> tuple[int, ...]:
    """Insert a due receipt using the configured age representation."""

    values = [int(value) for value in inventory]
    if not config.perishability.enabled:
        return (values[0] + int(incoming),)
    if incoming:
        life = int(lifetime or config.perishability.max_shelf_life)
        if life <= 0 or life > len(values):
            raise ValueError(
                f"realized lifetime {life} is incompatible with inventory schema"
            )
        values[life - 1] += int(incoming)
    return tuple(values)


def consume_inventory(
    config: InventoryModelConfig,
    inventory: tuple[int, ...],
    quantity: int,
) -> tuple[tuple[int, ...], int, int]:
    """Consume inventory using FEFO or the scalar nonperishable rule."""

    if config.perishability.enabled:
        return consume_fefo(inventory, quantity)
    remaining, served = consume_nonperishable(inventory[0], quantity)
    return (remaining,), served, int(quantity) - served


def age_and_project_inventory(
    config: InventoryModelConfig,
    inventory: tuple[int, ...],
) -> tuple[tuple[int, ...], int, int, int]:
    """Return next inventory, pre-aging holding quantity, expiry, and discard."""

    ending_inventory = int(sum(inventory))
    if config.perishability.enabled:
        expired = int(inventory[0])
        aged = tuple((*inventory[1:], 0))
        projected, discarded = project_inventory(aged, config.capacity)
        return projected, ending_inventory, expired, discarded
    projected, discarded = project_inventory(inventory, config.capacity)
    return projected, ending_inventory, 0, discarded


__all__ = [
    "age_and_project_inventory",
    "consume_fefo",
    "consume_inventory",
    "consume_nonperishable",
    "lifetime_choices",
    "project_inventory",
    "receive_inventory",
]
