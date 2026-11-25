"""Factory for the single compositional inventory-case definition."""
from __future__ import annotations

from typing import Mapping

from inventory.common.distributions import DiscreteDemand
from inventory.core.case import InventoryCase
from inventory.core.config import (
    BacklogConfig,
    InventoryModelConfig,
    LeadTimeConfig,
    PerishabilityConfig,
)
from inventory.core.transition import TransitionCosts


def inventory_case(
    *,
    name: str,
    demand: DiscreteDemand,
    n: int,
    costs: TransitionCosts | Mapping[str, object],
    lead_time: LeadTimeConfig | None = None,
    perishability: PerishabilityConfig | None = None,
    backlog: BacklogConfig | None = None,
    capacity: int | None = None,
    kind: str = "custom",
) -> InventoryCase:
    """Build one case by independently selecting the three state components.

    ``None`` disables that component.  Callers must pass typed component
    configs, which keeps deterministic and stochastic semantics explicit and
    avoids combination-specific case classes or constructor branches.
    """

    config = InventoryModelConfig(
        lead_time=lead_time,
        perishability=perishability,
        backlog=backlog,
        capacity=capacity,
    )
    return InventoryCase(name, config, costs, demand, n, kind)


__all__ = ["inventory_case"]
