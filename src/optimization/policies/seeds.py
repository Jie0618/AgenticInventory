"""Optimization-owned policy seed catalog selected from a canonical case."""

from __future__ import annotations

from inventory.core import InventoryCase
from inventory.policy.ast import AST
from optimization.policies.baselines.lead_time import baselines_for_lead_time
from optimization.policies.baselines.perishable import baselines_for_perishable


def policy_seed_asts_for_case(case: InventoryCase) -> tuple[tuple[str, AST], ...]:
    """Return optimized-backend baselines when the case supports that catalog."""

    if case.config.composition_key == "lead_time":
        rows = baselines_for_lead_time(case.lead_time, case.n)
    elif case.config.composition_key == "perishability":
        rows = baselines_for_perishable(
            case.config.perishability.max_shelf_life,
            case.n,
        )
    else:
        return ()
    return tuple((f"baseline:{spec.name}", spec.ast) for spec in rows)


__all__ = ["policy_seed_asts_for_case"]
