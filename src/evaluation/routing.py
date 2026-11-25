"""Automatic numerical-backend selection for canonical inventory cases."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from inventory.core import (
    BacklogMode,
    InventoryCase,
    LeadTimeMode,
    PerishabilityMode,
)
from evaluation.reference_cache import validate_identity


LEAD_TIME_ARRAY_BACKEND = "lead_time_array"
COMPOSED_BACKEND = "composed_finite_mdp"


@dataclass(frozen=True)
class BackendDecision:
    """Explain the deterministic hard switch selected for one case."""

    backend: str
    optimized: bool
    reasons: tuple[str, ...]


def select_evaluation_backend(case: InventoryCase) -> BackendDecision:
    """Select the established lead-time backend exactly when semantics match.

    The optimized implementation assumes a deterministic countdown pipeline,
    non-perishable scalar on-hand inventory, lost unmet demand, an ``n``-wide
    projected grid, and no per-order cost. Costs belonging to disabled
    components are inert and do not affect routing. Every other composition
    uses the canonical component kernel. This is an internal hard switch;
    callers never request acceleration.
    """

    if not isinstance(case, InventoryCase):
        raise TypeError("case must be an InventoryCase")

    config = case.config
    costs = case.costs
    reasons: list[str] = []
    if config.lead_time.mode is not LeadTimeMode.DETERMINISTIC:
        reasons.append("lead_time_is_not_deterministic")
    elif int(config.lead_time.periods or 0) == 0:
        # The array L=0 solver chooses an order-up-to level y <= n,
        # whereas the canonical model chooses an order quantity q <= n and
        # projects only after the transition.  They are not the same action
        # space away from an empty state, so L=0 must use the common kernel.
        reasons.append("immediate_replenishment_action_semantics_differ")
    if config.perishability.mode is not PerishabilityMode.DISABLED:
        reasons.append("perishability_is_enabled")
    if config.backlog.mode not in (BacklogMode.DISABLED, BacklogMode.LOST_SALES):
        reasons.append("backlog_is_enabled")
    if config.capacity != case.n:
        reasons.append("projection_capacity_differs_from_action_grid")
    # Costs attached to disabled components are semantically inert. Ordering
    # cost is different: it applies to every positive action and is not
    # represented by the established lead-time array Bellman recursion.
    if costs.ordering != 0.0:
        reasons.append("unsupported_cost_component")

    if reasons:
        return BackendDecision(COMPOSED_BACKEND, False, tuple(reasons))
    return BackendDecision(
        LEAD_TIME_ARRAY_BACKEND,
        True,
        ("deterministic_lead_time_only_semantics_match",),
    )


def solve_inventory_dp(
    case: InventoryCase,
    horizon: int,
    *,
    max_states: int = 2_000_000,
    max_table_cells: int = 50_000_000,
):
    """Solve a case through the automatically selected DP implementation."""

    decision = select_evaluation_backend(case)
    if decision.optimized:
        from evaluation.backends.lead_time.dp import solve_dp

        result = solve_dp(
            lead_time=case.lead_time,
            horizon=horizon,
            n=case.n,
            demand=case.demand,
            h=case.h,
            b=case.b,
        )
        return replace(
            result,
            metadata={
                **dict(result.metadata),
                "backend": decision.backend,
                "backend_reasons": decision.reasons,
                "case_signature": case.case_signature,
                "horizon": int(horizon),
            },
        )

    from evaluation.dp import solve_composed_dp

    result = solve_composed_dp(
        case,
        horizon,
        max_states=max_states,
        max_table_cells=max_table_cells,
    )
    return replace(
        result,
        metadata={
            **dict(result.metadata),
            "backend": decision.backend,
            "backend_reasons": decision.reasons,
            "case_signature": case.case_signature,
            "horizon": int(horizon),
        },
    )


def validate_reference_for_case(case: InventoryCase, reference: object, horizon: int) -> None:
    """Validate signed references and supported table-free lead-time geometry."""

    decision = select_evaluation_backend(case)
    if decision.optimized:
        from evaluation.backends.lead_time.dp import DPResult
        from evaluation.backends.lead_time.models import RollingDPReference

        expected_types = (DPResult, RollingDPReference)
    else:
        from evaluation.dp import ComposedDPResult

        expected_types = (ComposedDPResult,)
    if not isinstance(reference, expected_types):
        expected_names = " or ".join(row.__name__ for row in expected_types)
        raise TypeError(
            f"{decision.backend} requires {expected_names}, "
            f"got {type(reference).__name__}"
        )

    # RollingDPReference deliberately retains only lead-time geometry; the
    # separately cached optimal cost and supervised samples carry the horizon-
    # specific Bellman result.  Validate every mathematical input here, while
    # fit_inventory_policy validates those companion artifacts and evaluator.
    if decision.optimized and isinstance(reference, RollingDPReference):
        demand_matches = (
            np.array_equal(reference.demand.values, case.demand.values)
            and np.allclose(
                reference.demand.probs,
                case.demand.probs,
                rtol=0.0,
                atol=1.0e-14,
            )
        )
        if not (
            int(horizon) > 0
            and reference.lead_time == case.lead_time
            and reference.n == case.n
            and reference.h == case.h
            and reference.b == case.b
            and demand_matches
        ):
            raise ValueError("rolling reference inputs do not match case")
        return

    metadata = dict(getattr(reference, "metadata", {}))
    validate_identity(
        reference,
        case_signature=case.case_signature,
        horizon=horizon,
        expected_backend=decision.backend,
    )
    signature = metadata.get("case_signature")
    if signature is not None:
        if str(signature) != case.case_signature:
            raise ValueError("reference does not belong to the supplied case")
    elif decision.optimized:
        demand_matches = (
            np.array_equal(reference.demand.values, case.demand.values)
            and np.allclose(
                reference.demand.probs,
                case.demand.probs,
                rtol=0.0,
                atol=1.0e-14,
            )
        )
        if not (
            reference.lead_time == case.lead_time
            and reference.n == case.n
            and reference.h == case.h
            and reference.b == case.b
            and demand_matches
        ):
            raise ValueError("legacy reference inputs do not match case")
    else:
        raise ValueError(
            "unsigned composed reference cannot be verified against the case"
        )


__all__ = [
    "BackendDecision",
    "COMPOSED_BACKEND",
    "LEAD_TIME_ARRAY_BACKEND",
    "select_evaluation_backend",
    "solve_inventory_dp",
    "validate_reference_for_case",
]
