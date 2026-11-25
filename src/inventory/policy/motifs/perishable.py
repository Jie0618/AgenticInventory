"""Expectation-free age summaries for perishable operational policies."""

from __future__ import annotations

from inventory.policy.ast import AST, add, const, cost, dist, mul, safe_ratio, state, sub


def _sum(nodes: list[AST]) -> AST:
    if not nodes:
        return const(0.0)
    return nodes[0] if len(nodes) == 1 else add(*nodes)


def perishable_motifs(
    shelf_life: int,
    *,
    decision_includes_freshest: bool = False,
) -> dict[str, AST]:
    if int(shelf_life) <= 0:
        raise ValueError("shelf_life must be positive")
    ages = [state(f"a{index}") for index in range(1, int(shelf_life) + 1)]
    total = _sum(ages)
    # With zero replenishment lead time and the canonical event order, a new
    # order occupies a_M only within the current transition.  After aging, the
    # next pre-order state has carry-over ages a1..a(M-1); a_M is zero on every
    # state reachable from the empty initial condition.  Keep total_inventory
    # for the full schema contract, and expose decision_inventory explicitly
    # for pure-perishable policies and experiments.
    decision_ages = ages if decision_includes_freshest else ages[:-1]
    decision_total = _sum(decision_ages)
    weighted = _sum(
        [mul(const(float(index)), node) for index, node in enumerate(ages, start=1)]
    )
    decision_weighted = _sum(
        [
            mul(const(float(index)), node)
            for index, node in enumerate(decision_ages, start=1)
        ]
    )
    # Experimental, expectation-free summary for agentic SOF discovery. It
    # measures concentration of the observable pre-order inventory profile.
    # Pure immediate-replenishment models exclude the temporary receipt bucket
    # a_M; positive-lead-time compositions include a_M because it is genuine
    # carried on-hand inventory at the next decision epoch.
    decision_second_moment = _sum(
        [mul(node, node) for node in decision_ages]
    )
    return {
        "total_inventory": total,
        "decision_inventory": decision_total,
        "expiring_inventory": ages[0],
        "fresh_inventory": decision_ages[-1] if decision_ages else const(0.0),
        "surviving_inventory": sub(total, ages[0]),
        "inventory_age_moment": weighted,
        "mean_remaining_life": safe_ratio(weighted, total),
        "decision_age_moment": decision_weighted,
        "decision_inventory_second_moment": decision_second_moment,
        "decision_mean_remaining_life": safe_ratio(
            decision_weighted, decision_total
        ),
        "demand_mean": dist("mu"),
        "demand_cv": dist("cv"),
        "cost_ratio": safe_ratio(cost("b"), cost("h")),
        "expiry_holding_ratio": safe_ratio(cost("w"), cost("h")),
    }
