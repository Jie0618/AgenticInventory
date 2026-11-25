"""Canonical budgets and selection rules for constant optimization."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModeRules:
    """Proposal/Exact budgets and route controls for the current pipeline."""

    name: str
    proposal_budget: int
    exact_budget_total: int
    soft_wall_time_seconds: float
    exact_enum_limit: int
    mixed_outer_branch_limit: int
    initial_exact_budget: int
    rolling_exact_budget: int
    final_exact_budget: int
    final_shortlist_budget: int
    overlap_budget: int
    exact_local_budget: int
    behavior_pilot_proposals: int
    local_elite_count: int
    audit_count: int


PIPELINE_RULES = {
    "fast": ModeRules(
        "fast", 512, 32, 120.0, 64, 8, 4, 4, 24, 16, 4, 4, 64, 3, 4
    ),
    "standard": ModeRules(
        "standard", 5_000, 96, 900.0, 256, 32, 8, 16, 72, 40, 16, 16, 128, 5, 8
    ),
    "certify": ModeRules(
        "certify", 20_000, 256, 3_600.0, 512, 64, 16, 48, 192, 96, 48, 48, 256, 8, 16
    ),
}


def certify_dimension_multiplier(parameter_count: int) -> int:
    """Return the effective extra-budget multiplier for a formula dimension.

    Certify uses the configured budget for formulas with at most three free
    constants.  Above that point it adds one unit per parameter, capped at ten
    parameters, so an unusually large AST cannot create an unbounded worker
    budget.  The returned value is used directly as the additive multiplier
    in the Certify budget schedule.
    """

    dimension = max(0, int(parameter_count))
    return max(0, min(dimension, 10) - 3)


def resolve_certify_rules(
    base_rules: ModeRules,
    *,
    parameter_count: int,
    discrete_combination_count: int | None,
    explicit_proposal_override: bool = False,
) -> ModeRules:
    """Resolve Certify budgets for one formula without changing route semantics.

    ``R1_SMALL_DISCRETE_EXACT`` already visits every legal discrete point, so
    its result does not benefit from a larger budget.  Explicit
    ``total_evaluations`` overrides are also left untouched for reproducible
    diagnostic experiments.  All other Certify formulas use the dimension
    schedule documented in ``docs/methods/OPTIMIZATION_RULES.md``.
    """

    if base_rules.name != "certify" or explicit_proposal_override:
        return base_rules
    if (
        discrete_combination_count is not None
        and int(discrete_combination_count) <= int(base_rules.exact_enum_limit)
    ):
        return base_rules

    multiplier = certify_dimension_multiplier(parameter_count)
    if multiplier <= 0:
        return base_rules

    # Dimension-aware Certify schedule.  Exact and shortlist budgets grow
    # alongside proposals because each additional proposal can expose a new
    # basin that must be arbitrated by strict Exact evaluation.
    proposal_budget = 20_000 + 8_000 * multiplier
    exact_budget_total = min(1_024, 256 + 48 * multiplier)
    final_shortlist_budget = min(384, 96 + 18 * multiplier)

    # Keep the phase ceilings consistent with the global Exact ceiling.  The
    # additional reserve is concentrated in final arbitration/local search.
    initial_exact_budget = 16 + 2 * multiplier
    rolling_exact_budget = 48 + 4 * multiplier
    final_exact_budget = exact_budget_total - initial_exact_budget - rolling_exact_budget

    return ModeRules(
        base_rules.name,
        proposal_budget,
        exact_budget_total,
        base_rules.soft_wall_time_seconds,
        base_rules.exact_enum_limit,
        base_rules.mixed_outer_branch_limit,
        initial_exact_budget,
        rolling_exact_budget,
        final_exact_budget,
        final_shortlist_budget,
        min(96, base_rules.overlap_budget + 8 * multiplier),
        min(96, base_rules.exact_local_budget + 8 * multiplier),
        base_rules.behavior_pilot_proposals,
        min(16, base_rules.local_elite_count + multiplier),
        min(32, base_rules.audit_count + 2 * multiplier),
    )


def resolve_pil_scalar_rules(
    base_rules: ModeRules,
    *,
    explicit_proposal_override: bool = False,
) -> ModeRules:
    """Use a moderate Certify budget for PIL's expensive scalar action.

    PIL has only one fitted scalar, so the generic 20k-point differential-
    evolution budget spends most of its wall time revisiting an already dense
    one-dimensional interval.  The dedicated route keeps the same two-tier MC
    contract while reserving enough strict evaluations to preserve solution
    quality. Explicit experiment overrides remain authoritative.
    """

    if base_rules.name != "certify" or explicit_proposal_override:
        return base_rules
    return ModeRules(
        name=base_rules.name,
        proposal_budget=8_000,
        exact_budget_total=144,
        soft_wall_time_seconds=1_200.0,
        exact_enum_limit=base_rules.exact_enum_limit,
        mixed_outer_branch_limit=base_rules.mixed_outer_branch_limit,
        initial_exact_budget=12,
        rolling_exact_budget=24,
        final_exact_budget=108,
        final_shortlist_budget=64,
        overlap_budget=24,
        exact_local_budget=20,
        behavior_pilot_proposals=base_rules.behavior_pilot_proposals,
        local_elite_count=6,
        audit_count=12,
    )
