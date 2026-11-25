"""Exact and Monte Carlo evaluators for canonical inventory cases.

Both evaluators consume the same :class:`ComposedTransitionKernel`.  Exact
evaluation expands the finite exogenous support returned by ``outcomes``;
Monte Carlo evaluation samples that kernel through ``sample``.  This keeps
stochastic lead-time, known-arrival lifetime, and geometric abandonment
semantics in one place instead of duplicating them in model-specific runners.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from evaluation.models import ExactEvaluationResult, MonteCarloEvaluationResult
from inventory.common.numeric import integer_scalar
from inventory.core.case import InventoryCase
from inventory.core.state import InventoryState
from inventory.core.transition import ComposedTransitionKernel


CanonicalPolicy = Callable[[InventoryState, int], int]


def _integer(value: Any, *, name: str, minimum: int | None = None) -> int:
    """Normalize an integer-valued public argument without truncation."""

    return integer_scalar(value, name=name, minimum=minimum)


def _validate_horizon_paths(
    horizon: int, paths: Optional[int] = None
) -> tuple[int, Optional[int]]:
    normalized_horizon = _integer(horizon, name="horizon", minimum=1)
    if paths is None:
        return normalized_horizon, None
    return normalized_horizon, _integer(paths, name="paths", minimum=1)


def _kernel_for(
    case: InventoryCase, kernel: Optional[ComposedTransitionKernel]
) -> ComposedTransitionKernel:
    if not isinstance(case, InventoryCase):
        raise TypeError("case must be an InventoryCase")
    if kernel is None:
        return case.kernel
    if not isinstance(kernel, ComposedTransitionKernel):
        raise TypeError("kernel must be a ComposedTransitionKernel")
    if kernel.config.composition_hash != case.config.composition_hash:
        raise ValueError("kernel configuration does not match case configuration")
    if kernel.costs != case.costs:
        raise ValueError("kernel costs do not match case costs")
    if kernel.schema != case.schema:
        raise ValueError("kernel schema does not match case schema")
    if kernel.action_cap != case.action_cap:
        raise ValueError("kernel action cap does not match case action cap")
    return kernel


def _policy_action(value: Any) -> int:
    """Normalize one policy action without silently truncating it."""

    return _integer(value, name="policy action")


def evaluate_composed_policy_exact(
    case: InventoryCase,
    policy: CanonicalPolicy,
    horizon: int,
    *,
    kernel: Optional[ComposedTransitionKernel] = None,
    initial_state: Optional[InventoryState | Sequence[int]] = None,
    max_reachable_states: int = 2_000_000,
) -> ExactEvaluationResult:
    """Evaluate a policy by forward propagation over demand and event support."""

    horizon_i, _ = _validate_horizon_paths(horizon)
    if not callable(policy):
        raise TypeError("policy must be callable")
    reachable_limit = _integer(
        max_reachable_states, name="max_reachable_states", minimum=1
    )
    transition = _kernel_for(case, kernel)
    state = case.initial_state if initial_state is None else initial_state
    if not isinstance(state, InventoryState):
        state = InventoryState.from_flat(state, case.schema)
    case.schema.validate(state)
    distribution: dict[InventoryState, float] = {state: 1.0}
    total_cost = 0.0
    expected_discarded = 0.0
    max_reachable = 1
    demand_values = np.asarray(case.demand.values, dtype=int)
    demand_probs = np.asarray(case.demand.probs, dtype=float)
    for period in range(horizon_i):
        remaining = horizon_i - period
        next_distribution: defaultdict[InventoryState, float] = defaultdict(float)
        for current, current_probability in distribution.items():
            if float(current_probability) == 0.0:
                continue
            action = _policy_action(policy(current, remaining))
            for demand, demand_probability in zip(demand_values, demand_probs):
                if float(demand_probability) == 0.0:
                    continue
                demand_mass = float(current_probability) * float(demand_probability)
                if demand_mass == 0.0:
                    continue
                for branch in transition.outcomes(current, action, int(demand)):
                    probability = demand_mass * float(branch.probability)
                    if probability == 0.0:
                        continue
                    total_cost += probability * branch.total_cost
                    expected_discarded += probability * float(branch.outcome.discarded)
                    if (
                        branch.next_state not in next_distribution
                        and len(next_distribution) >= reachable_limit
                    ):
                        raise MemoryError(
                            "composed exact evaluation reached at least "
                            f"{len(next_distribution) + 1} states, "
                            f"limit is {reachable_limit}"
                        )
                    next_distribution[branch.next_state] += probability
        distribution = dict(next_distribution)
        max_reachable = max(max_reachable, len(distribution))
    mass = float(sum(distribution.values()))
    if not np.isclose(mass, 1.0, atol=1.0e-10):
        raise RuntimeError(f"reachable probability mass drifted to {mass}")
    return ExactEvaluationResult(
        cost=float(total_cost),
        completed=True,
        metadata={
            "method": "composed_forward_finite_support_exact",
            "composition_id": case.config.composition_id,
            "max_reachable_states": int(max_reachable),
            "final_reachable_states": int(len(distribution)),
            "probability_mass": mass,
            "expected_cumulative_discarded": float(expected_discarded),
            "projection_capacity": case.config.capacity,
        },
    )


def evaluate_composed_policy_mc(
    case: InventoryCase,
    policy: CanonicalPolicy,
    horizon: int,
    *,
    paths: int,
    seed: int,
    kernel: Optional[ComposedTransitionKernel] = None,
    initial_state: Optional[InventoryState | Sequence[int]] = None,
) -> MonteCarloEvaluationResult:
    """Evaluate a policy by sampling demand and component events."""

    horizon_i, paths_i = _validate_horizon_paths(horizon, paths)
    assert paths_i is not None
    if not callable(policy):
        raise TypeError("policy must be callable")
    seed_i = _integer(seed, name="seed")
    transition = _kernel_for(case, kernel)
    initial = case.initial_state if initial_state is None else initial_state
    if not isinstance(initial, InventoryState):
        initial = InventoryState.from_flat(initial, case.schema)
    case.schema.validate(initial)
    rng = np.random.default_rng(seed_i)
    demand_values = np.asarray(case.demand.values, dtype=int)
    demand_probs = np.asarray(case.demand.probs, dtype=float)
    path_costs = np.zeros(paths_i, dtype=float)
    path_discarded = np.zeros(paths_i, dtype=float)
    for path in range(paths_i):
        state = initial
        cost = 0.0
        discarded = 0
        for period in range(horizon_i):
            remaining = horizon_i - period
            action = _policy_action(policy(state, remaining))
            demand = int(rng.choice(demand_values, p=demand_probs))
            outcome = transition.sample(state, action, demand, rng=rng)
            cost += outcome.total_cost
            discarded += int(outcome.discarded)
            state = outcome.next_state
        path_costs[path] = cost
        path_discarded[path] = float(discarded)
    standard_error = (
        0.0 if paths_i == 1 else float(np.std(path_costs, ddof=1) / np.sqrt(paths_i))
    )
    discarded_standard_error = (
        0.0
        if paths_i == 1
        else float(np.std(path_discarded, ddof=1) / np.sqrt(paths_i))
    )
    return MonteCarloEvaluationResult(
        mean_cost=float(np.mean(path_costs)),
        standard_error=standard_error,
        paths=paths_i,
        seed=seed_i,
        metadata={
            "method": "composed_monte_carlo",
            "composition_id": case.config.composition_id,
            "mean_cumulative_discarded": float(np.mean(path_discarded)),
            "discarded_standard_error": discarded_standard_error,
            "projection_capacity": case.config.capacity,
        },
    )


# Short aliases for callers that already use the core package as their model
# boundary.  The explicit ``composed`` names remain the least ambiguous when
# imported alongside ``inventory.common.evaluate_policy_*``.
evaluate_policy_exact = evaluate_composed_policy_exact
evaluate_policy_mc = evaluate_composed_policy_mc


__all__ = [
    "CanonicalPolicy",
    "evaluate_composed_policy_exact",
    "evaluate_composed_policy_mc",
    "evaluate_policy_exact",
    "evaluate_policy_mc",
]
