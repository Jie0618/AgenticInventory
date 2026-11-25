"""Small finite-state Bellman solver for the compositional inventory model.

This solver is a correctness/reference backend, not a replacement for the
specialized high-throughput lead-time solvers used by the existing research
benchmarks.  It intentionally calls the same ``outcomes`` method as the
canonical exact evaluator, so deterministic transitions are simply one-point
stochastic supports and all component combinations share one implementation.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from inventory.common.numeric import integer_scalar
from inventory.core.case import InventoryCase
from inventory.core.state import CanonicalStateSchema, InventoryState
from evaluation.compiled import (
    DeterministicTransitionTable,
    compile_deterministic_transition_table,
)


State = InventoryState
DEFAULT_MAX_TABLE_CELLS = 50_000_000


def _integer(value: object, *, name: str, minimum: int | None = None) -> int:
    """Normalize an integer-valued public argument without truncation."""

    return integer_scalar(value, name=name, minimum=minimum)


def _state_capacity(case: InventoryCase) -> int:
    capacity = case.config.capacity
    if capacity is None:
        raise ValueError(
            "composed finite-grid DP requires explicit config.capacity; "
            "Exact/MC evaluation can run without state projection"
        )
    return int(capacity)


def _bounded_vectors(
    width: int, cap: int, *, total_cap: bool = False
) -> tuple[tuple[int, ...], ...]:
    width = int(width)
    cap = int(cap)
    if width == 0:
        return ((),)
    if total_cap:
        rows: list[tuple[int, ...]] = []
        for row in itertools.product(range(cap + 1), repeat=width):
            if sum(row) <= cap:
                rows.append(tuple(int(value) for value in row))
        return tuple(rows)
    return tuple(
        tuple(int(value) for value in row)
        for row in itertools.product(range(cap + 1), repeat=width)
    )


def composed_state_count(case: InventoryCase) -> int:
    """Return the exact size of the finite canonical grid without materializing it."""

    if not isinstance(case, InventoryCase):
        raise TypeError("case must be an InventoryCase")
    schema = case.schema
    cap = _state_capacity(case)
    if case.config.perishability.enabled:
        inventory_count = math.comb(
            cap + schema.inventory_dimension, schema.inventory_dimension
        )
    else:
        inventory_count = cap + 1
    pipeline_count = (cap + 1) ** schema.pipeline_dimension
    if case.config.backlog.mode.value == "deterministic":
        # Deterministic patience uses age buckets, but ``n`` caps the total
        # outstanding demand rather than every cohort independently.
        backlog_count = math.comb(
            cap + schema.backlog_dimension, schema.backlog_dimension
        )
    elif schema.backlog_dimension:
        backlog_count = cap + 1
    else:
        backlog_count = 1
    return int(inventory_count * pipeline_count * backlog_count)


def enumerate_composed_states(case: InventoryCase) -> tuple[InventoryState, ...]:
    """Enumerate canonical states in a stable inventory/pipeline/backlog order."""

    if not isinstance(case, InventoryCase):
        raise TypeError("case must be an InventoryCase")
    schema = case.schema
    cap = _state_capacity(case)
    inventory = _bounded_vectors(
        schema.inventory_dimension,
        cap,
        total_cap=case.config.perishability.enabled,
    )
    pipeline = _bounded_vectors(schema.pipeline_dimension, cap)
    backlog = _bounded_vectors(
        schema.backlog_dimension,
        cap,
        total_cap=case.config.backlog.mode.value == "deterministic",
    )
    return tuple(
        InventoryState(a, p, b) for a in inventory for p in pipeline for b in backlog
    )


@dataclass(frozen=True)
class ComposedDPResult:
    """Finite-horizon value and integer-action tables for canonical states."""

    model_id: str
    horizon: int
    n: int
    states: tuple[InventoryState, ...]
    initial_state: InventoryState
    value_table: np.ndarray
    policy_table: np.ndarray
    discard_table: np.ndarray
    state_index: Mapping[InventoryState, int]
    schema: CanonicalStateSchema
    backend: str = "composed_scalar_projected"
    metadata: Mapping[str, object] = field(default_factory=dict)
    compiled_transition: DeterministicTransitionTable | None = field(
        default=None, repr=False, compare=False
    )

    def normalize_state(self, state: InventoryState | Sequence[int]) -> InventoryState:
        normalized = (
            state
            if isinstance(state, InventoryState)
            else InventoryState.from_flat(state, self.schema)
        )
        try:
            self.state_index[normalized]
        except KeyError as error:
            raise ValueError(
                f"state is outside the feasible grid: {normalized}"
            ) from error
        return normalized

    def value(self, period: int, state: InventoryState | Sequence[int]) -> float:
        period_i = _integer(period, name="period", minimum=0)
        if period_i > self.horizon:
            raise ValueError("value period must lie in [0, horizon]")
        normalized = self.normalize_state(state)
        return float(self.value_table[period_i, self.state_index[normalized]])

    def action(self, period: int, state: InventoryState | Sequence[int]) -> int:
        period_i = _integer(period, name="period", minimum=0)
        if period_i >= self.horizon:
            raise ValueError("policy period must lie in [0, horizon)")
        normalized = self.normalize_state(state)
        return int(self.policy_table[period_i, self.state_index[normalized]])

    def expected_discarded(
        self, period: int, state: InventoryState | Sequence[int]
    ) -> float:
        """Expected cumulative projected units under the selected policy."""

        period_i = _integer(period, name="period", minimum=0)
        if period_i > self.horizon:
            raise ValueError("discard period must lie in [0, horizon]")
        normalized = self.normalize_state(state)
        return float(self.discard_table[period_i, self.state_index[normalized]])


def solve_composed_dp(
    case: InventoryCase,
    horizon: int,
    *,
    max_states: int = 2_000_000,
    max_table_cells: int = DEFAULT_MAX_TABLE_CELLS,
) -> ComposedDPResult:
    """Solve an explicitly projected finite-grid case by Bellman recursion."""

    if not isinstance(case, InventoryCase):
        raise TypeError("case must be an InventoryCase")
    horizon = _integer(horizon, name="horizon", minimum=1)
    max_states = _integer(max_states, name="max_states", minimum=1)
    max_table_cells = _integer(max_table_cells, name="max_table_cells", minimum=1)
    count = composed_state_count(case)
    if count > max_states:
        raise MemoryError(
            f"composed scalar oracle requires {count} states, limit is {max_states}"
        )
    # Value and discard tables each have horizon + 1 rows; the policy table
    # has horizon rows.  Count all three arrays before allocating them.
    table_cells = (3 * horizon + 2) * count
    if table_cells > max_table_cells:
        raise MemoryError(
            "composed scalar oracle value/policy/discard tables require "
            f"{table_cells} cells, limit is {max_table_cells}"
        )
    states = enumerate_composed_states(case)
    if len(states) != count:
        raise RuntimeError(
            f"state enumeration produced {len(states)} states, expected {count}"
        )
    state_index = {state: index for index, state in enumerate(states)}
    values = np.zeros((horizon + 1, count), dtype=float)
    policies = np.zeros((horizon, count), dtype=int)
    discards = np.zeros((horizon + 1, count), dtype=float)
    compiled = compile_deterministic_transition_table(case, states, state_index)
    demand_values = np.asarray(case.demand.values, dtype=int)
    demand_probs = np.asarray(case.demand.probs, dtype=float)
    if compiled is not None:
        probabilities = compiled.demand_probabilities.reshape(1, 1, -1)
        for period in range(horizon - 1, -1, -1):
            continuation = values[period + 1][compiled.next_state_ids]
            action_values = np.sum(
                probabilities * (compiled.costs + continuation), axis=2
            )
            best_actions = np.argmin(action_values, axis=1).astype(np.int32)
            state_ids = np.arange(count, dtype=np.int32)
            values[period] = action_values[state_ids, best_actions]
            policies[period] = best_actions
            selected_next = compiled.next_state_ids[state_ids, best_actions]
            selected_discard = compiled.discarded[state_ids, best_actions]
            discards[period] = np.sum(
                compiled.demand_probabilities.reshape(1, -1)
                * (selected_discard + discards[period + 1][selected_next]),
                axis=1,
            )
    else:
        transition = case.kernel
        for period in range(horizon - 1, -1, -1):
            for state_id, state in enumerate(states):
                best_cost = float("inf")
                best_action = 0
                best_discarded = 0.0
                for action in case.feasible_actions(state):
                    expected = 0.0
                    expected_discarded = 0.0
                    for demand, demand_probability in zip(demand_values, demand_probs):
                        if float(demand_probability) == 0.0:
                            continue
                        for branch in transition.outcomes(state, action, int(demand)):
                            if float(branch.probability) == 0.0:
                                continue
                            try:
                                next_id = state_index[branch.next_state]
                            except KeyError as error:
                                raise RuntimeError(
                                    "canonical transition left its finite state grid: "
                                    f"{branch.next_state}"
                                ) from error
                            weight = float(demand_probability) * float(branch.probability)
                            expected += weight * (
                                branch.total_cost + values[period + 1, next_id]
                            )
                            expected_discarded += weight * (
                                float(branch.outcome.discarded)
                                + discards[period + 1, next_id]
                            )
                    if expected < best_cost:
                        best_cost = expected
                        best_action = action
                        best_discarded = expected_discarded
                values[period, state_id] = best_cost
                policies[period, state_id] = best_action
                discards[period, state_id] = best_discarded
    initial_state_id = state_index[case.initial_state]
    result = ComposedDPResult(
        model_id=case.model_id,
        horizon=horizon,
        n=case.n,
        states=states,
        initial_state=case.initial_state,
        value_table=values,
        policy_table=policies,
        discard_table=discards,
        state_index=state_index,
        schema=case.schema,
        backend=(
            "composed_compiled_finite_mdp"
            if compiled is not None
            else "composed_scalar_projected"
        ),
        metadata={
            "expected_cumulative_discarded": float(discards[0, initial_state_id]),
            "projection_capacity": case.config.capacity,
            "case_signature": case.case_signature,
            "compiled_transition": compiled is not None,
        },
        compiled_transition=compiled,
    )
    return result


__all__ = [
    "ComposedDPResult",
    "DEFAULT_MAX_TABLE_CELLS",
    "State",
    "composed_state_count",
    "enumerate_composed_states",
    "solve_composed_dp",
]
