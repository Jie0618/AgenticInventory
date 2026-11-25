"""Compiled finite-state transitions shared by generic DP and policy evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from inventory.core import InventoryCase, InventoryState


@dataclass(frozen=True)
class DeterministicTransitionTable:
    """Dense lookup table for models deterministic conditional on demand.

    The canonical kernel remains the source of truth.  Compilation calls it
    once for every finite ``(state, action, demand)`` triple, after which hot
    DP/Exact/MC loops use integer state ids and NumPy gathers only.
    """

    states: tuple[InventoryState, ...]
    state_array: np.ndarray
    next_state_ids: np.ndarray
    costs: np.ndarray
    discarded: np.ndarray
    demand_values: np.ndarray
    demand_probabilities: np.ndarray
    initial_state_id: int

    @property
    def state_count(self) -> int:
        return int(self.state_array.shape[0])

    @property
    def action_count(self) -> int:
        return int(self.next_state_ids.shape[1])


def compile_deterministic_transition_table(
    case: InventoryCase,
    states: Sequence[InventoryState],
    state_index: Mapping[InventoryState, int],
    *,
    max_entries: int = 20_000_000,
) -> DeterministicTransitionTable | None:
    """Compile a deterministic-given-demand model, or return ``None``.

    Random lead times, random shelf lives, and abandonment can produce more
    than one branch after demand is fixed.  Those compositions retain the
    canonical scalar evaluator until a sampled-branch table is introduced.
    """

    canonical_states = tuple(states)
    state_array = np.asarray(
        [state.flatten(case.schema) for state in canonical_states], dtype=np.int32
    )
    all_demand_values = np.asarray(case.demand.values, dtype=np.int32)
    all_demand_probabilities = np.asarray(case.demand.probs, dtype=float)
    positive = all_demand_probabilities > 0.0
    demand_values = all_demand_values[positive]
    demand_probabilities = all_demand_probabilities[positive]
    shape = (len(canonical_states), case.n + 1, len(demand_values))
    if int(np.prod(shape, dtype=np.int64)) > int(max_entries):
        return None
    next_state_ids = np.empty(shape, dtype=np.int32)
    costs = np.empty(shape, dtype=float)
    discarded = np.empty(shape, dtype=float)
    kernel = case.kernel
    for state_id, state in enumerate(canonical_states):
        for action in range(case.n + 1):
            for demand_id, demand in enumerate(demand_values):
                branches = kernel.outcomes(state, action, int(demand))
                if len(branches) != 1 or not np.isclose(
                    float(branches[0].probability), 1.0, rtol=0.0, atol=1.0e-14
                ):
                    return None
                outcome = branches[0].outcome
                try:
                    next_id = state_index[outcome.next_state]
                except KeyError as error:
                    raise RuntimeError(
                        "canonical transition left its finite state grid: "
                        f"{outcome.next_state}"
                    ) from error
                next_state_ids[state_id, action, demand_id] = int(next_id)
                costs[state_id, action, demand_id] = float(outcome.total_cost)
                discarded[state_id, action, demand_id] = float(outcome.discarded)
    return DeterministicTransitionTable(
        states=canonical_states,
        state_array=state_array,
        next_state_ids=next_state_ids,
        costs=costs,
        discarded=discarded,
        demand_values=demand_values,
        demand_probabilities=demand_probabilities,
        initial_state_id=int(state_index[case.initial_state]),
    )


__all__ = [
    "DeterministicTransitionTable",
    "compile_deterministic_transition_table",
]
