"""Occupancy-weighted information loss under a state abstraction."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Hashable, Mapping

import numpy as np


@dataclass
class StateAbstractionGroup:
    mass: float
    action_mass: np.ndarray
    q_cost_mass: np.ndarray
    optimal_q_cost_mass: float


def observe_state_group(
    groups: dict[Hashable, StateAbstractionGroup],
    key: Hashable,
    mass: float,
    action: int,
    q_costs: np.ndarray,
) -> None:
    """Accumulate one concrete state's occupancy and Bellman action costs."""

    costs = np.asarray(q_costs, dtype=float)
    group = groups.get(key)
    if group is None:
        group = StateAbstractionGroup(
            mass=0.0,
            action_mass=np.zeros(len(costs), dtype=float),
            q_cost_mass=np.zeros(len(costs), dtype=float),
            optimal_q_cost_mass=0.0,
        )
        groups[key] = group
    if costs.shape != group.q_cost_mass.shape:
        raise ValueError("all states in one abstraction must share an action space")
    if not 0 <= int(action) < len(costs):
        raise ValueError("action lies outside the Q-cost vector")
    if float(mass) < 0.0 or not np.isfinite(float(mass)):
        raise ValueError("occupancy mass must be finite and nonnegative")
    group.mass += float(mass)
    group.action_mass[int(action)] += float(mass)
    group.q_cost_mass += float(mass) * costs
    group.optimal_q_cost_mass += float(mass) * float(np.min(costs))


def summarize_state_groups(
    groups: Mapping[Hashable, StateAbstractionGroup],
    *,
    reference_average_cost: float,
    action_mass_tolerance: float = 1.0e-10,
) -> dict[str, float | int]:
    """Measure action ambiguity and one-step regret caused by an abstraction."""

    total_mass = float(sum(group.mass for group in groups.values()))
    if total_mass <= 0.0:
        raise ValueError("cannot summarize empty state groups")
    majority_correct = 0.0
    action_squared_error = 0.0
    ambiguous_mass = 0.0
    bellman_regret = 0.0
    for group in groups.values():
        majority_correct += float(np.max(group.action_mass))
        action_grid = np.arange(len(group.action_mass))
        action_mean = float(np.dot(action_grid, group.action_mass) / group.mass)
        action_squared_error += float(
            np.dot((action_grid - action_mean) ** 2, group.action_mass)
        )
        meaningful_actions = int(
            np.sum(group.action_mass > action_mass_tolerance * group.mass)
        )
        if meaningful_actions > 1:
            ambiguous_mass += group.mass
        bellman_regret += max(
            0.0,
            float(np.min(group.q_cost_mass)) - float(group.optimal_q_cost_mass),
        )
    regret_per_visited_period = bellman_regret / total_mass
    return {
        "groups": len(groups),
        "visited_mass": total_mass,
        "majority_action_error": 1.0 - majority_correct / total_mass,
        "conditional_action_std": math.sqrt(
            max(action_squared_error / total_mass, 0.0)
        ),
        "ambiguous_group_mass": ambiguous_mass / total_mass,
        "bellman_regret_per_visited_period": regret_per_visited_period,
        "bellman_regret_pct_of_dp_average_cost": (
            100.0
            * regret_per_visited_period
            / max(float(reference_average_cost), 1.0e-12)
        ),
    }


__all__ = [
    "StateAbstractionGroup",
    "observe_state_group",
    "summarize_state_groups",
]
