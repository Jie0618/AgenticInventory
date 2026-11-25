"""Dense exact policy-cost helpers for lead-time inventory."""

from __future__ import annotations

from itertools import product
from typing import Any, Callable, Mapping

import numpy as np

from evaluation.backends.lead_time.dp import DPResult
from inventory.policy.ast import AST

from optimization.batch.analysis import _BatchEvaluationContext, _node_dependencies
from optimization.batch.eval import _evaluate_ast_batch
from optimization.batch.features import _round_actions
from optimization.batch.samples import _BatchSamples
from optimization.support.utils import _initial_state, _state_dict


def evaluate_ast_policy_cost_exact(
    ast: AST,
    dp_result: DPResult,
    params: Mapping[str, float],
    motifs: Mapping[str, AST],
    rounding: str = "round",
    action_cap: int | None = None,
    initial_state: tuple[int, ...] | None = None,
) -> float:
    """Evaluate exact policy cost while batch-evaluating AST actions by period."""

    table_shape = (dp_result.horizon + 1,) + (dp_result.n + 1,) * (dp_result.lead_time + 1)
    value_table = np.zeros(table_shape, dtype=float)
    dist_features = dp_result.demand.feature_dict()
    cost_features = {"h": dp_result.h, "b": dp_result.b}
    demand_values = dp_result.demand.values.astype(int)
    demand_probs = dp_result.demand.probs.astype(float)
    state_ranges = [range(dp_result.n + 1) for _ in range(dp_result.lead_time + 1)]
    state_tuples = list(product(*state_ranges))
    state_array = np.asarray(state_tuples, dtype=int)
    state_dims = (dp_result.n + 1,) * (dp_result.lead_time + 1)
    action_cap = dp_result.n if action_cap is None else action_cap
    dependencies = _node_dependencies(ast, motifs)
    fixed_actions: np.ndarray | None = None
    if "remaining_horizon" not in dependencies.states:
        fixed_batch = _BatchSamples.from_state_array(state_array, remaining_horizon=1.0)
        fixed_context = _BatchEvaluationContext(fixed_batch, motifs)
        fixed_raw = _evaluate_ast_batch(
            ast,
            fixed_batch,
            params,
            dist_features,
            cost_features,
            motifs,
            context=fixed_context,
        )
        if not np.all(np.isfinite(fixed_raw)):
            raise ValueError("Policy formula produced NaN or Inf")
        fixed_actions = np.clip(_round_actions(fixed_raw, rounding, action_cap).astype(int), 0, dp_result.n)

    for t in range(dp_result.horizon - 1, -1, -1):
        if fixed_actions is None:
            batch = _BatchSamples.from_state_array(state_array, remaining_horizon=float(dp_result.horizon - t))
            context = _BatchEvaluationContext(batch, motifs)
            raw_actions = _evaluate_ast_batch(
                ast,
                batch,
                params,
                dist_features,
                cost_features,
                motifs,
                context=context,
            )
            if not np.all(np.isfinite(raw_actions)):
                raise ValueError("Policy formula produced NaN or Inf")
            actions = np.clip(_round_actions(raw_actions, rounding, action_cap).astype(int), 0, dp_result.n)
        else:
            actions = fixed_actions
        expected_values = np.zeros(len(state_tuples), dtype=float)
        next_values_flat = value_table[t + 1].reshape(-1)
        if dp_result.lead_time == 0:
            x = state_array[:, 0]
            y = np.minimum(x + actions, dp_result.n)
            for demand_value, demand_prob in zip(demand_values, demand_probs):
                next_x = np.clip(y - int(demand_value), 0, dp_result.n)
                cost = dp_result.h * np.maximum(y - int(demand_value), 0) + dp_result.b * np.maximum(int(demand_value) - y, 0)
                expected_values += float(demand_prob) * (cost + next_values_flat[next_x])
        else:
            x = state_array[:, 0]
            pipeline = state_array[:, 1:]
            available = x + pipeline[:, 0]
            for demand_value, demand_prob in zip(demand_values, demand_probs):
                next_x = np.clip(available - int(demand_value), 0, dp_result.n)
                next_columns = [next_x]
                if dp_result.lead_time > 1:
                    next_columns.extend([pipeline[:, idx] for idx in range(1, dp_result.lead_time)])
                next_columns.append(actions)
                next_index = np.ravel_multi_index(tuple(next_columns), state_dims)
                cost = dp_result.h * np.maximum(available - int(demand_value), 0) + dp_result.b * np.maximum(int(demand_value) - available, 0)
                expected_values += float(demand_prob) * (cost + next_values_flat[next_index])
        value_table[t].reshape(-1)[:] = expected_values

    state = _initial_state(dp_result, initial_state)
    return float(value_table[(0,) + state])

def evaluate_policy_cost_exact(
    policy: Callable[[Any, Mapping[str, Any], Mapping[str, float]], int],
    dp_result: DPResult,
    initial_state: tuple[int, ...] | None = None,
) -> float:
    """Evaluate one fixed policy by exact finite-horizon DP recursion.

    The policy can be any callable with the same signature returned by
    `compile_policy`.  This is the policy-cost objective used by constant
    optimization; it is exact on the truncated grid, not Monte Carlo.
    """

    table_shape = (dp_result.horizon + 1,) + (dp_result.n + 1,) * (dp_result.lead_time + 1)
    value_table = np.zeros(table_shape, dtype=float)
    dist_features = dp_result.demand.feature_dict()
    cost_features = {"h": dp_result.h, "b": dp_result.b}
    demand_values = dp_result.demand.values.astype(int)
    demand_probs = dp_result.demand.probs.astype(float)
    state_ranges = [range(dp_result.n + 1) for _ in range(dp_result.lead_time + 1)]

    for t in range(dp_result.horizon - 1, -1, -1):
        for state_tuple in product(*state_ranges):
            state_features = _state_dict(tuple(state_tuple), dp_result.horizon - t)
            q = int(policy(state_features, dist_features, cost_features))
            q = min(max(q, 0), dp_result.n)
            expected_cost = 0.0
            if dp_result.lead_time == 0:
                x = state_tuple[0]
                y = min(x + q, dp_result.n)
                for demand_value, demand_prob in zip(demand_values, demand_probs):
                    next_x = min(max(y - int(demand_value), 0), dp_result.n)
                    cost = dp_result.h * max(y - int(demand_value), 0) + dp_result.b * max(int(demand_value) - y, 0)
                    expected_cost += float(demand_prob) * (cost + float(value_table[(t + 1, next_x)]))
            else:
                x = state_tuple[0]
                pipeline = state_tuple[1:]
                available = x + pipeline[0]
                for demand_value, demand_prob in zip(demand_values, demand_probs):
                    next_x = min(max(available - int(demand_value), 0), dp_result.n)
                    next_state = (next_x,) + pipeline[1:] + (q,)
                    cost = dp_result.h * max(available - int(demand_value), 0) + dp_result.b * max(int(demand_value) - available, 0)
                    expected_cost += float(demand_prob) * (cost + float(value_table[(t + 1,) + next_state]))
            value_table[(t,) + tuple(state_tuple)] = expected_cost

    state = _initial_state(dp_result, initial_state)
    return float(value_table[(0,) + state])
