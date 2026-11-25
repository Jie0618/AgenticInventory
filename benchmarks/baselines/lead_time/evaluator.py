"""Adapter from lead-time action providers to the lead-time exact evaluator."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from evaluation.backends.lead_time.dp import DPResult
from inventory.policy.ast import AST, add, state
from optimization import SparseExactPolicyCostEvaluator
from optimization.models.types import EvaluationResult

from benchmarks.baselines.lead_time.policies import (
    ComputationalActionProvider,
)


def dependency_sentinel_ast(lead_time: int, *, horizon_dependent: bool) -> AST:
    """Describe dependencies to the generic exact engine without adding AST logic."""

    terms = [state("x"), *(state(f"p{index}") for index in range(1, lead_time + 1))]
    if horizon_dependent:
        terms.append(state("remaining_horizon"))
    return add(*terms)


class ComputationalActionAdapter:
    """Lightweight provider adapter for trajectory-based evaluators.

    Unlike ``ComputationalBaselineEvaluator``, this adapter builds no exact
    state geometry. It is therefore safe for MC benchmarks at L=10 while
    preserving provider validation and randomized-action semantics.
    """

    def __init__(self, provider: ComputationalActionProvider, reference: DPResult) -> None:
        self.provider = provider
        self.reference = reference
        self._randomized_actions = getattr(provider, "action_distribution", None)

    @property
    def optimization_route_hint(self) -> str | None:
        value = getattr(self.provider, "optimization_route_hint", None)
        return None if value is None else str(value)

    def _actions(
        self,
        states: np.ndarray,
        params: Mapping[str, float],
        dist_features: Mapping[str, Any],
        cost_features: Mapping[str, float],
        remaining_horizon: int,
    ) -> np.ndarray:
        del dist_features, cost_features
        actions = np.asarray(
            self.provider.actions(states, params, remaining_horizon), dtype=np.int32
        )
        self._validate_actions(actions, len(states))
        return actions

    def _actions_many(
        self,
        states: np.ndarray,
        params_batch: tuple[Mapping[str, float], ...],
        remaining_horizon: int,
    ) -> np.ndarray:
        method = getattr(self.provider, "actions_many", None)
        if callable(method):
            actions = np.asarray(
                method(states, params_batch, remaining_horizon), dtype=np.int32
            )
        else:
            actions = np.stack(
                [
                    self.provider.actions(states[index], params, remaining_horizon)
                    for index, params in enumerate(params_batch)
                ],
                axis=0,
            ).astype(np.int32, copy=False)
        expected = (len(params_batch), int(states.shape[1]))
        if actions.shape != expected:
            raise ValueError("computational baseline returned an invalid action matrix")
        self._validate_actions(actions, actions.size)
        return actions

    def _action_distribution(
        self,
        states: np.ndarray,
        params: Mapping[str, float],
        remaining_horizon: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        if not callable(self._randomized_actions):
            return None
        return self._validate_distribution(
            self._randomized_actions(states, params, remaining_horizon),
            (len(states),),
        )

    def _action_distribution_many(
        self,
        states: np.ndarray,
        params_batch: tuple[Mapping[str, float], ...],
        remaining_horizon: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        if not callable(self._randomized_actions):
            return None
        rows = [
            self._randomized_actions(states[index], params, remaining_horizon)
            for index, params in enumerate(params_batch)
        ]
        lower = np.stack([row[0] for row in rows], axis=0)
        upper = np.stack([row[1] for row in rows], axis=0)
        probability = np.stack([row[2] for row in rows], axis=0)
        return self._validate_distribution(
            (lower, upper, probability),
            (len(params_batch), int(states.shape[1])),
        )

    def _validate_actions(self, actions: np.ndarray, expected_size: int) -> None:
        if actions.size != int(expected_size):
            raise ValueError("computational baseline returned an invalid action vector")
        if np.any(actions < 0) or np.any(actions > self.reference.n):
            raise ValueError("computational baseline action lies outside [0,n]")

    def _validate_distribution(
        self,
        values: tuple[np.ndarray, np.ndarray, np.ndarray],
        expected_shape: tuple[int, ...],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lower = np.asarray(values[0], dtype=np.int32)
        upper = np.asarray(values[1], dtype=np.int32)
        probability = np.asarray(values[2], dtype=float)
        if (
            lower.shape != expected_shape
            or upper.shape != expected_shape
            or probability.shape != expected_shape
            or np.any(lower < 0)
            or np.any(upper > self.reference.n)
            or np.any(lower > upper)
            or not np.all(np.isfinite(probability))
            or np.any(probability < 0.0)
            or np.any(probability > 1.0)
        ):
            raise ValueError("computational baseline returned an invalid action distribution")
        return lower, upper, probability


class ComputationalBaselineEvaluator(SparseExactPolicyCostEvaluator):
    """Use a benchmark-local provider while reusing exact probability propagation."""

    def __init__(
        self,
        provider: ComputationalActionProvider,
        reference: DPResult,
        horizon: int,
        **kwargs: Any,
    ) -> None:
        self.provider = provider
        self._randomized_actions = getattr(provider, "action_distribution", None)
        sentinel = dependency_sentinel_ast(
            reference.lead_time,
            horizon_dependent=bool(provider.horizon_dependent),
        )
        super().__init__(
            sentinel,
            reference,
            horizon,
            motifs={},
            use_action_projection=False,
            **kwargs,
        )

    @property
    def optimization_route_hint(self) -> str | None:
        value = getattr(self.provider, "optimization_route_hint", None)
        return None if value is None else str(value)

    def evaluate(
        self, params: Mapping[str, float], incumbent: float | None = None
    ) -> EvaluationResult:
        if not callable(self._randomized_actions):
            return super().evaluate(params, incumbent=incumbent)
        return self._evaluate_randomized(params, incumbent)

    def _evaluate_randomized(
        self, params: Mapping[str, float], incumbent: float | None
    ) -> EvaluationResult:
        """Exactly propagate a provider's two-point randomized actions."""

        dp = self.reference
        base = dp.n + 1
        dimensions = dp.lead_time + 1
        state_count = base**dimensions
        initial_index = self._encode_state(self.initial_state, base)
        active_indexes = np.asarray([initial_index], dtype=np.int64)
        active_probs = np.asarray([1.0], dtype=float)
        demand_values = dp.demand.values.astype(int)
        demand_probs = dp.demand.probs.astype(float)
        cumulative_cost = 0.0
        visited_states = 0
        max_reachable = 1

        for step in range(self.horizon):
            states = self._states_for_indexes(active_indexes)
            remaining_horizon = self.horizon - step
            lower, upper, upper_probability = self._randomized_actions(
                states, params, remaining_horizon
            )
            lower = np.asarray(lower, dtype=np.int32)
            upper = np.asarray(upper, dtype=np.int32)
            upper_probability = np.asarray(upper_probability, dtype=float)
            expected_shape = (len(active_indexes),)
            if (
                lower.shape != expected_shape
                or upper.shape != expected_shape
                or upper_probability.shape != expected_shape
                or np.any(lower < 0)
                or np.any(upper > dp.n)
                or np.any(lower > upper)
                or not np.all(np.isfinite(upper_probability))
                or np.any(upper_probability < 0.0)
                or np.any(upper_probability > 1.0)
            ):
                raise ValueError(
                    "computational baseline returned an invalid action distribution"
                )

            available, pipeline_suffix = self._geometry_for_indexes(active_indexes)
            cumulative_cost += float(
                np.dot(active_probs, self._expected_cost_by_available[available])
            )
            visited_states += len(active_indexes)
            max_reachable = max(max_reachable, len(active_indexes))

            joined_indexes: list[np.ndarray] = []
            joined_probs: list[np.ndarray] = []
            lower_probability = 1.0 - upper_probability
            for demand, demand_probability in zip(demand_values, demand_probs):
                next_x = np.clip(available - int(demand), 0, dp.n).astype(np.int64)
                prefix = next_x * self._next_x_multiplier + pipeline_suffix
                for actions, action_probability in (
                    (lower, lower_probability),
                    (upper, upper_probability),
                ):
                    branch_probability = (
                        active_probs * float(demand_probability) * action_probability
                    )
                    keep = branch_probability > 0.0
                    if np.any(keep):
                        joined_indexes.append((prefix[keep] * base + actions[keep]))
                        joined_probs.append(branch_probability[keep])

            all_indexes = np.concatenate(joined_indexes)
            all_probs = np.concatenate(joined_probs)
            if (
                not self.lazy_state_geometry
                and len(all_indexes) >= self.dense_switch_fraction * state_count
            ):
                dense_probs = np.bincount(
                    all_indexes, weights=all_probs, minlength=state_count
                )
                active_indexes = np.flatnonzero(dense_probs)
                active_probs = dense_probs[active_indexes]
            else:
                active_indexes, inverse = np.unique(all_indexes, return_inverse=True)
                active_probs = np.bincount(
                    inverse, weights=all_probs, minlength=len(active_indexes)
                )

            if incumbent is not None:
                remaining = self.horizon - step - 1
                lower_bound = cumulative_cost + remaining * self._minimum_period_cost
                tolerance = 1.0e-12 * max(1.0, abs(float(incumbent)))
                if lower_bound > float(incumbent) + tolerance:
                    return EvaluationResult(
                        cost=float(lower_bound),
                        completed=False,
                        early_stopped=True,
                        visited_states=visited_states,
                        metadata={
                            "fidelity": "exact_lower_bound",
                            "horizon": self.horizon,
                            "engine": "randomized_sparse",
                            "bound_valid": True,
                            "max_reachable_states": max_reachable,
                        },
                    )

        return EvaluationResult(
            cost=float(cumulative_cost),
            completed=True,
            visited_states=visited_states,
            metadata={
                "fidelity": "exact",
                "horizon": self.horizon,
                "engine": "randomized_sparse",
                "randomized_actions_exactly_propagated": True,
                "max_reachable_states": max_reachable,
            },
        )

    def _actions(
        self,
        states: np.ndarray,
        params: Mapping[str, float],
        dist_features: Mapping[str, Any],
        cost_features: Mapping[str, float],
        remaining_horizon: int,
    ) -> np.ndarray:
        del dist_features, cost_features
        actions = np.asarray(
            self.provider.actions(states, params, remaining_horizon), dtype=np.int32
        )
        if actions.shape != (len(states),):
            raise ValueError("computational baseline returned an invalid action vector")
        if np.any(actions < 0) or np.any(actions > self.reference.n):
            raise ValueError("computational baseline action lies outside [0,n]")
        return actions


__all__ = [
    "ComputationalActionAdapter",
    "ComputationalBaselineEvaluator",
    "dependency_sentinel_ast",
]
