"""Generic stationary proposal evaluator for compiled finite MDPs.

This is deliberately a proposal-only evaluator.  It propagates the state
distribution under a time-homogeneous policy and, once the distribution and
one-step cost stabilize, extrapolates the remaining horizon.  If the guard is
hit before stabilization it returns an incomplete result so the adaptive
router can use fixed-path MC.  Strict final evaluation remains unchanged.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

import numpy as np

from optimization.models.types import EvaluationResult


class CompiledStationaryEvaluator:
    """Guarded stationary propagation over a deterministic transition table."""

    exact_behavior_class_count = 0

    def __init__(
        self,
        action_evaluator: Any,
        transition_table: Any,
        *,
        horizon: int,
        initial_state_id: int,
        tolerance: float = 1.0e-10,
        required_consecutive_hits: int = 3,
        max_periods: int | None = 128,
        max_active_states: int | None = 20_000,
        max_seconds: float | None = 1.0,
    ) -> None:
        self.action_evaluator = action_evaluator
        self.transition_table = transition_table
        self.horizon = int(horizon)
        self.initial_state_id = int(initial_state_id)
        self.tolerance = float(tolerance)
        self.required_consecutive_hits = max(1, int(required_consecutive_hits))
        self.max_periods = None if max_periods is None else max(1, int(max_periods))
        self.max_active_states = (
            None if max_active_states is None else max(1, int(max_active_states))
        )
        self.max_seconds = None if max_seconds is None else max(0.0, float(max_seconds))
        self._cache: dict[bytes, EvaluationResult] = {}

    def set_debug_context(self, context: Mapping[str, object] | None) -> None:
        del context

    def annotate_last(self, updates: Mapping[str, object]) -> None:
        del updates

    def clear_cache(self) -> None:
        self._cache.clear()

    def exact_action_signature_key(self, params: Mapping[str, float]) -> bytes:
        return self.action_signature(params).tobytes()

    def action_signature(self, params: Mapping[str, float]) -> np.ndarray:
        table = self.transition_table
        return np.asarray(
            self.action_evaluator.actions_array(table.state_array, params, 1),
            dtype=np.int32,
        )

    def evaluate(
        self, params: Mapping[str, float], incumbent: float | None = None
    ) -> EvaluationResult:
        return self.evaluate_many((params,), incumbent=incumbent)[0]

    def evaluate_many(
        self,
        params_batch: Sequence[Mapping[str, float]],
        incumbent: float | None = None,
    ) -> list[EvaluationResult]:
        """Propagate a candidate batch while vectorizing AST evaluation."""

        params_list = list(params_batch)
        if not params_list:
            return []
        signatures = [self.exact_action_signature_key(params) for params in params_list]
        outputs: list[EvaluationResult | None] = [None] * len(params_list)
        pending: list[Mapping[str, float]] = []
        pending_keys: list[bytes] = []
        pending_positions: dict[bytes, list[int]] = {}
        for index, key in enumerate(signatures):
            cached = self._cache.get(key)
            if cached is not None:
                outputs[index] = cached
                continue
            if key not in pending_positions:
                pending_positions[key] = []
                pending.append(params_list[index])
                pending_keys.append(key)
            pending_positions[key].append(index)
        if pending:
            computed = self._evaluate_many_uncached(
                pending,
                action_signatures=pending_keys,
                incumbent=incumbent,
            )
            for key, result in zip(pending_keys, computed):
                self._cache[key] = result
                for index in pending_positions[key]:
                    outputs[index] = result
        assert all(result is not None for result in outputs)
        return [result for result in outputs if result is not None]

    def _evaluate_many_uncached(
        self,
        params_batch: Sequence[Mapping[str, float]],
        action_signatures: Sequence[bytes] | None = None,
        incumbent: float | None = None,
    ) -> list[EvaluationResult]:
        """Compute one result per unique stationary action signature."""

        del incumbent
        params_list = list(params_batch)
        if not params_list:
            return []
        started = time.perf_counter()
        table = self.transition_table
        state_count = int(table.state_array.shape[0])
        candidates = len(params_list)
        distribution = np.zeros((candidates, state_count), dtype=float)
        distribution[:, self.initial_state_id] = 1.0
        demand_probabilities = np.asarray(table.demand_probabilities, dtype=float)
        state_ids = np.arange(state_count, dtype=np.int32)[None, :]
        if action_signatures is None:
            actions = np.asarray(
                [self.action_signature(params) for params in params_list],
                dtype=np.int32,
            )
        else:
            actions = np.asarray(
                [
                    np.frombuffer(signature, dtype=np.int32, count=state_count)
                    for signature in action_signatures
                ],
                dtype=np.int32,
            )
        cumulative = np.zeros(candidates, dtype=float)
        previous_step_cost = np.full(candidates, np.nan, dtype=float)
        hits = np.zeros(candidates, dtype=np.int32)
        done = np.zeros(candidates, dtype=bool)
        guard_reasons: list[str | None] = [None] * candidates
        cycle_at: list[int | None] = [None] * candidates
        periods = np.zeros(candidates, dtype=np.int32)
        max_periods = self.max_periods

        for period in range(self.horizon):
            active_candidates = np.flatnonzero(~done)
            if len(active_candidates) == 0:
                break
            if max_periods is not None and period >= max_periods:
                for index in active_candidates:
                    guard_reasons[int(index)] = "max_periods"
                    done[int(index)] = True
                break
            if self.max_seconds is not None and time.perf_counter() - started > self.max_seconds:
                for index in active_candidates:
                    guard_reasons[int(index)] = "max_seconds"
                    done[int(index)] = True
                break
            active_states = np.count_nonzero(distribution > 0.0, axis=1)
            if self.max_active_states is not None:
                over_limit = np.flatnonzero(
                    (~done) & (active_states > self.max_active_states)
                )
                for index in over_limit:
                    guard_reasons[int(index)] = "max_active_states"
                    done[int(index)] = True
                active_candidates = np.flatnonzero(~done)
                if len(active_candidates) == 0:
                    break
            selected_next = table.next_state_ids[state_ids, actions]
            selected_cost = table.costs[state_ids, actions]
            step_cost_by_state = np.sum(
                demand_probabilities[None, None, :] * selected_cost, axis=2
            )
            step_cost = np.sum(distribution * step_cost_by_state, axis=1)
            next_distribution = np.zeros_like(distribution)
            for candidate in active_candidates:
                for demand_id, probability in enumerate(demand_probabilities):
                    next_distribution[candidate] += np.bincount(
                        selected_next[candidate, :, demand_id],
                        weights=distribution[candidate] * float(probability),
                        minlength=state_count,
                    )
            cumulative[active_candidates] += step_cost[active_candidates]
            delta = np.max(np.abs(next_distribution - distribution), axis=1)
            cost_delta = np.abs(step_cost - previous_step_cost)
            converged = (
                (~done)
                & np.isfinite(previous_step_cost)
                & (delta <= self.tolerance)
                & (cost_delta <= self.tolerance)
            )
            hits[converged] += 1
            hits[(~done) & ~converged] = 0
            previous_step_cost = step_cost
            distribution[active_candidates] = next_distribution[active_candidates]
            periods[active_candidates] = period + 1
            newly_stationary = np.flatnonzero(
                (~done)
                & (hits >= self.required_consecutive_hits)
                & (period + 1 < self.horizon)
            )
            for index in newly_stationary:
                remaining = self.horizon - period - 1
                cumulative[index] += remaining * step_cost[index]
                cycle_at[int(index)] = period + 1
                done[index] = True

        elapsed = time.perf_counter() - started
        outputs: list[EvaluationResult] = []
        for index in range(candidates):
            guard_reason = guard_reasons[index]
            completed = guard_reason is None
            outputs.append(
                EvaluationResult(
                    cost=float(cumulative[index]),
                    completed=completed,
                    visited_states=state_count * int(periods[index]),
                    metadata={
                        "fidelity": "stationary_fast",
                        "backend": "compiled_finite_mdp_stationary",
                        "bound_valid": False,
                        "stationary_triggered": cycle_at[index] is not None,
                        "cycle_detected_at": cycle_at[index],
                        "stationary_hit_streak": int(hits[index]),
                        "proposal_probe_seconds": elapsed / candidates,
                        "proposal_batch_seconds": elapsed,
                        "proposal_batch_size": candidates,
                        "proposal_guard_triggered": guard_reason is not None,
                        "proposal_guard_abort_reason": guard_reason,
                        "compiled_states": state_count,
                    },
                )
            )
        return outputs


__all__ = ["CompiledStationaryEvaluator"]
