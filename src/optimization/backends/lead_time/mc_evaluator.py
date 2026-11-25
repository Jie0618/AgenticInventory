"""Fixed-R Monte Carlo proposal evaluation for lead-time inventory.

The module intentionally contains no incumbent/shortlist policy logic.  It is
just a vectorized simulator over a shared demand trajectory bank.  The
pipeline uses its result for ranking only; strict Exact remains authoritative.
"""

from __future__ import annotations

from runtime.artifacts import stable_key
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from evaluation.backends.lead_time.dp import DPResult
from inventory.policy.ast import AST, has_cyclic_cap
from optimization.batch.analysis import _BatchEvaluationContext
from optimization.batch.eval import _evaluate_ast_batch
from optimization.batch.features import _BatchEvaluationUnsupported, _round_actions
from optimization.batch.samples import _BatchSamples
from optimization.models.types import EvaluationResult


@dataclass(frozen=True)
class DemandTrajectoryBank:
    """One deterministic set of sampled demands shared by all proposals."""

    demand_indices: np.ndarray
    demand_values: np.ndarray
    num_paths: int
    horizon: int
    seed: int
    policy_uniforms: np.ndarray | None = None

    def __post_init__(self) -> None:
        indices = np.asarray(self.demand_indices, dtype=np.int32)
        values = np.asarray(self.demand_values, dtype=np.int32)
        if indices.shape != (int(self.num_paths), int(self.horizon)):
            raise ValueError("demand_indices must have shape (num_paths, horizon)")
        if values.shape != indices.shape:
            raise ValueError("demand_values must have the same shape as demand_indices")
        uniforms = self.policy_uniforms
        if uniforms is None:
            uniforms = np.random.default_rng(int(self.seed) ^ 0x5EED5EED).random(
                indices.shape
            )
        uniforms = np.asarray(uniforms, dtype=float)
        if uniforms.shape != indices.shape or not np.all(np.isfinite(uniforms)):
            raise ValueError("policy_uniforms must be finite with shape (num_paths, horizon)")
        if np.any(uniforms < 0.0) or np.any(uniforms >= 1.0):
            raise ValueError("policy_uniforms must lie in [0,1)")
        object.__setattr__(self, "demand_indices", indices)
        object.__setattr__(self, "demand_values", values)
        object.__setattr__(self, "policy_uniforms", uniforms)

    @classmethod
    def create(
        cls,
        demand: Any,
        horizon: int,
        num_paths: int,
        seed: int = 0,
    ) -> "DemandTrajectoryBank":
        paths = int(num_paths)
        periods = int(horizon)
        if paths <= 0 or periods <= 0:
            raise ValueError("num_paths and horizon must be positive")
        values = np.asarray(demand.values, dtype=np.int32)
        probs = np.asarray(demand.probs, dtype=float)
        rng = np.random.default_rng(int(seed))
        indices = rng.choice(len(values), size=(paths, periods), p=probs).astype(np.int32)
        uniforms = rng.random((paths, periods))
        return cls(indices, values[indices], paths, periods, int(seed), uniforms)


@dataclass(frozen=True)
class MonteCarloEvaluationResult:
    """Diagnostic view of one fixed-R MC estimate."""

    score: float
    raw_mean: float
    raw_se: float
    paired_se: float | None
    variance_reduction_ratio: float | None
    path_costs: np.ndarray
    anchor_key: str | None


@dataclass(frozen=True)
class _PreparedManyActionBatch:
    """Period-reusable views for one candidate x trajectory chunk."""

    batch: _BatchSamples
    parameter_arrays: Mapping[str, np.ndarray]
    context: _BatchEvaluationContext
    candidates: int
    paths: int


class MonteCarloPolicyCostEvaluator:
    """Evaluate a policy on a shared demand bank with vectorized trajectories."""

    def __init__(
        self,
        ast: AST,
        reference: DPResult,
        bank: DemandTrajectoryBank,
        motifs: Mapping[str, AST] | None = None,
        *,
        rounding: str = "round",
        action_cap: int | None = None,
        initial_state: tuple[int, ...] | None = None,
        use_exact_anchor_control_variate: bool = True,
        use_common_random_numbers: bool = True,
        collect_pathwise_costs: bool = False,
        collect_mc_standard_error: bool = True,
        action_evaluator: Any | None = None,
        horizon: int | None = None,
        # Proposal MC is evaluated in candidate x trajectory chunks.  A
        # larger chunk amortizes the 1,000-period Python/NumPy loop and AST
        # setup; 64 remains small enough for the current H=1000, R<=1024
        # proposal workloads without creating an oversized state tensor.
        batch_size: int = 64,
    ) -> None:
        evaluation_horizon = int(reference.horizon if horizon is None else horizon)
        if int(bank.horizon) != evaluation_horizon:
            raise ValueError("MC bank horizon must match the evaluation horizon")
        if rounding != "round":
            raise ValueError("MC optimizer evaluators require canonical round-and-clip actions")
        self.ast = ast
        self.reference = reference
        self.horizon = evaluation_horizon
        self.bank = bank
        self.motifs = dict(motifs or {})
        self._time_dependent_cap = has_cyclic_cap(ast, self.motifs)
        if self._time_dependent_cap and action_evaluator is not None:
            raise ValueError("cyclic caps cannot use a stationary action evaluator")
        self.rounding = rounding
        self.action_cap = reference.n if action_cap is None else int(action_cap)
        if self.action_cap != reference.n:
            raise ValueError("MC optimizer evaluators require action_cap == state/action grid n")
        self.initial_state = (
            tuple(0 for _ in range(reference.lead_time + 1))
            if initial_state is None
            else tuple(int(value) for value in initial_state)
        )
        if len(self.initial_state) != reference.lead_time + 1:
            raise ValueError("initial_state has the wrong dimension")
        if any(value < 0 or value > reference.n for value in self.initial_state):
            raise ValueError("initial_state lies outside the state grid")
        self.use_exact_anchor_control_variate = bool(use_exact_anchor_control_variate)
        self.use_common_random_numbers = bool(use_common_random_numbers)
        self.collect_pathwise_costs = bool(collect_pathwise_costs)
        self.collect_mc_standard_error = bool(collect_mc_standard_error)
        self._action_evaluator = action_evaluator
        self.batch_size = max(1, int(batch_size))
        # These values are invariant over all paths and periods.  Rebuilding
        # them inside the horizon loop was a measurable part of the scalar MC
        # cost (especially distribution quantiles).
        self._dist_features = reference.demand.feature_dict()
        self._cost_features = {"h": float(reference.h), "b": float(reference.b)}
        self._debug_context: dict[str, Any] = {}
        self._anchor_params: dict[str, float] | None = None
        self._anchor_exact_cost: float | None = None
        self._anchor_path_costs: np.ndarray | None = None
        self._anchor_key: str | None = None
        self._last_result: MonteCarloEvaluationResult | None = None
        self._eval_cache: dict[tuple[tuple[str, float], str | None], EvaluationResult] = {}
        self._independent_eval_counter = 0
        self._last_batch_diagnostics: dict[str, float | int] = {}

    @property
    def exact_behavior_class_count(self) -> int:
        return int(getattr(self._action_evaluator, "exact_behavior_class_count", 0) or 0)

    @property
    def optimization_route_hint(self) -> str | None:
        value = getattr(self._action_evaluator, "optimization_route_hint", None)
        return None if value is None else str(value)

    def set_debug_context(self, context: Mapping[str, Any] | None) -> None:
        self._debug_context = dict(context or {})

    def annotate_last(self, updates: Mapping[str, Any]) -> None:
        # Kept for the pipeline's evaluator instrumentation contract.
        if self._last_result is not None:
            return

    def action_signature(self, params: Mapping[str, float]) -> np.ndarray:
        evaluator = self._action_evaluator
        if evaluator is not None and callable(getattr(evaluator, "action_signature", None)):
            return evaluator.action_signature(params)
        states = np.asarray(self.initial_state, dtype=int).reshape(1, -1)
        return self._policy_actions(states, params, remaining_horizon=1)

    def exact_action_signature_key(self, params: Mapping[str, float]) -> bytes | None:
        if self._time_dependent_cap:
            return None
        evaluator = self._action_evaluator
        method = getattr(evaluator, "exact_action_signature_key", None)
        return None if not callable(method) else method(params)

    @staticmethod
    def _key(params: Mapping[str, float]) -> tuple[tuple[str, float], ...]:
        return tuple(sorted((str(name), float(value)) for name, value in params.items()))

    @staticmethod
    def _anchor_digest(params: Mapping[str, float]) -> str:
        payload = repr(MonteCarloPolicyCostEvaluator._key(params)).encode("utf-8")
        return stable_key(payload)[:16]

    def set_exact_anchor(
        self,
        params: Mapping[str, float] | None,
        exact_cost: float | None,
    ) -> None:
        """Refresh the paired control-variate anchor when the incumbent changes."""

        if params is None or exact_cost is None or not np.isfinite(float(exact_cost)):
            self._anchor_params = None
            self._anchor_exact_cost = None
            self._anchor_path_costs = None
            self._anchor_key = None
            return
        key = self._anchor_digest(params)
        if key == self._anchor_key and self._anchor_path_costs is not None:
            return
        self._anchor_params = dict(params)
        self._anchor_exact_cost = float(exact_cost)
        self._anchor_key = key
        self._anchor_path_costs = (
            self.simulate_path_costs(self._anchor_params, bank=self.bank)
            if self.use_common_random_numbers
            else None
        )

    def _policy_actions(
        self,
        states: np.ndarray,
        params: Mapping[str, float],
        remaining_horizon: int,
    ) -> np.ndarray:
        evaluator = self._action_evaluator
        if evaluator is not None and callable(getattr(evaluator, "_actions", None)):
            return np.asarray(
                evaluator._actions(  # noqa: SLF001 - shared exact action kernel
                    states,
                    params,
                    self._dist_features,
                    self._cost_features,
                    int(remaining_horizon),
                ),
                dtype=np.int32,
            )
        batch = _BatchSamples.from_state_array(states, remaining_horizon=float(remaining_horizon))
        if self._time_dependent_cap:
            batch.features["time"] = np.full(len(states), self.horizon - remaining_horizon, dtype=float)
        raw = _evaluate_ast_batch(
            self.ast,
            batch,
            params,
            self._dist_features,
            self._cost_features,
            self.motifs,
            context=_BatchEvaluationContext(batch, self.motifs),
        )
        if not np.all(np.isfinite(raw)):
            raise ValueError("Policy formula produced NaN or Inf")
        return np.clip(
            _round_actions(raw, self.rounding, self.action_cap).astype(np.int32),
            0,
            self.reference.n,
        )

    def _computational_actions_many(
        self,
        states: np.ndarray,
        params_batch: tuple[Mapping[str, float], ...],
        remaining_horizon: int,
        uniforms: np.ndarray,
    ) -> np.ndarray:
        """Evaluate provider actions, including randomized two-point actions."""

        evaluator = self._action_evaluator
        assert evaluator is not None
        distribution_method = getattr(evaluator, "_action_distribution_many", None)
        if callable(distribution_method):
            distribution = distribution_method(states, params_batch, remaining_horizon)
            if distribution is not None:
                lower, upper, upper_probability = distribution
                return np.where(
                    uniforms[None, :] < upper_probability, upper, lower
                ).astype(np.int32)
        method = getattr(evaluator, "_actions_many", None)
        if callable(method):
            return np.asarray(
                method(states, params_batch, remaining_horizon), dtype=np.int32
            )
        return np.stack(
            [
                self._policy_actions(states[index], params, remaining_horizon)
                for index, params in enumerate(params_batch)
            ],
            axis=0,
        )

    def _computational_actions_one(
        self,
        states: np.ndarray,
        params: Mapping[str, float],
        remaining_horizon: int,
        uniforms: np.ndarray,
    ) -> np.ndarray:
        evaluator = self._action_evaluator
        assert evaluator is not None
        distribution_method = getattr(evaluator, "_action_distribution", None)
        if callable(distribution_method):
            distribution = distribution_method(states, params, remaining_horizon)
            if distribution is not None:
                lower, upper, upper_probability = distribution
                return np.where(uniforms < upper_probability, upper, lower).astype(
                    np.int32
                )
        return self._policy_actions(states, params, remaining_horizon)

    def _prepare_policy_actions_many(
        self,
        states: np.ndarray,
        params_batch: Sequence[Mapping[str, float]],
        paths: int,
    ) -> _PreparedManyActionBatch:
        """Prepare parameter broadcasts and mutable state views once per chunk."""

        candidates = len(params_batch)
        state_width = int(states.shape[2])
        flat_states = states.reshape(candidates * int(paths), state_width)
        size = int(flat_states.shape[0])
        remaining = np.empty(size, dtype=float)
        features: dict[str, np.ndarray] = {
            "x": flat_states[:, 0],
            "remaining_horizon": remaining,
        }
        if self._time_dependent_cap:
            features["time"] = np.empty(size, dtype=float)
        for index in range(1, state_width):
            features[f"p{index}"] = flat_states[:, index]
        batch = _BatchSamples(
            features=features,
            targets=np.empty(0, dtype=float),
            weights=np.empty(0, dtype=float),
            size=size,
        )
        parameter_arrays = {
            str(name): np.repeat(
                np.asarray([float(params[name]) for params in params_batch], dtype=float),
                int(paths),
            )
            for name in params_batch[0]
        }
        return _PreparedManyActionBatch(
            batch=batch,
            parameter_arrays=parameter_arrays,
            context=_BatchEvaluationContext(
                batch,
                self.motifs,
                cache_state_independent=True,
                cache_within_evaluation=True,
            ),
            candidates=candidates,
            paths=int(paths),
        )

    def _policy_actions_many_prepared(
        self,
        prepared: _PreparedManyActionBatch,
        remaining_horizon: int,
    ) -> np.ndarray:
        """Evaluate one period using a prepared candidate x path AST batch.

        The generic SOF batch evaluator already vectorizes state rows.  The
        state arrays are mutable views into the simulation tensor.  Parameters
        and state-independent AST subtrees stay cached for the whole chunk.
        """

        remaining = np.asarray(prepared.batch.features["remaining_horizon"])
        remaining.fill(float(remaining_horizon))
        if self._time_dependent_cap:
            np.asarray(prepared.batch.features["time"]).fill(self.horizon - remaining_horizon)
        prepared.context.begin_evaluation()
        raw = _evaluate_ast_batch(
            self.ast,
            prepared.batch,
            prepared.parameter_arrays,
            self._dist_features,
            self._cost_features,
            self.motifs,
            context=prepared.context,
        )
        if not np.all(np.isfinite(raw)):
            raise ValueError("Policy formula produced NaN or Inf")
        # This evaluator accepts only canonical round-and-clip actions and the
        # finite check above has already established that nan_to_num is not
        # needed.  Keep the generic helper for other callers, but avoid its
        # extra full-array copies in the 1,000-period MC hot loop.
        actions_float = np.rint(raw)
        np.clip(actions_float, 0.0, float(self.reference.n), out=actions_float)
        actions = actions_float.astype(np.int32)
        return actions.reshape(prepared.candidates, prepared.paths)

    def simulate_path_costs(
        self,
        params: Mapping[str, float],
        *,
        bank: DemandTrajectoryBank | None = None,
    ) -> np.ndarray:
        """Return one total cost per demand trajectory; only the horizon is looped."""

        dp = self.reference
        trajectory_bank = self.bank if bank is None else bank
        paths = int(trajectory_bank.num_paths)
        states = np.tile(np.asarray(self.initial_state, dtype=np.int32), (paths, 1))
        path_costs = np.zeros(paths, dtype=float)
        demands = trajectory_bank.demand_values
        policy_uniforms = np.asarray(trajectory_bank.policy_uniforms)
        for period in range(self.horizon):
            remaining_horizon = self.horizon - period
            actions = (
                self._computational_actions_one(
                    states,
                    params,
                    remaining_horizon,
                    policy_uniforms[:, period],
                )
                if self._action_evaluator is not None
                else self._policy_actions(states, params, remaining_horizon)
            )
            if dp.lead_time == 0:
                available = np.minimum(states[:, 0] + actions, dp.n)
            else:
                available = states[:, 0].astype(np.int32) + states[:, 1].astype(np.int32)
            demand = demands[:, period].astype(np.int32, copy=False)
            path_costs += (
                dp.h * np.maximum(available - demand, 0)
                + dp.b * np.maximum(demand - available, 0)
            )
            next_x = np.clip(available - demand, 0, dp.n).astype(np.int32)
            if dp.lead_time == 0:
                states[:, 0] = next_x
            else:
                if dp.lead_time > 1:
                    states[:, 1:-1] = states[:, 2:]
                states[:, dp.lead_time] = actions
                states[:, 0] = next_x
        return path_costs

    def simulate_path_costs_many(
        self,
        params_batch: Sequence[Mapping[str, float]],
        *,
        bank: DemandTrajectoryBank | None = None,
    ) -> np.ndarray:
        """Return costs with one row per candidate and one column per path.

        The horizon is still the only explicit time loop, but each iteration
        evaluates all candidates and trajectories in one NumPy AST batch.  A
        small candidate chunk is used by ``evaluate_many`` to bound memory.
        """

        if not params_batch:
            return np.empty((0, 0), dtype=float)
        trajectory_bank = self.bank if bank is None else bank
        paths = int(trajectory_bank.num_paths)
        candidates = len(params_batch)
        dp = self.reference
        state_width = len(self.initial_state)
        # float64 keeps feature columns as zero-copy views. Inventory and
        # pipeline values remain exact integers throughout the simulation.
        states = np.broadcast_to(
            np.asarray(self.initial_state, dtype=float),
            (candidates, paths, state_width),
        ).copy()
        path_costs = np.zeros((candidates, paths), dtype=float)
        demands = trajectory_bank.demand_values
        flat_params = tuple(params_batch)
        try:
            prepare_started = time.perf_counter()
            prepared = (
                None
                if self._action_evaluator is not None
                else self._prepare_policy_actions_many(states, flat_params, paths)
            )
            prepare_seconds = time.perf_counter() - prepare_started
            for period in range(self.horizon):
                remaining_horizon = self.horizon - period
                actions = (
                    self._computational_actions_many(
                        states,
                        flat_params,
                        remaining_horizon,
                        np.asarray(trajectory_bank.policy_uniforms)[:, period],
                    )
                    if prepared is None
                    else self._policy_actions_many_prepared(
                        prepared,
                        remaining_horizon,
                    )
                )
                if dp.lead_time == 0:
                    available = np.minimum(states[:, :, 0] + actions, dp.n)
                else:
                    available = states[:, :, 0] + states[:, :, 1]
                demand = demands[:, period][None, :]
                net_inventory = available - demand
                # Reuse the same net-inventory array for holding, shortage,
                # and the next on-hand state.  The former expression computed
                # available-demand twice and then performed a third clip/cast
                # pass over the full candidate×path matrix.
                next_x = np.maximum(net_inventory, 0.0)
                shortage = np.maximum(-net_inventory, 0.0)
                path_costs += dp.h * next_x + dp.b * shortage
                np.minimum(next_x, float(dp.n), out=next_x)
                if dp.lead_time == 0:
                    states[:, :, 0] = next_x
                else:
                    if dp.lead_time > 1:
                        states[:, :, 1:-1] = states[:, :, 2:]
                    states[:, :, dp.lead_time] = actions
                    states[:, :, 0] = next_x
            self._last_batch_diagnostics = {
                "mc_batch_prepare_seconds": float(prepare_seconds),
                "mc_batch_ast_cache_hits": int(
                    0 if prepared is None else prepared.context.cache_hits
                ),
                "mc_batch_ast_cache_misses": int(
                    0 if prepared is None else prepared.context.cache_misses
                ),
            }
        except _BatchEvaluationUnsupported:
            # Preserve support for formulas outside the small vectorized
            # grammar.  They remain correct, only without candidate batching.
            return np.vstack(
                [self.simulate_path_costs(params, bank=trajectory_bank) for params in params_batch]
            )
        return path_costs

    def _result_from_path_costs(
        self,
        params: Mapping[str, float],
        path_costs: np.ndarray,
        evaluation_bank: DemandTrajectoryBank,
        elapsed_seconds: float,
        *,
        batch_size: int = 1,
    ) -> EvaluationResult:
        raw_mean = float(np.mean(path_costs))
        raw_std = float(np.std(path_costs, ddof=1)) if len(path_costs) > 1 else 0.0
        raw_se = raw_std / np.sqrt(max(1, len(path_costs)))
        score = raw_mean
        paired_se: float | None = None
        variance_reduction_ratio: float | None = None
        used_cv = False
        if (
            self.use_exact_anchor_control_variate
            and self.use_common_random_numbers
            and self._anchor_path_costs is not None
            and self._anchor_exact_cost is not None
        ):
            delta = path_costs - self._anchor_path_costs
            delta_std = float(np.std(delta, ddof=1)) if len(delta) > 1 else 0.0
            paired_se = delta_std / np.sqrt(max(1, len(delta)))
            score = float(self._anchor_exact_cost + np.mean(delta))
            delta_var = float(np.var(delta))
            raw_var = float(np.var(path_costs))
            variance_reduction_ratio = raw_var / delta_var if delta_var > 1.0e-30 else float("inf")
            used_cv = True
        result = MonteCarloEvaluationResult(
            score=float(score),
            raw_mean=raw_mean,
            raw_se=float(raw_se),
            paired_se=paired_se,
            variance_reduction_ratio=variance_reduction_ratio,
            path_costs=path_costs,
            anchor_key=self._anchor_key,
        )
        self._last_result = result
        # Attribute a fair amortized cost to each result.  The actual batch
        # elapsed time is retained separately for diagnostics.
        per_candidate_seconds = float(elapsed_seconds) / max(1, int(batch_size))
        metadata: dict[str, Any] = {
            "fidelity": "mc_fixed",
            "engine": "vectorized_mc_batch" if int(batch_size) > 1 else "vectorized_mc",
            "horizon": int(self.horizon),
            "mc_paths": int(evaluation_bank.num_paths),
            "mc_seed": int(evaluation_bank.seed),
            "mc_crn": self.use_common_random_numbers,
            "mc_raw_mean": raw_mean,
            "mc_raw_se": float(raw_se),
            "mc_score": float(score),
            "mc_se": float(paired_se if paired_se is not None else raw_se),
            "mc_paired_se": paired_se,
            "mc_variance_reduction_ratio": variance_reduction_ratio,
            "mc_control_variate_used": used_cv,
            "mc_anchor_key": self._anchor_key,
            "mc_anchor_exact_cost": self._anchor_exact_cost,
            "mc_seconds": per_candidate_seconds,
            "mc_batch_seconds": float(elapsed_seconds),
            "mc_batch_size": int(batch_size),
            "visited_states": int(self.bank.num_paths * self.horizon),
        }
        if int(batch_size) > 1:
            metadata.update(self._last_batch_diagnostics)
        if self.collect_pathwise_costs:
            metadata["mc_path_costs"] = path_costs.tolist()
        return EvaluationResult(
            cost=float(score),
            completed=True,
            visited_states=int(self.bank.num_paths * self.horizon),
            metadata=metadata,
        )

    def evaluate(
        self,
        params: Mapping[str, float],
        incumbent: float | None = None,
    ) -> EvaluationResult:
        del incumbent  # MC is never a rigorous incumbent-bound evaluator.
        started = time.perf_counter()
        key = self._key(params)
        cache_key = (key, self._anchor_key)
        if self.use_common_random_numbers:
            cached = self._eval_cache.get(cache_key)
            if cached is not None:
                return cached
            evaluation_bank = self.bank
        else:
            # Independent-sampling mode is an A/B diagnostic.  It disables
            # paired CV and deliberately does not reuse a score cache.
            evaluation_bank = DemandTrajectoryBank.create(
                self.reference.demand,
                horizon=self.horizon,
                num_paths=self.bank.num_paths,
                seed=self.bank.seed + self._independent_eval_counter,
            )
            self._independent_eval_counter += 1
        path_costs = self.simulate_path_costs(params, bank=evaluation_bank)
        output = self._result_from_path_costs(
            params,
            path_costs,
            evaluation_bank,
            time.perf_counter() - started,
        )
        if self.use_common_random_numbers:
            self._eval_cache[cache_key] = output
        return output

    def evaluate_many(
        self,
        params_batch: Sequence[Mapping[str, float]],
        incumbent: float | None = None,
    ) -> list[EvaluationResult]:
        """Evaluate multiple proposal candidates in candidate×path batches."""

        del incumbent
        params_list = list(params_batch)
        if not params_list:
            return []
        if not self.use_common_random_numbers:
            return [self.evaluate(params) for params in params_list]
        started = time.perf_counter()
        outputs: list[EvaluationResult | None] = [None] * len(params_list)
        pending: list[tuple[int, Mapping[str, float], tuple[tuple[str, float], ...]]] = []
        for index, params in enumerate(params_list):
            key = self._key(params)
            cached = self._eval_cache.get((key, self._anchor_key))
            if cached is not None:
                outputs[index] = cached
            else:
                pending.append((index, params, key))
        if pending:
            for start_index in range(0, len(pending), self.batch_size):
                chunk = pending[start_index : start_index + self.batch_size]
                batch_params = [item[1] for item in chunk]
                batch_started = time.perf_counter()
                path_costs = self.simulate_path_costs_many(batch_params, bank=self.bank)
                batch_elapsed = time.perf_counter() - batch_started
                for row_index, (output_index, params, key) in enumerate(chunk):
                    result = self._result_from_path_costs(
                        params,
                        path_costs[row_index],
                        self.bank,
                        batch_elapsed,
                        batch_size=len(chunk),
                    )
                    result = result
                    outputs[output_index] = result
                    self._eval_cache[(key, self._anchor_key)] = result
        assert all(result is not None for result in outputs)
        return [result for result in outputs if result is not None]

    def clear_cache(self) -> None:
        self._eval_cache.clear()
