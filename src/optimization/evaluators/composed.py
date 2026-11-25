"""General policy-cost evaluators over the canonical component kernel."""

from __future__ import annotations

import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

import numpy as np

from inventory.core import InventoryCase, InventoryState
from inventory.core.config import BacklogMode, LeadTimeMode, PerishabilityMode
from inventory.policy.ast import AST, ast_check, has_cyclic_cap
from optimization.batch.analysis import _BatchEvaluationContext
from optimization.batch.analysis import _node_dependencies
from optimization.batch.eval import _evaluate_ast_batch
from optimization.batch.features import _round_actions
from optimization.batch.samples import _BatchSamples
from optimization.models.types import EvaluationResult, MonteCarloProposalConfig, OptimizationConfig
from optimization.solvers.proposal_evaluator import AdaptiveProposalEvaluator
from optimization.backends.finite_mdp.stationary import CompiledStationaryEvaluator
from evaluation.compiled import DeterministicTransitionTable


_MC_PATHS_BY_MODE = {"fast": 256, "standard": 512, "certify": 1024}
_STATIONARY_GUARD_BY_MODE = {
    "fast": {"max_periods": 64, "max_active_states": 5_000, "max_seconds": 0.2},
    "standard": {"max_periods": 96, "max_active_states": 10_000, "max_seconds": 0.5},
    "certify": {"max_periods": 128, "max_active_states": 20_000, "max_seconds": 1.0},
}


class ComposedActionProvider(Protocol):
    """Computational policy interface shared by composed Exact and MC.

    Providers may additionally expose ``action_distributions`` returning one
    sequence of (integer action, probability) pairs per state. Exact integrates
    these probabilities; MC samples using an explicit independent uniform bank.
    """

    def actions(
        self,
        states: Sequence[InventoryState],
        params: Mapping[str, float],
        remaining_horizon: int,
    ) -> np.ndarray: ...


class _ValidatedProviderActions:
    def __init__(self, provider: ComposedActionProvider, action_cap: int) -> None:
        self.provider = provider
        self.action_cap = action_cap

    def distributions(self, states, params, remaining_horizon):
        method = getattr(self.provider, "action_distributions", None)
        if method is None:
            return [((int(q), 1.0),) for q in self.actions(states, params, remaining_horizon)]
        rows = method(states, params, remaining_horizon)
        if len(rows) != len(states):
            raise ValueError("one action distribution required per state")
        for row in rows:
            if not row or any(not isinstance(q, (int, np.integer)) or not 0 <= q <= self.action_cap
                              or not np.isfinite(p) or p < 0 for q, p in row):
                raise ValueError("invalid computational action distribution")
            if not np.isclose(sum(p for _, p in row), 1.0, rtol=0, atol=1e-12):
                raise ValueError("action probabilities must sum to one")
        return rows

    def sample(self, states, params, remaining_horizon, uniforms):
        if not hasattr(self.provider, "action_distributions"):
            return self.actions(states, params, remaining_horizon)
        rows = self.distributions(states, params, remaining_horizon)
        if len(uniforms) != len(rows):
            raise ValueError("one policy uniform required per state")
        output = []
        for row, uniform in zip(rows, uniforms):
            if not np.isfinite(uniform) or not 0 <= uniform < 1:
                raise ValueError("policy uniforms must lie in [0,1)")
            cumulative = 0.0
            action = row[-1][0]
            for candidate, probability in row:
                cumulative += probability
                if uniform < cumulative:
                    action = candidate
                    break
            output.append(action)
        return np.asarray(output, dtype=np.int32)

    def actions(
        self,
        states: Sequence[InventoryState],
        params: Mapping[str, float],
        remaining_horizon: int,
    ) -> np.ndarray:
        actions = np.asarray(self.provider.actions(states, params, remaining_horizon))
        if actions.shape != (len(states),) or not np.issubdtype(actions.dtype, np.integer):
            raise ValueError("computational policy must return one integer action per state")
        if np.any(actions < 0) or np.any(actions > self.action_cap):
            raise ValueError("computational policy action lies outside [0,n]")
        return actions.astype(np.int32, copy=False)


def cost_features_for_case(case: InventoryCase) -> dict[str, float]:
    costs = case.costs
    return {
        "h": float(costs.holding),
        "b": float(costs.shortage),
        "w": float(costs.expiry),
        "g": float(costs.backlog),
        "a": float(costs.abandonment),
        "k": float(costs.ordering),
        "holding": float(costs.holding),
        "shortage": float(costs.shortage),
        "expiry": float(costs.expiry),
        "backlog": float(costs.backlog),
        "abandonment": float(costs.abandonment),
        "ordering": float(costs.ordering),
    }


class _ComposedASTActions:
    def __init__(
        self,
        ast: AST,
        case: InventoryCase,
        motifs: Mapping[str, AST],
        *,
        rounding: str,
        action_cap: int,
    ) -> None:
        checked = ast_check(
            ast,
            motifs=motifs,
            allowed_state_names=case.schema.allowed_state_names,
        )
        self.ast = checked.ast
        self.case = case
        self.motifs = dict(motifs)
        self.rounding = rounding
        self.action_cap = int(action_cap)
        self.dist_features = case.demand.feature_dict()
        self.cost_features = cost_features_for_case(case)

    def actions(
        self,
        states: Sequence[InventoryState],
        params: Mapping[str, float],
        remaining_horizon: int,
    ) -> np.ndarray:
        if not states:
            return np.empty(0, dtype=np.int32)
        matrix = np.asarray(
            [state.flatten(self.case.schema) for state in states], dtype=np.int32
        )
        return self.actions_array(matrix, params, remaining_horizon)

    def actions_array(
        self,
        matrix: np.ndarray,
        params: Mapping[str, float | np.ndarray],
        remaining_horizon: int,
    ) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.int32)
        if matrix.ndim != 2 or matrix.shape[1] != self.case.schema.dimension:
            raise ValueError("state matrix has the wrong shape")
        if len(matrix) == 0:
            return np.empty(0, dtype=np.int32)
        batch = _BatchSamples.from_state_array(
            matrix,
            remaining_horizon=float(remaining_horizon),
            state_names=self.case.schema.names,
        )
        context = _BatchEvaluationContext(
            batch,
            self.motifs,
            cache_state_independent=True,
            cache_within_evaluation=True,
        )
        context.begin_evaluation()
        raw = _evaluate_ast_batch(
            self.ast,
            batch,
            params,
            self.dist_features,
            self.cost_features,
            self.motifs,
            context=context,
        )
        if not np.all(np.isfinite(raw)):
            raise ValueError("Policy formula produced NaN or Inf")
        return np.asarray(
            _round_actions(raw, self.rounding, self.action_cap), dtype=np.int32
        )


@dataclass(frozen=True)
class _PreparedComposedActionBatch:
    """Reusable candidate x trajectory views for vectorized MC actions."""

    batch: _BatchSamples
    parameter_arrays: Mapping[str, np.ndarray]
    context: _BatchEvaluationContext
    candidates: int
    paths: int


class ComposedExactPolicyCostEvaluator:
    """Strict sparse forward propagation for any canonical composition."""

    exact_behavior_class_count = 0

    def __init__(
        self,
        ast: AST,
        case: InventoryCase,
        horizon: int,
        motifs: Mapping[str, AST] | None = None,
        *,
        rounding: str = "round",
        action_cap: int | None = None,
        initial_state: InventoryState | Sequence[int] | None = None,
        max_reachable_states: int = 2_000_000,
        action_provider: ComposedActionProvider | None = None,
        compiled_transition: DeterministicTransitionTable | None = None,
        _mc_timeaware: bool = False,
    ) -> None:
        if not isinstance(case, InventoryCase):
            raise TypeError("case must be an InventoryCase")
        if int(horizon) <= 0:
            raise ValueError("horizon must be positive")
        if has_cyclic_cap(ast, motifs) and not _mc_timeaware:
            raise ValueError("cyclic caps require an explicit MC trajectory evaluator")
        if rounding != "round":
            raise ValueError("composed optimizer requires canonical round-and-clip actions")
        self.case = case
        self.horizon = int(horizon)
        self.motifs = dict(motifs or {})
        self.rounding = rounding
        self.action_cap = case.n if action_cap is None else int(action_cap)
        if self.action_cap != case.n:
            raise ValueError("action_cap must equal the case action cap")
        if initial_state is None:
            self.initial_state = case.initial_state
        elif isinstance(initial_state, InventoryState):
            case.schema.validate(initial_state)
            self.initial_state = initial_state
        else:
            self.initial_state = InventoryState.from_flat(initial_state, case.schema)
        self.max_reachable_states = int(max_reachable_states)
        if self.max_reachable_states <= 0:
            raise ValueError("max_reachable_states must be positive")
        if action_provider is None:
            self._actions = _ComposedASTActions(
                ast, case, self.motifs, rounding=rounding, action_cap=self.action_cap
            )
            self.ast = self._actions.ast
            self._horizon_dependent = (
                "remaining_horizon"
                in _node_dependencies(self.ast, self.motifs).states
            )
        else:
            self._actions = _ValidatedProviderActions(action_provider, self.action_cap)
            self.ast = ast
            # A generic computational provider may use the supplied horizon even
            # when its placeholder AST does not.  Only canonical AST policies can
            # safely opt into stationary action reuse.
            self._horizon_dependent = True
        self._kernel = case.kernel
        self._compiled = compiled_transition
        if self._compiled is not None:
            if self._compiled.action_count != self.action_cap + 1:
                raise ValueError("compiled transition action grid does not match evaluator")
            try:
                self._compiled_initial_state_id = self._compiled.states.index(
                    self.initial_state
                )
            except ValueError as error:
                raise ValueError(
                    "evaluator initial state is outside the compiled state grid"
                ) from error
        else:
            self._compiled_initial_state_id = None
        self._cache: dict[tuple[tuple[str, float], ...], EvaluationResult] = {}

    @staticmethod
    def _key(params: Mapping[str, float]) -> tuple[tuple[str, float], ...]:
        return tuple(sorted((str(name), float(value)) for name, value in params.items()))

    def make_fast_evaluator(
        self,
        config: OptimizationConfig,
        mode: str,
        route: str | None = None,
        parameter_count: int | None = None,
    ) -> "AdaptiveProposalEvaluator":
        del route, parameter_count
        supplied = config.mc_proposal
        # Keep a concrete settings object for default-config routing without
        # changing the public optional field on OptimizationConfig.
        settings = supplied or MonteCarloProposalConfig()
        final_paths = (
            _MC_PATHS_BY_MODE.get(mode, 512)
            if supplied is None or supplied.paths is None
            else int(supplied.paths)
        )
        if supplied is not None and (
            not supplied.enabled or supplied.routing_mode == "never"
        ):
            return self
        seed = int(config.random_seed) + (0 if supplied is None else int(supplied.seed))
        proposal_paths = min(final_paths, 512) if mode == "certify" else final_paths

        def make_mc(paths: int) -> ComposedMonteCarloPolicyCostEvaluator:
            return ComposedMonteCarloPolicyCostEvaluator(self, paths=paths, seed=seed)

        stationary = None
        if (
            self._compiled is not None
            and isinstance(self._actions, _ComposedASTActions)
            and "remaining_horizon"
            not in _node_dependencies(self.ast, self.motifs).states
            and settings.stationary_probe_enabled
        ):
            guard = _STATIONARY_GUARD_BY_MODE.get(
                mode, _STATIONARY_GUARD_BY_MODE["standard"]
            )
            stationary = CompiledStationaryEvaluator(
                self._actions,
                self._compiled,
                horizon=self.horizon,
                initial_state_id=int(self._compiled_initial_state_id),
                tolerance=1.0e-10,
                required_consecutive_hits=3,
                max_periods=(
                    guard["max_periods"]
                    if settings.stationary_probe_max_periods is None
                    else settings.stationary_probe_max_periods
                ),
                max_active_states=(
                    guard["max_active_states"]
                    if settings.stationary_probe_max_active_states is None
                    else settings.stationary_probe_max_active_states
                ),
                max_seconds=(
                    guard["max_seconds"]
                    if settings.stationary_probe_max_seconds is None
                    else settings.stationary_probe_max_seconds
                ),
            )
        # Keep strict exact evaluation as the final authority.  Stationary is
        # only a guarded proposal route; non-stationary formulas use MC directly.
        return AdaptiveProposalEvaluator(
            stationary if stationary is not None else self,
            None if supplied is None or supplied.lazy_init else make_mc(proposal_paths),
            mc_factory=lambda: make_mc(proposal_paths),
            rerank_mc_factory=(
                (lambda: make_mc(final_paths))
                if proposal_paths < final_paths
                else None
            ),
            routing_mode=settings.routing_mode,
            warmup_candidates=settings.warmup_candidates,
            sentinel_candidates=settings.sentinel_candidates,
            min_mc_observations=settings.min_mc_observations,
            adaptive_sparse_sentinels=mode == "certify",
        )

    def evaluate(
        self, params: Mapping[str, float], incumbent: float | None = None
    ) -> EvaluationResult:
        key = self._key(params)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        if self._compiled is not None:
            return self._evaluate_compiled(params, incumbent, key)

        started = time.perf_counter()
        distribution: dict[InventoryState, float] = {self.initial_state: 1.0}
        cumulative_cost = 0.0
        visited_states = 0
        peak_states = 1
        demand_values = self.case.demand.values.astype(int)
        demand_probs = self.case.demand.probs.astype(float)
        for period in range(self.horizon):
            states = tuple(distribution)
            probabilities = np.fromiter(
                (distribution[state] for state in states), dtype=float
            )
            if isinstance(self._actions, _ValidatedProviderActions):
                action_rows = self._actions.distributions(states, params, self.horizon-period)
            else:
                action_rows = [((int(q), 1.0),) for q in
                               self._actions.actions(states, params, self.horizon-period)]
            next_distribution: defaultdict[InventoryState, float] = defaultdict(float)
            for state, state_probability, action_row in zip(states, probabilities, action_rows):
                if state_probability == 0.0:
                    continue
                for action, action_probability in action_row:
                    if action_probability == 0.0:
                        continue
                    for demand, demand_probability in zip(demand_values, demand_probs):
                        if demand_probability == 0.0:
                            continue
                        mass = float(state_probability) * float(demand_probability) * action_probability
                        for branch in self._kernel.outcomes(state, int(action), int(demand)):
                            weight = mass * float(branch.probability)
                            if weight == 0.0:
                                continue
                            cumulative_cost += weight * branch.total_cost
                            next_distribution[branch.next_state] += weight
                            if len(next_distribution) > self.max_reachable_states:
                                raise MemoryError(
                                    "composed optimization exceeded reachable-state limit "
                                    f"{self.max_reachable_states}"
                                )
            visited_states += len(states)
            distribution = dict(next_distribution)
            peak_states = max(peak_states, len(distribution))
            if incumbent is not None and cumulative_cost > float(incumbent):
                return EvaluationResult(
                    cost=float(cumulative_cost),
                    completed=False,
                    early_stopped=True,
                    visited_states=visited_states,
                    metadata={
                        "fidelity": "exact",
                        "backend": "composed_sparse_exact",
                        "bound_valid": True,
                        "stop_reason": "incumbent_bound",
                        "peak_reachable_states": peak_states,
                    },
                )

        result = EvaluationResult(
            cost=float(cumulative_cost),
            completed=True,
            visited_states=visited_states,
            metadata={
                "fidelity": "exact",
                "backend": "composed_sparse_exact",
                "bound_valid": True,
                "peak_reachable_states": peak_states,
                "probability_mass": float(sum(distribution.values())),
                "exact_seconds": time.perf_counter() - started,
            },
        )
        self._cache[key] = result
        return result

    def _evaluate_compiled(
        self,
        params: Mapping[str, float],
        incumbent: float | None,
        key: tuple[tuple[str, float], ...],
    ) -> EvaluationResult:
        """Evaluate a policy by dense gathers over a compiled finite MDP."""

        del incumbent
        assert self._compiled is not None
        assert self._compiled_initial_state_id is not None
        started = time.perf_counter()
        table = self._compiled
        state_ids = np.arange(table.state_count, dtype=np.int32)
        values = np.zeros(table.state_count, dtype=float)
        probabilities = table.demand_probabilities.reshape(1, -1)
        visited_states = 0
        stationary_actions = None
        if not self._horizon_dependent:
            assert isinstance(self._actions, _ComposedASTActions)
            stationary_actions = self._actions.actions_array(
                table.state_array, params, 1
            )
        for remaining_horizon in range(1, self.horizon + 1):
            if (isinstance(self._actions, _ValidatedProviderActions)
                    and hasattr(self._actions.provider, "action_distributions")):
                rows = self._actions.distributions(table.states, params, remaining_horizon)
                following = np.zeros_like(values)
                for state_id, row in enumerate(rows):
                    for action, probability in row:
                        following[state_id] += probability * np.sum(
                            table.demand_probabilities * (table.costs[state_id, action]
                            + values[table.next_state_ids[state_id, action]]))
                values = following
                visited_states += table.state_count
                continue
            if stationary_actions is not None:
                actions = stationary_actions
            elif isinstance(self._actions, _ComposedASTActions):
                actions = self._actions.actions_array(
                    table.state_array, params, remaining_horizon
                )
            else:
                actions = self._actions.actions(
                    table.states, params, remaining_horizon
                )
            selected_next = table.next_state_ids[state_ids, actions]
            selected_cost = table.costs[state_ids, actions]
            values = np.sum(
                probabilities * (selected_cost + values[selected_next]), axis=1
            )
            visited_states += table.state_count
            # Shorter-horizon values are not a lower bound when the formula
            # explicitly depends on remaining_horizon. The compiled path is
            # fast enough to finish instead of applying conditional pruning.
        result = EvaluationResult(
            cost=float(values[self._compiled_initial_state_id]),
            completed=True,
            visited_states=visited_states,
            metadata={
                "fidelity": "exact",
                "backend": "compiled_finite_mdp_exact",
                "bound_valid": True,
                "compiled_states": table.state_count,
                "exact_seconds": time.perf_counter() - started,
            },
        )
        self._cache[key] = result
        return result


class ComposedMonteCarloPolicyCostEvaluator:
    """Shared-trajectory MC evaluator for any canonical composition.

    The constructor accepts a bound ``ComposedExactPolicyCostEvaluator``
    so proposal MC can reuse a compiled transition table.  ``from_case`` is the
    MC-only constructor: it binds the policy and transition kernel directly to
    an :class:`~inventory.core.InventoryCase` and never builds or requires a DP
    reference.  The two paths intentionally share the same trajectory kernel so
    their costs are comparable under the same demand bank.
    """

    exact_behavior_class_count = 0

    def __init__(
        self,
        exact_evaluator: ComposedExactPolicyCostEvaluator | None = None,
        *,
        ast: AST | None = None,
        case: InventoryCase | None = None,
        horizon: int | None = None,
        motifs: Mapping[str, AST] | None = None,
        rounding: str = "round",
        action_cap: int | None = None,
        initial_state: InventoryState | Sequence[int] | None = None,
        action_provider: ComposedActionProvider | None = None,
        paths: int,
        seed: int,
        batch_size: int | None = None,
    ) -> None:
        if action_provider is not None:
            raise TypeError(
                "ComposedMonteCarloPolicyCostEvaluator does not support "
                "computational action providers; use "
                "ComputationalMonteCarloPolicyCostEvaluator"
            )
        self.cyclic = ast is not None and has_cyclic_cap(ast, motifs)
        self.policy_ast = ast
        if self.cyclic:
            if case is None:
                raise TypeError("cyclic MC requires case")
            checked = ast_check(ast, motifs=motifs, allowed_state_names=case.schema.allowed_state_names)
            self.policy_ast = checked.ast
            ast = checked.ast
        if exact_evaluator is None:
            if ast is None or case is None or horizon is None:
                raise TypeError(
                    "MC-only construction requires ast, case, and horizon"
                )
            exact_evaluator = ComposedExactPolicyCostEvaluator(
                ast,
                case,
                int(horizon),
                motifs,
                rounding=rounding,
                action_cap=action_cap,
                initial_state=initial_state,
                action_provider=action_provider,
                _mc_timeaware=True,
            )
            self.mc_only = True
        else:
            if any(value is not None for value in (ast, case, horizon, motifs, action_provider)):
                raise TypeError(
                    "ast/case/horizon/motifs/action_provider cannot be supplied "
                    "with exact_evaluator"
                )
            self.mc_only = False
        if int(paths) <= 0:
            raise ValueError("paths must be positive")
        self.exact_evaluator = exact_evaluator
        self.paths = int(paths)
        self.seed = int(seed)
        self.batch_size = (
            max(1, min(128, 16_384 // self.paths))
            if batch_size is None
            else max(1, int(batch_size))
        )
        self._cache: dict[tuple[tuple[str, float], ...], EvaluationResult] = {}
        demand = exact_evaluator.case.demand
        compiled = exact_evaluator._compiled
        demand_values = (
            demand.values.astype(np.int32)
            if compiled is None
            else compiled.demand_values
        )
        demand_probabilities = (
            demand.probs.astype(float)
            if compiled is None
            else compiled.demand_probabilities
        )
        rng = np.random.default_rng(self.seed)
        self._demand_indices = rng.choice(
            len(demand_values),
            size=(self.paths, exact_evaluator.horizon),
            p=demand_probabilities,
        ).astype(np.int32)
        self._demands = demand_values[self._demand_indices]

    @classmethod
    def from_case(
        cls,
        ast: AST,
        case: InventoryCase,
        horizon: int,
        *,
        motifs: Mapping[str, AST] | None = None,
        rounding: str = "round",
        action_cap: int | None = None,
        initial_state: InventoryState | Sequence[int] | None = None,
        action_provider: ComposedActionProvider | None = None,
        paths: int,
        seed: int,
        batch_size: int | None = None,
    ) -> "ComposedMonteCarloPolicyCostEvaluator":
        """Construct an MC evaluator without solving or loading a DP."""

        return cls(
            None,
            ast=ast,
            case=case,
            horizon=horizon,
            motifs=motifs,
            rounding=rounding,
            action_cap=action_cap,
            initial_state=initial_state,
            action_provider=action_provider,
            paths=paths,
            seed=seed,
            batch_size=batch_size,
        )

    def set_exact_anchor(
        self, params: Mapping[str, float] | None, exact_cost: float | None
    ) -> None:
        del params, exact_cost

    def set_debug_context(self, context: Mapping[str, object] | None) -> None:
        """Accept the shared proposal-debug hook used by adaptive routing."""

        del context

    def annotate_last(self, updates: Mapping[str, object]) -> None:
        del updates

    def clear_cache(self) -> None:
        self._cache.clear()

    def evaluate(
        self, params: Mapping[str, float], incumbent: float | None = None
    ) -> EvaluationResult:
        warnings.warn(
            "Scalar composed-MC evaluate() was called. This path is intended "
            "only for one-off final evaluation; optimizer code must call "
            "evaluate_many() so parameter points are simulated in batches.",
            RuntimeWarning,
            stacklevel=2,
        )
        return self.evaluate_many((params,), incumbent=incumbent)[0]

    def evaluate_many(
        self,
        params_batch: Sequence[Mapping[str, float]],
        incumbent: float | None = None,
    ) -> list[EvaluationResult]:
        del incumbent
        params_list = list(params_batch)
        if not params_list:
            return []
        outputs: list[EvaluationResult | None] = [None] * len(params_list)
        pending: list[tuple[int, Mapping[str, float]]] = []
        for index, params in enumerate(params_list):
            cached = self._cache.get(self.exact_evaluator._key(params))
            if cached is None:
                pending.append((index, params))
            else:
                outputs[index] = cached
        for start in range(0, len(pending), self.batch_size):
            chunk = pending[start : start + self.batch_size]
            chunk_params = tuple(item[1] for item in chunk)
            if (
                not self.cyclic
                and
                self.exact_evaluator._compiled is not None
                and isinstance(self.exact_evaluator._actions, _ComposedASTActions)
            ):
                results = self._evaluate_compiled_many(chunk_params)
            else:
                results = self._evaluate_canonical_many(chunk_params)
            for (output_index, params), result in zip(chunk, results):
                outputs[output_index] = result
                self._cache[self.exact_evaluator._key(params)] = result
        assert all(result is not None for result in outputs)
        return [result for result in outputs if result is not None]

    def _prepare_canonical_actions_many(
        self,
        states: np.ndarray,
        params_batch: Sequence[Mapping[str, float]],
    ) -> _PreparedComposedActionBatch:
        actions_provider = self.exact_evaluator._actions
        if not isinstance(actions_provider, _ComposedASTActions):
            raise TypeError(
                "batched composed MC currently requires a canonical AST action provider"
            )
        candidates, paths, width = states.shape
        flat_states = states.reshape(candidates * paths, width)
        remaining = np.empty(candidates * paths, dtype=float)
        features = {
            name: flat_states[:, index]
            for index, name in enumerate(self.exact_evaluator.case.schema.names)
        }
        features["remaining_horizon"] = remaining
        features["time"] = np.empty(candidates * paths, dtype=float)
        batch = _BatchSamples(
            features=features,
            targets=np.empty(0, dtype=float),
            weights=np.empty(0, dtype=float),
            size=candidates * paths,
        )
        parameter_names = tuple(sorted(params_batch[0]))
        if any(set(params) != set(parameter_names) for params in params_batch[1:]):
            raise ValueError("all batched parameter mappings must use the same keys")
        parameter_arrays = {
            name: np.repeat(
                np.asarray([float(params[name]) for params in params_batch], dtype=float),
                paths,
            )
            for name in parameter_names
        }
        return _PreparedComposedActionBatch(
            batch=batch,
            parameter_arrays=parameter_arrays,
            context=_BatchEvaluationContext(
                batch,
                actions_provider.motifs,
                cache_state_independent=True,
                cache_within_evaluation=True,
            ),
            candidates=candidates,
            paths=paths,
        )

    def _canonical_actions_many(
        self,
        prepared: _PreparedComposedActionBatch,
        remaining_horizon: int,
    ) -> np.ndarray:
        actions_provider = self.exact_evaluator._actions
        assert isinstance(actions_provider, _ComposedASTActions)
        np.asarray(prepared.batch.features["remaining_horizon"]).fill(
            float(remaining_horizon)
        )
        np.asarray(prepared.batch.features["time"]).fill(self.exact_evaluator.horizon - remaining_horizon)
        prepared.context.begin_evaluation()
        raw = _evaluate_ast_batch(
            actions_provider.ast,
            prepared.batch,
            prepared.parameter_arrays,
            actions_provider.dist_features,
            actions_provider.cost_features,
            actions_provider.motifs,
            context=prepared.context,
        )
        if not np.all(np.isfinite(raw)):
            raise ValueError("Policy formula produced NaN or Inf")
        actions_float = np.rint(raw)
        np.clip(actions_float, 0.0, float(actions_provider.action_cap), out=actions_float)
        return actions_float.astype(np.int32).reshape(
            prepared.candidates, prepared.paths
        )

    @staticmethod
    def _consume_many(inventory: np.ndarray, quantity: np.ndarray) -> np.ndarray:
        """Consume candidate x path inventory FEFO and return unmet quantity."""

        need = np.broadcast_to(
            np.asarray(quantity, dtype=np.int32), inventory.shape[:2]
        ).copy()
        for index in range(inventory.shape[2]):
            used = np.minimum(inventory[:, :, index], need)
            inventory[:, :, index] -= used
            need -= used
        return need

    @staticmethod
    def _project_many(
        values: np.ndarray,
        capacity: int | None,
        *,
        freshest_first: bool,
    ) -> None:
        if capacity is None:
            return
        excess = np.maximum(np.sum(values, axis=2) - int(capacity), 0)
        indexes = (
            range(values.shape[2] - 1, -1, -1)
            if freshest_first
            else range(values.shape[2])
        )
        for index in indexes:
            removed = np.minimum(values[:, :, index], excess)
            values[:, :, index] -= removed
            excess -= removed
        if np.any(excess):
            raise RuntimeError("batched finite-grid projection failed")

    def simulate_path_costs(self, params: Mapping[str, float]) -> np.ndarray:
        """Return final-bank path costs using the same batched transition kernel."""
        kernel = (
            self._evaluate_compiled_many
            if self.exact_evaluator._compiled is not None and not self.cyclic
            else self._evaluate_canonical_many
        )
        return np.asarray(kernel((params,), pathwise=True)[0], dtype=float)

    def bind_demand_paths(self, demands: np.ndarray) -> None:
        """Bind an explicit shared bank, including a prefix of a larger bank."""
        values = np.asarray(demands)
        support = np.asarray(
            self.exact_evaluator.case.demand.values
            if self.exact_evaluator._compiled is None
            else self.exact_evaluator._compiled.demand_values
        )
        if values.shape != (self.paths, self.exact_evaluator.horizon):
            raise ValueError("demand bank shape does not match evaluator")
        if not np.isin(values, support).all():
            raise ValueError("demand bank contains values outside the case support")
        self._demands = values
        if self.exact_evaluator._compiled is not None:
            order = np.argsort(support)
            self._demand_indices = order[np.searchsorted(support[order], values)].astype(np.int32)
        self.clear_cache()

    def _evaluate_canonical_many(
        self, params_batch: Sequence[Mapping[str, float]], *, pathwise: bool = False
    ) -> list[EvaluationResult] | np.ndarray:
        """Vectorize candidates and paths over deterministic canonical transitions."""

        evaluator = self.exact_evaluator
        case = evaluator.case
        config = case.config
        if config.transition_mode.value != "deterministic":
            raise NotImplementedError(
                "batched composed MC does not yet support stochastic lead time, "
                "receipt lifetime, or geometric backlog; refusing scalar fallback"
            )
        if not isinstance(evaluator._actions, _ComposedASTActions):
            raise TypeError(
                "batched composed MC does not support computational action providers"
            )

        candidates = len(params_batch)
        started = time.perf_counter()
        initial = np.asarray(
            evaluator.initial_state.flatten(case.schema), dtype=np.int32
        )
        states = np.broadcast_to(
            initial, (candidates, self.paths, case.schema.dimension)
        ).copy()
        costs = np.zeros((candidates, self.paths), dtype=float)
        prepared = self._prepare_canonical_actions_many(states, params_batch)

        inventory_stop = case.schema.inventory_dimension
        pipeline_stop = inventory_stop + case.schema.pipeline_dimension
        inventory = states[:, :, :inventory_stop]
        pipeline = states[:, :, inventory_stop:pipeline_stop]
        backlog = states[:, :, pipeline_stop:]
        capacity = config.capacity
        lead_mode = config.lead_time.mode
        perishability_mode = config.perishability.mode
        backlog_mode = config.backlog.mode
        lead_time = int(config.lead_time.periods or 0)
        shelf_life = int(config.perishability.shelf_life or 0)
        cost = case.costs

        if lead_mode not in (LeadTimeMode.DISABLED, LeadTimeMode.DETERMINISTIC):
            raise NotImplementedError("known-ETA stochastic lead time needs a batched event bank")
        if perishability_mode not in (
            PerishabilityMode.DISABLED,
            PerishabilityMode.DETERMINISTIC,
        ):
            raise NotImplementedError(
                "known-arrival stochastic shelf life needs a batched event bank"
            )
        if backlog_mode is BacklogMode.GEOMETRIC:
            raise NotImplementedError("geometric backlog needs a batched abandonment bank")

        for period in range(evaluator.horizon):
            actions = self._canonical_actions_many(
                prepared, evaluator.horizon - period
            )

            if lead_mode is LeadTimeMode.DISABLED or lead_time == 0:
                incoming = actions.copy()
            else:
                incoming = pipeline[:, :, 0].copy()
                if pipeline.shape[2] > 1:
                    pipeline[:, :, :-1] = pipeline[:, :, 1:]
                pipeline[:, :, -1] = 0
                target = pipeline[:, :, lead_time - 1]
                target += actions
                if capacity is not None:
                    np.minimum(target, int(capacity), out=target)

            if perishability_mode is PerishabilityMode.DISABLED:
                inventory[:, :, 0] += incoming
            elif shelf_life > 0:
                inventory[:, :, shelf_life - 1] += incoming

            if backlog.shape[2]:
                if config.backlog.uses_scalar_total:
                    original = backlog[:, :, 0].copy()
                    unmet_backlog = self._consume_many(inventory, original)
                    backlog[:, :, 0] = unmet_backlog
                else:
                    for index in range(backlog.shape[2] - 1, -1, -1):
                        original = backlog[:, :, index].copy()
                        backlog[:, :, index] = self._consume_many(inventory, original)

            unmet = self._consume_many(
                inventory, self._demands[:, period][None, :]
            )
            ending_inventory = np.sum(inventory, axis=2)
            costs += float(cost.holding) * ending_inventory
            if backlog_mode in (BacklogMode.DISABLED, BacklogMode.LOST_SALES):
                costs += float(cost.shortage) * unmet
            costs += float(cost.ordering) * actions

            if perishability_mode is PerishabilityMode.DETERMINISTIC:
                expired = inventory[:, :, 0].copy()
                costs += float(cost.expiry) * expired
                if inventory.shape[2] > 1:
                    inventory[:, :, :-1] = inventory[:, :, 1:]
                inventory[:, :, -1] = 0
            self._project_many(inventory, capacity, freshest_first=True)

            if backlog_mode in (BacklogMode.DISABLED, BacklogMode.LOST_SALES):
                pass
            elif backlog_mode is BacklogMode.DETERMINISTIC:
                abandoned = backlog[:, :, -1].copy()
                if backlog.shape[2] > 1:
                    backlog[:, :, 1:] = backlog[:, :, :-1]
                backlog[:, :, 0] = unmet
                self._project_many(backlog, capacity, freshest_first=False)
                costs += float(cost.abandonment) * abandoned
                costs += float(cost.backlog) * np.sum(backlog, axis=2)
            elif backlog_mode is BacklogMode.FULL:
                backlog[:, :, 0] += unmet
                self._project_many(backlog, capacity, freshest_first=False)
                costs += float(cost.backlog) * backlog[:, :, 0]
            else:  # pragma: no cover - guarded above
                raise NotImplementedError(f"unsupported backlog mode {backlog_mode}")

        if pathwise:
            return costs
        elapsed = time.perf_counter() - started
        outputs = []
        for index in range(candidates):
            path_costs = costs[index]
            outputs.append(
                EvaluationResult(
                    cost=float(np.mean(path_costs)),
                    completed=True,
                    visited_states=self.paths * evaluator.horizon,
                    metadata={
                        "fidelity": "mc_fixed",
                        "proposal_fidelity": "mc_fixed",
                        "backend": "composed_vectorized_transition_mc",
                        "action_execution": "cyclic_cap_round" if self.cyclic else "round",
                        "mc_only": bool(self.mc_only),
                        "paths": self.paths,
                        "seed": self.seed,
                        "mc_standard_error": (
                            0.0
                            if self.paths == 1
                            else float(np.std(path_costs, ddof=1) / np.sqrt(self.paths))
                        ),
                        "proposal_mc_seconds": elapsed / max(1, candidates),
                        "mc_batch_seconds": elapsed,
                        "mc_batch_size": candidates,
                    },
                )
            )
        return outputs

    def _evaluate_compiled_many(
        self, params_batch: Sequence[Mapping[str, float]], *, pathwise: bool = False
    ) -> list[EvaluationResult] | np.ndarray:
        """Vectorize candidates and paths over the compiled transition table."""

        evaluator = self.exact_evaluator
        table = evaluator._compiled
        actions_provider = evaluator._actions
        assert table is not None
        assert isinstance(actions_provider, _ComposedASTActions)
        candidates = len(params_batch)
        started = time.perf_counter()
        state_ids = np.full(
            (candidates, self.paths),
            evaluator._compiled_initial_state_id,
            dtype=np.int32,
        )
        costs = np.zeros((candidates, self.paths), dtype=float)
        parameter_arrays = {
            name: np.repeat(
                np.asarray([float(params[name]) for params in params_batch]), self.paths
            )
            for name in params_batch[0]
        }
        stationary_action_map = None
        if not evaluator._horizon_dependent:
            tiled_states = np.tile(table.state_array, (candidates, 1))
            stationary_parameter_arrays = {
                name: np.repeat(
                    np.asarray([float(params[name]) for params in params_batch]),
                    table.state_count,
                )
                for name in params_batch[0]
            }
            stationary_action_map = actions_provider.actions_array(
                tiled_states,
                stationary_parameter_arrays,
                1,
            ).reshape(candidates, table.state_count)
        candidate_ids = np.arange(candidates, dtype=np.int32)[:, None]
        for period in range(evaluator.horizon):
            if stationary_action_map is None:
                flat_states = table.state_array[state_ids.reshape(-1)]
                flat_actions = actions_provider.actions_array(
                    flat_states,
                    parameter_arrays,
                    evaluator.horizon - period,
                )
                actions = flat_actions.reshape(candidates, self.paths)
            else:
                actions = stationary_action_map[candidate_ids, state_ids]
            demand_ids = self._demand_indices[:, period][None, :]
            costs += table.costs[state_ids, actions, demand_ids]
            state_ids = table.next_state_ids[state_ids, actions, demand_ids]
        if pathwise:
            return costs
        elapsed = time.perf_counter() - started
        outputs: list[EvaluationResult] = []
        for index, params in enumerate(params_batch):
            path_costs = costs[index]
            outputs.append(
                EvaluationResult(
                    cost=float(np.mean(path_costs)),
                    completed=True,
                    visited_states=self.paths * evaluator.horizon,
                    metadata={
                        "fidelity": "mc_fixed",
                        "proposal_fidelity": "mc_fixed",
                        "backend": "compiled_finite_mdp_mc",
                        "mc_only": bool(self.mc_only),
                        "paths": self.paths,
                        "seed": self.seed,
                        "mc_standard_error": (
                            0.0
                            if self.paths == 1
                            else float(np.std(path_costs, ddof=1) / np.sqrt(self.paths))
                        ),
                        "proposal_mc_seconds": elapsed / max(1, candidates),
                        "mc_batch_seconds": elapsed,
                        "mc_batch_size": candidates,
                    },
                )
            )
        return outputs


class ComputationalMonteCarloPolicyCostEvaluator:
    """MC evaluator for Python computational policies.

    Fixed-lifetime FIFO providers can opt into batched integer action intervals.
    Other compositions/providers retain the canonical scalar kernel.
    """

    def __init__(
        self,
        case: InventoryCase,
        horizon: int,
        action_provider: ComposedActionProvider,
        *,
        paths: int,
        seed: int,
        action_cap: int | None = None,
        initial_state: InventoryState | Sequence[int] | None = None,
        require_batch: bool = False,
    ) -> None:
        if int(paths) <= 0:
            raise ValueError("paths must be positive")
        self.case = case
        self.horizon = int(horizon)
        self.paths = int(paths)
        self.seed = int(seed)
        self.require_batch = bool(require_batch)
        if self.require_batch and not (
                case.composition_key == "perishability"
                and case.config.perishability.shelf_life is not None
                and callable(getattr(action_provider, "batch_action_intervals", None))):
            raise ValueError("production computational MC requires a batched action provider")
        self._evaluator = ComposedExactPolicyCostEvaluator(
            {"op": "const", "value": 0.0},
            case,
            self.horizon,
            action_cap=action_cap,
            initial_state=initial_state,
            action_provider=action_provider,
        )
        demand = case.demand
        rng = np.random.default_rng(self.seed)
        indices = rng.choice(
            len(demand.values), size=(self.paths, self.horizon), p=demand.probs
        )
        self._demands = np.asarray(demand.values, dtype=np.int32)[indices]
        # Separate stream from demand and exogenous transition events.
        self._policy_uniforms = np.random.default_rng(self.seed ^ 0xD1B54A32).random(
            (self.paths, self.horizon))

    def bind_policy_uniforms(self, uniforms: np.ndarray) -> None:
        values = np.asarray(uniforms, dtype=float)
        if values.shape != (self.paths, self.horizon) or not np.all(
            np.isfinite(values) & (values >= 0) & (values < 1)
        ):
            raise ValueError("invalid policy uniform bank")
        self._policy_uniforms = values.copy()

    def bind_demand_paths(self, demands: np.ndarray) -> None:
        values = np.asarray(demands)
        if values.shape != (self.paths, self.horizon):
            raise ValueError("demand bank shape does not match evaluator")
        if not np.isin(values, self.case.demand.values).all():
            raise ValueError("demand bank contains values outside the case support")
        self._demands = values

    def simulate_path_costs(self, params: Mapping[str, float]) -> np.ndarray:
        provider = self._evaluator._actions.provider
        if (self.case.composition_key == "perishability"
                and self.case.config.perishability.shelf_life is not None
                and callable(getattr(provider, "batch_action_intervals", None))):
            return self._simulate_batched_perishable_path_costs(params)
        if self.require_batch:
            raise ValueError("batched action provider unavailable; refusing scalar fallback")
        return self._simulate_scalar_path_costs(params)

    def _simulate_batched_perishable_path_costs(self, params) -> np.ndarray:
        """Same FIFO/holding-before-expiry/projection rules as the kernel."""
        provider = self._evaluator._actions.provider
        inventory = np.tile(np.asarray(self._evaluator.initial_state.inventory,
                                       dtype=np.int64), (self.paths, 1))
        costs = np.zeros(self.paths, dtype=float)
        capacity = self.case.config.capacity
        for period in range(self.horizon):
            lo, hi, probability = provider.batch_action_intervals(
                inventory, params, self.horizon - period)
            lo, hi, probability = np.asarray(lo), np.asarray(hi), np.asarray(probability)
            if (lo.shape != (self.paths,) or hi.shape != lo.shape
                    or probability.shape != lo.shape
                    or not np.issubdtype(lo.dtype, np.integer)
                    or not np.issubdtype(hi.dtype, np.integer)
                    or np.any(lo < 0) or np.any(hi < lo)
                    or np.any(hi > self.case.n)
                    or not np.all(np.isfinite(probability) & (probability >= 0)
                                  & (probability <= 1))):
                raise ValueError("invalid batched action intervals")
            # Scalar sampling selects lo for u < 1-P(hi), including boundaries.
            actions = np.where(self._policy_uniforms[:, period] < 1-probability, lo, hi)
            inventory[:, -1] += actions
            need = self._demands[:, period].copy()
            for age in range(inventory.shape[1]):
                used = np.minimum(inventory[:, age], need)
                inventory[:, age] -= used
                need -= used
            holding = self.case.costs.holding * inventory.sum(axis=1)
            shortage = self.case.costs.shortage * need
            expiry = self.case.costs.expiry * inventory[:, 0]
            ordering = self.case.costs.ordering * actions
            costs += ((holding + shortage) + expiry) + ordering
            inventory[:, :-1] = inventory[:, 1:]
            inventory[:, -1] = 0
            if capacity is not None:
                excess = np.maximum(inventory.sum(axis=1) - capacity, 0)
                for age in range(inventory.shape[1]-1, -1, -1):
                    removed = np.minimum(inventory[:, age], excess)
                    inventory[:, age] -= removed
                    excess -= removed
        return costs

    def _simulate_scalar_path_costs(self, params: Mapping[str, float]) -> np.ndarray:
        rng = np.random.default_rng(self.seed ^ 0x9E3779B9)
        path_costs = np.zeros(self.paths, dtype=float)
        for path in range(self.paths):
            state = self._evaluator.initial_state
            for period in range(self.horizon):
                remaining = self.horizon - period
                action = int(self._evaluator._actions.sample(
                    (state,), params, remaining, (self._policy_uniforms[path, period],))[0])
                branches = self._evaluator._kernel.outcomes(
                    state, action, int(self._demands[path, period])
                )
                if len(branches) == 1:
                    branch = branches[0]
                else:
                    probabilities = np.asarray(
                        [item.probability for item in branches], dtype=float
                    )
                    branch = branches[int(rng.choice(
                        len(branches), p=probabilities / probabilities.sum()
                    ))]
                path_costs[path] += branch.outcome.total_cost
                state = branch.next_state
        return path_costs

    def evaluate(self, params: Mapping[str, float]) -> EvaluationResult:
        started = time.perf_counter()
        path_costs = self.simulate_path_costs(params)
        return EvaluationResult(
            cost=float(np.mean(path_costs)),
            completed=True,
            visited_states=self.paths * self.horizon,
            metadata={
                "fidelity": "mc_fixed",
                "backend": "composed_computational_mc",
                "paths": self.paths,
                "horizon": self.horizon,
                "seed": self.seed,
                "mc_standard_error": (
                    0.0 if self.paths == 1
                    else float(np.std(path_costs, ddof=1) / np.sqrt(self.paths))
                ),
                "proposal_mc_seconds": time.perf_counter() - started,
            },
        )


class TwoFidelityComposedMCEvaluator:
    """Low-path proposals, high-path empirical arbitration for composed cases."""

    two_fidelity_exhaustive_route = True

    def __init__(
        self,
        high: ComposedMonteCarloPolicyCostEvaluator,
        low: ComposedMonteCarloPolicyCostEvaluator,
    ) -> None:
        self.high = high
        self.low = low

    def make_fast_evaluator(self, config, mode, route=None, parameter_count=None):
        return self.low

    def evaluate(self, params, incumbent=None) -> EvaluationResult:
        result = self.high.evaluate_many((params,), incumbent=incumbent)[0]
        return EvaluationResult(
            cost=result.cost,
            completed=result.completed,
            visited_states=result.visited_states,
            metadata={**result.metadata, "fidelity": "exact", "mc_empirical_strict": True},
        )

    def evaluate_many(self, params_batch, incumbent=None) -> list[EvaluationResult]:
        return [
            EvaluationResult(
                cost=result.cost,
                completed=result.completed,
                early_stopped=False,
                visited_states=result.visited_states,
                metadata={
                    **result.metadata,
                    "fidelity": "exact",
                    "mc_empirical_strict": True,
                },
            )
            for result in self.high.evaluate_many(
                params_batch, incumbent=incumbent
            )
        ]


__all__ = [
    "ComposedActionProvider",
    "ComposedExactPolicyCostEvaluator",
    "ComposedMonteCarloPolicyCostEvaluator",
    "ComputationalMonteCarloPolicyCostEvaluator",
    "TwoFidelityComposedMCEvaluator",
    "cost_features_for_case",
]
