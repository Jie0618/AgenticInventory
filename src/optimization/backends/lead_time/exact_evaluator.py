"""Production Exact optimization kernels for lead-time inventory."""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping

import numpy as np

from evaluation.backends.lead_time.dp import DPResult
from inventory.policy.ast import AST

from optimization.batch.analysis import _BatchEvaluationContext, _node_dependencies
from optimization.batch.eval import _evaluate_ast_batch
from optimization.batch.features import _round_actions
from optimization.batch.samples import _BatchSamples
from optimization.models.types import EvaluationResult
from optimization.backends.lead_time.policy_cost import evaluate_ast_policy_cost_exact
from optimization.policies.vbs import detect_vbs_ast, evaluate_vbs_actions


class DenseExactPolicyCostEvaluator:
    """Bind an AST to the existing full-grid exact finite-horizon evaluator."""

    def __init__(
        self,
        ast: AST,
        dp_result: DPResult,
        motifs: Mapping[str, AST] | None = None,
        *,
        rounding: str = "round",
        action_cap: int | None = None,
        initial_state: tuple[int, ...] | None = None,
    ) -> None:
        self.ast = ast
        self.dp_result = dp_result
        self.motifs = dict(motifs or {})
        if rounding != "round":
            raise ValueError("Exact optimizer evaluators require canonical round-and-clip actions")
        self.rounding = rounding
        self.action_cap = dp_result.n if action_cap is None else int(action_cap)
        if self.action_cap != dp_result.n:
            raise ValueError("Exact optimizer evaluators require action_cap == state/action grid n")
        self.initial_state = initial_state
        self._signature_states: np.ndarray | None = None

    def evaluate(self, params: Mapping[str, float], incumbent: float | None = None) -> EvaluationResult:
        cost = evaluate_ast_policy_cost_exact(
            self.ast,
            self.dp_result,
            params,
            self.motifs,
            rounding=self.rounding,
            action_cap=self.action_cap,
            initial_state=self.initial_state,
        )
        states = (self.dp_result.n + 1) ** (self.dp_result.lead_time + 1)
        return EvaluationResult(
            cost=float(cost),
            completed=True,
            visited_states=states * self.dp_result.horizon,
            metadata={"fidelity": "exact", "horizon": self.dp_result.horizon, "engine": "dense"},
        )

    def configure_signature_bank(self, size: int) -> None:
        self._signature_states = _build_signature_states(self.dp_result, size)

    def action_signature(self, params: Mapping[str, float]) -> np.ndarray:
        if self._signature_states is None:
            self.configure_signature_bank(500)
        assert self._signature_states is not None
        return _action_signature(
            self.ast,
            self._signature_states,
            params,
            self.dp_result,
            self.motifs,
            self.action_cap,
        )


class SparseExactPolicyCostEvaluator:
    """Exact forward probability propagation over reachable states only."""

    @staticmethod
    def uses_special_initial_anchors(specs: Any) -> bool:
        """Report the lead-time VBS anchor rule to the shared controller."""

        from optimization.policies.vbs import is_vbs_parameterization

        return is_vbs_parameterization(specs)

    def make_fast_evaluator(
        self,
        config: Any,
        mode: str,
        route: str | None = None,
        parameter_count: int | None = None,
    ) -> Any:
        """Construct the lead-time Stationary/MC proposal evaluator."""

        from optimization.policies.proposal import make_fast_evaluator

        return make_fast_evaluator(self, config, mode, route, parameter_count)

    def __init__(
        self,
        ast: AST,
        reference: DPResult,
        horizon: int,
        motifs: Mapping[str, AST] | None = None,
        *,
        rounding: str = "round",
        action_cap: int | None = None,
        initial_state: tuple[int, ...] | None = None,
        convergence_tolerance: float = 1.0e-14,
        collect_reachable_actions: bool = False,
        dense_switch_fraction: float = 0.10,
        max_cycle_period: int | None = 0,
        stationary_required_consecutive_hits: int = 1,
        probability_prune_threshold: float = 0.0,
        use_inventory_position_reduction: bool = True,
        use_stable_transition_matrix: bool = True,
        use_action_projection: bool = True,
        stable_dense_cell_limit: int = 1_000_000,
        use_csr_sparse_transition: bool = True,
        use_adaptive_reachable_csr: bool = True,
        adaptive_csr_warmup_periods: int = 8,
        adaptive_csr_max_support_states: int = 1_000_000,
        adaptive_csr_max_build_seconds: float = 10.0,
        adaptive_csr_max_attempts: int = 3,
        certified_screen_max_states: int = 100_000,
        certified_screen_max_seconds: float = 60.0,
        use_stationary_continuation_bound: bool = True,
        full_state_geometry_limit: int | None = None,
        debug_trace: Callable[[Mapping[str, Any]], None] | None = None,
        proposal_complexity_guard: Mapping[str, Any] | None = None,
    ) -> None:
        if reference.lead_time <= 0:
            raise ValueError("Sparse evaluator requires positive lead time")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        self.ast = ast
        self.reference = reference
        self.horizon = int(horizon)
        self.motifs = dict(motifs or {})
        if rounding != "round":
            raise ValueError("Exact optimizer evaluators require canonical round-and-clip actions")
        self.rounding = rounding
        self.action_cap = reference.n if action_cap is None else int(action_cap)
        if self.action_cap != reference.n:
            raise ValueError("Exact optimizer evaluators require action_cap == state/action grid n")
        self.initial_state = tuple(0 for _ in range(reference.lead_time + 1)) if initial_state is None else tuple(initial_state)
        self.convergence_tolerance = float(convergence_tolerance)
        self.collect_reachable_actions = bool(collect_reachable_actions)
        if dense_switch_fraction < 0.0:
            raise ValueError("dense_switch_fraction must be nonnegative")
        self.dense_switch_fraction = float(dense_switch_fraction)
        self.max_cycle_period = (
            2 * (reference.lead_time + 1)
            if max_cycle_period is None
            else int(max_cycle_period)
        )
        if self.max_cycle_period < 0:
            raise ValueError("max_cycle_period must be nonnegative")
        if stationary_required_consecutive_hits <= 0:
            raise ValueError("stationary_required_consecutive_hits must be positive")
        self.stationary_required_consecutive_hits = int(stationary_required_consecutive_hits)
        if probability_prune_threshold < 0.0:
            raise ValueError("probability_prune_threshold must be nonnegative")
        self.probability_prune_threshold = float(probability_prune_threshold)
        self.use_inventory_position_reduction = bool(use_inventory_position_reduction)
        self.use_stable_transition_matrix = bool(use_stable_transition_matrix)
        self.use_action_projection = bool(use_action_projection)
        self.use_csr_sparse_transition = bool(use_csr_sparse_transition)
        self.use_adaptive_reachable_csr = bool(use_adaptive_reachable_csr)
        if adaptive_csr_warmup_periods <= 0:
            raise ValueError("adaptive_csr_warmup_periods must be positive")
        self.adaptive_csr_warmup_periods = int(adaptive_csr_warmup_periods)
        if adaptive_csr_max_support_states <= 0:
            raise ValueError("adaptive_csr_max_support_states must be positive")
        if adaptive_csr_max_build_seconds <= 0.0:
            raise ValueError("adaptive_csr_max_build_seconds must be positive")
        if adaptive_csr_max_attempts <= 0:
            raise ValueError("adaptive_csr_max_attempts must be positive")
        self.adaptive_csr_max_support_states = int(adaptive_csr_max_support_states)
        self.adaptive_csr_max_build_seconds = float(adaptive_csr_max_build_seconds)
        self.adaptive_csr_max_attempts = int(adaptive_csr_max_attempts)
        if certified_screen_max_states <= 0:
            raise ValueError("certified_screen_max_states must be positive")
        if certified_screen_max_seconds <= 0.0:
            raise ValueError("certified_screen_max_seconds must be positive")
        self.certified_screen_max_states = int(certified_screen_max_states)
        self.certified_screen_max_seconds = float(certified_screen_max_seconds)
        self.use_stationary_continuation_bound = bool(
            use_stationary_continuation_bound
        )
        if full_state_geometry_limit is not None and int(full_state_geometry_limit) < 0:
            raise ValueError("full_state_geometry_limit must be nonnegative or None")
        self.full_state_geometry_limit = (
            None if full_state_geometry_limit is None else int(full_state_geometry_limit)
        )
        self.proposal_complexity_guard = (
            None if proposal_complexity_guard is None else dict(proposal_complexity_guard)
        )
        # Disabled by default; debug runs can collect period-level state traces
        # without changing the normal evaluator's allocation or I/O path.
        self._debug_trace = debug_trace
        self._debug_context: dict[str, Any] = {}
        if stable_dense_cell_limit < 0:
            raise ValueError("stable_dense_cell_limit must be nonnegative")
        self.stable_dense_cell_limit = int(stable_dense_cell_limit)
        self._horizon_dependent = "remaining_horizon" in _node_dependencies(ast, self.motifs).states
        self._is_vector_base_stock = detect_vbs_ast(ast, reference.lead_time)
        self._state_partition_tokens = _state_partition_tokens(
            ast, self.motifs, reference.lead_time, frozenset()
        )
        self._signature_states: np.ndarray | None = None
        state_count = (reference.n + 1) ** (reference.lead_time + 1)
        self.lazy_state_geometry = bool(
            self.full_state_geometry_limit is not None
            and state_count > self.full_state_geometry_limit
        )
        self._action_values = (
            None
            if self._horizon_dependent or self.lazy_state_geometry
            else np.empty(state_count, dtype=np.int32)
        )
        self._action_stamps = (
            None
            if self._horizon_dependent or self.lazy_state_geometry
            else np.zeros(state_count, dtype=np.uint32)
        )
        self._action_generation = 0
        self._verified_behaviors: dict[bytes, list[tuple[float, np.ndarray, np.ndarray]]] = {}
        self._verified_behavior_count = 0
        self._verified_behavior_limit = 4096 if state_count <= 5000 else 32
        # Geometry and one-period cost depend on the finite state grid and
        # demand distribution, not on policy constants.  Materializing them
        # once avoids repeated base conversion and pipeline encoding at every
        # horizon step of every parameter evaluation.
        self._decoded_states: np.ndarray | None = None
        self._action_representative_states: np.ndarray | None = None
        self._action_class_by_index: np.ndarray | None = None
        if not self.lazy_state_geometry:
            all_indexes = np.arange(state_count, dtype=np.int64)
            decoded_states = self._decode(all_indexes, reference.n + 1, reference.lead_time + 1)
            self._decoded_states = decoded_states.astype(np.int32, copy=False)
            (
                self._action_representative_states,
                self._action_class_by_index,
            ) = _stationary_action_partition(
                self._state_partition_tokens,
                reference,
                self._decoded_states,
                max_classes=50_000,
            )
            if not self.use_action_projection:
                self._action_representative_states = None
                self._action_class_by_index = None
        # Action projection and proof-level behavior dedup have different cost
        # profiles.  Projecting every candidate over a large partition can be
        # slower than evaluating only reachable states, so propagation keeps
        # the conservative 50k-class guard above.  Behavior dedup, however,
        # evaluates only a small fingerprint for new buckets and materializes
        # the complete class map solely on a possible cache hit.  A separate
        # larger guard therefore enables exact L=2 behavior reuse without
        # imposing a full-state action pass on every candidate.
        if self.lazy_state_geometry:
            behavior_representatives = None
        elif self._action_representative_states is not None:
            behavior_representatives = self._action_representative_states
        else:
            assert self._decoded_states is not None
            behavior_representatives, _behavior_classes = _stationary_action_partition(
                self._state_partition_tokens,
                reference,
                self._decoded_states,
                max_classes=250_000,
            )
        self._exact_behavior_states = behavior_representatives
        self.exact_behavior_class_count = (
            0 if behavior_representatives is None else len(behavior_representatives)
        )
        self.action_projection_class_count = (
            0
            if self._action_representative_states is None
            else len(self._action_representative_states)
        )
        self._behavior_fingerprint_states = (
            self._exact_behavior_states[
                np.linspace(
                    0,
                    len(self._exact_behavior_states) - 1,
                    min(64, len(self._exact_behavior_states)),
                    dtype=int,
                )
            ]
            if not self._horizon_dependent and self._exact_behavior_states is not None
            else None
        )
        self._dist_features = reference.demand.feature_dict()
        self._cost_features = {"h": reference.h, "b": reference.b}
        # Reused by every reachable-state action batch for this AST. Dependency
        # analysis is structural and does not depend on states or constants.
        self._batch_dependency_cache: dict[int, Any] = {}
        self._exact_behavior_buckets: dict[
            bytes, list[tuple[bytes, dict[str, float], np.ndarray | None]]
        ] = {}
        self._next_behavior_id = 0
        self._proof_behavior_cache_active = False
        self._available_by_index: np.ndarray | None = None
        self._pipeline_suffix_by_index: np.ndarray | None = None
        if self._decoded_states is not None:
            self._available_by_index = (
                self._decoded_states[:, 0].astype(np.int32)
                + self._decoded_states[:, 1].astype(np.int32)
            )
            if reference.lead_time > 1:
                suffix = self._decoded_states[:, 2].astype(np.int64)
                for pipeline_index in range(3, reference.lead_time + 1):
                    suffix = suffix * (reference.n + 1) + self._decoded_states[:, pipeline_index]
                self._pipeline_suffix_by_index = suffix
            else:
                self._pipeline_suffix_by_index = np.zeros(state_count, dtype=np.int64)
        self._next_x_multiplier = (reference.n + 1) ** max(0, reference.lead_time - 1)
        available_grid = np.arange(2 * reference.n + 1, dtype=np.int32)
        expected_cost = np.zeros(len(available_grid), dtype=float)
        for demand, probability in zip(reference.demand.values.astype(int), reference.demand.probs.astype(float)):
            expected_cost += float(probability) * (
                reference.h * np.maximum(available_grid - int(demand), 0)
                + reference.b * np.maximum(int(demand) - available_grid, 0)
            )
        self._expected_cost_by_available = expected_cost
        # Every future period incurs at least the globally best one-period
        # newsvendor cost, even under a relaxation that may choose available
        # inventory independently each period.  Multiplying this floor by the
        # remaining horizon is therefore a rigorous continuation lower bound.
        # It is especially useful for long-horizon incumbent pruning when the
        # finite DP reference is shorter than the policy evaluation horizon.
        self._minimum_period_cost = float(np.min(expected_cost))

    def screening_copy(self, probability_prune_threshold: float = 1.0e-8) -> "SparseExactPolicyCostEvaluator":
        """Create an independent same-horizon evaluator for approximate screening."""

        return SparseExactPolicyCostEvaluator(
            self.ast,
            self.reference,
            self.horizon,
            self.motifs,
            rounding=self.rounding,
            action_cap=self.action_cap,
            initial_state=self.initial_state,
            convergence_tolerance=self.convergence_tolerance,
            collect_reachable_actions=False,
            dense_switch_fraction=self.dense_switch_fraction,
            max_cycle_period=1,
            stationary_required_consecutive_hits=3,
            probability_prune_threshold=probability_prune_threshold,
            use_inventory_position_reduction=self.use_inventory_position_reduction,
            use_stable_transition_matrix=self.use_stable_transition_matrix,
            use_action_projection=self.use_action_projection,
            stable_dense_cell_limit=self.stable_dense_cell_limit,
            use_csr_sparse_transition=self.use_csr_sparse_transition,
            use_adaptive_reachable_csr=self.use_adaptive_reachable_csr,
            adaptive_csr_warmup_periods=self.adaptive_csr_warmup_periods,
            adaptive_csr_max_support_states=self.adaptive_csr_max_support_states,
            adaptive_csr_max_build_seconds=self.adaptive_csr_max_build_seconds,
            adaptive_csr_max_attempts=self.adaptive_csr_max_attempts,
            certified_screen_max_states=self.certified_screen_max_states,
            certified_screen_max_seconds=self.certified_screen_max_seconds,
            use_stationary_continuation_bound=self.use_stationary_continuation_bound,
            full_state_geometry_limit=self.full_state_geometry_limit,
            debug_trace=self._debug_trace,
            proposal_complexity_guard=None,
        )

    def stationary_copy(
        self,
        *,
        tolerance: float = 1.0e-10,
        required_consecutive_hits: int = 3,
        proposal_complexity_guard: Mapping[str, Any] | None = None,
    ) -> "SparseExactPolicyCostEvaluator":
        """Create the guarded stationary-fast evaluator used by the pipeline."""

        return SparseExactPolicyCostEvaluator(
            self.ast,
            self.reference,
            self.horizon,
            self.motifs,
            rounding=self.rounding,
            action_cap=self.action_cap,
            initial_state=self.initial_state,
            convergence_tolerance=float(tolerance),
            collect_reachable_actions=False,
            dense_switch_fraction=self.dense_switch_fraction,
            max_cycle_period=1,
            stationary_required_consecutive_hits=int(required_consecutive_hits),
            probability_prune_threshold=0.0,
            use_inventory_position_reduction=self.use_inventory_position_reduction,
            use_stable_transition_matrix=self.use_stable_transition_matrix,
            use_action_projection=self.use_action_projection,
            stable_dense_cell_limit=self.stable_dense_cell_limit,
            use_csr_sparse_transition=self.use_csr_sparse_transition,
            use_adaptive_reachable_csr=self.use_adaptive_reachable_csr,
            adaptive_csr_warmup_periods=self.adaptive_csr_warmup_periods,
            adaptive_csr_max_support_states=self.adaptive_csr_max_support_states,
            adaptive_csr_max_build_seconds=self.adaptive_csr_max_build_seconds,
            adaptive_csr_max_attempts=self.adaptive_csr_max_attempts,
            certified_screen_max_states=self.certified_screen_max_states,
            certified_screen_max_seconds=self.certified_screen_max_seconds,
            use_stationary_continuation_bound=self.use_stationary_continuation_bound,
            full_state_geometry_limit=self.full_state_geometry_limit,
            debug_trace=self._debug_trace,
            proposal_complexity_guard=proposal_complexity_guard,
        )

    def set_debug_context(self, context: Mapping[str, Any] | None) -> None:
        """Set candidate metadata attached to subsequent period trace events."""

        self._debug_context = dict(context or {})

    def evaluate(self, params: Mapping[str, float], incumbent: float | None = None) -> EvaluationResult:
        dp = self.reference
        fidelity = "exact" if self.probability_prune_threshold == 0.0 else "probability_pruned"
        evaluation_started = time.perf_counter()
        evaluation_deadline_at = self._debug_context.get("evaluation_deadline_at")
        evaluation_timeout_seconds = self._debug_context.get("evaluation_timeout_seconds")
        if (
            self.use_inventory_position_reduction
            and dp.lead_time == 1
            and self._state_partition_tokens == frozenset({"inventory_position"})
            and not self._horizon_dependent
            and not self.collect_reachable_actions
            and dp.horizon < self.horizon
        ):
            return self._evaluate_l1_inventory_position(params, incumbent, fidelity)
        base = dp.n + 1
        dimensions = dp.lead_time + 1
        state_count = base**dimensions
        initial_index = self._encode_state(self.initial_state, base)
        active_indexes = np.asarray([initial_index], dtype=np.int64)
        active_probs = np.asarray([1.0], dtype=float)
        action_values = self._action_values
        action_stamps = self._action_stamps
        if action_values is not None:
            self._action_generation += 1
            if self._action_generation >= np.iinfo(np.uint32).max:
                assert action_stamps is not None
                action_stamps.fill(0)
                self._action_generation = 1
        generation = self._action_generation
        reachable_actions: dict[tuple[int, ...], int] | None = {} if self.collect_reachable_actions else None
        cumulative_cost = 0.0
        visited_states = 0
        max_reachable = 1
        dense_steps = 0
        sparse_steps = 0
        stable_matrix_steps = 0
        stable_sparse_steps = 0
        aggregation_seconds = 0.0
        cycle_period = 0
        cycle_detected_at = 0
        stationary_hit_streak = 0
        transition_cache_hits = 0
        cumulative_pruned_mass = 0.0
        stable_transition_source: np.ndarray | None = None
        stable_joined_indexes: np.ndarray | None = None
        stable_transition_targets: np.ndarray | None = None
        stable_target_positions: np.ndarray | None = None
        stable_transition_matrix: np.ndarray | None = None
        stable_transition_sparse_operator: Any | None = None
        stable_source_step_costs: np.ndarray | None = None
        # Adaptive reachable-union CSR path.  For eligible exact L>=2
        # evaluations it is enabled by default; callers can still disable it
        # explicitly.  The first few periods use the existing exact
        # sparse/dense propagation, then a policy-specific transition operator
        # is built over the closure of the reachable support.  The operator is
        # transposed (target x source), matching the stable CSR path below.
        adaptive_csr_requested = bool(
            self.use_adaptive_reachable_csr
            and not self._horizon_dependent
            and dp.lead_time > 1
            and self.probability_prune_threshold == 0.0
        )
        adaptive_csr_active = adaptive_csr_requested
        adaptive_csr_operator: Any | None = None
        adaptive_csr_support: np.ndarray | None = None
        adaptive_csr_actions: np.ndarray | None = None
        adaptive_csr_source_step_costs: np.ndarray | None = None
        adaptive_csr_steps = 0
        adaptive_csr_rebuilds = 0
        adaptive_csr_expansions = 0
        adaptive_csr_build_seconds = 0.0
        adaptive_csr_warmup_completed = 0
        adaptive_csr_build_attempts = 0
        adaptive_csr_build_aborts = 0
        adaptive_csr_last_abort_reason: str | None = None
        adaptive_csr_peak_attempted_support = 0
        adaptive_csr_next_attempt_period = self.adaptive_csr_warmup_periods
        certified_screen_attempted = False
        certified_screen_proved = False
        certified_screen_seconds = 0.0
        certified_screen_periods = 0
        certified_screen_peak_states = 0
        certified_screen_retained_mass = 1.0
        certified_screen_lower_bound: float | None = None
        certified_screen_discarded_mass = 0.0
        certified_screen_discarded_continuation_floor = 0.0
        adaptive_csr_matrix_factory: Any | None = None
        guard = self.proposal_complexity_guard
        if guard is not None and not bool(guard.get("enabled", True)):
            guard = None
        guard_generated_branches = 0
        guard_previous_active_states = 1
        guard_growth_hits = 0
        guard_peak_active_states = 1
        guard_abort_reason: str | None = None
        if adaptive_csr_active:
            try:
                from scipy.sparse import csr_matrix as _csr_matrix

                adaptive_csr_matrix_factory = _csr_matrix
            except ImportError:
                # SciPy is an optional acceleration dependency.  A missing
                # import must never change correctness or disable the scalar
                # evaluator, so silently fall back to the existing path.
                adaptive_csr_active = False
        distribution_history: list[tuple[np.ndarray, np.ndarray]] = [
            (active_indexes.copy(), active_probs.copy())
        ]
        step_cost_history: list[float] = []
        demand_values = dp.demand.values.astype(int)
        demand_probs = dp.demand.probs.astype(float)
        dist_features = dp.demand.feature_dict()
        cost_features = {"h": dp.h, "b": dp.b}
        behavior_key: bytes | None = None
        projected_actions: np.ndarray | None = None
        debug_context = dict(self._debug_context)
        debug_detail = bool(debug_context.get("trace_detail", False))
        debug_periods = {int(value) for value in debug_context.get("trace_periods", ())}
        debug_thresholds = tuple(sorted({int(value) for value in debug_context.get("trace_thresholds", ())}))
        debug_thresholds_seen: set[int] = set()
        debug_state_union: np.ndarray | set[int] | None = None
        debug_state_action_union: np.ndarray | set[int] | None = None
        if debug_detail:
            if self.lazy_state_geometry:
                debug_state_union = set()
                debug_state_action_union = set()
            else:
                debug_state_union = np.zeros(state_count, dtype=bool)
                debug_state_action_union = np.zeros(state_count * (dp.n + 1), dtype=bool)
        debug_max_unique_x = 0
        debug_max_pipeline_tuples = 0
        debug_max_inventory_positions = 0
        debug_max_actions = 0
        debug_deadline_period: int | None = None
        debug_deadline_states: int | None = None
        debug_deadline_elapsed: float | None = None
        debug_cost_bound_period: int | None = None
        debug_cost_bound_states: int | None = None
        debug_cost_bound_elapsed: float | None = None
        debug_cost_bound_value: float | None = None
        debug_pilots: dict[str, dict[str, float]] = {}
        debug_started = time.perf_counter()
        debug_deadline_at = debug_context.get("worker_deadline")
        # Keep the incumbent supplied in the candidate context for diagnostics;
        # the actual ``incumbent`` argument remains the authority for stopping
        # (and is unset for the first anchor when no valid incumbent exists).
        debug_incumbent = debug_context.get("incumbent_cost_before")
        if self._action_representative_states is not None:
            projected_actions = self._actions(
                self._action_representative_states,
                params,
                dist_features,
                cost_features,
                1,
            )
        if not self._horizon_dependent and not self.lazy_state_geometry:
            fingerprint_states = (
                self._behavior_fingerprint_states
                if self._behavior_fingerprint_states is not None
                else (
                    np.asarray([self.initial_state], dtype=int)
                    if self._signature_states is None
                    else self._signature_states[:64]
                )
            )
            behavior_key = self._actions(
                fingerprint_states, params, dist_features, cost_features, 1
            ).astype(np.int32, copy=False).tobytes()
            for cached_cost, cached_states, cached_actions in self._verified_behaviors.get(behavior_key, ()):
                candidate_actions = self._actions(cached_states, params, dist_features, cost_features, 1)
                if np.array_equal(candidate_actions, cached_actions):
                    return EvaluationResult(
                        cost=float(cached_cost),
                        completed=True,
                        visited_states=0,
                        metadata={
                            "fidelity": fidelity,
                            "horizon": self.horizon,
                            "engine": "sparse",
                            "verified_action_map_cache_hit": True,
                            "verified_reachable_states": len(cached_states),
                        },
                    )

        internal_reachable_actions: dict[tuple[int, ...], int] | None = (
            {}
            if behavior_key is not None and not self._proof_behavior_cache_active
            else None
        )

        for step in range(self.horizon):
            period_started = time.perf_counter() if self._debug_trace is not None else 0.0
            source_indexes = active_indexes
            visited_states += len(active_indexes)
            max_reachable = max(max_reachable, len(active_indexes))
            using_stable_transition = (
                not adaptive_csr_active
                and adaptive_csr_operator is None
                and stable_transition_source is not None
                and stable_joined_indexes is not None
                and stable_transition_targets is not None
                and stable_target_positions is not None
                and stable_source_step_costs is not None
                and (
                    active_indexes is stable_transition_source
                    or np.array_equal(active_indexes, stable_transition_source)
                )
            )
            using_adaptive_transition = (
                adaptive_csr_operator is not None
                and adaptive_csr_support is not None
                and adaptive_csr_source_step_costs is not None
                and (
                    active_indexes is adaptive_csr_support
                    or np.array_equal(active_indexes, adaptive_csr_support)
                )
            )
            if using_adaptive_transition:
                actions = None
                newly_evaluated_states = None
                newly_evaluated_actions = None
            elif using_stable_transition:
                actions = None
                newly_evaluated_states = None
                newly_evaluated_actions = None
            elif projected_actions is not None:
                assert self._action_class_by_index is not None
                actions = projected_actions[
                    self._action_class_by_index[active_indexes]
                ].astype(int, copy=False)
                assert action_stamps is not None
                missing = action_stamps[active_indexes] != generation
                newly_evaluated_states = None
                newly_evaluated_actions = None
                if np.any(missing):
                    newly_evaluated_states = self._states_for_indexes(active_indexes[missing])
                    newly_evaluated_actions = actions[missing]
                    action_stamps[active_indexes[missing]] = generation
            elif action_values is None:
                states = self._states_for_indexes(active_indexes)
                actions = self._actions(states, params, dist_features, cost_features, self.horizon - step)
                newly_evaluated_states = states
                newly_evaluated_actions = actions
            else:
                assert action_stamps is not None
                actions = action_values[active_indexes].astype(int)
                missing = action_stamps[active_indexes] != generation
                newly_evaluated_states = None
                newly_evaluated_actions = None
                if np.any(missing):
                    newly_evaluated_states = self._states_for_indexes(active_indexes[missing])
                    evaluated = self._actions(
                        newly_evaluated_states,
                        params,
                        dist_features,
                        cost_features,
                        self.horizon - step,
                    )
                    actions[missing] = evaluated
                    action_values[active_indexes[missing]] = evaluated.astype(np.int32)
                    action_stamps[active_indexes[missing]] = generation
                    newly_evaluated_actions = evaluated
            if reachable_actions is not None and not using_stable_transition and not using_adaptive_transition:
                assert actions is not None
                states = self._states_for_indexes(active_indexes)
                reachable_actions.update(
                    {tuple(int(value) for value in state): int(action) for state, action in zip(states, actions)}
                )
            if (
                internal_reachable_actions is not None
                and newly_evaluated_states is not None
                and newly_evaluated_actions is not None
            ):
                internal_reachable_actions.update(
                    {
                        tuple(int(value) for value in state): int(action)
                        for state, action in zip(newly_evaluated_states, newly_evaluated_actions)
                    }
                )

            if using_adaptive_transition:
                assert adaptive_csr_support is not None
                assert adaptive_csr_source_step_costs is not None
                step_cost = float(np.dot(active_probs, adaptive_csr_source_step_costs))
                available = None
                pipeline_suffix = None
            elif using_stable_transition:
                assert stable_source_step_costs is not None
                step_cost = float(np.dot(active_probs, stable_source_step_costs))
                available = None
                pipeline_suffix = None
            else:
                available, pipeline_suffix = self._geometry_for_indexes(active_indexes)
                step_cost = float(np.dot(active_probs, self._expected_cost_by_available[available]))
            next_indexes: list[np.ndarray] = []
            next_probs: list[np.ndarray] = []
            if using_adaptive_transition:
                assert adaptive_csr_operator is not None
                assert adaptive_csr_support is not None
                joined_indexes = adaptive_csr_support
                transition_cache_hits += 1
                aggregation_started = time.perf_counter()
                active_probs = np.asarray(
                    adaptive_csr_operator.dot(active_probs)
                ).reshape(-1)
                adaptive_csr_steps += 1
                active_indexes = adaptive_csr_support
                aggregation_seconds += time.perf_counter() - aggregation_started
                used_stable_matrix = True
            elif using_stable_transition:
                assert stable_joined_indexes is not None
                assert stable_transition_targets is not None
                assert stable_target_positions is not None
                joined_indexes = stable_joined_indexes
                transition_cache_hits += 1
                aggregation_started = time.perf_counter()
                if stable_transition_matrix is not None:
                    active_probs = active_probs @ stable_transition_matrix
                    stable_matrix_steps += 1
                elif stable_transition_sparse_operator is not None:
                    active_probs = np.asarray(
                        stable_transition_sparse_operator.dot(active_probs)
                    ).reshape(-1)
                    stable_sparse_steps += 1
                else:
                    joined_probs = (
                        demand_probs[:, None] * active_probs[None, :]
                    ).reshape(-1)
                    active_probs = np.bincount(
                        stable_target_positions,
                        weights=joined_probs,
                        minlength=len(stable_transition_targets),
                    )
                    stable_sparse_steps += 1
                active_indexes = stable_transition_targets
                aggregation_seconds += time.perf_counter() - aggregation_started
                used_stable_matrix = True
            else:
                used_stable_matrix = False
                assert available is not None
                assert pipeline_suffix is not None
                assert actions is not None
                for demand in demand_values:
                    next_x = np.clip(available - int(demand), 0, dp.n).astype(np.int64)
                    encoded = (
                        next_x * self._next_x_multiplier + pipeline_suffix
                    ) * base + actions
                    next_indexes.append(encoded)
                joined_indexes = np.concatenate(next_indexes)
            if not used_stable_matrix:
                for probability in demand_probs:
                    next_probs.append(active_probs * float(probability))
                joined_probs = np.concatenate(next_probs)
                aggregation_started = time.perf_counter()
                # Sorting/uniquing is ideal while the reachable set is sparse.  Once
                # transitions cover a material fraction of the finite state grid,
                # direct integer scatter is substantially cheaper and remains exact.
                # Lazy geometry is selected specifically to avoid allocations
                # proportional to the Cartesian state space.  A full-length
                # bincount would silently reintroduce that memory cost.
                use_dense = (
                    not self.lazy_state_geometry
                    and joined_indexes.size >= self.dense_switch_fraction * state_count
                )
                if use_dense:
                    dense_probs = np.bincount(joined_indexes, weights=joined_probs, minlength=state_count)
                    active_indexes = np.flatnonzero(dense_probs)
                    active_probs = dense_probs[active_indexes]
                    dense_steps += 1
                else:
                    active_indexes, inverse = np.unique(joined_indexes, return_inverse=True)
                    active_probs = np.bincount(
                        inverse,
                        weights=joined_probs,
                        minlength=len(active_indexes),
                    )
                    sparse_steps += 1
                aggregation_seconds += time.perf_counter() - aggregation_started
            if self.probability_prune_threshold > 0.0:
                keep = active_probs >= self.probability_prune_threshold
                if not np.all(keep):
                    if not np.any(keep):
                        keep[int(np.argmax(active_probs))] = True
                    retained_mass = float(np.sum(active_probs[keep]))
                    cumulative_pruned_mass += max(0.0, 1.0 - retained_mass)
                    active_indexes = active_indexes[keep]
                    active_probs = active_probs[keep] / retained_mass
            if guard is not None:
                # This guard is intentionally proposal-only.  A triggered
                # probe never publishes its partial deterministic cost; the
                # caller must fall back to MC.  Strict Exact evaluators pass
                # ``proposal_complexity_guard=None`` and never enter here.
                guard_generated_branches += int(len(source_indexes) * len(demand_values))
                guard_peak_active_states = max(guard_peak_active_states, int(len(active_indexes)))
                period_number = step + 1
                elapsed_guard = time.perf_counter() - evaluation_started
                current_states = int(len(active_indexes))
                ratio = current_states / max(1, guard_previous_active_states)
                if (
                    bool(guard.get("growth_guard_enabled", True))
                    and current_states >= int(guard.get("growth_min_states", 2_000))
                    and ratio >= float(guard.get("growth_ratio", 2.0))
                ):
                    guard_growth_hits += 1
                else:
                    guard_growth_hits = 0
                guard_previous_active_states = max(1, current_states)
                if (
                    int(guard.get("max_active_states", 0) or 0) > 0
                    and current_states > int(guard["max_active_states"])
                ):
                    guard_abort_reason = "active_state_limit"
                elif (
                    int(guard.get("max_generated_branches", 0) or 0) > 0
                    and guard_generated_branches > int(guard["max_generated_branches"])
                ):
                    guard_abort_reason = "transition_work_limit"
                elif (
                    float(guard.get("max_seconds", 0.0) or 0.0) > 0.0
                    and elapsed_guard >= float(guard["max_seconds"])
                ):
                    guard_abort_reason = "probe_time_limit"
                elif (
                    int(guard.get("growth_hits", 0) or 0) > 0
                    and guard_growth_hits >= int(guard["growth_hits"])
                ):
                    guard_abort_reason = "state_growth"
                elif (
                    bool(guard.get("enabled", True))
                    and int(guard.get("max_periods", 0) or 0) > 0
                    and period_number
                    > int(guard["max_periods"]) + self.stationary_required_consecutive_hits - 1
                ):
                    if not bool(guard.get("project_period_limit", False)):
                        guard_abort_reason = "period_limit"
                    else:
                        # Period count alone is not a useful complexity signal.
                        # Small recurrent chains can need more than 128 periods
                        # to satisfy the tight stationary tolerance while still
                        # costing only milliseconds to finish. Treat max_periods
                        # as the point at which a probe must justify continuation:
                        # extrapolate measured transition work and wall time to H,
                        # and fall back to MC only when that projection exceeds a
                        # configured guard. This retains protection for growing
                        # L>=3/4 chains without misrouting cheap L=1/2 candidates.
                        projected_branches = (
                            float(guard_generated_branches)
                            * float(self.horizon)
                            / max(1.0, float(period_number))
                        )
                        projected_seconds = (
                            float(elapsed_guard)
                            * float(self.horizon)
                            / max(1.0, float(period_number))
                        )
                        branch_limit = int(guard.get("max_generated_branches", 0) or 0)
                        seconds_limit = float(guard.get("max_seconds", 0.0) or 0.0)
                        if branch_limit > 0 and projected_branches > float(branch_limit):
                            guard_abort_reason = "projected_transition_work_limit"
                        elif seconds_limit > 0.0 and projected_seconds > seconds_limit:
                            guard_abort_reason = "projected_probe_time_limit"
                        elif branch_limit <= 0 and seconds_limit <= 0.0:
                            guard_abort_reason = "period_limit"
                if guard_abort_reason is not None:
                    return EvaluationResult(
                        cost=float("inf"),
                        completed=False,
                        early_stopped=False,
                        visited_states=visited_states,
                        reachable_actions=reachable_actions,
                        metadata={
                            "fidelity": "proposal_guard_abort",
                            "horizon": self.horizon,
                            "engine": "adaptive_sparse_dense",
                            "state_geometry": "lazy" if self.lazy_state_geometry else "eager",
                            "proposal_guard_triggered": True,
                            "proposal_guard_abort_reason": guard_abort_reason,
                            "probe_periods": period_number,
                            "probe_peak_active_states": guard_peak_active_states,
                            "probe_generated_branches": guard_generated_branches,
                            "probe_elapsed_seconds": elapsed_guard,
                            "bound_valid": False,
                            "partial_cost": float(cumulative_cost),
                            "partial_cost_is_bound": False,
                            "adaptive_csr_requested": adaptive_csr_requested,
                            "adaptive_csr_warmup_completed": adaptive_csr_warmup_completed,
                        },
                    )
            if (
                not self._horizon_dependent
                and self.use_stable_transition_matrix
                and not adaptive_csr_active
                and adaptive_csr_operator is None
                and stable_transition_source is None
                and np.array_equal(active_indexes, source_indexes)
            ):
                stable_transition_source = source_indexes.copy()
                stable_available, _stable_suffix = self._geometry_for_indexes(
                    stable_transition_source
                )
                stable_source_step_costs = self._expected_cost_by_available[
                    stable_available
                ]
                stable_joined_indexes = joined_indexes.copy()
                stable_transition_targets = np.unique(stable_joined_indexes)
                if np.array_equal(stable_transition_source, stable_transition_targets):
                    # Canonicalize the equal support arrays so subsequent
                    # stationary periods use an identity check, not a full
                    # elementwise comparison on every horizon step.
                    stable_transition_source = stable_transition_targets
                stable_target_positions = np.searchsorted(
                    stable_transition_targets, stable_joined_indexes
                )
                dense_cells = len(stable_transition_source) * len(stable_transition_targets)
                source_positions = np.tile(
                    np.arange(len(stable_transition_source), dtype=np.int64),
                    len(demand_probs),
                )
                transition_weights = np.repeat(
                    demand_probs, len(stable_transition_source)
                )
                if dense_cells <= self.stable_dense_cell_limit:
                    stable_transition_matrix = np.zeros(
                        (len(stable_transition_source), len(stable_transition_targets)),
                        dtype=float,
                    )
                    np.add.at(
                        stable_transition_matrix,
                        (source_positions, stable_target_positions),
                        transition_weights,
                    )
                elif self.use_csr_sparse_transition:
                    try:
                        from scipy.sparse import csr_matrix

                        # Store the transposed transition directly so each
                        # stable update is one compiled CSR matrix-vector
                        # product: next[target] = sum_source P[source,target]
                        # * current[source].  This avoids allocating the
                        # demand-by-support probability expansion every period.
                        stable_transition_sparse_operator = csr_matrix(
                            (
                                transition_weights,
                                (stable_target_positions, source_positions),
                            ),
                            shape=(
                                len(stable_transition_targets),
                                len(stable_transition_source),
                            ),
                        )
                    except ImportError:
                        stable_transition_sparse_operator = None
            cumulative_cost += step_cost
            step_cost_history.append(step_cost)

            if (
                evaluation_deadline_at is not None
                and time.perf_counter() >= float(evaluation_deadline_at)
            ):
                return EvaluationResult(
                    cost=float("inf"),
                    completed=False,
                    early_stopped=False,
                    visited_states=visited_states,
                    reachable_actions=reachable_actions,
                    metadata={
                        "fidelity": "timeout_incomplete",
                        "horizon": self.horizon,
                        "engine": "adaptive_sparse_dense",
                        "state_geometry": "lazy" if self.lazy_state_geometry else "eager",
                        "stop_reason": "timeout",
                        "timed_out": True,
                        "bound_valid": False,
                        "partial_cost": float(cumulative_cost),
                        "partial_cost_is_bound": False,
                        "evaluation_timeout_seconds": (
                            float(evaluation_timeout_seconds)
                            if evaluation_timeout_seconds is not None
                            else None
                        ),
                        "timeout_elapsed_seconds": time.perf_counter() - evaluation_started,
                        "max_reachable_states": max_reachable,
                        "dense_steps": dense_steps,
                        "sparse_steps": sparse_steps,
                        "stable_matrix_steps": stable_matrix_steps,
                        "stable_sparse_steps": stable_sparse_steps,
                        "adaptive_csr_requested": adaptive_csr_requested,
                        "adaptive_csr_active": adaptive_csr_operator is not None,
                        "adaptive_csr_warmup_periods": self.adaptive_csr_warmup_periods,
                        "adaptive_csr_warmup_completed": adaptive_csr_warmup_completed,
                        "adaptive_csr_steps": adaptive_csr_steps,
                        "adaptive_csr_rebuilds": adaptive_csr_rebuilds,
                        "adaptive_csr_expansions": adaptive_csr_expansions,
                        "adaptive_csr_support_size": (
                            len(adaptive_csr_support)
                            if adaptive_csr_support is not None
                            else 0
                        ),
                        "adaptive_csr_build_seconds": adaptive_csr_build_seconds,
                        "adaptive_csr_build_attempts": adaptive_csr_build_attempts,
                        "adaptive_csr_build_aborts": adaptive_csr_build_aborts,
                        "adaptive_csr_last_abort_reason": adaptive_csr_last_abort_reason,
                        "adaptive_csr_peak_attempted_support": adaptive_csr_peak_attempted_support,
                        "adaptive_csr_max_support_states": self.adaptive_csr_max_support_states,
                        "adaptive_csr_max_build_seconds": self.adaptive_csr_max_build_seconds,
                        "adaptive_csr_max_attempts": self.adaptive_csr_max_attempts,
                        "certified_screen_attempted": certified_screen_attempted,
                        "certified_screen_proved": certified_screen_proved,
                        "certified_screen_seconds": certified_screen_seconds,
                        "certified_screen_periods": certified_screen_periods,
                        "certified_screen_peak_states": certified_screen_peak_states,
                        "certified_screen_retained_mass": certified_screen_retained_mass,
                        "certified_screen_discarded_mass": certified_screen_discarded_mass,
                        "certified_screen_discarded_continuation_floor": certified_screen_discarded_continuation_floor,
                        "certified_screen_lower_bound": certified_screen_lower_bound,
                        "aggregation_seconds": aggregation_seconds,
                    },
                )

            period_number = step + 1
            if debug_detail:
                assert debug_state_union is not None
                assert debug_state_action_union is not None
                if isinstance(debug_state_union, set):
                    debug_state_union.update(int(value) for value in source_indexes)
                else:
                    debug_state_union[source_indexes] = True
                if actions is not None:
                    action_keys = source_indexes * (dp.n + 1) + np.asarray(actions, dtype=np.int64)
                    if isinstance(debug_state_action_union, set):
                        debug_state_action_union.update(int(value) for value in action_keys)
                    else:
                        debug_state_action_union[action_keys] = True
            unique_x = pipeline_tuple_count = unique_inventory_positions = unique_actions = 0

            elapsed_now = time.perf_counter() - debug_started
            if debug_deadline_at is not None and debug_deadline_period is None and time.perf_counter() >= float(debug_deadline_at):
                debug_deadline_period = period_number
                debug_deadline_states = int(visited_states)
                debug_deadline_elapsed = elapsed_now

            for pilot_period in (25, 50, 100):
                if period_number == pilot_period:
                    rate = visited_states / max(elapsed_now, 1.0e-12)
                    predicted_states = visited_states + max(0, self.horizon - period_number) * len(source_indexes)
                    debug_pilots[str(pilot_period)] = {
                        "predicted_states_total": float(predicted_states),
                        "predicted_runtime_seconds": float(elapsed_now + max(0, predicted_states - visited_states) / max(rate, 1.0e-12)),
                        "observed_states_per_second": float(rate),
                    }

            trace_threshold_hit = False
            if debug_detail:
                for threshold in debug_thresholds:
                    if threshold not in debug_thresholds_seen and len(source_indexes) >= threshold:
                        debug_thresholds_seen.add(threshold)
                        trace_threshold_hit = True
            trace_period_hit = period_number in debug_periods or trace_threshold_hit
            # Compute the lower bound before emitting the period snapshot so
            # the snapshot can state whether this period has already crossed
            # the shadow cost bound.  ``incumbent`` controls real early stop;
            # ``debug_incumbent`` is diagnostics-only when incumbent is None.
            incumbent_for_bound = incumbent if incumbent is not None else debug_incumbent
            lower_bound = float(cumulative_cost)
            if incumbent_for_bound is not None:
                lower_bound += self._dp_remaining_lower_bound(
                    step + 1, active_indexes, active_probs, base, dimensions
                )
            incumbent_tolerance = (
                1.0e-12 * max(1.0, abs(float(incumbent_for_bound)))
                if incumbent_for_bound is not None
                else 0.0
            )
            bound_failed = bool(
                incumbent_for_bound is not None
                and lower_bound > float(incumbent_for_bound) + incumbent_tolerance
            )
            if bound_failed and debug_cost_bound_period is None:
                debug_cost_bound_period = period_number
                debug_cost_bound_states = int(visited_states)
                debug_cost_bound_elapsed = float(elapsed_now)
                debug_cost_bound_value = float(lower_bound)

            if self._debug_trace is not None and debug_detail and trace_period_hit:
                states = self._states_for_indexes(source_indexes)
                unique_x = int(np.unique(states[:, 0]).size) if len(states) else 0
                pipeline_tuple_count = (
                    int(np.unique(states[:, 1:], axis=0).shape[0])
                    if states.shape[1] > 1 and len(states)
                    else 1
                )
                inventory_positions = (
                    states[:, 0] + np.sum(states[:, 1:], axis=1)
                    if states.shape[1] > 1
                    else states[:, 0]
                )
                unique_inventory_positions = int(np.unique(inventory_positions).size) if len(states) else 0
                unique_actions = int(np.unique(actions).size) if actions is not None and len(actions) else 0
                debug_max_unique_x = max(debug_max_unique_x, unique_x)
                debug_max_pipeline_tuples = max(debug_max_pipeline_tuples, pipeline_tuple_count)
                debug_max_inventory_positions = max(debug_max_inventory_positions, unique_inventory_positions)
                debug_max_actions = max(debug_max_actions, unique_actions)
                order_min = order_max = None
                if actions is not None and len(actions):
                    order_min = int(np.min(actions))
                    order_max = int(np.max(actions))
                self._debug_trace(
                    {
                        "kind": "period",
                        **self._debug_context,
                        "t": int(period_number),
                        "active_states": int(len(source_indexes)),
                        "states_total": int(visited_states),
                        "states_max": int(max_reachable),
                        "cumulative_state_visits": int(visited_states),
                        "unique_states_union": _debug_seen_count(debug_state_union),
                        "unique_state_action_pairs_union": _debug_seen_count(debug_state_action_union),
                        "unique_x": int(unique_x),
                        "unique_pipeline_tuples": int(pipeline_tuple_count),
                        "unique_inventory_positions": int(unique_inventory_positions),
                        "unique_actions": int(unique_actions),
                        "pipeline_states": int(pipeline_tuple_count),
                        "transition_count": int(len(source_indexes) * len(demand_values)),
                        "inventory_min": int(np.min(states[:, 0])) if len(states) else None,
                        "inventory_max": int(np.max(states[:, 0])) if len(states) else None,
                        "order_min": order_min,
                        "order_max": order_max,
                        "period_expected_cost": float(step_cost),
                        "cumulative_expected_cost": float(cumulative_cost),
                        "cost_bound_already_failed": bound_failed,
                        "incumbent_cost_for_bound": (
                            float(incumbent_for_bound)
                            if incumbent_for_bound is not None
                            else None
                        ),
                        "deadline_remaining_seconds": (
                            float(debug_deadline_at - time.perf_counter())
                            if debug_deadline_at is not None
                            else None
                        ),
                        "elapsed_seconds": time.perf_counter() - period_started,
                    }
                )

            if (
                incumbent is not None
                and lower_bound > float(incumbent) + incumbent_tolerance
            ):
                return EvaluationResult(
                    cost=float(lower_bound),
                    completed=False,
                    early_stopped=True,
                    visited_states=visited_states,
                    reachable_actions=reachable_actions,
                    metadata={
                        "fidelity": fidelity,
                        "horizon": self.horizon,
                        "engine": "adaptive_sparse_dense",
                        "state_geometry": "lazy" if self.lazy_state_geometry else "eager",
                        "stop_reason": "incumbent_bound",
                        "timed_out": False,
                        "bound_valid": True,
                        "lower_bound": lower_bound,
                        "incumbent_tolerance": incumbent_tolerance,
                        "max_reachable_states": max_reachable,
                        "dense_steps": dense_steps,
                        "sparse_steps": sparse_steps,
                        "stable_matrix_steps": stable_matrix_steps,
                        "stable_sparse_steps": stable_sparse_steps,
                        "action_projection_classes": (
                            len(self._action_representative_states)
                            if self._action_representative_states is not None
                            else 0
                        ),
                        "stable_transition_mode": (
                            "dense"
                            if stable_transition_matrix is not None
                            else "csr"
                            if stable_transition_sparse_operator is not None
                            else "sparse"
                            if stable_target_positions is not None
                            else "none"
                        ),
                        "stable_transition_source_states": (
                            len(stable_transition_source)
                            if stable_transition_source is not None
                            else 0
                        ),
                        "stable_transition_target_states": (
                            len(stable_transition_targets)
                            if stable_transition_targets is not None
                            else 0
                        ),
                        "adaptive_csr_requested": adaptive_csr_requested,
                        "adaptive_csr_active": adaptive_csr_operator is not None,
                        "adaptive_csr_warmup_periods": self.adaptive_csr_warmup_periods,
                        "adaptive_csr_warmup_completed": adaptive_csr_warmup_completed,
                        "adaptive_csr_steps": adaptive_csr_steps,
                        "adaptive_csr_rebuilds": adaptive_csr_rebuilds,
                        "adaptive_csr_expansions": adaptive_csr_expansions,
                        "adaptive_csr_support_size": (
                            len(adaptive_csr_support)
                            if adaptive_csr_support is not None
                            else 0
                        ),
                        "adaptive_csr_build_seconds": adaptive_csr_build_seconds,
                        "adaptive_csr_build_attempts": adaptive_csr_build_attempts,
                        "adaptive_csr_build_aborts": adaptive_csr_build_aborts,
                        "adaptive_csr_last_abort_reason": adaptive_csr_last_abort_reason,
                        "adaptive_csr_peak_attempted_support": adaptive_csr_peak_attempted_support,
                        "adaptive_csr_max_support_states": self.adaptive_csr_max_support_states,
                        "adaptive_csr_max_build_seconds": self.adaptive_csr_max_build_seconds,
                        "adaptive_csr_max_attempts": self.adaptive_csr_max_attempts,
                        "certified_screen_attempted": certified_screen_attempted,
                        "certified_screen_proved": certified_screen_proved,
                        "certified_screen_seconds": certified_screen_seconds,
                        "certified_screen_periods": certified_screen_periods,
                        "certified_screen_peak_states": certified_screen_peak_states,
                        "certified_screen_retained_mass": certified_screen_retained_mass,
                        "certified_screen_discarded_mass": certified_screen_discarded_mass,
                        "certified_screen_discarded_continuation_floor": certified_screen_discarded_continuation_floor,
                        "certified_screen_lower_bound": certified_screen_lower_bound,
                        "aggregation_seconds": aggregation_seconds,
                        "debug_trace_summary": {
                            "cumulative_state_visits": int(visited_states),
                            "unique_states_union": _debug_seen_count(debug_state_union),
                            "state_revisit_ratio": (
                                float(visited_states / max(1, int(_debug_seen_count(debug_state_union) or 0)))
                                if debug_state_union is not None else None
                            ),
                            "unique_state_action_pairs_union": _debug_seen_count(debug_state_action_union),
                            "state_action_revisit_ratio": (
                                float(visited_states / max(1, int(_debug_seen_count(debug_state_action_union) or 0)))
                                if debug_state_action_union is not None else None
                            ),
                            "max_unique_x": debug_max_unique_x,
                            "max_pipeline_tuples": debug_max_pipeline_tuples,
                            "max_inventory_positions": debug_max_inventory_positions,
                            "max_unique_actions": debug_max_actions,
                            "deadline_abort_period": debug_deadline_period,
                            "deadline_abort_states": debug_deadline_states,
                            "deadline_abort_elapsed_seconds": debug_deadline_elapsed,
                            "cost_bound_period": debug_cost_bound_period,
                            "cost_bound_states": debug_cost_bound_states,
                            "cost_bound_elapsed_seconds": debug_cost_bound_elapsed,
                            "partial_cost_at_bound": debug_cost_bound_value,
                            "complexity_pilot": debug_pilots,
                        },
                    },
                )
            if (
                adaptive_csr_active
                and adaptive_csr_operator is None
                and step + 1 >= adaptive_csr_next_attempt_period
                and step + 1 < self.horizon
                and adaptive_csr_matrix_factory is not None
            ):
                build_started = time.perf_counter()
                adaptive_csr_build_attempts += 1
                build_result, abort_reason, attempted_support, attempted_expansions = (
                    self._build_adaptive_reachable_csr(
                        params,
                        active_indexes,
                        demand_values,
                        demand_probs,
                        base,
                        projected_actions,
                        dist_features,
                        cost_features,
                        adaptive_csr_matrix_factory,
                        max_support_states=self.adaptive_csr_max_support_states,
                        deadline_at=(
                            min(
                                float(evaluation_deadline_at),
                                build_started + self.adaptive_csr_max_build_seconds,
                            )
                            if evaluation_deadline_at is not None
                            else build_started + self.adaptive_csr_max_build_seconds
                        ),
                    )
                )
                adaptive_csr_build_seconds += time.perf_counter() - build_started
                adaptive_csr_peak_attempted_support = max(
                    adaptive_csr_peak_attempted_support, attempted_support
                )
                if build_result is None:
                    adaptive_csr_build_aborts += 1
                    adaptive_csr_last_abort_reason = abort_reason
                    adaptive_csr_expansions = max(
                        adaptive_csr_expansions, attempted_expansions
                    )
                    # Failed speculative builds are retried only at exponential
                    # period backoff.  Once the attempt budget is exhausted,
                    # disable this accelerator for the candidate and retain the
                    # strict period-by-period sparse evaluator.
                    adaptive_csr_next_attempt_period = max(step + 2, 2 * (step + 1))
                    if (
                        abort_reason == "support_limit"
                        or adaptive_csr_build_attempts >= self.adaptive_csr_max_attempts
                    ):
                        adaptive_csr_active = False
                    if abort_reason == "support_limit" and incumbent is not None:
                        certified_screen_attempted = True
                        screen_started = time.perf_counter()
                        screen = self._certified_substochastic_screen(
                            params,
                            active_indexes,
                            active_probs,
                            cumulative_cost,
                            step + 1,
                            float(incumbent),
                            max_states=self.certified_screen_max_states,
                            deadline_at=(
                                min(
                                    float(evaluation_deadline_at),
                                    screen_started + self.certified_screen_max_seconds,
                                )
                                if evaluation_deadline_at is not None
                                else screen_started + self.certified_screen_max_seconds
                            ),
                        )
                        certified_screen_proved = bool(screen["proved"])
                        certified_screen_seconds += float(screen["seconds"])
                        certified_screen_periods += int(screen["periods"])
                        certified_screen_peak_states = max(
                            certified_screen_peak_states,
                            int(screen["peak_states"]),
                        )
                        certified_screen_retained_mass = float(
                            screen["retained_mass"]
                        )
                        certified_screen_lower_bound = float(screen["lower_bound"])
                        certified_screen_discarded_mass = float(
                            screen["discarded_mass"]
                        )
                        certified_screen_discarded_continuation_floor = float(
                            screen["discarded_continuation_floor"]
                        )
                        if certified_screen_proved:
                            return EvaluationResult(
                                cost=certified_screen_lower_bound,
                                completed=False,
                                early_stopped=True,
                                visited_states=(
                                    visited_states + int(screen["visited_states"])
                                ),
                                reachable_actions=reachable_actions,
                                metadata={
                                    "fidelity": fidelity,
                                    "horizon": self.horizon,
                                    "engine": "certified_substochastic_screen",
                                    "state_geometry": (
                                        "lazy" if self.lazy_state_geometry else "eager"
                                    ),
                                    "stop_reason": "certified_substochastic_bound",
                                    "timed_out": False,
                                    "bound_valid": True,
                                    "lower_bound": certified_screen_lower_bound,
                                    "incumbent_tolerance": (
                                        1.0e-12
                                        * max(1.0, abs(float(incumbent)))
                                    ),
                                    "max_reachable_states": max_reachable,
                                    "adaptive_csr_requested": adaptive_csr_requested,
                                    "adaptive_csr_active": False,
                                    "adaptive_csr_build_attempts": adaptive_csr_build_attempts,
                                    "adaptive_csr_build_aborts": adaptive_csr_build_aborts,
                                    "adaptive_csr_last_abort_reason": adaptive_csr_last_abort_reason,
                                    "adaptive_csr_peak_attempted_support": adaptive_csr_peak_attempted_support,
                                    "adaptive_csr_build_seconds": adaptive_csr_build_seconds,
                                    "certified_screen_attempted": True,
                                    "certified_screen_proved": True,
                                    "certified_screen_seconds": certified_screen_seconds,
                                    "certified_screen_periods": certified_screen_periods,
                                    "certified_screen_completed_periods": int(
                                        screen["completed_periods"]
                                    ),
                                    "certified_screen_peak_states": certified_screen_peak_states,
                                    "certified_screen_retained_mass": certified_screen_retained_mass,
                                    "certified_screen_discarded_mass": certified_screen_discarded_mass,
                                    "certified_screen_discarded_continuation_floor": certified_screen_discarded_continuation_floor,
                                    "certified_screen_max_states": self.certified_screen_max_states,
                                    "certified_screen_max_seconds": self.certified_screen_max_seconds,
                                },
                            )
                        if bool(
                            self._debug_context.get(
                                "stop_after_inconclusive_certified_screen", False
                            )
                        ):
                            return EvaluationResult(
                                cost=certified_screen_lower_bound,
                                completed=False,
                                early_stopped=False,
                                visited_states=(
                                    visited_states + int(screen["visited_states"])
                                ),
                                reachable_actions=reachable_actions,
                                metadata={
                                    "fidelity": "certified_lower_bound_inconclusive",
                                    "horizon": self.horizon,
                                    "engine": "certified_substochastic_screen",
                                    "stop_reason": "diagnostic_screen_inconclusive",
                                    "timed_out": False,
                                    "bound_valid": True,
                                    "lower_bound": certified_screen_lower_bound,
                                    "adaptive_csr_build_attempts": adaptive_csr_build_attempts,
                                    "adaptive_csr_build_aborts": adaptive_csr_build_aborts,
                                    "adaptive_csr_last_abort_reason": adaptive_csr_last_abort_reason,
                                    "adaptive_csr_peak_attempted_support": adaptive_csr_peak_attempted_support,
                                    "adaptive_csr_build_seconds": adaptive_csr_build_seconds,
                                    "certified_screen_attempted": True,
                                    "certified_screen_proved": False,
                                    "certified_screen_seconds": certified_screen_seconds,
                                    "certified_screen_periods": certified_screen_periods,
                                    "certified_screen_completed_periods": int(
                                        screen["completed_periods"]
                                    ),
                                    "certified_screen_peak_states": certified_screen_peak_states,
                                    "certified_screen_retained_mass": certified_screen_retained_mass,
                                    "certified_screen_discarded_mass": certified_screen_discarded_mass,
                                    "certified_screen_discarded_continuation_floor": certified_screen_discarded_continuation_floor,
                                    "certified_screen_max_states": self.certified_screen_max_states,
                                    "certified_screen_max_seconds": self.certified_screen_max_seconds,
                                },
                            )
                else:
                    (
                        adaptive_csr_support,
                        adaptive_csr_operator,
                        adaptive_csr_source_step_costs,
                        adaptive_csr_expansions,
                        adaptive_csr_actions,
                    ) = build_result
                    adaptive_csr_rebuilds = 1 + int(adaptive_csr_expansions)
                    adaptive_csr_warmup_completed = step + 1
                    # The CSR operator is defined on the complete closure
                    # support. Embed the current sparse distribution once.
                    support_positions = np.searchsorted(
                        adaptive_csr_support, active_indexes
                    )
                    support_probs = np.zeros(len(adaptive_csr_support), dtype=float)
                    support_probs[support_positions] = active_probs
                    active_indexes = adaptive_csr_support
                    active_probs = support_probs
                    if adaptive_csr_actions is not None:
                        support_states = self._states_for_indexes(adaptive_csr_support)
                        if reachable_actions is not None:
                            reachable_actions.update(
                                {
                                    tuple(int(value) for value in state): int(action)
                                    for state, action in zip(
                                        support_states, adaptive_csr_actions
                                    )
                                }
                            )
                        if internal_reachable_actions is not None:
                            internal_reachable_actions.update(
                                {
                                    tuple(int(value) for value in state): int(action)
                                    for state, action in zip(
                                        support_states, adaptive_csr_actions
                                    )
                                }
                            )
            if not self._horizon_dependent and self.max_cycle_period > 0:
                available_periods = min(self.max_cycle_period, len(distribution_history))
                matched_period = 0
                for period in range(1, available_periods + 1):
                    historical_indexes, historical_probs = distribution_history[-period]
                    if not np.array_equal(active_indexes, historical_indexes):
                        continue
                    if np.max(np.abs(active_probs - historical_probs), initial=0.0) > self.convergence_tolerance:
                        continue
                    if step + 1 < 8:
                        continue
                    if len(step_cost_history) >= 2:
                        cost_delta = abs(step_cost_history[-1] - step_cost_history[-2])
                        if cost_delta > 1.0e-12 * max(1.0, abs(step_cost_history[-1])):
                            continue
                    matched_period = period
                    break
                if matched_period:
                    if matched_period == 1:
                        stationary_hit_streak += 1
                    else:
                        # The pipeline currently guards period-1 extrapolation. Keep the
                        # general period bookkeeping conservative for callers
                        # that explicitly request a longer cycle period.
                        stationary_hit_streak += 1
                    if stationary_hit_streak >= self.stationary_required_consecutive_hits:
                        cycle_costs = step_cost_history[-matched_period:]
                        remaining = self.horizon - step - 1
                        full_cycles, remainder = divmod(remaining, matched_period)
                        cumulative_cost += full_cycles * float(sum(cycle_costs))
                        cumulative_cost += float(sum(cycle_costs[:remainder]))
                        cycle_period = matched_period
                        cycle_detected_at = step + 1
                        break
                else:
                    stationary_hit_streak = 0
                if cycle_period:
                    break
                distribution_history.append((active_indexes.copy(), active_probs.copy()))
                if len(distribution_history) > self.max_cycle_period:
                    distribution_history.pop(0)
                if len(step_cost_history) > self.max_cycle_period:
                    step_cost_history.pop(0)

        if (
            behavior_key is not None
            and internal_reachable_actions
            and self._verified_behavior_count < self._verified_behavior_limit
        ):
            behavior_states = np.asarray(list(internal_reachable_actions), dtype=int)
            behavior_actions = np.asarray(
                [internal_reachable_actions[state] for state in internal_reachable_actions], dtype=int
            )
            self._verified_behaviors.setdefault(behavior_key, []).append(
                (float(cumulative_cost), behavior_states, behavior_actions)
            )
            self._verified_behavior_count += 1

        return EvaluationResult(
            cost=float(cumulative_cost),
            completed=True,
            visited_states=visited_states,
            reachable_actions=reachable_actions,
            metadata={
                "fidelity": (
                    fidelity
                    if fidelity != "exact"
                    else f"stationary_{self.convergence_tolerance:.0e}"
                    if cycle_period
                    else "exact_fallback"
                    if self.max_cycle_period > 0
                    else "exact"
                ),
                "horizon": self.horizon,
                "engine": "adaptive_sparse_dense",
                "state_geometry": "lazy" if self.lazy_state_geometry else "eager",
                "max_reachable_states": max_reachable,
                "verified_action_map_cache_hit": False,
                "dense_steps": dense_steps,
                "sparse_steps": sparse_steps,
                "stable_matrix_steps": stable_matrix_steps,
                "stable_sparse_steps": stable_sparse_steps,
                "action_projection_classes": (
                    len(self._action_representative_states)
                    if self._action_representative_states is not None
                    else 0
                ),
                "stable_transition_mode": (
                    "dense"
                    if stable_transition_matrix is not None
                    else "csr"
                    if stable_transition_sparse_operator is not None
                    else "sparse"
                    if stable_target_positions is not None
                    else "none"
                ),
                "stable_transition_source_states": (
                    len(stable_transition_source)
                    if stable_transition_source is not None
                    else 0
                ),
                "stable_transition_target_states": (
                    len(stable_transition_targets)
                    if stable_transition_targets is not None
                    else 0
                ),
                "adaptive_csr_requested": adaptive_csr_requested,
                "adaptive_csr_active": adaptive_csr_operator is not None,
                "adaptive_csr_warmup_periods": self.adaptive_csr_warmup_periods,
                "adaptive_csr_warmup_completed": adaptive_csr_warmup_completed,
                "adaptive_csr_steps": adaptive_csr_steps,
                "adaptive_csr_rebuilds": adaptive_csr_rebuilds,
                "adaptive_csr_expansions": adaptive_csr_expansions,
                "adaptive_csr_support_size": (
                    len(adaptive_csr_support)
                    if adaptive_csr_support is not None
                    else 0
                ),
                "adaptive_csr_build_seconds": adaptive_csr_build_seconds,
                "adaptive_csr_build_attempts": adaptive_csr_build_attempts,
                "adaptive_csr_build_aborts": adaptive_csr_build_aborts,
                "adaptive_csr_last_abort_reason": adaptive_csr_last_abort_reason,
                "adaptive_csr_peak_attempted_support": adaptive_csr_peak_attempted_support,
                "adaptive_csr_max_support_states": self.adaptive_csr_max_support_states,
                "adaptive_csr_max_build_seconds": self.adaptive_csr_max_build_seconds,
                "adaptive_csr_max_attempts": self.adaptive_csr_max_attempts,
                "certified_screen_attempted": certified_screen_attempted,
                "certified_screen_proved": certified_screen_proved,
                "certified_screen_seconds": certified_screen_seconds,
                "certified_screen_periods": certified_screen_periods,
                "certified_screen_peak_states": certified_screen_peak_states,
                "certified_screen_retained_mass": certified_screen_retained_mass,
                "certified_screen_discarded_mass": certified_screen_discarded_mass,
                "certified_screen_discarded_continuation_floor": certified_screen_discarded_continuation_floor,
                "certified_screen_lower_bound": certified_screen_lower_bound,
                "aggregation_seconds": aggregation_seconds,
                "cycle_period": cycle_period,
                "cycle_detected_at": cycle_detected_at,
                "stationary_triggered": bool(cycle_period),
                "exact_fallback": bool(
                    self.max_cycle_period > 0 and cycle_period == 0
                ),
                "stationary_hit_streak": stationary_hit_streak,
                "transition_cache_hits": transition_cache_hits,
                "probability_prune_threshold": self.probability_prune_threshold,
                "cumulative_pruned_mass": cumulative_pruned_mass,
                "final_probability_support": {
                    str(threshold): int(np.count_nonzero(active_probs >= threshold))
                    for threshold in (1.0e-6, 1.0e-9, 1.0e-12, 1.0e-15)
                },
                "final_probability_mass_below": {
                    str(threshold): float(np.sum(active_probs[active_probs < threshold]))
                    for threshold in (1.0e-6, 1.0e-9, 1.0e-12, 1.0e-15)
                },
                "debug_trace_summary": {
                    "cumulative_state_visits": int(visited_states),
                    "unique_states_union": _debug_seen_count(debug_state_union),
                    "state_revisit_ratio": (
                        float(visited_states / max(1, int(_debug_seen_count(debug_state_union) or 0)))
                        if debug_state_union is not None else None
                    ),
                    "unique_state_action_pairs_union": _debug_seen_count(debug_state_action_union),
                    "state_action_revisit_ratio": (
                        float(visited_states / max(1, int(_debug_seen_count(debug_state_action_union) or 0)))
                        if debug_state_action_union is not None else None
                    ),
                    "max_unique_x": debug_max_unique_x,
                    "max_pipeline_tuples": debug_max_pipeline_tuples,
                    "max_inventory_positions": debug_max_inventory_positions,
                    "max_unique_actions": debug_max_actions,
                    "deadline_abort_period": debug_deadline_period,
                    "deadline_abort_states": debug_deadline_states,
                    "deadline_abort_elapsed_seconds": debug_deadline_elapsed,
                    "cost_bound_period": debug_cost_bound_period,
                    "cost_bound_states": debug_cost_bound_states,
                    "cost_bound_elapsed_seconds": debug_cost_bound_elapsed,
                    "partial_cost_at_bound": debug_cost_bound_value,
                    "complexity_pilot": debug_pilots,
                },
            },
        )

    def _evaluate_l1_inventory_position(
        self,
        params: Mapping[str, float],
        incumbent: float | None,
        fidelity: str,
    ) -> EvaluationResult:
        """Exact L=1 propagation on the inventory-position Markov state."""

        evaluation_started = time.perf_counter()
        evaluation_deadline_at = self._debug_context.get("evaluation_deadline_at")
        evaluation_timeout_seconds = self._debug_context.get("evaluation_timeout_seconds")
        dp = self.reference
        assert self._exact_behavior_states is not None
        actions = self._actions(
            self._exact_behavior_states,
            params,
            self._dist_features,
            self._cost_features,
            remaining_horizon=1,
        )
        state_count = 2 * dp.n + 1
        inventory_positions = np.arange(state_count, dtype=np.int64)
        demand_values = dp.demand.values.astype(int)
        demand_probs = dp.demand.probs.astype(float)
        transitions = np.empty((len(demand_values), state_count), dtype=np.int64)
        for demand_index, demand in enumerate(demand_values):
            remaining = np.clip(inventory_positions - int(demand), 0, dp.n)
            transitions[demand_index] = remaining + actions

        initial_position = int(sum(self.initial_state))
        active_indexes = np.asarray([initial_position], dtype=np.int64)
        active_probs = np.asarray([1.0], dtype=float)
        cumulative_cost = 0.0
        cumulative_pruned_mass = 0.0
        visited_states = 0
        max_reachable = 1
        aggregation_seconds = 0.0
        distribution_history: list[tuple[np.ndarray, np.ndarray]] = [
            (active_indexes.copy(), active_probs.copy())
        ]
        step_cost_history: list[float] = []
        stationary_hit_streak = 0
        cycle_period = 0
        cycle_detected_at = 0
        guard = self.proposal_complexity_guard
        if guard is not None and not bool(guard.get("enabled", True)):
            guard = None
        guard_generated_branches = 0
        guard_previous_active_states = 1
        guard_growth_hits = 0
        guard_peak_active_states = 1
        for _step in range(self.horizon):
            visited_states += len(active_indexes)
            max_reachable = max(max_reachable, len(active_indexes))
            step_cost = float(
                np.dot(active_probs, self._expected_cost_by_available[active_indexes])
            )
            cumulative_cost += step_cost
            step_cost_history.append(step_cost)
            joined_indexes = transitions[:, active_indexes].reshape(-1)
            joined_probs = (demand_probs[:, None] * active_probs[None, :]).reshape(-1)
            aggregation_started = time.perf_counter()
            dense_probs = np.bincount(
                joined_indexes, weights=joined_probs, minlength=state_count
            )
            active_indexes = np.flatnonzero(dense_probs)
            active_probs = dense_probs[active_indexes]
            aggregation_seconds += time.perf_counter() - aggregation_started
            if self.probability_prune_threshold > 0.0:
                keep = active_probs >= self.probability_prune_threshold
                if not np.all(keep):
                    if not np.any(keep):
                        keep[int(np.argmax(active_probs))] = True
                    retained_mass = float(np.sum(active_probs[keep]))
                    cumulative_pruned_mass += max(0.0, 1.0 - retained_mass)
                    active_indexes = active_indexes[keep]
                    active_probs = active_probs[keep] / retained_mass
            if guard is not None:
                guard_generated_branches += int(len(active_indexes) * len(demand_values))
                guard_peak_active_states = max(guard_peak_active_states, int(len(active_indexes)))
                period_number = _step + 1
                current_states = int(len(active_indexes))
                ratio = current_states / max(1, guard_previous_active_states)
                if (
                    bool(guard.get("growth_guard_enabled", True))
                    and current_states >= int(guard.get("growth_min_states", 2_000))
                    and ratio >= float(guard.get("growth_ratio", 2.0))
                ):
                    guard_growth_hits += 1
                else:
                    guard_growth_hits = 0
                guard_previous_active_states = max(1, current_states)
                guard_abort_reason: str | None = None
                elapsed_guard = time.perf_counter() - evaluation_started
                if int(guard.get("max_active_states", 0) or 0) > 0 and current_states > int(guard["max_active_states"]):
                    guard_abort_reason = "active_state_limit"
                elif int(guard.get("max_generated_branches", 0) or 0) > 0 and guard_generated_branches > int(guard["max_generated_branches"]):
                    guard_abort_reason = "transition_work_limit"
                elif float(guard.get("max_seconds", 0.0) or 0.0) > 0.0 and elapsed_guard >= float(guard["max_seconds"]):
                    guard_abort_reason = "probe_time_limit"
                elif int(guard.get("growth_hits", 0) or 0) > 0 and guard_growth_hits >= int(guard["growth_hits"]):
                    guard_abort_reason = "state_growth"
                elif (
                    int(guard.get("max_periods", 0) or 0) > 0
                    and period_number
                    > int(guard["max_periods"]) + self.stationary_required_consecutive_hits - 1
                ):
                    if not bool(guard.get("project_period_limit", False)):
                        guard_abort_reason = "period_limit"
                    else:
                        projected_branches = (
                            float(guard_generated_branches)
                            * float(self.horizon)
                            / max(1.0, float(period_number))
                        )
                        projected_seconds = (
                            float(elapsed_guard)
                            * float(self.horizon)
                            / max(1.0, float(period_number))
                        )
                        branch_limit = int(guard.get("max_generated_branches", 0) or 0)
                        seconds_limit = float(guard.get("max_seconds", 0.0) or 0.0)
                        if branch_limit > 0 and projected_branches > float(branch_limit):
                            guard_abort_reason = "projected_transition_work_limit"
                        elif seconds_limit > 0.0 and projected_seconds > seconds_limit:
                            guard_abort_reason = "projected_probe_time_limit"
                        elif branch_limit <= 0 and seconds_limit <= 0.0:
                            guard_abort_reason = "period_limit"
                if guard_abort_reason is not None:
                    return EvaluationResult(
                        cost=float("inf"),
                        completed=False,
                        early_stopped=False,
                        visited_states=visited_states,
                        metadata={
                            "fidelity": "proposal_guard_abort",
                            "horizon": self.horizon,
                            "engine": "l1_inventory_position",
                            "proposal_guard_triggered": True,
                            "proposal_guard_abort_reason": guard_abort_reason,
                            "probe_periods": period_number,
                            "probe_peak_active_states": guard_peak_active_states,
                            "probe_generated_branches": guard_generated_branches,
                            "probe_elapsed_seconds": elapsed_guard,
                            "bound_valid": False,
                            "partial_cost": float(cumulative_cost),
                            "partial_cost_is_bound": False,
                        },
                    )
            if (
                evaluation_deadline_at is not None
                and time.perf_counter() >= float(evaluation_deadline_at)
            ):
                return EvaluationResult(
                    cost=float("inf"),
                    completed=False,
                    early_stopped=False,
                    visited_states=visited_states,
                    metadata={
                        "fidelity": "timeout_incomplete",
                        "horizon": self.horizon,
                        "engine": "l1_inventory_position",
                        "stop_reason": "timeout",
                        "timed_out": True,
                        "bound_valid": False,
                        "partial_cost": float(cumulative_cost),
                        "partial_cost_is_bound": False,
                        "evaluation_timeout_seconds": (
                            float(evaluation_timeout_seconds)
                            if evaluation_timeout_seconds is not None
                            else None
                        ),
                        "timeout_elapsed_seconds": time.perf_counter() - evaluation_started,
                        "max_reachable_states": max_reachable,
                        "dense_steps": _step + 1,
                        "sparse_steps": 0,
                        "aggregation_seconds": aggregation_seconds,
                    },
                )
            if self.max_cycle_period > 0:
                historical_indexes, historical_probs = distribution_history[-1]
                period_cost_ok = True
                if len(step_cost_history) >= 2:
                    period_cost_ok = abs(step_cost_history[-1] - step_cost_history[-2]) <= (
                        1.0e-12 * max(1.0, abs(step_cost_history[-1]))
                    )
                if (
                    _step + 1 >= 8
                    and
                    np.array_equal(active_indexes, historical_indexes)
                    and np.max(np.abs(active_probs - historical_probs), initial=0.0)
                    <= self.convergence_tolerance
                    and period_cost_ok
                ):
                    stationary_hit_streak += 1
                    if stationary_hit_streak >= self.stationary_required_consecutive_hits:
                        remaining = self.horizon - _step - 1
                        cumulative_cost += remaining * step_cost
                        cycle_period = 1
                        cycle_detected_at = _step + 1
                        break
                else:
                    stationary_hit_streak = 0
                distribution_history[0] = (active_indexes.copy(), active_probs.copy())
            tolerance = (
                1.0e-12 * max(1.0, abs(float(incumbent)))
                if incumbent is not None
                else 0.0
            )
            if incumbent is not None and cumulative_cost > float(incumbent) + tolerance:
                return EvaluationResult(
                    cost=float(cumulative_cost),
                    completed=False,
                    early_stopped=True,
                    visited_states=visited_states,
                    metadata={
                        "fidelity": fidelity,
                        "horizon": self.horizon,
                        "engine": "l1_inventory_position",
                        "stop_reason": "incumbent_bound",
                        "timed_out": False,
                        "bound_valid": True,
                        "lower_bound": cumulative_cost,
                        "incumbent_tolerance": tolerance,
                        "max_reachable_states": max_reachable,
                        "dense_steps": self.horizon,
                        "sparse_steps": 0,
                        "aggregation_seconds": aggregation_seconds,
                    },
                )
        return EvaluationResult(
            cost=float(cumulative_cost),
            completed=True,
            visited_states=visited_states,
            metadata={
                "fidelity": (
                    fidelity
                    if fidelity != "exact"
                    else f"stationary_{self.convergence_tolerance:.0e}"
                    if cycle_period
                    else "exact_fallback"
                    if self.max_cycle_period > 0
                    else "exact"
                ),
                "horizon": self.horizon,
                "engine": "l1_inventory_position",
                "max_reachable_states": max_reachable,
                "verified_action_map_cache_hit": False,
                "dense_steps": self.horizon,
                "sparse_steps": 0,
                "aggregation_seconds": aggregation_seconds,
                "cycle_period": cycle_period,
                "cycle_detected_at": cycle_detected_at,
                "stationary_triggered": bool(cycle_period),
                "exact_fallback": bool(self.max_cycle_period > 0 and cycle_period == 0),
                "stationary_hit_streak": stationary_hit_streak,
                "transition_cache_hits": 0,
                "probability_prune_threshold": self.probability_prune_threshold,
                "cumulative_pruned_mass": cumulative_pruned_mass,
                "final_probability_support": {
                    str(threshold): int(np.count_nonzero(active_probs >= threshold))
                    for threshold in (1.0e-6, 1.0e-9, 1.0e-12, 1.0e-15)
                },
                "final_probability_mass_below": {
                    str(threshold): float(np.sum(active_probs[active_probs < threshold]))
                    for threshold in (1.0e-6, 1.0e-9, 1.0e-12, 1.0e-15)
                },
            },
        )

    def _build_adaptive_reachable_csr(
        self,
        params: Mapping[str, float],
        seed_indexes: np.ndarray,
        demand_values: np.ndarray,
        demand_probs: np.ndarray,
        base: int,
        projected_actions: np.ndarray | None,
        dist_features: Mapping[str, Any],
        cost_features: Mapping[str, float],
        csr_matrix_factory: Any,
        *,
        max_support_states: int,
        deadline_at: float,
    ) -> tuple[
        tuple[np.ndarray, Any, np.ndarray, int, np.ndarray] | None,
        str | None,
        int,
        int,
    ]:
        """Build an exact CSR operator over the reachable transition closure.

        ``seed_indexes`` is the support observed after the warm-up periods.
        We repeatedly add all policy/demand successors until the union is
        closed.  This is policy-specific (and therefore cannot be shared
        across parameter points), but it removes the per-period full-grid
        ``bincount`` once the closure has been materialized.
        """

        support = np.unique(np.asarray(seed_indexes, dtype=np.int64))
        expansions = 0
        peak_support = len(support)
        while True:
            if time.perf_counter() >= deadline_at:
                return None, "build_time_limit", peak_support, expansions
            if len(support) > max_support_states:
                return None, "support_limit", peak_support, expansions
            if projected_actions is not None:
                assert self._action_class_by_index is not None
                actions = projected_actions[
                    self._action_class_by_index[support]
                ].astype(np.int64, copy=False)
            else:
                actions = self._actions(
                    self._states_for_indexes(support),
                    params,
                    dist_features,
                    cost_features,
                    remaining_horizon=1,
                ).astype(np.int64, copy=False)
            available, pipeline_suffix = self._geometry_for_indexes(support)
            next_indexes: list[np.ndarray] = []
            for demand in demand_values:
                next_x = np.clip(available - int(demand), 0, self.reference.n).astype(np.int64)
                next_indexes.append(
                    (next_x * self._next_x_multiplier + pipeline_suffix) * base + actions
                )
            joined_indexes = np.concatenate(next_indexes)
            targets = np.unique(joined_indexes)
            new_targets = np.setdiff1d(targets, support, assume_unique=True)
            if len(new_targets) == 0:
                break
            next_support_size = len(support) + len(new_targets)
            peak_support = max(peak_support, next_support_size)
            if next_support_size > max_support_states:
                return None, "support_limit", peak_support, expansions
            if time.perf_counter() >= deadline_at:
                return None, "build_time_limit", peak_support, expansions
            support = np.union1d(support, new_targets)
            expansions += 1

        target_positions = np.searchsorted(support, joined_indexes)
        source_positions = np.tile(
            np.arange(len(support), dtype=np.int64),
            len(demand_probs),
        )
        transition_weights = np.repeat(demand_probs, len(support))
        operator = csr_matrix_factory(
            (transition_weights, (target_positions, source_positions)),
            shape=(len(support), len(support)),
        )
        operator.sum_duplicates()
        support_available, _support_suffix = self._geometry_for_indexes(support)
        source_step_costs = self._expected_cost_by_available[support_available]
        return (
            support,
            operator,
            source_step_costs,
            expansions,
            actions,
        ), None, peak_support, expansions

    def _certified_substochastic_screen(
        self,
        params: Mapping[str, float],
        indexes: np.ndarray,
        probabilities: np.ndarray,
        cumulative_cost: float,
        elapsed_periods: int,
        incumbent: float,
        *,
        max_states: int,
        deadline_at: float,
    ) -> dict[str, Any]:
        """Try to reject a candidate using a bounded, rigorous lower bound.

        Low-probability states are discarded without renormalization.  Since
        all period costs are nonnegative, the surviving substochastic process
        accumulates no more cost than the full policy process.  Crossing the
        completed incumbent therefore proves rejection; failure to cross is
        only inconclusive and never changes the strict evaluator state.
        """

        active_indexes = np.asarray(indexes, dtype=np.int64).copy()
        active_probs = np.asarray(probabilities, dtype=float).copy()
        demand_values = self.reference.demand.values.astype(int)
        demand_probs = self.reference.demand.probs.astype(float)
        dist_features = self.reference.demand.feature_dict()
        cost_features = {"h": self.reference.h, "b": self.reference.b}
        base = self.reference.n + 1
        dimensions = self.reference.lead_time + 1
        started = time.perf_counter()
        visited_states = 0
        peak_states = len(active_indexes)
        periods = 0
        lower_bound = float(cumulative_cost)
        discarded_mass = 0.0
        discarded_continuation_floor = 0.0
        tolerance = 1.0e-12 * max(1.0, abs(float(incumbent)))

        for period in range(int(elapsed_periods), self.horizon):
            if time.perf_counter() >= deadline_at:
                break
            visited_states += len(active_indexes)
            peak_states = max(peak_states, len(active_indexes))
            states = self._states_for_indexes(active_indexes)
            actions = self._actions(
                states,
                params,
                dist_features,
                cost_features,
                self.horizon - period,
            )
            available, pipeline_suffix = self._geometry_for_indexes(active_indexes)
            cumulative_cost += float(
                np.dot(active_probs, self._expected_cost_by_available[available])
            )
            next_indexes = []
            next_probs = []
            for demand, probability in zip(demand_values, demand_probs):
                next_x = np.clip(available - int(demand), 0, self.reference.n).astype(
                    np.int64
                )
                next_indexes.append(
                    (next_x * self._next_x_multiplier + pipeline_suffix) * base
                    + actions
                )
                next_probs.append(active_probs * float(probability))
            joined_indexes = np.concatenate(next_indexes)
            joined_probs = np.concatenate(next_probs)
            active_indexes, inverse = np.unique(joined_indexes, return_inverse=True)
            active_probs = np.bincount(
                inverse, weights=joined_probs, minlength=len(active_indexes)
            )
            if len(active_indexes) > max_states:
                selected = np.argpartition(active_probs, -max_states)[-max_states:]
                retained_mass = float(np.sum(active_probs[selected]))
                newly_discarded_mass = max(
                    0.0, float(np.sum(active_probs)) - retained_mass
                )
                # Every discarded trajectory has already contributed its cost
                # through ``completed_periods``.  For its untracked future we
                # can still bank the state-independent one-period DP floor.
                # Banking it at the moment of deletion avoids both restoring
                # the state and silently replacing all of its future cost by 0.
                discarded_mass += newly_discarded_mass
                discarded_continuation_floor += (
                    newly_discarded_mass
                    * max(0, self.horizon - (period + 1))
                    * self._minimum_period_cost
                )
                selected = selected[
                    np.argsort(active_indexes[selected], kind="stable")
                ]
                active_indexes = active_indexes[selected]
                active_probs = active_probs[selected]
            periods += 1
            completed_periods = period + 1
            lower_bound = float(cumulative_cost + discarded_continuation_floor)
            if completed_periods < self.horizon:
                lower_bound += self._dp_remaining_lower_bound(
                    completed_periods,
                    active_indexes,
                    active_probs,
                    base,
                    dimensions,
                )
            if lower_bound > float(incumbent) + tolerance:
                return {
                    "proved": True,
                    "lower_bound": lower_bound,
                    "periods": periods,
                    "completed_periods": completed_periods,
                    "visited_states": visited_states,
                    "peak_states": peak_states,
                    "retained_mass": float(np.sum(active_probs)),
                    "discarded_mass": discarded_mass,
                    "discarded_continuation_floor": discarded_continuation_floor,
                    "seconds": time.perf_counter() - started,
                }

        return {
            "proved": False,
            "lower_bound": lower_bound,
            "periods": periods,
            "completed_periods": int(elapsed_periods) + periods,
            "visited_states": visited_states,
            "peak_states": peak_states,
            "retained_mass": float(np.sum(active_probs)),
            "discarded_mass": discarded_mass,
            "discarded_continuation_floor": discarded_continuation_floor,
            "seconds": time.perf_counter() - started,
        }

    def _states_for_indexes(self, indexes: np.ndarray) -> np.ndarray:
        """Decode only requested states when full Cartesian geometry is disabled."""

        normalized = np.asarray(indexes, dtype=np.int64)
        if self._decoded_states is not None:
            return self._decoded_states[normalized]
        return self._decode(
            normalized,
            self.reference.n + 1,
            self.reference.lead_time + 1,
        ).astype(np.int32, copy=False)

    def _geometry_for_indexes(
        self,
        indexes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return available inventory and encoded pipeline suffix on demand."""

        normalized = np.asarray(indexes, dtype=np.int64)
        if self._available_by_index is not None:
            assert self._pipeline_suffix_by_index is not None
            return (
                self._available_by_index[normalized],
                self._pipeline_suffix_by_index[normalized],
            )
        states = self._states_for_indexes(normalized)
        available = (
            states[:, 0].astype(np.int32, copy=False)
            + states[:, 1].astype(np.int32, copy=False)
        )
        if self.reference.lead_time <= 1:
            suffix = np.zeros(len(states), dtype=np.int64)
        else:
            suffix = states[:, 2].astype(np.int64, copy=True)
            base = self.reference.n + 1
            for pipeline_index in range(3, self.reference.lead_time + 1):
                suffix *= base
                suffix += states[:, pipeline_index]
        return available, suffix

    def _actions(
        self,
        states: np.ndarray,
        params: Mapping[str, float],
        dist_features: Mapping[str, Any],
        cost_features: Mapping[str, float],
        remaining_horizon: int,
    ) -> np.ndarray:
        if self._is_vector_base_stock:
            actions, _active_constraints = evaluate_vbs_actions(
                states,
                params,
                self.reference.lead_time,
                self.reference.n,
            )
            return actions
        batch = _BatchSamples.from_state_array(states, remaining_horizon=float(remaining_horizon))
        raw = _evaluate_ast_batch(
            self.ast,
            batch,
            params,
            dist_features,
            cost_features,
            self.motifs,
            context=_BatchEvaluationContext(
                batch,
                self.motifs,
                cache_within_evaluation=True,
                dependency_cache=self._batch_dependency_cache,
            ),
        )
        if not np.all(np.isfinite(raw)):
            raise ValueError("Policy formula produced NaN or Inf")
        actions = np.rint(raw)
        np.clip(actions, 0.0, float(self.reference.n), out=actions)
        return actions.astype(np.int32)

    def configure_signature_bank(self, size: int) -> None:
        self._signature_states = _build_signature_states(self.reference, size)

    def disable_large_behavior_proof(self, class_limit: int = 5000) -> None:
        """Prefer reachable-policy reuse when a complete map is too expensive."""

        if self.exact_behavior_class_count <= int(class_limit):
            return
        self._exact_behavior_states = None
        self._behavior_fingerprint_states = None
        self._exact_behavior_buckets.clear()
        self.exact_behavior_class_count = 0
        self._proof_behavior_cache_active = False

    def action_signature(self, params: Mapping[str, float]) -> np.ndarray:
        if self._signature_states is None:
            self.configure_signature_bank(500)
        assert self._signature_states is not None
        return self._actions(
            self._signature_states,
            params,
            self.reference.demand.feature_dict(),
            {"h": self.reference.h, "b": self.reference.b},
            remaining_horizon=1,
        )

    def exact_action_signature_key(self, params: Mapping[str, float]) -> bytes | None:
        """Return a proof-level behavior key for small stationary models.

        The optimizer may use this as an exact behavioral cache key.  It is
        deliberately unavailable for horizon-dependent formulas and
        structural partitions above the guarded class limit.  Complete maps
        are materialized only after a fingerprint collision, so standard L=2
        grids participate without paying a full-state pass for every unique
        bucket.

        A 64-state fingerprint selects a small comparison bucket.  A repeated
        fingerprint is never trusted by itself: the complete rounded action
        maps are compared before an existing key is returned.  New fingerprint
        buckets therefore avoid a full-state pass, while every cache hit still
        has an exact all-state proof.
        """

        if self._horizon_dependent or self._exact_behavior_states is None:
            return None
        self._proof_behavior_cache_active = True
        assert self._behavior_fingerprint_states is not None
        fingerprint = self._actions(
            self._behavior_fingerprint_states,
            params,
            self._dist_features,
            self._cost_features,
            remaining_horizon=1,
        )
        fingerprint_key = self._compact_action_bytes(fingerprint)
        bucket = self._exact_behavior_buckets.get(fingerprint_key)
        if not bucket:
            token = self._new_behavior_token()
            self._exact_behavior_buckets[fingerprint_key] = [(token, dict(params), None)]
            return token

        candidate_actions = self._complete_action_map(params)
        for index, (token, representative, representative_actions) in enumerate(bucket):
            if representative_actions is None:
                representative_actions = self._complete_action_map(representative)
                bucket[index] = (token, representative, representative_actions)
            if np.array_equal(candidate_actions, representative_actions):
                return token
        token = self._new_behavior_token()
        bucket.append((token, dict(params), candidate_actions))
        return token

    def _complete_action_map(self, params: Mapping[str, float]) -> np.ndarray:
        return self._actions(
            self._exact_behavior_states,
            params,
            self._dist_features,
            self._cost_features,
            remaining_horizon=1,
        ).astype(np.int32, copy=False)

    def _new_behavior_token(self) -> bytes:
        self._next_behavior_id += 1
        return self._next_behavior_id.to_bytes(8, byteorder="little", signed=False)

    def _compact_action_bytes(self, actions: np.ndarray) -> bytes:
        if self.action_cap <= np.iinfo(np.uint8).max:
            return actions.astype(np.uint8, copy=False).tobytes()
        if self.action_cap <= np.iinfo(np.uint16).max:
            return actions.astype(np.uint16, copy=False).tobytes()
        return actions.astype(np.uint32, copy=False).tobytes()

    def _dp_remaining_lower_bound(
        self,
        elapsed_periods: int,
        indexes: np.ndarray,
        probabilities: np.ndarray,
        base: int,
        dimensions: int,
    ) -> float:
        dp = self.reference
        if not self.use_stationary_continuation_bound:
            return 0.0
        remaining_periods = max(0, self.horizon - int(elapsed_periods))
        if remaining_periods == 0:
            return 0.0
        probability_mass = float(np.sum(probabilities))
        continuation_values = getattr(dp, "continuation_values", None)
        continuation_horizon = int(getattr(dp, "continuation_horizon", 0) or 0)
        if continuation_values is not None and continuation_horizon > 0:
            used_periods = min(continuation_horizon, remaining_periods)
            # A stored V_K cannot bound a shorter-than-K tail.  In that final
            # window retain only the universally valid one-period floor.
            if used_periods == continuation_horizon:
                tail_count = base ** max(0, dimensions - 2)
                prefix = indexes // tail_count
                available = prefix // base + prefix % base
                tail = indexes % tail_count
                compressed_indexes = available * tail_count + tail
                layer = np.asarray(continuation_values).reshape(-1)
                dp_bound = float(np.dot(probabilities, layer[compressed_indexes]))
                relaxed_tail = (
                    remaining_periods - continuation_horizon
                ) * self._minimum_period_cost * probability_mass
                return float(dp_bound + relaxed_tail)
            return float(
                remaining_periods * self._minimum_period_cost * probability_mass
            )
        if int(getattr(dp, "horizon", 0) or 0) <= 0 or not hasattr(dp, "value_table"):
            # Rolling references deliberately discard all Bellman layers.
            # The global minimum expected one-period cost remains a strict,
            # zero-memory lower bound for every reachable state and policy.
            return float(
                remaining_periods * self._minimum_period_cost * probability_mass
            )
        # The finite-horizon DP is time-homogeneous.  Its V_K table is a valid
        # lower bound for the next K periods from *any* evaluation time, not
        # only when its original time index happens to match elapsed_periods.
        # Use the longest stored continuation that fits, then relax all later
        # periods independently to the global one-period cost floor.
        continuation_periods = min(int(dp.horizon), remaining_periods)
        value_index = int(dp.horizon) - continuation_periods
        remaining_values = dp.value_table[value_index].reshape(-1)[indexes]
        dp_bound = float(np.dot(probabilities, remaining_values))
        relaxed_tail = (
            (remaining_periods - continuation_periods)
            * self._minimum_period_cost
            * probability_mass
        )
        return float(dp_bound + relaxed_tail)

    @staticmethod
    def _decode(indexes: np.ndarray, base: int, dimensions: int) -> np.ndarray:
        states = np.empty((len(indexes), dimensions), dtype=int)
        remaining = np.asarray(indexes, dtype=np.int64).copy()
        for column in range(dimensions - 1, -1, -1):
            states[:, column] = remaining % base
            remaining //= base
        return states

    @staticmethod
    def _encode_state(state: tuple[int, ...], base: int) -> int:
        result = 0
        for value in state:
            result = result * base + int(value)
        return result


def _build_signature_states(dp: DPResult, size: int) -> np.ndarray:
    dimensions = dp.lead_time + 1
    base = dp.n + 1
    total = base**dimensions
    target = min(max(1, int(size)), total)
    if total <= 2 * target:
        indexes = np.arange(total, dtype=np.int64)
        return SparseExactPolicyCostEvaluator._decode(indexes, base, dimensions)[:target]
    rng = np.random.default_rng(20260802 + 31 * dp.lead_time + dp.n)
    rows: list[tuple[int, ...]] = [tuple(0 for _ in range(dimensions))]
    for dimension in range(dimensions):
        low = [0] * dimensions
        high = [0] * dimensions
        high[dimension] = dp.n
        rows.extend((tuple(low), tuple(high)))
    if dp.lead_time >= 2:
        for total_pipeline in np.linspace(0, dp.n, min(32, dp.n + 1), dtype=int):
            near = [0] * dimensions
            far = [0] * dimensions
            near[1] = int(total_pipeline)
            far[-1] = int(total_pipeline)
            rows.extend((tuple(near), tuple(far)))
    unique = dict.fromkeys(rows)
    while len(unique) < target:
        row = tuple(int(value) for value in rng.integers(0, base, size=dimensions))
        unique[row] = None
    return np.asarray(list(unique)[:target], dtype=int)


def _debug_seen_count(values: np.ndarray | set[int] | None) -> int | None:
    if values is None:
        return None
    if isinstance(values, set):
        return len(values)
    return int(np.count_nonzero(values))


def _stationary_action_partition(
    tokens: frozenset[str],
    dp: DPResult,
    decoded_states: np.ndarray,
    *,
    max_classes: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Partition states by the exact state features read by a stationary AST.

    The partition is structural and formula-agnostic.  It recognizes the
    validated inventory-position motif as one scalar feature and otherwise
    uses the explicit state coordinates reached through arbitrary motifs.
    """

    dimensions = dp.lead_time + 1
    if not tokens:
        return (
            np.zeros((1, dimensions), dtype=np.int32),
            np.zeros(len(decoded_states), dtype=np.uint8),
        )
    allowed = {"inventory_position"} | {
        f"state:{name}" for name in ("x", *(f"p{i}" for i in range(1, dp.lead_time + 1)))
    }
    if not tokens <= allowed:
        return None, None

    base = dp.n + 1
    raw_names = sorted(
        (token.split(":", 1)[1] for token in tokens if token.startswith("state:")),
        key=lambda name: 0 if name == "x" else int(name[1:]),
    )
    columns = [0 if name == "x" else int(name[1:]) for name in raw_names]
    upper_bound = 1
    code = np.zeros(len(decoded_states), dtype=np.int64)
    if "inventory_position" in tokens:
        ip_count = (dp.lead_time + 1) * dp.n + 1
        upper_bound *= ip_count
        code = np.sum(decoded_states, axis=1, dtype=np.int64)
    for column in columns:
        upper_bound *= base
        if upper_bound > max_classes:
            return None, None
        code = code * base + decoded_states[:, column].astype(np.int64)
    if upper_bound > max_classes:
        return None, None
    _unique, indexes, inverse = np.unique(
        code, return_index=True, return_inverse=True
    )
    representatives = decoded_states[np.asarray(indexes, dtype=np.int64)]
    class_count = len(representatives)
    if class_count <= np.iinfo(np.uint8).max:
        inverse = inverse.astype(np.uint8)
    elif class_count <= np.iinfo(np.uint16).max:
        inverse = inverse.astype(np.uint16)
    else:
        inverse = inverse.astype(np.uint32)
    return representatives, inverse


def _state_partition_tokens(
    node: AST,
    motifs: Mapping[str, AST],
    lead_time: int,
    visiting: frozenset[str],
) -> frozenset[str]:
    op = str(node.get("op", ""))
    if op == "state":
        return frozenset({f"state:{node['name']}"})
    if op == "motif":
        name = str(node["name"])
        if name == "inventory_position" and _is_inventory_position_sum(
            motifs.get(name), lead_time
        ):
            return frozenset({"inventory_position"})
        if name in visiting or name not in motifs:
            return frozenset({f"motif:{name}"})
        return _state_partition_tokens(
            motifs[name], motifs, lead_time, visiting | frozenset({name})
        )
    result = frozenset()
    for value in node.values():
        if isinstance(value, dict) and "op" in value:
            result |= _state_partition_tokens(value, motifs, lead_time, visiting)
        elif isinstance(value, list):
            for child in value:
                if isinstance(child, dict) and "op" in child:
                    result |= _state_partition_tokens(child, motifs, lead_time, visiting)
    return result


def _is_inventory_position_sum(node: AST | None, lead_time: int) -> bool:
    if node is None:
        return False

    def flatten(current: AST) -> list[str] | None:
        op = str(current.get("op", ""))
        if op == "state":
            return [str(current["name"])]
        if op != "add":
            return None
        names: list[str] = []
        for child in current.get("args", ()):
            flattened = flatten(child)
            if flattened is None:
                return None
            names.extend(flattened)
        return names

    names = flatten(node)
    expected = ["x", *(f"p{index}" for index in range(1, lead_time + 1))]
    return names is not None and sorted(names) == sorted(expected)


def _action_signature(
    ast: AST,
    states: np.ndarray,
    params: Mapping[str, float],
    dp: DPResult,
    motifs: Mapping[str, AST],
    action_cap: int,
) -> np.ndarray:
    batch = _BatchSamples.from_state_array(states, remaining_horizon=1.0)
    raw = _evaluate_ast_batch(
        ast,
        batch,
        params,
        dp.demand.feature_dict(),
        {"h": dp.h, "b": dp.b},
        motifs,
        context=_BatchEvaluationContext(batch, motifs),
    )
    if not np.all(np.isfinite(raw)):
        raise ValueError("Policy formula produced NaN or Inf")
    return np.clip(_round_actions(raw, "round", action_cap).astype(int), 0, dp.n)
