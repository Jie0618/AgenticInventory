"""Current constant-optimization pipeline.

This module implements the single-worker pipeline described in
``docs/methods/OPTIMIZATION_RULES.md``. The search
objective is a guarded stationary evaluator, while every value that can become
the returned incumbent is produced by the strict unpruned evaluator.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import replace
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from optimization.models.types import (
    EvaluationResult,
    MonteCarloProposalConfig,
    OptimizationConfig,
    ParameterSpec,
    PolicyOptimizationResult,
)
from optimization.solvers.parameters import (
    canonical_parameter_key,
    canonicalize_params,
    decode_params,
    discrete_space_size,
    encode_params,
    iter_discrete_params,
)
from optimization.solvers.rules import (
    ModeRules,
    PIPELINE_RULES,
    certify_dimension_multiplier,
    resolve_certify_rules,
    resolve_pil_scalar_rules,
)


# Initial anchors are strict evaluations, but a pathological anchor must not
# consume an unbounded fraction of a worker's wall time.  These are soft
# per-anchor deadlines: the evaluator checks them between propagation periods.
ANCHOR_TIMEOUT_SECONDS = {
    "fast": 2.0,
    "standard": 5.0,
    "certify": 10.0,
}


def _strict_exact_fidelity(result: EvaluationResult) -> bool:
    """Return whether a completed result is safe to publish as an incumbent.

    Stationary extrapolations are useful for proposal ranking only.  The
    exact evaluator may report ``exact`` directly or ``exact_fallback`` when
    its guarded stationary probe did not trigger and it completed the full
    horizon.
    """

    fidelity = str(result.metadata.get("fidelity", "exact"))
    return fidelity in {"exact", "exact_fallback"}


def _uses_special_initial_anchors(evaluator: Any, specs: Sequence[ParameterSpec]) -> bool:
    """Ask a model evaluator whether it owns a special anchor strategy."""

    method = getattr(evaluator, "uses_special_initial_anchors", None)
    if callable(method):
        return bool(method(specs))
    # Compatibility for existing evaluator test doubles and serialized VBS
    # specs. New model integrations should expose the evaluator hook above.
    return bool(
        len(specs) >= 2
        and all(
            getattr(spec, "monotone_group", None) == "vector_base_stock_targets"
            and getattr(spec, "monotone_index", None) is not None
            and str(getattr(spec, "name", ""))
            == f"S{int(getattr(spec, 'monotone_index'))}"
            for spec in specs
        )
        and sorted(int(getattr(spec, "monotone_index")) for spec in specs)
        == list(range(len(specs)))
    )


@dataclass(frozen=True)
class StructuralAnalysis:
    parameter_kind: str
    parameter_count: int
    continuous_dimension: int
    discrete_dimension: int
    discrete_combination_count: int | None
    mixed_branch_count: int | None
    stationary_formula: bool
    horizon_dependent_formula: bool
    uses_rounding_or_integer_action: bool
    structural_features: tuple[str, ...]
    structural_class_count: int | None
    analytic_behavior_cells_available: bool
    behavior_cell_count: int | None
    has_default_constants: bool
    warm_start_count: int
    optimization_route_hint: str | None = None


@dataclass
class Candidate:
    candidate_id: int
    params: dict[str, float]
    key: tuple[Any, ...]
    source: str
    proposal_index: int = 0
    fast: EvaluationResult | None = None
    exact: EvaluationResult | None = None
    behavior_key: bytes | None = None
    parameter_cache_hit: bool = False
    behavior_cache_hit: bool = False
    exact_source: str | None = None
    # Normalized parameter coordinates are cached at proposal time.  The
    # shortlist builder must not repeatedly decode/re-encode the same record.
    encoded_params: np.ndarray | None = None

    @property
    def fast_completed(self) -> bool:
        return bool(self.fast is not None and self.fast.completed and np.isfinite(self.fast.cost))

    @property
    def fast_cost(self) -> float:
        if self.fast is None:
            return float("inf")
        return float(self.fast.cost) if np.isfinite(self.fast.cost) else float("inf")

    @property
    def proposal_score(self) -> float:
        """Compatibility alias for the mixed-fidelity proposal score."""

        return self.fast_cost

    @property
    def proposal_fidelity(self) -> str | None:
        return None if self.fast is None else str(self.fast.metadata.get("proposal_fidelity", self.fast.metadata.get("fidelity")))

    @property
    def exact_completed(self) -> bool:
        return bool(
            self.exact is not None
            and self.exact.completed
            and np.isfinite(self.exact.cost)
            and _strict_exact_fidelity(self.exact)
        )


@dataclass
class _Archive:
    specs: tuple[ParameterSpec, ...]
    exact_evaluator: Any
    fast_evaluator: Any
    rules: ModeRules
    started: float
    proposal_count: int = 0
    exact_count: int = 0
    completed_fast_count: int = 0
    early_stopped_count: int = 0
    exact_fallback_count: int = 0
    fast_evaluator_calls: int = 0
    fast_stationary_count: int = 0
    fast_exact_fallback_count: int = 0
    fast_behavior_reuse_count: int = 0
    fast_incomplete_count: int = 0
    mc_fallback_count: int = 0
    mc_rerank_count: int = 0
    mc_fallback_reasons: dict[str, int] = field(default_factory=dict)
    fast_stationary_periods: list[float] = field(default_factory=list)
    budget_recovery_count: int = 0
    parameter_cache_hits: int = 0
    behavior_proof_attempts: int = 0
    behavior_cache_hits: int = 0
    records: dict[tuple[Any, ...], Candidate] = field(default_factory=dict)
    behavior_fast: dict[bytes, EvaluationResult] = field(default_factory=dict)
    behavior_exact: dict[bytes, EvaluationResult] = field(default_factory=dict)
    exact_best_cost: float = float("inf")
    exact_best_params: dict[str, float] | None = None
    exact_best_record: Candidate | None = None
    exact_sources: dict[str, int] = field(default_factory=dict)
    phase_exact_used: dict[str, int] = field(default_factory=dict)
    next_rolling_milestone: int = 1
    rolling_milestones: tuple[int, ...] = ()
    final_exact_used: int = 0
    wall_time_limited: bool = False
    exhaustive_override: bool = False
    timings: dict[str, float] = field(default_factory=dict)
    shortlist_diagnostics: dict[str, Any] = field(default_factory=dict)
    anchor_timeout_count: int = 0
    anchor_early_stop_count: int = 0
    anchor_timeout_seconds: float = 0.0
    mc_anchor_ready: bool = False

    def __post_init__(self) -> None:
        count = self.rules.rolling_exact_budget
        if count:
            self.rolling_milestones = tuple(
                max(1, int(math.ceil(i * self.rules.proposal_budget / (count + 1))))
                for i in range(1, count + 1)
            )

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def proposal_deadline_reached(self, reserve_seconds: float = 0.0) -> bool:
        """Mark and report the proposal-stage wall-clock deadline.

        This is intentionally a controller-level guard.  It does not interrupt
        an in-flight proposal evaluator, and it must never gate the later
        shortlist/final-Exact phase.
        """

        if self.elapsed + max(0.0, float(reserve_seconds)) >= self.rules.soft_wall_time_seconds:
            self.wall_time_limited = True
            return True
        return False

    # Alternate-format name for proposal-controller callers outside this
    # module.  Exact evaluation no longer calls this guard.
    def deadline_reached(self, reserve_seconds: float = 0.0, *, phase_guard: bool = False) -> bool:
        del phase_guard
        return self.proposal_deadline_reached(reserve_seconds)

    def add_timing(self, name: str, seconds: float) -> None:
        self.timings[name] = self.timings.get(name, 0.0) + max(0.0, float(seconds))

    @staticmethod
    def _set_evaluator_context(evaluator: Any, context: Mapping[str, Any]) -> None:
        setter = getattr(evaluator, "set_debug_context", None)
        if callable(setter):
            setter(context)

    @staticmethod
    def _annotate_evaluator(evaluator: Any, updates: Mapping[str, Any]) -> None:
        annotator = getattr(evaluator, "annotate_last", None)
        if callable(annotator):
            annotator(updates)

    @property
    def proposal_remaining(self) -> int:
        return max(0, self.rules.proposal_budget - self.proposal_count)

    @property
    def exact_remaining(self) -> int:
        return max(0, self.rules.exact_budget_total - self.exact_count)

    @property
    def final_exact_limit(self) -> int:
        """Final budget plus any initial-anchor slots that were not needed."""

        if not _uses_special_initial_anchors(self.exact_evaluator, self.specs):
            return self.rules.final_exact_budget
        unused_initial = max(
            0,
            self.rules.initial_exact_budget
            - self.phase_exact_used.get("initial", 0),
        )
        return self.rules.final_exact_budget + unused_initial

    def _canonical(self, params: Mapping[str, float]) -> tuple[dict[str, float], tuple[Any, ...]]:
        canonical = canonicalize_params(params, self.specs)
        return canonical, canonical_parameter_key(canonical, self.specs)

    def _behavior_key(self, evaluator: Any, params: Mapping[str, float]) -> bytes | None:
        if not self.specs:
            return None
        class_count = int(getattr(evaluator, "exact_behavior_class_count", 0) or 0)
        if class_count > 50_000:
            return None
        method = getattr(evaluator, "exact_action_signature_key", None)
        if not callable(method):
            return None
        self.behavior_proof_attempts += 1
        key = method(params)
        return None if key is None else bytes(key)

    def _get_or_create(self, params: Mapping[str, float], source: str) -> Candidate | None:
        canonical, key = self._canonical(params)
        existing = self.records.get(key)
        if existing is not None:
            existing.parameter_cache_hit = True
            self.parameter_cache_hits += 1
            return existing
        candidate = Candidate(len(self.records) + 1, canonical, key, source)
        self.records[key] = candidate
        return candidate

    def propose_fast(self, params: Mapping[str, float], source: str) -> Candidate | None:
        """Consume one proposal slot and evaluate with Stationary-fast."""

        if self.deadline_reached():
            return None
        if self.proposal_count >= self.rules.proposal_budget:
            return None
        self.proposal_count += 1
        if source == "budget_recovery":
            self.budget_recovery_count += 1
        candidate = self._get_or_create(params, source)
        assert candidate is not None
        if candidate.encoded_params is None:
            encode_started = time.perf_counter()
            candidate.encoded_params = np.asarray(encode_params(candidate.params, self.specs), dtype=float)
            self.add_timing("parameter_encoding_seconds", time.perf_counter() - encode_started)
        if candidate.fast is not None:
            candidate.proposal_index = self.proposal_count
            self._maybe_rolling()
            return candidate
        fast_started = time.perf_counter()
        candidate.proposal_index = self.proposal_count
        candidate.behavior_key = self._behavior_key(self.fast_evaluator, candidate.params)
        behavior_reused = False
        fresh_proposal_evaluation = False
        if candidate.behavior_key is not None and candidate.behavior_key in self.behavior_fast:
            candidate.fast = self.behavior_fast[candidate.behavior_key]
            candidate.behavior_cache_hit = True
            self.behavior_cache_hits += 1
            self.fast_behavior_reuse_count += 1
            behavior_reused = True
        else:
            self.fast_evaluator_calls += 1
            fresh_proposal_evaluation = True
            incumbent_before = self.exact_best_cost if np.isfinite(self.exact_best_cost) else None
            self._set_evaluator_context(
                self.fast_evaluator,
                {
                    "candidate_id": candidate.candidate_id,
                    "candidate_source": source,
                    "selection_reason": source,
                    "parent_candidate_id": None,
                    "raw_params": dict(params),
                    "canonical_params": dict(candidate.params),
                    "incumbent_cost_before": incumbent_before,
                },
            )
            try:
                evaluator_many = getattr(self.fast_evaluator, "evaluate_many", None)
                candidate.fast = (
                    evaluator_many((candidate.params,), incumbent=incumbent_before)[0]
                    if callable(evaluator_many)
                    else self.fast_evaluator.evaluate(
                        candidate.params,
                        incumbent=incumbent_before,
                    )
                )
            except (FloatingPointError, OverflowError, ValueError):
                candidate.fast = EvaluationResult(
                    cost=float("inf"),
                    completed=False,
                    early_stopped=False,
                    metadata={"invalid": True, "fidelity": "lower_bound_only"},
                )
            if candidate.behavior_key is not None and candidate.fast.completed:
                self.behavior_fast[candidate.behavior_key] = candidate.fast
        self.add_timing("proposal_fast_seconds", time.perf_counter() - fast_started)
        if candidate.fast is not None and fresh_proposal_evaluation:
            self.add_timing(
                "proposal_probe_seconds",
                float(candidate.fast.metadata.get("proposal_probe_seconds", candidate.fast.metadata.get("probe_seconds", 0.0))),
            )
            self.add_timing(
                "proposal_mc_seconds",
                float(candidate.fast.metadata.get("proposal_mc_seconds", candidate.fast.metadata.get("mc_seconds", 0.0))),
            )
            if candidate.fast.metadata.get("mc_fallback"):
                self.mc_fallback_count += 1
                reason = str(candidate.fast.metadata.get("mc_fallback_reason", "unknown"))
                self.mc_fallback_reasons[reason] = self.mc_fallback_reasons.get(reason, 0) + 1
        if candidate.fast.completed:
            self.completed_fast_count += 1
            if not behavior_reused and candidate.fast.metadata.get("stationary_triggered"):
                self.fast_stationary_count += 1
                self.fast_stationary_periods.append(float(candidate.fast.metadata.get("cycle_detected_at", 0.0)))
            if not behavior_reused and candidate.fast.metadata.get("exact_fallback"):
                self.fast_exact_fallback_count += 1
        else:
            self.fast_incomplete_count += 1
        if candidate.fast.early_stopped:
            self.early_stopped_count += 1
        if candidate.fast.completed and candidate.fast.metadata.get("exact_fallback"):
            candidate.exact = candidate.fast
            candidate.exact_source = "exact_fallback"
            self.exact_fallback_count += 1
            self._update_exact_best(candidate)
            self._annotate_evaluator(
                self.fast_evaluator,
                {
                    "incumbent_cost_after": self.exact_best_cost if np.isfinite(self.exact_best_cost) else None,
                    "became_incumbent": False,
                },
            )
        self._maybe_rolling()
        return candidate

    def propose_fast_batch(
        self,
        params_batch: Sequence[Mapping[str, float]],
        source: str,
    ) -> list[Candidate]:
        """Consume/evaluate proposal candidates in one MC candidate batch.

        Parameter canonicalization, behavior-cache reuse, proposal counters,
        and rolling milestones intentionally follow :meth:`propose_fast`.
        Only the fresh evaluator calls are grouped; duplicate or proven
        behavior classes still avoid any numerical evaluation.
        """

        candidates: list[Candidate] = []
        pending: list[Candidate] = []
        processed: list[Candidate] = []
        for params in params_batch:
            if self.deadline_reached() or self.proposal_count >= self.rules.proposal_budget:
                break
            self.proposal_count += 1
            if source == "budget_recovery":
                self.budget_recovery_count += 1
            candidate = self._get_or_create(params, source)
            assert candidate is not None
            candidate.proposal_index = self.proposal_count
            if candidate.encoded_params is None:
                encode_started = time.perf_counter()
                candidate.encoded_params = np.asarray(encode_params(candidate.params, self.specs), dtype=float)
                self.add_timing("parameter_encoding_seconds", time.perf_counter() - encode_started)
            candidates.append(candidate)
            if candidate.fast is not None:
                continue
            candidate.behavior_key = self._behavior_key(self.fast_evaluator, candidate.params)
            if candidate.behavior_key is not None and candidate.behavior_key in self.behavior_fast:
                candidate.fast = self.behavior_fast[candidate.behavior_key]
                candidate.behavior_cache_hit = True
                self.behavior_cache_hits += 1
                self.fast_behavior_reuse_count += 1
                processed.append(candidate)
                continue
            pending.append(candidate)

        if pending:
            self.fast_evaluator_calls += len(pending)
            incumbent_before = self.exact_best_cost if np.isfinite(self.exact_best_cost) else None
            self._set_evaluator_context(
                self.fast_evaluator,
                {
                    "candidate_source": source,
                    "selection_reason": source,
                    "batch_candidate_ids": [candidate.candidate_id for candidate in pending],
                    "incumbent_cost_before": incumbent_before,
                },
            )
            started = time.perf_counter()
            try:
                evaluator_many = getattr(self.fast_evaluator, "evaluate_many", None)
                if callable(evaluator_many):
                    results = evaluator_many([candidate.params for candidate in pending], incumbent=incumbent_before)
                else:
                    results = [self.fast_evaluator.evaluate(candidate.params, incumbent=incumbent_before) for candidate in pending]
            except (FloatingPointError, OverflowError, ValueError):
                # A single formula/parameter can be outside the vectorized
                # grammar.  Fall back per candidate rather than converting a
                # whole batch into invalid results.
                results = []
                for candidate in pending:
                    try:
                        evaluator_many = getattr(self.fast_evaluator, "evaluate_many", None)
                        results.append(
                            evaluator_many((candidate.params,), incumbent=incumbent_before)[0]
                            if callable(evaluator_many)
                            else self.fast_evaluator.evaluate(
                                candidate.params, incumbent=incumbent_before
                            )
                        )
                    except (FloatingPointError, OverflowError, ValueError):
                        results.append(
                            EvaluationResult(
                                cost=float("inf"),
                                completed=False,
                                early_stopped=False,
                                metadata={"invalid": True, "fidelity": "lower_bound_only"},
                            )
                        )
            elapsed = time.perf_counter() - started
            self.add_timing("proposal_fast_seconds", elapsed)
            for candidate, result in zip(pending, results):
                candidate.fast = result
                processed.append(candidate)
                if candidate.behavior_key is not None and result.completed:
                    self.behavior_fast[candidate.behavior_key] = result
                self.add_timing(
                    "proposal_probe_seconds",
                    float(result.metadata.get("proposal_probe_seconds", result.metadata.get("probe_seconds", 0.0))),
                )
                self.add_timing(
                    "proposal_mc_seconds",
                    float(result.metadata.get("proposal_mc_seconds", result.metadata.get("mc_seconds", 0.0))),
                )
                if result.metadata.get("mc_fallback"):
                    self.mc_fallback_count += 1
                    reason = str(result.metadata.get("mc_fallback_reason", "unknown"))
                    self.mc_fallback_reasons[reason] = self.mc_fallback_reasons.get(reason, 0) + 1

        for candidate in processed:
            if candidate.fast is not None and candidate.fast.completed:
                self.completed_fast_count += 1
                if candidate.fast.metadata.get("stationary_triggered"):
                    self.fast_stationary_count += 1
                    self.fast_stationary_periods.append(float(candidate.fast.metadata.get("cycle_detected_at", 0.0)))
                if candidate.fast.metadata.get("exact_fallback"):
                    self.fast_exact_fallback_count += 1
            else:
                self.fast_incomplete_count += 1
            if candidate.fast is not None and candidate.fast.early_stopped:
                self.early_stopped_count += 1
            if candidate.fast is not None and candidate.fast.completed and candidate.fast.metadata.get("exact_fallback"):
                candidate.exact = candidate.fast
                candidate.exact_source = "exact_fallback"
                self.exact_fallback_count += 1
                self._update_exact_best(candidate)
        for _candidate in candidates:
            self._maybe_rolling()
        return candidates

    def _phase_allowed(self, phase: str) -> bool:
        if phase == "initial":
            if self.exhaustive_override:
                return True
            return self.phase_exact_used.get(phase, 0) < self.rules.initial_exact_budget
        if phase == "rolling":
            return self.phase_exact_used.get(phase, 0) < self.rules.rolling_exact_budget
        return self.final_exact_used < self.final_exact_limit

    def evaluate_exact(
        self,
        params: Mapping[str, float],
        *,
        source: str,
        phase: str = "final",
        allow_early_stop: bool = True,
        timeout_seconds: float | None = None,
    ) -> Candidate | None:
        """Run one strict Exact candidate, respecting phase and optional timeout."""

        candidate = self._get_or_create(params, source)
        assert candidate is not None
        if candidate.exact_completed:
            return candidate
        if not self._phase_allowed(phase) or (
            self.exact_count >= self.rules.exact_budget_total and not self.exhaustive_override
        ):
            return None
        exact_started = time.perf_counter()
        behavior_key = candidate.behavior_key or self._behavior_key(self.exact_evaluator, candidate.params)
        candidate.behavior_key = behavior_key
        if behavior_key is not None and behavior_key in self.behavior_exact:
            candidate.exact = self.behavior_exact[behavior_key]
            candidate.behavior_cache_hit = True
            self.behavior_cache_hits += 1
            self._update_exact_best(candidate)
            self.add_timing(f"exact_{phase}_seconds", time.perf_counter() - exact_started)
            return candidate
        incumbent = self.exact_best_cost if allow_early_stop and np.isfinite(self.exact_best_cost) else None
        incumbent_before = self.exact_best_cost if np.isfinite(self.exact_best_cost) else None
        evaluation_context = {
            "candidate_id": candidate.candidate_id,
            "candidate_source": source,
            "selection_reason": source,
            "parent_candidate_id": None,
            "raw_params": dict(params),
            "canonical_params": dict(candidate.params),
            "incumbent_cost_before": incumbent_before,
        }
        if timeout_seconds is not None:
            timeout_value = max(0.0, float(timeout_seconds))
            evaluation_context.update(
                {
                    "evaluation_kind": "anchor",
                    "evaluation_timeout_seconds": timeout_value,
                    "evaluation_deadline_at": time.perf_counter() + timeout_value,
                }
            )
        self._set_evaluator_context(
            self.exact_evaluator,
            evaluation_context,
        )
        try:
            evaluator_many = getattr(self.exact_evaluator, "evaluate_many", None)
            result = (
                evaluator_many((candidate.params,), incumbent=incumbent)[0]
                if callable(evaluator_many)
                else self.exact_evaluator.evaluate(candidate.params, incumbent=incumbent)
            )
        except (FloatingPointError, OverflowError, ValueError):
            result = EvaluationResult(cost=float("inf"), completed=False, metadata={"invalid": True})
        candidate.exact = result
        self.add_timing(f"exact_{phase}_seconds", time.perf_counter() - exact_started)
        candidate.exact_source = source
        self.exact_count += 1
        self.phase_exact_used[phase] = self.phase_exact_used.get(phase, 0) + 1
        self.exact_sources[source] = self.exact_sources.get(source, 0) + 1
        if phase == "final":
            self.final_exact_used += 1
        if phase == "initial":
            if result.metadata.get("timed_out", False):
                self.anchor_timeout_count += 1
                self.anchor_timeout_seconds += time.perf_counter() - exact_started
            elif result.early_stopped:
                self.anchor_early_stop_count += 1
        if behavior_key is not None and result.completed:
            self.behavior_exact[behavior_key] = result
        old_best = self.exact_best_cost
        if result.completed:
            self._update_exact_best(candidate)
        self._annotate_evaluator(
            self.exact_evaluator,
            {
                "incumbent_cost_after": self.exact_best_cost if np.isfinite(self.exact_best_cost) else None,
                "became_incumbent": bool(result.completed and self.exact_best_cost < old_best),
                "exact_rank": (
                    1 + sum(
                        1
                        for row in self.records.values()
                        if row.exact_completed
                        and row.exact is not None
                        and result.completed
                        and float(row.exact.cost) < float(result.cost)
                    )
                    if result.completed
                    else None
                ),
            },
        )
        return candidate

    def evaluate_exact_batch(
        self,
        params_batch: Sequence[Mapping[str, float]],
        *,
        source: str,
        phase: str = "final",
        allow_early_stop: bool = True,
    ) -> list[Candidate]:
        """Batch strict evaluations when the evaluator exposes evaluate_many.

        This is primarily for empirical-MC strict tiers: Exact
        evaluators remain on the scalar path, while MC finalists avoid one
        Python/NumPy trajectory simulation per candidate.
        """
        rows = [self._get_or_create(params, source) for params in params_batch]
        candidates = [row for row in rows if row is not None and not row.exact_completed]
        if not candidates or not self._phase_allowed(phase):
            return [row for row in rows if row is not None]
        remaining = self.rules.exact_budget_total - self.exact_count
        if not self.exhaustive_override:
            candidates = candidates[: max(0, remaining)]
        if not candidates:
            return [row for row in rows if row is not None]
        method = getattr(self.exact_evaluator, "evaluate_many", None)
        if not callable(method):
            for row in candidates:
                self.evaluate_exact(row.params, source=source, phase=phase, allow_early_stop=allow_early_stop)
            return [row for row in rows if row is not None]
        incumbent = self.exact_best_cost if allow_early_stop and np.isfinite(self.exact_best_cost) else None
        self._set_evaluator_context(self.exact_evaluator, {"candidate_source": source, "selection_reason": source})
        started = time.perf_counter()
        try:
            results = method([row.params for row in candidates], incumbent=incumbent)
        except (FloatingPointError, OverflowError, ValueError):
            results = [EvaluationResult(cost=float("inf"), completed=False, metadata={}) for _ in candidates]
        elapsed = time.perf_counter() - started
        self.add_timing(f"exact_{phase}_seconds", elapsed)
        for row, result in zip(candidates, results):
            row.exact = result
            row.exact_source = source
            self.exact_count += 1
            self.phase_exact_used[phase] = self.phase_exact_used.get(phase, 0) + 1
            self.exact_sources[source] = self.exact_sources.get(source, 0) + 1
            if phase == "final":
                self.final_exact_used += 1
            if result.completed:
                self._update_exact_best(row)
        return [row for row in rows if row is not None]

    def _update_exact_best(self, candidate: Candidate) -> None:
        if candidate.exact_completed and candidate.exact is not None:
            if float(candidate.exact.cost) < self.exact_best_cost:
                self.exact_best_cost = float(candidate.exact.cost)
                self.exact_best_params = dict(candidate.params)
                self.exact_best_record = candidate
                setter = getattr(self.fast_evaluator, "set_exact_anchor", None)
                if self.mc_anchor_ready and callable(setter):
                    setter(self.exact_best_params, self.exact_best_cost)

    def _maybe_rolling(self) -> None:
        while (
            self.next_rolling_milestone <= len(self.rolling_milestones)
            and self.proposal_count >= self.rolling_milestones[self.next_rolling_milestone - 1]
        ):
            completed = [
                row for row in self.records.values()
                if row.fast_completed and not row.exact_completed
            ]
            if completed:
                chosen = min(completed, key=lambda row: row.fast_cost)
                self.evaluate_exact(
                    chosen.params,
                    source="rolling",
                    phase="rolling",
                    allow_early_stop=True,
                )
            self.next_rolling_milestone += 1


def analyze_policy_structure(
    specs: Sequence[ParameterSpec],
    evaluator: Any | None = None,
    *,
    warm_start_count: int = 0,
) -> StructuralAnalysis:
    specs = tuple(specs)
    discrete = tuple(spec for spec in specs if spec.kind in {"integer", "grid"})
    continuous = tuple(spec for spec in specs if spec.kind == "continuous")
    if not specs:
        kind = "none"
    elif discrete and continuous:
        kind = "mixed"
    elif discrete:
        kind = "discrete"
    else:
        kind = "continuous"
    horizon_dependent = bool(getattr(evaluator, "_horizon_dependent", False))
    features = tuple(sorted(str(value) for value in getattr(evaluator, "_state_partition_tokens", ())))
    class_count = getattr(evaluator, "exact_behavior_class_count", None)
    class_count = None if class_count is None else int(class_count)
    route_hint = getattr(evaluator, "optimization_route_hint", None)
    route_hint = None if route_hint is None else str(route_hint)
    branch_count = discrete_space_size(discrete) if discrete else None
    return StructuralAnalysis(
        parameter_kind=kind,
        parameter_count=len(specs),
        continuous_dimension=len(continuous),
        discrete_dimension=len(discrete),
        discrete_combination_count=branch_count if discrete and not continuous else None,
        mixed_branch_count=branch_count if discrete and continuous else None,
        stationary_formula=not horizon_dependent,
        horizon_dependent_formula=horizon_dependent,
        uses_rounding_or_integer_action=True,
        structural_features=features,
        structural_class_count=class_count,
        analytic_behavior_cells_available=False,
        behavior_cell_count=None,
        has_default_constants=all(spec.default is not None for spec in specs),
        warm_start_count=int(warm_start_count),
        optimization_route_hint=route_hint,
    )


def choose_route(analysis: StructuralAnalysis, rules: ModeRules) -> str:
    if analysis.optimization_route_hint == "pil_scalar":
        if analysis.parameter_kind != "continuous" or analysis.parameter_count != 1:
            raise ValueError("pil_scalar route requires exactly one continuous parameter")
        return "R8_PIL_SCALAR"
    if analysis.parameter_kind == "none":
        return "R0_PARAMETER_FREE"
    if (
        analysis.parameter_kind == "discrete"
        and analysis.discrete_combination_count is not None
        and analysis.discrete_combination_count <= rules.exact_enum_limit
    ):
        return "R1_SMALL_DISCRETE_EXACT"
    if analysis.analytic_behavior_cells_available:
        return "R2_BEHAVIOR_CELLS"
    if analysis.parameter_kind == "discrete":
        if (analysis.discrete_combination_count or 0) <= rules.proposal_budget:
            return "R3_ENUMERABLE_DISCRETE"
        return "R4_LARGE_DISCRETE"
    if analysis.parameter_kind == "continuous":
        return "R5_CONTINUOUS_DE"
    branches = analysis.mixed_branch_count or 0
    minimum_points = max(8, 4 * analysis.continuous_dimension + 4)
    if branches <= rules.mixed_outer_branch_limit and branches * minimum_points <= 0.40 * rules.proposal_budget:
        return "R6_MIXED_OUTER"
    return "R7_MIXED_JOINT"


_MC_PATHS_BY_MODE = {"fast": 256, "standard": 512, "certify": 1024}
_MC_GUARD_BY_MODE = {
    "fast": {
        "max_periods": 64,
        "max_active_states": 5_000,
        "max_generated_branches": 200_000,
        "max_seconds": 0.2,
    },
    "standard": {
        "max_periods": 96,
        "max_active_states": 10_000,
        "max_generated_branches": 500_000,
        "max_seconds": 0.5,
    },
    "certify": {
        "max_periods": 128,
        "max_active_states": 20_000,
        "max_generated_branches": 1_000_000,
        "max_seconds": 1.0,
    },
}


def _resolved_mc_config(config: OptimizationConfig, mode: str) -> MonteCarloProposalConfig:
    supplied = config.mc_proposal or MonteCarloProposalConfig()
    defaults = _MC_GUARD_BY_MODE.get(mode, _MC_GUARD_BY_MODE["standard"])
    return replace(
        supplied,
        paths=(
            _MC_PATHS_BY_MODE.get(mode, 512)
            if supplied.paths is None
            else int(supplied.paths)
        ),
        seed=int(supplied.seed) + int(config.random_seed),
        stationary_probe_max_periods=(
            defaults["max_periods"]
            if supplied.stationary_probe_max_periods is None
            else int(supplied.stationary_probe_max_periods)
        ),
        stationary_probe_max_active_states=(
            defaults["max_active_states"]
            if supplied.stationary_probe_max_active_states is None
            else int(supplied.stationary_probe_max_active_states)
        ),
        stationary_probe_max_generated_branches=(
            defaults["max_generated_branches"]
            if supplied.stationary_probe_max_generated_branches is None
            else int(supplied.stationary_probe_max_generated_branches)
        ),
        stationary_probe_max_seconds=(
            defaults["max_seconds"]
            if supplied.stationary_probe_max_seconds is None
            else float(supplied.stationary_probe_max_seconds)
        ),
    )


def _make_fast_evaluator(
    evaluator: Any,
    config: OptimizationConfig,
    mode: str,
    route: str | None = None,
    parameter_count: int | None = None,
) -> Any:
    factory = getattr(evaluator, "make_fast_evaluator", None)
    if callable(factory):
        return factory(config, mode, route, parameter_count)
    stationary_copy = getattr(evaluator, "stationary_copy", None)
    if callable(stationary_copy):
        try:
            return stationary_copy(
                tolerance=1.0e-10,
                required_consecutive_hits=3,
                proposal_complexity_guard=None,
            )
        except TypeError:
            return stationary_copy(
                tolerance=1.0e-10,
                required_consecutive_hits=3,
            )
    return evaluator


def _reported_mc_paths(fast_evaluator: Any, config: OptimizationConfig, mode: str) -> tuple[int, int]:
    """Report the paths actually used by the routed evaluator.

    Composed finite-MDP evaluators expose their complete proposal budget
    directly.  The lead-time adaptive evaluator may create a 512-path lazy
    proposal and a 1024-path reranker, so retain the fallback when
    those evaluators do not expose an instance yet.
    """
    resolved = int(_resolved_mc_config(config, mode).paths or 0)
    direct = getattr(fast_evaluator, "paths", None)
    if direct is not None:
        value = int(direct)
        return value, value
    proposal = getattr(fast_evaluator, "mc_evaluator", None)
    proposal_paths = getattr(proposal, "paths", None)
    if proposal_paths is None:
        proposal_value = min(resolved, 512) if mode == "certify" else resolved
    else:
        proposal_value = int(proposal_paths)
    rerank = getattr(fast_evaluator, "rerank_mc_evaluator", None)
    rerank_paths = getattr(rerank, "paths", None)
    rerank_value = resolved if rerank_paths is None else int(rerank_paths)
    return proposal_value, rerank_value


def _default_params(specs: Sequence[ParameterSpec]) -> dict[str, float]:
    return {spec.name: spec.initial_value() for spec in specs}


def _center_params(specs: Sequence[ParameterSpec]) -> dict[str, float]:
    return {
        spec.name: float(spec.values[len(spec.values) // 2]) if spec.kind == "grid" and spec.values else (spec.lower + spec.upper) / 2.0
        for spec in specs
    }


def _bound_params(specs: Sequence[ParameterSpec], upper: bool) -> dict[str, float]:
    return {spec.name: spec.upper if upper else spec.lower for spec in specs}


_SOBOL_LOCK = threading.Lock()


def _sobol(count: int, dimensions: int, seed: int) -> np.ndarray:
    if count <= 0 or dimensions <= 0:
        return np.empty((0, dimensions), dtype=float)
    try:
        from scipy.stats import qmc
        exponent = int(math.ceil(math.log2(max(1, count))))
        # SciPy lazily initializes shared direction-number tables. Concurrent
        # constructors with different dimensions can race on their first use.
        with _SOBOL_LOCK:
            return qmc.Sobol(dimensions, scramble=True, seed=int(seed)).random_base2(exponent)[:count]
    except Exception:
        return np.random.default_rng(int(seed)).random((count, dimensions))


def _initial_exact_anchors(archive: _Archive, warm_starts: Sequence[Mapping[str, float]]) -> None:
    specs = archive.specs
    candidates: list[tuple[Mapping[str, float], str]] = []
    candidates.extend((params, "warm_start") for params in warm_starts)
    candidates.append((_default_params(specs), "default"))
    if _uses_special_initial_anchors(archive.exact_evaluator, specs):
        # VBS center/upper anchors can induce an almost-full L-dimensional
        # reachable state closure before proposal quality is known.  Keep only
        # the registered default and the safe zero target before proposal
        # search; proposal-ranked candidates own the remaining Exact budget.
        candidates.append((_bound_params(specs, False), "lower_bound"))
        archive.shortlist_diagnostics["initial_anchor_strategy"] = "vbs_safe_default_and_lower"
    else:
        candidates.extend((params, "anchor") for params in (_center_params(specs), _bound_params(specs, False), _bound_params(specs, True)))
        for point, source in zip(_sobol(max(0, archive.rules.initial_exact_budget * 2), len(specs), 1009), range(archive.rules.initial_exact_budget * 2)):
            candidates.append((decode_params(point, specs), "anchor_sobol"))
        archive.shortlist_diagnostics["initial_anchor_strategy"] = "default_center_bounds_and_sobol"
    seen: set[tuple[Any, ...]] = set()
    anchor_timeout = ANCHOR_TIMEOUT_SECONDS.get(archive.rules.name)
    for params, source in candidates:
        canonical = canonicalize_params(params, specs)
        key = canonical_parameter_key(canonical, specs)
        if key in seen:
            continue
        seen.add(key)
        if archive.phase_exact_used.get("initial", 0) >= archive.rules.initial_exact_budget:
            break
        archive.evaluate_exact(
            canonical,
            source=source,
            phase="initial",
            allow_early_stop=np.isfinite(archive.exact_best_cost),
            timeout_seconds=anchor_timeout,
        )


def _mixed_radix_params(index: int, specs: Sequence[ParameterSpec]) -> dict[str, float]:
    discrete = [spec for spec in specs if spec.kind in {"integer", "grid"}]
    values: list[Sequence[float]] = [spec.grid_values(spec.cardinality() or 1) for spec in discrete]
    output: dict[str, float] = {}
    remainder = int(index)
    for spec, options in reversed(list(zip(discrete, values))):
        choice = remainder % len(options)
        remainder //= len(options)
        output[spec.name] = float(options[choice])
    return output


def _discrete_strata(specs: Sequence[ParameterSpec], count: int) -> list[dict[str, float]]:
    total = discrete_space_size(tuple(specs))
    if total <= 0 or count <= 0:
        return []
    indexes = [min(total - 1, int((i + 0.5) * total / count)) for i in range(count)]
    return [_mixed_radix_params(index, specs) for index in indexes]


def _neighbors(params: Mapping[str, float], specs: Sequence[ParameterSpec], scale: int = 1) -> list[dict[str, float]]:
    result: list[dict[str, float]] = []
    for spec in specs:
        if spec.kind not in {"integer", "grid"}:
            continue
        options = list(spec.grid_values(spec.cardinality() or 1))
        current = canonicalize_params(params, specs)[spec.name]
        try:
            position = min(range(len(options)), key=lambda index: abs(float(options[index]) - current))
        except ValueError:
            continue
        for direction in (-1, 1):
            index = position + direction * max(1, int(scale))
            if 0 <= index < len(options):
                row = dict(params)
                row[spec.name] = float(options[index])
                result.append(row)
    return result


def _unit_neighbors(unit: np.ndarray, specs: Sequence[ParameterSpec], step: float) -> list[np.ndarray]:
    output: list[np.ndarray] = []
    for index in range(len(specs)):
        for direction in (-1.0, 1.0):
            row = unit.copy()
            row[index] = float(np.clip(row[index] + direction * step, 0.0, 1.0))
            output.append(row)
    return output


def _propose_discrete(archive: _Archive, specs: Sequence[ParameterSpec], route: str) -> None:
    def propose_rows(rows: Sequence[Mapping[str, float]], source: str) -> None:
        # Keep proposal chunks aligned with the MC evaluator's candidate
        # batch size so discrete proposal batches actually exercise the
        # larger vectorized MC kernel.
        chunk_size = 64
        for start in range(0, len(rows), chunk_size):
            if archive.wall_time_limited or archive.proposal_count >= archive.rules.proposal_budget:
                return
            archive.propose_fast_batch(rows[start : start + chunk_size], source)

    budget = archive.rules.proposal_budget
    total = discrete_space_size(tuple(specs))
    if route == "R3_ENUMERABLE_DISCRETE":
        rows: list[Mapping[str, float]] = []
        for params in iter_discrete_params(specs):
            if archive.proposal_count + len(rows) >= budget:
                break
            rows.append(params)
            if len(rows) >= 32:
                propose_rows(rows, "enumeration")
                rows.clear()
        if rows:
            propose_rows(rows, "enumeration")
        return
    global_count = int(round(0.40 * budget))
    neighborhood_count = int(round(0.50 * budget))
    audit_count = budget - global_count - neighborhood_count
    propose_rows(list(_discrete_strata(specs, global_count)), "global_stratified")
    scales = (max(1, max((spec.cardinality() or 1) for spec in specs) // 4), max(1, max((spec.cardinality() or 1) for spec in specs) // 8), 1)
    for scale in scales:
        if archive.wall_time_limited:
            return
        elites = sorted((row for row in archive.records.values() if row.fast_completed), key=lambda row: row.fast_cost)[: max(1, archive.rules.local_elite_count)]
        if not elites:
            break
        for elite in elites:
            rows: list[Mapping[str, float]] = []
            for params in _neighbors(elite.params, specs, scale):
                if archive.wall_time_limited or archive.proposal_count >= global_count + neighborhood_count:
                    break
                rows.append(params)
            propose_rows(rows, f"neighborhood_scale_{scale}")
    ranked = sorted(archive.records.values(), key=lambda row: row.fast_cost)
    if ranked:
        rows = [ranked[int(index)].params for index in np.linspace(0, len(ranked) - 1, max(1, audit_count), dtype=int)]
        propose_rows(rows, "audit")
    while not archive.wall_time_limited and archive.proposal_count < budget and total > 0:
        rows = []
        while archive.proposal_count + len(rows) < budget and len(rows) < 32:
            rows.append(_mixed_radix_params((archive.proposal_count + len(rows)) % total, specs))
        propose_rows(rows, "audit_boundary")


def _propose_continuous(archive: _Archive, specs: Sequence[ParameterSpec], route: str, seed: int) -> None:
    def propose_rows(rows: Sequence[Mapping[str, float]], source: str) -> list[Candidate]:
        output: list[Candidate] = []
        # Keep proposal chunks aligned with the MC evaluator's candidate
        # batch size so continuous/mixed proposal batches actually exercise
        # the larger vectorized MC kernel.
        chunk_size = 64
        for start in range(0, len(rows), chunk_size):
            if archive.wall_time_limited or archive.proposal_count >= archive.rules.proposal_budget:
                break
            output.extend(archive.propose_fast_batch(rows[start : start + chunk_size], source))
        return output

    budget = archive.rules.proposal_budget
    global_budget = int(round(0.25 * budget))
    de_budget = int(round(0.55 * budget))
    local_budget = budget - global_budget - de_budget
    propose_rows([decode_params(point, specs) for point in _sobol(max(1, global_budget), len(specs), seed)], "sobol_global")
    population_size = min(
        max(16, 8 * len(specs)) if archive.rules.name == "fast" else max(32, 12 * len(specs)) if archive.rules.name == "standard" else max(48, 16 * len(specs)),
        max(4, global_budget),
    )
    rows = sorted((row for row in archive.records.values() if row.fast_completed), key=lambda row: row.fast_cost)
    population = [encode_params(row.params, specs) for row in rows[:population_size]]
    if len(population) < 4:
        population.extend(list(_sobol(4 - len(population), len(specs), seed + 71)))
    rng = np.random.default_rng(int(seed) + 17)
    consumed = 0
    while not archive.wall_time_limited and archive.proposal_count < global_budget + de_budget and consumed < de_budget and population:
        old_population = [row.copy() for row in population]
        old_costs = [archive.records[canonical_parameter_key(decode_params(row, specs), specs)].fast_cost if canonical_parameter_key(decode_params(row, specs), specs) in archive.records else float("inf") for row in old_population]
        trials: list[np.ndarray] = []
        target_indices: list[int] = []
        for target_index, target in enumerate(old_population):
            if archive.wall_time_limited or archive.proposal_count >= global_budget + de_budget or consumed >= de_budget:
                break
            choices = [i for i in range(len(old_population)) if i != target_index]
            if len(choices) < 3:
                break
            a, b, c = rng.choice(choices, size=3, replace=False)
            mutant = np.clip(old_population[a] + 0.7 * (old_population[b] - old_population[c]), 0.0, 1.0)
            mask = rng.random(len(specs)) < 0.9
            mask[int(rng.integers(0, len(specs)))] = True
            trial = np.where(mask, mutant, target)
            trials.append(trial)
            target_indices.append(target_index)
            consumed += 1
        trial_candidates = propose_rows([decode_params(trial, specs) for trial in trials], "de_trial")
        for trial, target_index, candidate in zip(trials, target_indices, trial_candidates):
            trial_cost = candidate.fast_cost if candidate.fast_completed else float("inf")
            if trial_cost < old_costs[target_index]:
                population[target_index] = trial
    elites = sorted((row for row in archive.records.values() if row.fast_completed), key=lambda row: row.fast_cost)[: archive.rules.local_elite_count]
    step = 0.10
    while not archive.wall_time_limited and archive.proposal_count < budget and local_budget > 0 and step >= 0.01:
        improved = False
        for elite in list(elites):
            center = encode_params(elite.params, specs)
            trials = _unit_neighbors(center, specs, step)
            candidates = propose_rows([decode_params(trial, specs) for trial in trials], "fast_local")
            for trial, candidate in zip(trials, candidates):
                if archive.wall_time_limited or archive.proposal_count >= budget or local_budget <= 0:
                    break
                local_budget -= 1
                if candidate is not None and candidate.fast_completed and candidate.fast_cost < elite.fast_cost:
                    improved = True
        if not improved:
            step *= 0.5

    # Local refinement has a data-dependent number of useful neighbors and
    # therefore cannot be allowed to strand the nominal proposal budget.  Any
    # unused slots are deterministically returned to the search as a Sobol
    # audit/restart stream.  This keeps ``proposal_budget`` an actual maximum
    # search budget rather than an accidental phase-allocation total.
    recovery = archive.proposal_remaining
    if not archive.wall_time_limited and recovery > 0:
        recovery_started = time.perf_counter()
        propose_rows([decode_params(point, specs) for point in _sobol(recovery, len(specs), seed + 100_003)], "budget_recovery")
        archive.add_timing("budget_recovery_controller_seconds", time.perf_counter() - recovery_started)


def _propose_pil_scalar(
    archive: _Archive,
    specs: Sequence[ParameterSpec],
    seed: int,
) -> None:
    """Search PIL's single target with progressive global/local resolution."""

    if len(specs) != 1 or specs[0].kind != "continuous":
        raise ValueError("R8_PIL_SCALAR requires one continuous parameter")

    def propose_units(units: Sequence[float], source: str) -> None:
        chunk_size = 64
        rows = [decode_params((float(np.clip(unit, 0.0, 1.0)),), specs) for unit in units]
        for start in range(0, len(rows), chunk_size):
            if archive.wall_time_limited or archive.proposal_count >= archive.rules.proposal_budget:
                break
            archive.propose_fast_batch(rows[start : start + chunk_size], source)

    # A 513-point dyadic grid covers the full interval before the route spends
    # time on refinement. Endpoints/defaults have already been strict anchors,
    # but retaining them here makes the low-fidelity ranking self-contained.
    global_count = min(513, archive.rules.proposal_budget)
    propose_units(np.linspace(0.0, 1.0, global_count), "pil_global_grid")

    # Refine several basins, not just the noisy low-fidelity winner. Each
    # stage re-ranks the archive so an improved basin receives the next scale.
    for step in (1.0 / 512.0, 1.0 / 2_048.0, 1.0 / 8_192.0):
        if archive.wall_time_limited or archive.proposal_count >= archive.rules.proposal_budget:
            break
        elites = sorted(
            (row for row in archive.records.values() if row.fast_completed),
            key=lambda row: row.fast_cost,
        )[:8]
        units: list[float] = []
        for elite in elites:
            center = float(encode_params(elite.params, specs)[0])
            units.extend(center + offset * step for offset in (-4, -3, -2, -1, 1, 2, 3, 4))
        propose_units(units, "pil_multiscale_local")

    # Use the remaining moderate budget as a low-discrepancy audit. Sobol
    # prefixes cover the whole interval even when the proposal wall interrupts
    # a high-L run before all 8k points are consumed.
    remaining = archive.proposal_remaining
    if not archive.wall_time_limited and remaining > 0:
        started = time.perf_counter()
        audit = _sobol(remaining, 1, int(seed) + 80_081).reshape(-1)
        propose_units(audit, "pil_scalar_audit")
        archive.add_timing(
            "pil_scalar_audit_controller_seconds", time.perf_counter() - started
        )


def _build_shortlist(archive: _Archive, count: int) -> list[Candidate]:
    """Build the Exact finalist set without quadratic Python bookkeeping.

    The old implementation re-encoded every candidate for every selected
    point.  For a 16k-record R7 archive and a 96-point shortlist that creates
    millions of Python calls.  We now cache normalized coordinates on each
    proposal, restrict diversity to a bounded reduced pool, and update the
    nearest-selected distance with one NumPy vector operation per selected
    point.  The fast-cost ordering and the best-first portion of the contract
    are unchanged.
    """

    if count <= 0:
        archive.shortlist_diagnostics.update(
            {"shortlist_input_records": 0, "shortlist_selected_count": 0}
        )
        return []

    sort_started = time.perf_counter()
    ranked = sorted(
        (row for row in archive.records.values() if row.fast_completed),
        key=lambda row: row.fast_cost,
    )
    archive.add_timing("shortlist_sort_seconds", time.perf_counter() - sort_started)
    if not ranked:
        archive.shortlist_diagnostics.update(
            {"shortlist_input_records": 0, "shortlist_selected_count": 0}
        )
        return []

    encode_started = time.perf_counter()
    for row in ranked:
        if row.encoded_params is None:
            row.encoded_params = np.asarray(encode_params(row.params, archive.specs), dtype=float)
    archive.add_timing("shortlist_encode_seconds", time.perf_counter() - encode_started)

    target = min(int(count), len(ranked))
    best_target = min(target, max(1, int(count) * 2 // 3))

    # A bounded pool keeps diversity selection predictable while preserving a
    # small audit reservoir from the full archive.  Behavior champions are
    # included only up to a bounded cap; the final Exact stage still owns the
    # strict quality decision.
    reduced_limit = min(len(ranked), max(1024, int(count) * 8))
    pool_indices: list[int] = list(range(reduced_limit))
    pool_index_set = set(pool_indices)
    reservoir_limit = min(256, max(0, target - best_target) * 4)
    if reservoir_limit and len(ranked) > reduced_limit:
        reservoir = np.linspace(reduced_limit, len(ranked) - 1, reservoir_limit, dtype=int).tolist()
        for index in reservoir:
            if index not in pool_index_set:
                pool_indices.append(int(index))
                pool_index_set.add(int(index))
    champion_limit = min(256, max(0, target - best_target) * 4)
    if champion_limit:
        seen_behaviors: set[bytes] = set()
        for index, row in enumerate(ranked):
            behavior = row.behavior_key
            if behavior is None or behavior in seen_behaviors:
                continue
            seen_behaviors.add(behavior)
            if index not in pool_index_set:
                pool_indices.append(index)
                pool_index_set.add(index)
            if len(seen_behaviors) >= champion_limit:
                break

    pool_started = time.perf_counter()
    pool_rows = [ranked[index] for index in pool_indices]
    pool_vectors = np.vstack([row.encoded_params for row in pool_rows]).astype(float, copy=False)
    archive.add_timing("shortlist_pool_materialization_seconds", time.perf_counter() - pool_started)

    selected: list[Candidate] = list(ranked[:best_target])
    selected_keys = {row.key for row in selected}
    selected_best_count = len(selected)

    diversity_started = time.perf_counter()
    if selected:
        selected_vectors = np.vstack([row.encoded_params for row in selected])
        min_distance = np.min(
            np.linalg.norm(pool_vectors[:, None, :] - selected_vectors[None, :, :], axis=2),
            axis=1,
        )
    else:
        min_distance = np.full(len(pool_rows), np.inf, dtype=float)

    # Preserve ranked order for the diverse part, but avoid recomputing all
    # pairwise distances in Python.  Each accepted row updates one vector of
    # nearest-selected distances.
    for pool_pos, row in enumerate(pool_rows):
        if len(selected) >= target:
            break
        if row.key in selected_keys:
            continue
        if min_distance[pool_pos] > 0.05:
            selected.append(row)
            selected_keys.add(row.key)
            min_distance = np.minimum(
                min_distance,
                np.linalg.norm(pool_vectors - row.encoded_params, axis=1),
            )

    selected_diverse_count = len(selected) - selected_best_count
    for row in ranked:
        if len(selected) >= target:
            break
        if row.key not in selected_keys:
            selected.append(row)
            selected_keys.add(row.key)
    archive.add_timing("shortlist_diversity_seconds", time.perf_counter() - diversity_started)

    archive.shortlist_diagnostics.update({
        "shortlist_input_records": len(ranked),
        "shortlist_unique_parameter_count": len({row.key for row in ranked}),
        "shortlist_unique_behavior_count": len({row.behavior_key for row in ranked if row.behavior_key is not None}),
        "shortlist_reduced_pool_count": len(pool_rows),
        "shortlist_target_best": best_target,
        "shortlist_target_diverse": max(0, target - best_target),
        "shortlist_selected_best": selected_best_count,
        "shortlist_selected_diverse": selected_diverse_count,
        "shortlist_selected_count": len(selected),
        "shortlist_distance_threshold": 0.05,
    })
    return selected


def _mc_rerank_pre_shortlist(archive: _Archive, count: int) -> None:
    """Unify the bounded pre-shortlist under the final CRN/MC anchor."""

    rerank = getattr(archive.fast_evaluator, "rerank", None)
    if not callable(rerank) or count <= 0:
        return
    rows = [row for row in archive.records.values() if row.fast_completed]
    if not rows:
        return
    fidelity_counts: dict[str, int] = {}
    for row in rows:
        fidelity = str(row.proposal_fidelity or "unknown")
        fidelity_counts[fidelity] = fidelity_counts.get(fidelity, 0) + 1
    pre_mc_count = int(fidelity_counts.get("mc_fixed", 0))
    pre_mc_fraction = pre_mc_count / max(1, len(rows))
    low_mixture_count_limit = max(1, int(count) // 4)
    archive.shortlist_diagnostics.update(
        {
            "mc_rerank_pre_fidelity_counts": fidelity_counts,
            "mc_rerank_pre_mc_count": pre_mc_count,
            "mc_rerank_pre_mc_fraction": pre_mc_fraction,
            "mc_rerank_low_mixture_count_limit": low_mixture_count_limit,
            "mc_rerank_low_mixture_fraction_limit": 0.10,
        }
    )
    # Standard and Certify always normalize the bounded pool. Even a small MC
    # share makes Stationary and MC scores incomparable; skipping
    # this pass caused a higher-budget search to miss lower-cost basins on
    # low-mixture formulas.
    rows.sort(key=lambda row: row.proposal_score)
    pool_size = min(len(rows), max(int(count) * 4, int(count)))
    pool = rows[:pool_size]
    # Keep a small deterministic tail reservoir so a noisy cached score
    # cannot permanently exclude a distinct behavior from the unified pass.
    if len(rows) > pool_size:
        reservoir_size = min(max(8, int(count)), len(rows) - pool_size)
        indexes = np.linspace(pool_size, len(rows) - 1, reservoir_size, dtype=int)
        seen = {row.key for row in pool}
        pool.extend(row for row in (rows[int(index)] for index in indexes) if row.key not in seen)
    started = time.perf_counter()
    changed = int(rerank(pool))
    archive.mc_rerank_count += changed
    archive.add_timing("mc_rerank_seconds", time.perf_counter() - started)
    archive.shortlist_diagnostics["mc_rerank_pool_count"] = len(pool)
    archive.shortlist_diagnostics["mc_rerank_evaluations"] = changed


def _directional_overlap(archive: _Archive, shortlist: Sequence[Candidate]) -> None:
    """Audit proposal/Exact disagreement within one total overlap reserve."""

    exact_rows = [row for row in archive.records.values() if row.exact_completed and row.fast_completed]
    downward = [max(0.0, row.fast_cost - float(row.exact.cost)) for row in exact_rows if row.exact is not None]
    radius = max(2.0 * max(downward, default=0.0), 1.0e-10 * max(1.0, abs(archive.exact_best_cost)))
    overlap_used = 0
    for _round in range(2):
        if overlap_used >= archive.rules.overlap_budget:
            break
        candidates = [
            row for row in archive.records.values()
            if row.fast_completed and not row.exact_completed
            and row.fast_cost - radius <= archive.exact_best_cost
        ]
        candidates.sort(key=lambda row: row.fast_cost - radius)
        round_before = archive.final_exact_used
        for row in candidates:
            if overlap_used >= archive.rules.overlap_budget:
                break
            before = archive.final_exact_used
            archive.evaluate_exact(row.params, source="overlap", phase="final", allow_early_stop=True)
            overlap_used += archive.final_exact_used - before
        if archive.final_exact_used == round_before:
            break


def _exact_local(archive: _Archive, specs: Sequence[ParameterSpec]) -> None:
    """Spend the Exact-local reserve as a monotone best-first pattern search.

    The former continuous loop spread each coordinate poll over every member
    of a static elite list.  With seven parameters, Certify's 48 evaluations
    could therefore be exhausted before even one improved point became the
    center of a second poll.  Here every accepted move becomes the next center
    immediately.  Since ``archive.evaluate_exact`` is authoritative and the
    archive incumbent only moves on a strict Exact improvement, this changes
    search efficiency but cannot degrade the returned policy cost.
    """

    # This is the final Exact phase.  Reclaim any reserve that shortlist or
    # overlap could not consume because of duplicate parameters, behavior
    # reuse, or early candidate exhaustion.  The phase/global guards remain
    # authoritative, so this never raises the mode's Exact budget; it merely
    # prevents already-reserved evaluations from being stranded.
    budget = max(
        0,
        min(
            archive.exact_remaining,
            archive.final_exact_limit - archive.final_exact_used,
        ),
    )
    starts = sorted((row for row in archive.records.values() if row.exact_completed), key=lambda row: float(row.exact.cost))[: archive.rules.local_elite_count]
    if not starts:
        return
    used = 0
    attempts = 0
    max_attempts = max(64, 8 * max(1, budget))
    if any(spec.kind == "continuous" for spec in specs):
        for seed in starts:
            if used >= budget or attempts >= max_attempts:
                break
            current = seed
            step = 0.05
            while step >= 1.0e-4 and used < budget and attempts < max_attempts:
                sweep_improved = False
                for coordinate, spec in enumerate(specs):
                    if used >= budget or attempts >= max_attempts:
                        break
                    if spec.kind in {"integer", "grid"}:
                        param_trials = _neighbors(current.params, (spec,), 1)
                    else:
                        center = encode_params(current.params, specs)
                        unit_trials: list[np.ndarray] = []
                        for direction in (-1.0, 1.0):
                            trial = center.copy()
                            value = float(center[coordinate])
                            if direction < 0.0 and value <= 2.0 * step:
                                trial[coordinate] = 0.0
                            elif direction > 0.0 and value >= 1.0 - 2.0 * step:
                                trial[coordinate] = 1.0
                            else:
                                trial[coordinate] = float(
                                    np.clip(value + direction * step, 0.0, 1.0)
                                )
                            unit_trials.append(trial)
                        param_trials = [decode_params(trial, specs) for trial in unit_trials]

                    center_cost = float(current.exact.cost)
                    best_move: Candidate | None = None
                    for params in param_trials:
                        if used >= budget or attempts >= max_attempts:
                            break
                        exact_before = archive.exact_count
                        candidate = archive.evaluate_exact(
                            params,
                            source="exact_local",
                            phase="final",
                            allow_early_stop=True,
                        )
                        attempts += 1
                        used += archive.exact_count - exact_before
                        if (
                            candidate is not None
                            and candidate.exact_completed
                            and float(candidate.exact.cost) < center_cost
                            and (
                                best_move is None
                                or float(candidate.exact.cost) < float(best_move.exact.cost)
                            )
                        ):
                            best_move = candidate
                    if best_move is not None:
                        current = best_move
                        sweep_improved = True
                if not sweep_improved:
                    step *= 0.5
    else:
        for start in starts:
            for params in _neighbors(start.params, specs, 1):
                if used >= budget or attempts >= max_attempts:
                    return
                exact_before = archive.exact_count
                candidate = archive.evaluate_exact(params, source="exact_local", phase="final", allow_early_stop=True)
                attempts += 1
                used += archive.exact_count - exact_before


def _decode_exact_candidate(archive: _Archive, unit: np.ndarray, specs: Sequence[ParameterSpec], source: str) -> Candidate | None:
    return archive.evaluate_exact(decode_params(unit, specs), source=source, phase="final", allow_early_stop=True)


def optimize_policy_constants_pipeline(
    specs: Sequence[ParameterSpec],
    evaluator: Any,
    *,
    config: OptimizationConfig = OptimizationConfig(),
    warm_starts: Sequence[Mapping[str, float]] = (),
) -> PolicyOptimizationResult:
    """Run the complete route/search/calibration/final-Exact pipeline."""

    mode = str(config.mode).lower()
    if mode not in PIPELINE_RULES:
        raise ValueError(f"Unknown optimization mode: {config.mode}")
    rules = PIPELINE_RULES[mode]
    explicit_proposal_override = config.total_evaluations is not None
    # ``total_evaluations`` remains a supported explicit experiment override.
    # It changes only the proposal budget; the mode's independent Exact reserve
    # is preserved as required by the pipeline contract.
    if explicit_proposal_override:
        requested = int(config.total_evaluations)
        if requested <= 0:
            raise ValueError("total_evaluations must be positive")
        rules = replace(rules, proposal_budget=requested)
    specs = tuple(specs)
    started = time.perf_counter()
    analysis_started = time.perf_counter()
    analysis = analyze_policy_structure(specs, evaluator, warm_start_count=len(warm_starts))
    rules = resolve_certify_rules(
        rules,
        parameter_count=analysis.parameter_count,
        discrete_combination_count=analysis.discrete_combination_count,
        explicit_proposal_override=explicit_proposal_override,
    )
    route = choose_route(analysis, rules)
    if route == "R8_PIL_SCALAR":
        rules = resolve_pil_scalar_rules(
            rules,
            explicit_proposal_override=explicit_proposal_override,
        )
    two_fidelity_exhaustive = bool(
        route == "R1_SMALL_DISCRETE_EXACT"
        and getattr(evaluator, "two_fidelity_exhaustive_route", False)
    )
    # A complete R1 certificate must consume one proposal slot per legal
    # combination.  If an explicit smoke override is smaller than that finite
    # domain, raise the effective proposal budget to the certificate size so
    # metadata remains internally consistent (and the override never silently
    # turns a global certificate into a partial scan).
    if route == "R1_SMALL_DISCRETE_EXACT":
        exhaustive_size = int(analysis.discrete_combination_count or 0)
        if exhaustive_size > rules.proposal_budget:
            rules = replace(rules, proposal_budget=exhaustive_size)
    analysis_seconds = time.perf_counter() - analysis_started
    setup_started = time.perf_counter()
    fast_evaluator = (
        evaluator
        if route == "R0_PARAMETER_FREE"
        or (route == "R1_SMALL_DISCRETE_EXACT" and not two_fidelity_exhaustive)
        else _make_fast_evaluator(
            evaluator,
            config,
            mode,
            route,
            analysis.parameter_count,
        )
    )
    setup_seconds = time.perf_counter() - setup_started
    archive = _Archive(specs, evaluator, fast_evaluator, rules, started)
    archive.add_timing("structural_analysis_seconds", analysis_seconds)
    archive.add_timing("evaluator_setup_seconds", setup_seconds)
    runs: list[Mapping[str, Any]] = []

    if route == "R0_PARAMETER_FREE":
        archive.evaluate_exact({}, source="anchor", phase="initial", allow_early_stop=False)
        status = "parameter_free_exact"
        certified = True
    elif route == "R1_SMALL_DISCRETE_EXACT" and not two_fidelity_exhaustive:
        total = int(analysis.discrete_combination_count or 0)
        archive.exhaustive_override = True
        archive.phase_exact_used["initial"] = 0
        exhaustive_points = list(iter_discrete_params(specs))
        priority_indexes: list[int] = []
        for params in warm_starts:
            canonical = canonicalize_params(params, specs)
            key = canonical_parameter_key(canonical, specs)
            for index, point in enumerate(exhaustive_points):
                if (
                    index not in priority_indexes
                    and canonical_parameter_key(canonicalize_params(point, specs), specs) == key
                ):
                    priority_indexes.append(index)
                    break
        priority = set(priority_indexes)
        ordered_points = [
            (exhaustive_points[index], "warm_start") for index in priority_indexes
        ] + [
            (point, "exhaustive")
            for index, point in enumerate(exhaustive_points)
            if index not in priority
        ]
        if len(ordered_points) != total:
            raise RuntimeError("R1 ordering did not preserve the complete discrete domain")
        for params, source in ordered_points:
            archive.proposal_count += 1
            # Once one complete incumbent exists, the strict remaining-cost
            # lower bound may safely reject a discrete candidate that cannot
            # improve it.  Every legal point is still visited, so the global
            # discrete certificate is unchanged; only dominated trajectories
            # stop before the full horizon.
            archive.evaluate_exact(
                params,
                source=source,
                phase="initial",
                allow_early_stop=archive.exact_best_params is not None,
            )
        runs.append(
            {
                "algorithm": "complete_discrete_enumeration",
                "evaluations": total,
                "priority_warm_starts": min(len(warm_starts), total),
            }
        )
        status = "certified_discrete_global"
        certified = True
    elif route == "R1_SMALL_DISCRETE_EXACT":
        # MC substitute for exhaustive Exact.  Preserve the R1
        # route and enumerate the complete legal domain, but score every point
        # with the low-cost evaluator first.  The unchanged Certify shortlist
        # and strict reserve then arbitrate finalists with the high-cost MC
        # evaluator.  Ordinary Exact evaluators never enter this branch.
        exhaustive_points = list(iter_discrete_params(specs))
        for start in range(0, len(exhaustive_points), 64):
            archive.propose_fast_batch(
                exhaustive_points[start : start + 64],
                "exhaustive_low_fidelity",
            )
        shortlist = _build_shortlist(archive, rules.final_shortlist_budget)
        best_count = min(
            len(shortlist), max(1, rules.final_shortlist_budget * 2 // 3)
        )
        if getattr(evaluator, "two_fidelity_exhaustive_route", False):
            best_rows = shortlist[:best_count]
            diverse_rows = shortlist[best_count:]
            archive.evaluate_exact_batch(
                [row.params for row in best_rows],
                source="final_best",
                phase="final",
                allow_early_stop=True,
            )
            archive.evaluate_exact_batch(
                [row.params for row in diverse_rows],
                source="final_diverse",
                phase="final",
                allow_early_stop=True,
            )
        else:
            for index, row in enumerate(shortlist):
                source = "final_best" if index < best_count else "final_diverse"
                archive.evaluate_exact(
                    row.params,
                    source=source,
                    phase="final",
                    allow_early_stop=True,
                )
        _directional_overlap(archive, shortlist)
        _exact_local(archive, specs)
        status = "discrete_best_found"
        certified = False
    else:
        initial_started = time.perf_counter()
        _initial_exact_anchors(archive, warm_starts)
        archive.mc_anchor_ready = True
        anchor_setter = getattr(archive.fast_evaluator, "set_exact_anchor", None)
        if callable(anchor_setter) and archive.exact_best_params is not None:
            anchor_setter(archive.exact_best_params, archive.exact_best_cost)
        archive.add_timing("initial_anchor_controller_seconds", time.perf_counter() - initial_started)
        shortlist: list[Candidate] = []
        # The soft wall belongs to proposal search only.  If the initial
        # anchors already consumed the proposal wall, skip proposal search,
        # but still allow any already collected shortlist to be exact-evaluated.
        if not archive.proposal_deadline_reached():
            search_started = time.perf_counter()
            if route in {"R3_ENUMERABLE_DISCRETE", "R4_LARGE_DISCRETE"}:
                _propose_discrete(archive, [spec for spec in specs if spec.kind in {"integer", "grid"}], route)
            elif route == "R8_PIL_SCALAR":
                _propose_pil_scalar(archive, specs, int(config.random_seed))
            elif route == "R6_MIXED_OUTER":
                discrete = [spec for spec in specs if spec.kind in {"integer", "grid"}]
                continuous = [spec for spec in specs if spec.kind == "continuous"]
                branches = list(iter_discrete_params(discrete))
                per_branch = max(8, 4 * len(continuous) + 4)
                for branch in branches:
                    for point in _sobol(per_branch, len(continuous), int(config.random_seed) + len(branch)):
                        params = dict(branch)
                        params.update({spec.name: decode_params(point, continuous)[spec.name] for spec in continuous})
                        archive.propose_fast(params, "mixed_branch_init")
                        if archive.proposal_count >= rules.proposal_budget:
                            break
                    if archive.proposal_count >= rules.proposal_budget:
                        break
                if archive.proposal_count < rules.proposal_budget and not archive.wall_time_limited:
                    _propose_continuous(archive, specs, route, int(config.random_seed))
            else:
                _propose_continuous(archive, specs, route, int(config.random_seed))
            archive.add_timing("proposal_search_controller_seconds", time.perf_counter() - search_started)
            runs.append({"algorithm": "stationary_fast_search", "route": route, "proposals": archive.proposal_count})
        # Proposal cutoff must not suppress final Exact.  Exact evaluation is
        # deliberately allowed to run beyond the proposal wall, and its own
        # phase budgets remain the only automatic limits here.
        rerank_started = time.perf_counter()
        # Standard/certify normalize a bounded finalist pool under one current
        # CRN/MC anchor. Fast remains a strict two-layer screen and spends its
        # small Exact budget directly on the shortlist.
        if rules.name != "fast":
            _mc_rerank_pre_shortlist(archive, rules.final_shortlist_budget)
        else:
            archive.shortlist_diagnostics["mc_rerank_disabled"] = "fast_two_layer_pipeline"
        archive.add_timing("mc_rerank_controller_seconds", time.perf_counter() - rerank_started)
        shortlist_build_started = time.perf_counter()
        shortlist = _build_shortlist(archive, rules.final_shortlist_budget)
        archive.add_timing("shortlist_build_seconds", time.perf_counter() - shortlist_build_started)
        best_count = min(len(shortlist), max(1, rules.final_shortlist_budget * 2 // 3))
        final_started = time.perf_counter()
        if getattr(evaluator, "two_fidelity_exhaustive_route", False):
            archive.evaluate_exact_batch(
                [row.params for row in shortlist[:best_count]],
                source="final_best", phase="final", allow_early_stop=True,
            )
            archive.evaluate_exact_batch(
                [row.params for row in shortlist[best_count:]],
                source="final_diverse", phase="final", allow_early_stop=True,
            )
        else:
            for index, row in enumerate(shortlist):
                source = "final_best" if index < best_count else "final_diverse"
                archive.evaluate_exact(row.params, source=source, phase="final", allow_early_stop=True)
        archive.add_timing("final_shortlist_seconds", time.perf_counter() - final_started)
        overlap_started = time.perf_counter()
        _directional_overlap(archive, shortlist)
        archive.add_timing("directional_overlap_seconds", time.perf_counter() - overlap_started)
        local_started = time.perf_counter()
        _exact_local(archive, specs)
        archive.add_timing("exact_local_refinement_seconds", time.perf_counter() - local_started)
        status = {
            "R2_BEHAVIOR_CELLS": "behavior_best_found",
            "R3_ENUMERABLE_DISCRETE": "discrete_best_found",
            "R4_LARGE_DISCRETE": "discrete_best_found",
            "R5_CONTINUOUS_DE": "continuous_best_found",
            "R6_MIXED_OUTER": "mixed_best_found",
            "R7_MIXED_JOINT": "mixed_best_found",
            "R8_PIL_SCALAR": "pil_scalar_best_found",
        }.get(route, "best_found")
        certified = False
    if archive.exact_best_params is None:
        raise RuntimeError("Pipeline produced no completed strict Exact candidate")
    if archive.wall_time_limited:
        status = "wall_time_limited_best_found"
    if archive.wall_time_limited:
        search_stop_reason = "soft_wall_time"
    elif route in {"R1_SMALL_DISCRETE_EXACT", "R3_ENUMERABLE_DISCRETE"}:
        search_stop_reason = "finite_domain_exhausted"
    elif archive.proposal_count >= rules.proposal_budget:
        search_stop_reason = "proposal_budget_exhausted"
    else:
        search_stop_reason = "route_search_terminated"
    shortlist_report = dict(archive.shortlist_diagnostics)
    mc_results = [
        row.fast
        for row in archive.records.values()
        if row.fast is not None and row.fast.metadata.get("fidelity") == "mc_fixed"
    ]
    mc_fidelity_counts: dict[str, int] = {}
    for row in archive.records.values():
        if row.fast is None:
            continue
        fidelity = str(row.fast.metadata.get("proposal_fidelity", row.fast.metadata.get("fidelity", "unknown")))
        mc_fidelity_counts[fidelity] = mc_fidelity_counts.get(fidelity, 0) + 1
    def _mc_median(name: str) -> float | None:
        values = [float(item.metadata[name]) for item in mc_results if item.metadata.get(name) is not None and np.isfinite(float(item.metadata[name]))]
        return float(np.median(values)) if values else None

    mc_summary = {
        "evaluations": len(mc_results),
        "fidelity_counts": mc_fidelity_counts,
        "raw_mean_median": _mc_median("mc_raw_mean"),
        "raw_se_median": _mc_median("mc_raw_se"),
        "paired_se_median": _mc_median("mc_paired_se"),
        "variance_reduction_ratio_median": _mc_median("mc_variance_reduction_ratio"),
        "fallback_reasons": dict(archive.mc_fallback_reasons),
    }
    route_state = str(getattr(fast_evaluator, "route_state", "disabled"))
    probe_count = int(getattr(fast_evaluator, "probe_count", 0) or 0)
    probe_abort_count = int(getattr(fast_evaluator, "probe_abort_count", 0) or 0)
    mc_created = bool(getattr(fast_evaluator, "mc_evaluator", None) is not None)
    reported_mc_proposal_paths, reported_mc_rerank_paths = _reported_mc_paths(
        fast_evaluator, config, mode
    )
    convergence = {
        "pipeline_version": "current",
        "route": route,
        "certified_global": certified,
        "proposal_budget": rules.proposal_budget,
        "parameter_count": analysis.parameter_count,
        "discrete_dimension": analysis.discrete_dimension,
        "continuous_dimension": analysis.continuous_dimension,
        "dimension_budget_multiplier": (
            certify_dimension_multiplier(analysis.parameter_count)
            if mode == "certify"
            and not explicit_proposal_override
            and not (
                analysis.discrete_combination_count is not None
                and int(analysis.discrete_combination_count) <= int(rules.exact_enum_limit)
            )
            else 0
        ),
        "dimension_budget_scaled": bool(
            mode == "certify"
            and not explicit_proposal_override
            and rules.proposal_budget != PIPELINE_RULES["certify"].proposal_budget
        ),
        "proposal_count": archive.proposal_count,
        "proposal_budget_total": rules.proposal_budget,
        "proposal_budget_used": archive.proposal_count,
        "proposal_budget_unused": max(0, rules.proposal_budget - archive.proposal_count),
        "proposal_budget_used_fraction": archive.proposal_count / max(1, rules.proposal_budget),
        "search_stop_reason": search_stop_reason,
        "search_stop_phase": "proposal" if archive.wall_time_limited else None,
        "proposal_wall_time_limited": archive.wall_time_limited,
        "exact_wall_time_limited": False,
        "exact_timeout_enabled": False,
        "budget_recovery_proposals": archive.budget_recovery_count,
        "exact_budget_total": rules.exact_budget_total,
        "exact_budget_override": bool(archive.exhaustive_override),
        "exhaustive_override": bool(archive.exhaustive_override),
        "exact_evaluations": archive.exact_count,
        "exact_budget_used_fraction": archive.exact_count / max(1, rules.exact_budget_total),
        "exact_evaluations_by_source": dict(archive.exact_sources),
        "initial_exact_evaluations": archive.phase_exact_used.get("initial", 0),
        "anchor_timeout_count": archive.anchor_timeout_count,
        "anchor_early_stop_count": archive.anchor_early_stop_count,
        "anchor_timeout_seconds": float(archive.anchor_timeout_seconds),
        "anchor_timeout_budget_seconds": (
            float(ANCHOR_TIMEOUT_SECONDS[archive.rules.name])
            if archive.rules.name in ANCHOR_TIMEOUT_SECONDS
            else None
        ),
        "rolling_exact_evaluations": archive.phase_exact_used.get("rolling", 0),
        "final_exact_evaluations": archive.phase_exact_used.get("final", 0),
        "completed_evaluations": archive.completed_fast_count,
        "early_stopped_evaluations": archive.early_stopped_count,
        "exact_fallback_evaluations": archive.exact_fallback_count,
        "fast_evaluator_calls": archive.fast_evaluator_calls,
        "fast_stationary_count": archive.fast_stationary_count,
        "fast_exact_fallback_count": archive.fast_exact_fallback_count,
        "fast_behavior_reuse_count": archive.fast_behavior_reuse_count,
        "fast_incomplete_count": archive.fast_incomplete_count,
        "mc_fallback_count": archive.mc_fallback_count,
        "mc_rerank_evaluations": archive.mc_rerank_count,
        "mc_fallback_reasons": dict(archive.mc_fallback_reasons),
        "unique_parameter_points": len(archive.records),
        "cache_hits": archive.parameter_cache_hits,
        "parameter_cache_hits": archive.parameter_cache_hits,
        "behavior_proof_attempts": archive.behavior_proof_attempts,
        "behavior_cache_hits": archive.behavior_cache_hits,
        "unique_behaviors": len(archive.behavior_fast),
        "stationary_evaluations": archive.fast_stationary_count,
        "stationary_trigger_rate": archive.fast_stationary_count / max(1, archive.fast_evaluator_calls),
        "stationary_period_median": float(np.median(archive.fast_stationary_periods)) if archive.fast_stationary_periods else None,
        # ``screening_fidelity=disabled`` denotes no probability pruning.
        # Mixed proposal fidelity is reported separately below.
        "screening_fidelity": "disabled",
        "proposal_fidelity": (
            "stationary_fast"
            if not bool(getattr(getattr(config, "mc_proposal", None), "enabled", True))
            else (
                "mc_fixed"
                if route_state == "mc_first" and probe_count == 0
                else "stationary_probe_plus_mc"
            )
        ),
        "screening_prune_threshold": 0.0,
        "mc_proposal_enabled": bool(getattr(getattr(config, "mc_proposal", None), "enabled", True)),
        "mc_active": mc_created,
        "mc_created": mc_created,
        "proposal_route_state": route_state,
        "proposal_probe_count": probe_count,
        "proposal_probe_abort_count": probe_abort_count,
        "proposal_probe_abort_rate": probe_abort_count / max(1, probe_count),
        "proposal_route_switch_reason": getattr(fast_evaluator, "route_switch_reason", None),
        "proposal_route_window_history": list(
            getattr(fast_evaluator, "route_window_history", ())
        ),
        "proposal_route_phase_states": dict(
            getattr(fast_evaluator, "phase_route_states", {})
        ),
        "proposal_routing_mode": getattr(fast_evaluator, "routing_mode", "disabled"),
        "mc_paths": int(
            (_resolved_mc_config(config, mode).paths or 0)
            if bool(getattr(getattr(config, "mc_proposal", None), "enabled", True))
            else 0
        ),
        "mc_proposal_paths": int(
            reported_mc_proposal_paths
            if bool(getattr(getattr(config, "mc_proposal", None), "enabled", True))
            else 0
        ),
        "mc_rerank_paths": int(
            reported_mc_rerank_paths
            if bool(getattr(getattr(config, "mc_proposal", None), "enabled", True))
            else 0
        ),
        "mc_seed": int(_resolved_mc_config(config, mode).seed),
        "mc_common_random_numbers": bool(_resolved_mc_config(config, mode).use_common_random_numbers),
        "mc_control_variate": bool(_resolved_mc_config(config, mode).use_exact_anchor_control_variate),
        "mc_summary": mc_summary,
        "max_reachable_states": max((int(max(row.fast.metadata.get("max_reachable_states", 0), row.fast.metadata.get("probe_peak_active_states", 0))) for row in archive.records.values() if row.fast is not None), default=0),
        "wall_time_limited": archive.wall_time_limited,
        "shortlist": shortlist_report,
        "analysis": analysis.__dict__,
        "optimizer_objective": "strict_exact_policy_cost",
        "best_fidelity": "exact",
        "phase_seconds": dict(archive.timings),
        "pipeline_elapsed_seconds": time.perf_counter() - started,
    }
    return PolicyOptimizationResult(
        best_params=dict(archive.exact_best_params),
        best_cost=float(archive.exact_best_cost),
        status=status,
        mode=mode,
        route=route,
        total_evaluations=archive.proposal_count + archive.exact_count,
        completed_evaluations=archive.completed_fast_count,
        early_stopped_evaluations=archive.early_stopped_count,
        unique_parameter_points=len(archive.records),
        cache_hits=archive.parameter_cache_hits + archive.behavior_cache_hits,
        best_found_at_evaluation=archive.exact_sources.get("final_best", archive.exact_count),
        optimizer_runs=tuple(runs),
        convergence_report=convergence,
        elapsed_seconds=time.perf_counter() - started,
    )
__all__ = [
    "StructuralAnalysis",
    "analyze_policy_structure",
    "choose_route",
    "optimize_policy_constants_pipeline",
]
