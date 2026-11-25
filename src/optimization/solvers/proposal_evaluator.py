"""Adaptive Stationary-probe/Monte-Carlo proposal evaluator."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np

from optimization.models.types import EvaluationResult


@dataclass
class _RoutePhaseState:
    """One search phase's reversible cost-routing window."""

    name: str
    mc_first: bool = False
    window_index: int = 0
    candidate_count: int = 0
    probe_count: int = 0
    abort_count: int = 0
    probe_seconds: float = 0.0
    mc_count: int = 0
    mc_seconds: float = 0.0
    probe_return_evidence_windows: int = 0
    sentinel_candidates: int = 0


class AdaptiveProposalEvaluator:
    """Use guarded stationary propagation, falling back to fixed-R MC."""

    def __init__(
        self,
        stationary_evaluator: Any,
        mc_evaluator: Any | None = None,
        *,
        mc_factory: Any | None = None,
        rerank_mc_factory: Any | None = None,
        routing_mode: str = "auto",
        warmup_candidates: int = 64,
        abort_rate_no_mc: float = 0.05,
        abort_rate_mc_first: float = 0.50,
        cost_switch_to_mc_ratio: float = 1.15,
        cost_switch_to_probe_ratio: float = 0.85,
        cost_switch_to_probe_confirmations: int = 3,
        sentinel_candidates: int = 8,
        adaptive_sparse_sentinels: bool = False,
        min_mc_observations: int = 4,
    ) -> None:
        if routing_mode not in {"auto", "always", "never", "probe_fallback"}:
            raise ValueError("routing_mode must be auto, always, never, or probe_fallback")
        self.stationary_evaluator = stationary_evaluator
        self._mc_evaluator = mc_evaluator
        self._mc_factory = mc_factory
        self._rerank_mc_evaluator: Any | None = None
        self._rerank_mc_factory = rerank_mc_factory
        self.routing_mode = routing_mode
        # Keep routing controls on the evaluator so all calls belonging to one
        # search phase share non-overlapping candidate windows.
        self.warmup_candidates = max(1, int(warmup_candidates))
        self.abort_rate_no_mc = float(abort_rate_no_mc)
        self.abort_rate_mc_first = float(abort_rate_mc_first)
        self.cost_switch_to_mc_ratio = float(cost_switch_to_mc_ratio)
        self.cost_switch_to_probe_ratio = float(cost_switch_to_probe_ratio)
        self.cost_switch_to_probe_confirmations = max(
            1, int(cost_switch_to_probe_confirmations)
        )
        self.sentinel_candidates = max(1, min(int(sentinel_candidates), self.warmup_candidates))
        self.adaptive_sparse_sentinels = bool(adaptive_sparse_sentinels)
        self.min_mc_observations = max(1, int(min_mc_observations))
        if not 0.0 < self.cost_switch_to_probe_ratio <= 1.0:
            raise ValueError("cost_switch_to_probe_ratio must lie in (0, 1]")
        if self.cost_switch_to_mc_ratio < 1.0:
            raise ValueError("cost_switch_to_mc_ratio must be at least 1")
        self.probe_count = 0
        self.probe_abort_count = 0
        self.route_window_history: list[dict[str, Any]] = []
        self.route_switch_reason: str | None = None
        self._phase_states: dict[str, _RoutePhaseState] = {}
        self._active_phase = "default"
        self._last_auto_route_mc_first = False
        self._mc_seconds_ewma: float | None = None
        self._mc_observations_total = 0
        self._anchor_params: Mapping[str, float] | None = None
        self._anchor_exact_cost: float | None = None
        self._debug_context: dict[str, Any] = {}
        self.fallback_count = 0
        self.stationary_count = 0
        self.fallback_reasons: dict[str, int] = {}
        self.rerank_count = 0

    @property
    def mc_evaluator(self) -> Any | None:
        """Return an already-created MC evaluator without forcing lazy init."""

        return self._mc_evaluator

    @property
    def rerank_mc_evaluator(self) -> Any | None:
        """Return an already-created full-fidelity rerank evaluator."""

        return self._rerank_mc_evaluator

    @property
    def route_state(self) -> str:
        if self.routing_mode == "never":
            return "no_mc"
        if self.routing_mode == "always":
            return "mc_first"
        if self.routing_mode == "probe_fallback":
            return "probe_then_mc_fallback"
        state = self._phase_state()
        if state.mc_first:
            return "mc_first"
        if state.window_index == 0:
            return "warmup_probe"
        return "probe_then_mc_fallback"

    @property
    def phase_route_states(self) -> Mapping[str, Mapping[str, Any]]:
        return {
            name: {
                "route": "mc_first" if state.mc_first else "probe_then_mc_fallback",
                "complete_windows": int(state.window_index),
                "partial_window_candidates": int(state.candidate_count),
            }
            for name, state in self._phase_states.items()
        }

    def _phase_name(self) -> str:
        source = self._debug_context.get("candidate_source", "default")
        return str(source or "default")

    def _phase_state(self) -> _RoutePhaseState:
        name = self._phase_name()
        self._active_phase = name
        state = self._phase_states.get(name)
        if state is None:
            state = _RoutePhaseState(
                name=name,
                mc_first=self._last_auto_route_mc_first,
                sentinel_candidates=self.sentinel_candidates,
            )
            self._phase_states[name] = state
        return state

    def _ensure_mc(self) -> Any | None:
        if self._mc_evaluator is None and self._mc_factory is not None:
            self._mc_evaluator = self._mc_factory()
            setter = getattr(self._mc_evaluator, "set_debug_context", None)
            if callable(setter):
                setter(self._debug_context)
            if self._anchor_params is not None:
                self._mc_evaluator.set_exact_anchor(self._anchor_params, self._anchor_exact_cost)
        return self._mc_evaluator

    def _ensure_rerank_mc(self) -> Any | None:
        if self._rerank_mc_evaluator is None and self._rerank_mc_factory is not None:
            self._rerank_mc_evaluator = self._rerank_mc_factory()
            setter = getattr(self._rerank_mc_evaluator, "set_debug_context", None)
            if callable(setter):
                setter(self._debug_context)
            if self._anchor_params is not None:
                self._rerank_mc_evaluator.set_exact_anchor(
                    self._anchor_params, self._anchor_exact_cost
                )
        return self._rerank_mc_evaluator

    def _observe_probe(
        self,
        state: _RoutePhaseState,
        guard_abort: bool,
        seconds: float,
    ) -> None:
        self.probe_count += 1
        if guard_abort:
            self.probe_abort_count += 1
        state.probe_count += 1
        if guard_abort:
            state.abort_count += 1
        state.probe_seconds += max(0.0, float(seconds))

    @staticmethod
    def _mc_seconds(result: EvaluationResult) -> float:
        return max(
            0.0,
            float(result.metadata.get("proposal_mc_seconds", result.metadata.get("mc_seconds", 0.0))),
        )

    def _observe_mc(self, state: _RoutePhaseState, result: EvaluationResult) -> None:
        seconds = self._mc_seconds(result)
        state.mc_count += 1
        state.mc_seconds += seconds
        if seconds > 0.0:
            self._mc_observations_total += 1
            self._mc_seconds_ewma = (
                seconds
                if self._mc_seconds_ewma is None
                else 0.20 * seconds + 0.80 * self._mc_seconds_ewma
            )

    def _close_route_window(self, state: _RoutePhaseState) -> None:
        """Choose the next window's route from measured candidate cost."""

        state.window_index += 1
        route_before = "mc_first" if state.mc_first else "probe_then_mc_fallback"
        abort_rate = state.abort_count / max(1, state.probe_count)
        mean_probe = state.probe_seconds / max(1, state.probe_count)
        mean_mc = state.mc_seconds / max(1, state.mc_count)
        reference_mc = (
            mean_mc
            if state.mc_count >= self.min_mc_observations and mean_mc > 0.0
            else self._mc_seconds_ewma
            if self._mc_observations_total >= self.min_mc_observations
            else None
        )
        estimated_probe_first = None
        estimated_mc_first = None
        decision = "hold"
        reason = "insufficient_mc_observations"
        if state.probe_count > 0 and reference_mc is not None and reference_mc > 0.0:
            estimated_probe_first = mean_probe + abort_rate * reference_mc
            estimated_mc_first = reference_mc
            if (
                not state.mc_first
                and estimated_probe_first > self.cost_switch_to_mc_ratio * estimated_mc_first
            ):
                state.mc_first = True
                state.probe_return_evidence_windows = 0
                decision = "switch_to_mc"
                reason = (
                    f"probe_cost={estimated_probe_first:.6f}>"
                    f"{self.cost_switch_to_mc_ratio:.3f}*mc_cost={estimated_mc_first:.6f}"
                )
            elif state.mc_first:
                if estimated_probe_first < self.cost_switch_to_probe_ratio * estimated_mc_first:
                    state.probe_return_evidence_windows += 1
                    if (
                        state.probe_return_evidence_windows
                        >= self.cost_switch_to_probe_confirmations
                    ):
                        state.mc_first = False
                        state.probe_return_evidence_windows = 0
                        decision = "switch_to_probe"
                        reason = (
                            f"probe_cost={estimated_probe_first:.6f}<"
                            f"{self.cost_switch_to_probe_ratio:.3f}*mc_cost={estimated_mc_first:.6f};"
                            f"confirmed={self.cost_switch_to_probe_confirmations}"
                        )
                    else:
                        reason = (
                            "await_probe_return_confirmation="
                            f"{state.probe_return_evidence_windows}/"
                            f"{self.cost_switch_to_probe_confirmations}"
                        )
                else:
                    state.probe_return_evidence_windows = 0
                    reason = "within_cost_hysteresis"
            else:
                state.probe_return_evidence_windows = 0
                reason = "within_cost_hysteresis"
        self._last_auto_route_mc_first = bool(state.mc_first)
        route_after = "mc_first" if state.mc_first else "probe_then_mc_fallback"
        sentinel_before = int(state.sentinel_candidates)
        if (
            self.adaptive_sparse_sentinels
            and state.mc_first
            and estimated_probe_first is not None
            and estimated_mc_first is not None
            and estimated_probe_first >= 2.0 * estimated_mc_first
        ):
            # One deterministic sentinel per complete window is sufficient to
            # keep the route reversible when probe cost is already decisively
            # above MC.  Retaining 8/64 sentinels in that regime made the
            # monitoring tax dominate Certify VBS wall time.
            state.sentinel_candidates = 1
        else:
            state.sentinel_candidates = self.sentinel_candidates
        record = {
            "phase": state.name,
            "window": state.window_index,
            "candidate_count": state.candidate_count,
            "probe_count": state.probe_count,
            "abort_count": state.abort_count,
            "abort_rate": abort_rate,
            "probe_seconds": state.probe_seconds,
            "mean_probe_seconds": mean_probe,
            "mc_count": state.mc_count,
            "mc_seconds": state.mc_seconds,
            "mean_mc_seconds": mean_mc,
            "reference_mc_seconds": reference_mc,
            "estimated_probe_first_seconds": estimated_probe_first,
            "estimated_mc_first_seconds": estimated_mc_first,
            "route_before": route_before,
            "route_after": route_after,
            "decision": decision,
            "decision_reason": reason,
            "probe_return_evidence_windows": state.probe_return_evidence_windows,
            "probe_return_confirmations_required": self.cost_switch_to_probe_confirmations,
            "sentinel_candidates_before": sentinel_before,
            "sentinel_candidates_after": int(state.sentinel_candidates),
            # Route decisions exposed to report consumers.
            "switched_to_mc": decision == "switch_to_mc",
            "switched_to_probe": decision == "switch_to_probe",
        }
        self.route_window_history.append(record)
        label = "warmup" if state.window_index == 1 else f"window_{state.window_index}"
        self.route_switch_reason = f"phase={state.name}:{label}:{decision}:{reason}"
        state.candidate_count = 0
        state.probe_count = 0
        state.abort_count = 0
        state.probe_seconds = 0.0
        state.mc_count = 0
        state.mc_seconds = 0.0

    @property
    def exact_behavior_class_count(self) -> int:
        return int(getattr(self.stationary_evaluator, "exact_behavior_class_count", 0) or 0)

    def set_debug_context(self, context: Mapping[str, Any] | None) -> None:
        self._debug_context = dict(context or {})
        for evaluator in (
            self.stationary_evaluator,
            self._mc_evaluator,
            self._rerank_mc_evaluator,
        ):
            setter = getattr(evaluator, "set_debug_context", None)
            if callable(setter):
                setter(self._debug_context)

    def annotate_last(self, updates: Mapping[str, Any]) -> None:
        for evaluator in (
            self.stationary_evaluator,
            self._mc_evaluator,
            self._rerank_mc_evaluator,
        ):
            annotator = getattr(evaluator, "annotate_last", None)
            if callable(annotator):
                annotator(updates)

    def exact_action_signature_key(self, params: Mapping[str, float]) -> bytes | None:
        method = getattr(self.stationary_evaluator, "exact_action_signature_key", None)
        return None if not callable(method) else method(params)

    def action_signature(self, params: Mapping[str, float]) -> np.ndarray:
        method = getattr(self.stationary_evaluator, "action_signature", None)
        if not callable(method):
            raise AttributeError("stationary evaluator does not expose action_signature")
        return method(params)

    def set_exact_anchor(self, params: Mapping[str, float] | None, exact_cost: float | None) -> None:
        self._anchor_params = None if params is None else dict(params)
        self._anchor_exact_cost = None if exact_cost is None else float(exact_cost)
        if self._mc_evaluator is not None:
            self._mc_evaluator.set_exact_anchor(params, exact_cost)
        if self._rerank_mc_evaluator is not None:
            self._rerank_mc_evaluator.set_exact_anchor(params, exact_cost)

    def evaluate(self, params: Mapping[str, float], incumbent: float | None = None) -> EvaluationResult:
        """Evaluate one proposal through the same window controller as batches."""

        return self.evaluate_many([params], incumbent=incumbent)[0]

    def _evaluate_window_segment(
        self,
        params_list: Sequence[Mapping[str, float]],
        state: _RoutePhaseState,
        incumbent: float | None,
    ) -> list[EvaluationResult]:
        outputs: list[EvaluationResult | None] = [None] * len(params_list)
        mc_indices: list[int] = []
        mc_params: list[Mapping[str, float]] = []
        mc_details: list[tuple[str, float, Mapping[str, Any], str] | None] = []

        if self.routing_mode == "always":
            probe_limit = 0
        elif self.routing_mode in {"never", "probe_fallback"} or not state.mc_first:
            probe_limit = len(params_list)
        else:
            # MC-first remains observable and reversible: a small deterministic
            # sentinel subset is still probed in every candidate window.
            probe_limit = min(
                len(params_list),
                max(0, state.sentinel_candidates - state.probe_count),
            )

        probe_results: list[EvaluationResult] | None = None
        probe_batch_seconds = 0.0
        evaluator_many = getattr(self.stationary_evaluator, "evaluate_many", None)
        if probe_limit and callable(evaluator_many):
            batch_started = time.perf_counter()
            try:
                probe_results = list(
                    evaluator_many(params_list[:probe_limit], incumbent=incumbent)
                )
                if len(probe_results) != probe_limit:
                    probe_results = None
            except (FloatingPointError, OverflowError, ValueError):
                probe_results = None
            probe_batch_seconds = time.perf_counter() - batch_started

        for index, params in enumerate(params_list):
            if index >= probe_limit:
                mc_indices.append(index)
                mc_params.append(params)
                mc_details.append(None)
                continue
            probe_route = "mc_first_sentinel_probe" if state.mc_first else self.route_state
            deterministic_started = time.perf_counter()
            if probe_results is not None:
                stationary = probe_results[index]
                measured_seconds = probe_batch_seconds / max(1, probe_limit)
            else:
                try:
                    stationary = self.stationary_evaluator.evaluate(
                        params, incumbent=incumbent
                    )
                except (FloatingPointError, OverflowError, ValueError):
                    stationary = EvaluationResult(
                        cost=float("inf"),
                        completed=False,
                        metadata={"fidelity": "proposal_probe_error", "bound_valid": False},
                    )
                measured_seconds = time.perf_counter() - deterministic_started
            probe_md = dict(stationary.metadata)
            probe_seconds = float(
                probe_md.get("proposal_probe_seconds", probe_md.get("probe_seconds", measured_seconds))
            )
            guard_abort = bool(probe_md.get("proposal_guard_triggered"))
            safe_reject = bool(stationary.early_stopped and probe_md.get("bound_valid"))
            self._observe_probe(state, guard_abort, probe_seconds)
            if (stationary.completed and not guard_abort) or safe_reject:
                if stationary.completed and not guard_abort:
                    self.stationary_count += 1
                probe_md.setdefault("proposal_fidelity", probe_md.get("fidelity", "stationary_fast"))
                probe_md.setdefault("probe_seconds", probe_seconds)
                probe_md.setdefault("proposal_route", probe_route)
                probe_md.setdefault("mc_lazy_created", self._mc_evaluator is not None)
                outputs[index] = replace(stationary, metadata=probe_md)
                continue

            reason = str(probe_md.get("proposal_guard_abort_reason", "stationary_probe_incomplete"))
            self.fallback_count += 1
            self.fallback_reasons[reason] = self.fallback_reasons.get(reason, 0) + 1
            if self.routing_mode == "never":
                probe_md.update(
                    {
                        "proposal_fidelity": "stationary_incomplete",
                        "mc_fallback": False,
                        "proposal_route": probe_route,
                        "probe_seconds": probe_seconds,
                    }
                )
                outputs[index] = replace(stationary, metadata=probe_md)
                continue
            mc_indices.append(index)
            mc_params.append(params)
            mc_details.append((reason, probe_seconds, probe_md, probe_route))

        if mc_params:
            mc = self._ensure_mc()
            if mc is None:
                for index, details in zip(mc_indices, mc_details):
                    if details is None:
                        outputs[index] = EvaluationResult(
                            cost=float("inf"),
                            completed=False,
                            metadata={"proposal_fidelity": "mc_unavailable", "proposal_route": "mc_first"},
                        )
                        continue
                    _reason, probe_seconds, probe_md, probe_route = details
                    incomplete_md = dict(probe_md)
                    incomplete_md.update(
                        {
                            "proposal_fidelity": "stationary_incomplete",
                            "mc_fallback": False,
                            "proposal_route": probe_route,
                            "probe_seconds": probe_seconds,
                        }
                    )
                    outputs[index] = EvaluationResult(
                        cost=float("inf"),
                        completed=False,
                        metadata=incomplete_md,
                    )
            else:
                mc.set_debug_context(self._debug_context)
                evaluator_many = getattr(mc, "evaluate_many", None)
                mc_results = (
                    evaluator_many(mc_params, incumbent=None)
                    if callable(evaluator_many)
                    else [mc.evaluate(params, incumbent=None) for params in mc_params]
                )
                for index, details, mc_result in zip(mc_indices, mc_details, mc_results):
                    self._observe_mc(state, mc_result)
                    if details is None:
                        combined = dict(mc_result.metadata)
                        combined.update({"proposal_route": "mc_first", "mc_lazy_created": True})
                        outputs[index] = replace(mc_result, metadata=combined)
                        continue
                    reason, probe_seconds, probe_md, probe_route = details
                    combined = dict(mc_result.metadata)
                    combined.update(
                        {
                            "proposal_fidelity": "mc_fixed",
                            "mc_fallback": True,
                            "mc_fallback_reason": reason,
                            "probe_seconds": probe_seconds,
                            "probe_periods": probe_md.get("probe_periods", probe_md.get("cycle_detected_at", 0)),
                            "probe_peak_active_states": probe_md.get("probe_peak_active_states", probe_md.get("max_reachable_states", 0)),
                            "probe_generated_branches": probe_md.get("probe_generated_branches", 0),
                            "proposal_probe_seconds": probe_seconds,
                            "proposal_mc_seconds": self._mc_seconds(mc_result),
                            "proposal_route": probe_route,
                            "mc_lazy_created": True,
                        }
                    )
                    outputs[index] = replace(mc_result, metadata=combined)

        assert all(result is not None for result in outputs)
        return [result for result in outputs if result is not None]

    def evaluate_many(
        self,
        params_batch: Sequence[Mapping[str, float]],
        incumbent: float | None = None,
    ) -> list[EvaluationResult]:
        """Evaluate proposals with reversible phase-local cost routing.

        Auto mode measures complete non-overlapping candidate windows. Probe
        mode evaluates every candidate and batches only its fallbacks; MC mode
        batches direct simulations while retaining a small sentinel probe set.
        The next window is selected from measured expected time with hysteresis.
        """

        params_list = list(params_batch)
        if not params_list:
            return []
        outputs: list[EvaluationResult] = []
        offset = 0
        while offset < len(params_list):
            state = self._phase_state()
            if self.routing_mode == "auto":
                capacity = max(1, self.warmup_candidates - state.candidate_count)
                stop = min(len(params_list), offset + capacity)
            else:
                stop = len(params_list)
            segment = params_list[offset:stop]
            outputs.extend(self._evaluate_window_segment(segment, state, incumbent))
            state.candidate_count += len(segment)
            offset = stop
            if self.routing_mode == "auto" and state.candidate_count >= self.warmup_candidates:
                self._close_route_window(state)
        return outputs

    def rerank(self, candidates: Sequence[Any]) -> int:
        """Recompute a bounded pool under one current anchor and fixed bank."""

        mc = self._ensure_rerank_mc()
        if mc is None:
            mc = self._ensure_mc() if self.routing_mode == "always" else self._mc_evaluator
        if mc is None:
            return 0
        mc.clear_cache()
        changed = 0
        params = [candidate.params for candidate in candidates]
        results = mc.evaluate_many(params, incumbent=None)
        for candidate, result in zip(candidates, results):
            candidate.fast = result
            changed += 1
        self.rerank_count += changed
        return changed

__all__ = ["AdaptiveProposalEvaluator"]
