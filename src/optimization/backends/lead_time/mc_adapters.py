"""Adapters that run fixed-bank MC through the canonical optimizer pipeline."""

from __future__ import annotations

from typing import Mapping, Sequence

from optimization.models.types import EvaluationResult

from optimization.backends.lead_time.mc_evaluator import MonteCarloPolicyCostEvaluator


class TwoFidelityMCEvaluator:
    """Use low-path MC for proposals and high-path MC for strict arbitration."""

    # Small discrete domains must be screened on the low-cost bank and then
    # arbitrated on the high-cost bank, rather than exhaustively consuming the
    # strict tier.  The canonical pipeline owns the budgets and route.
    two_fidelity_exhaustive_route = True

    def __init__(
        self,
        high: MonteCarloPolicyCostEvaluator,
        low: MonteCarloPolicyCostEvaluator,
    ) -> None:
        self.high = high
        self.low = low
        self.parameter_specs = None

    @property
    def optimization_route_hint(self) -> str | None:
        high = getattr(self.high, "optimization_route_hint", None)
        low = getattr(self.low, "optimization_route_hint", None)
        if high != low:
            raise ValueError("two-fidelity evaluators disagree on optimization route")
        return None if high is None else str(high)

    def make_fast_evaluator(
        self,
        config: object,
        mode: object,
        route: object = None,
        parameter_count: object = None,
    ) -> MonteCarloPolicyCostEvaluator:
        del config, mode, route, parameter_count
        return self.low

    @staticmethod
    def _strict_mc(result: EvaluationResult) -> EvaluationResult:
        metadata = dict(result.metadata)
        # The optimizer calls its authoritative tier "exact".
        # Mark high-path MC as strict for pipeline arbitration while retaining
        # an explicit audit flag that this is not an Exact/DP certificate.
        metadata.update({"fidelity": "exact", "mc_empirical_strict": True})
        return EvaluationResult(
            cost=result.cost,
            completed=result.completed,
            early_stopped=False,
            visited_states=result.visited_states,
            metadata=metadata,
        )

    def evaluate(
        self,
        params: Mapping[str, float],
        incumbent: float | None = None,
    ) -> EvaluationResult:
        return self._strict_mc(self.high.evaluate(params, incumbent=incumbent))

    def evaluate_many(
        self,
        params_batch: Sequence[Mapping[str, float]],
        incumbent: float | None = None,
    ) -> list[EvaluationResult]:
        return [
            self._strict_mc(result)
            for result in self.high.evaluate_many(params_batch, incumbent=incumbent)
        ]

    def action_signature(self, params: Mapping[str, float]):
        return self.high.action_signature(params)

    def exact_action_signature_key(self, params: Mapping[str, float]) -> bytes | None:
        return self.high.exact_action_signature_key(params)

    @property
    def exact_behavior_class_count(self) -> int | None:
        return self.high.exact_behavior_class_count

    def set_debug_context(self, context: object) -> None:
        self.high.set_debug_context(context)
        self.low.set_debug_context(context)

    def annotate_last(self, updates: Mapping[str, object]) -> None:
        self.high.annotate_last(updates)
        self.low.annotate_last(updates)


class ScaledMCEvaluator:
    """Scale a short-horizon proposal estimate to the full horizon."""

    def __init__(self, evaluator: MonteCarloPolicyCostEvaluator, scale: float) -> None:
        self.evaluator = evaluator
        self.scale = float(scale)

    def _scale_result(self, result: EvaluationResult) -> EvaluationResult:
        metadata = dict(result.metadata)
        metadata.update({"mc_short_horizon": True, "mc_horizon_scale": self.scale})
        return EvaluationResult(
            cost=float(result.cost) * self.scale,
            completed=result.completed,
            early_stopped=result.early_stopped,
            visited_states=result.visited_states,
            metadata=metadata,
        )

    def evaluate(self, params, incumbent=None):
        return self._scale_result(self.evaluator.evaluate(params, incumbent=incumbent))

    def evaluate_many(self, params_batch, incumbent=None):
        return [
            self._scale_result(row)
            for row in self.evaluator.evaluate_many(params_batch, incumbent=incumbent)
        ]

    def __getattr__(self, name):
        return getattr(self.evaluator, name)


__all__ = ["ScaledMCEvaluator", "TwoFidelityMCEvaluator"]
