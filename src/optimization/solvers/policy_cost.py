"""Public policy-cost optimizer entry point."""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence

from optimization.models.types import EvaluationResult, OptimizationConfig, ParameterSpec, PolicyOptimizationResult


class BoundPolicyCostEvaluator(Protocol):
    """Minimal evaluator interface required by the optimizer."""

    def evaluate(
        self,
        params: Mapping[str, float],
        incumbent: float | None = None,
    ) -> EvaluationResult:
        ...


def optimize_policy_constants(
    specs: Sequence[ParameterSpec],
    evaluator: BoundPolicyCostEvaluator,
    *,
    config: OptimizationConfig = OptimizationConfig(),
    warm_starts: Sequence[Mapping[str, float]] = (),
) -> PolicyOptimizationResult:
    """Optimize baseline constants with the current Stationary/Exact pipeline."""
    from optimization.solvers.pipeline import optimize_policy_constants_pipeline

    return optimize_policy_constants_pipeline(
        specs,
        evaluator,
        config=config,
        warm_starts=warm_starts,
    )


__all__ = ["BoundPolicyCostEvaluator", "optimize_policy_constants"]
