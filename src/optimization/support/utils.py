"""Shared helpers for lead-time DP constant fitting."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from evaluation.backends.lead_time.dp import DPResult

from optimization.models.types import OptimizationConfig, ParameterSpec


def _ordered_specs(param_names: Sequence[str], specs: Sequence[ParameterSpec]) -> list[ParameterSpec]:
    """Match user-provided parameter specs to AST parameter order."""

    by_name = {spec.name: spec for spec in specs}
    missing = [name for name in param_names if name not in by_name]
    if missing:
        raise ValueError(f"Missing parameter specs for: {missing}")
    return [by_name[name] for name in param_names]

def _loss_name(config: OptimizationConfig) -> str:
    """Return the configured loss name, supporting the `objective` alias."""

    return (config.loss if config.loss is not None else config.objective).strip().lower()

def _with_default_action_cap(config: OptimizationConfig, n: int) -> OptimizationConfig:
    """Set action cap to DP grid size unless the caller already configured it."""

    if config.action_cap is not None:
        return config
    return OptimizationConfig(
        strategy=config.strategy,
        mode=config.mode,
        total_evaluations=config.total_evaluations,
        maxiter=config.maxiter,
        refine_maxiter=config.refine_maxiter,
        grid_points=config.grid_points,
        hybrid_top_k=config.hybrid_top_k,
        rounding=config.rounding,
        action_cap=n,
        objective=config.objective,
        loss=config.loss,
        mse_kind=config.mse_kind,
        mse_weight=config.mse_weight,
        policy_weight=config.policy_weight,
        initial_state=config.initial_state,
        require_policy_cost=config.require_policy_cost,
        max_grid_evals=config.max_grid_evals,
        grid_sample_size=config.grid_sample_size,
        random_seed=config.random_seed,
        mc_proposal=config.mc_proposal,
    )

def _state_dict(state_tuple: tuple[int, ...], remaining_horizon: int) -> dict[str, object]:
    """Convert a DP table index into formula-evaluation state features."""

    return {"x": state_tuple[0], "p": list(state_tuple[1:]), "remaining_horizon": float(remaining_horizon)}

def _initial_state(dp_result: DPResult | None, configured: tuple[int, ...] | None) -> tuple[int, ...]:
    """Return a valid initial state for policy-cost evaluation."""

    if configured is not None:
        return tuple(int(value) for value in configured)
    if dp_result is None:
        return ()
    return tuple(0 for _ in range(dp_result.lead_time + 1))

def _dp_initial_cost(dp_result: DPResult | None, initial_state: tuple[int, ...]) -> float:
    """Return DP optimal cost scale used to normalize policy cost."""

    if dp_result is None:
        return 1.0
    return max(float(dp_result.value(0, initial_state)), 1.0e-12)
