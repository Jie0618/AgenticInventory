"""Lead-time Stationary/Monte-Carlo proposal evaluator construction."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from optimization.backends.lead_time.mc_evaluator import (
    DemandTrajectoryBank,
    MonteCarloPolicyCostEvaluator,
)
from optimization.models.types import MonteCarloProposalConfig, OptimizationConfig
from optimization.solvers.proposal_evaluator import AdaptiveProposalEvaluator


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


def resolved_mc_config(
    config: OptimizationConfig, mode: str
) -> MonteCarloProposalConfig:
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


def make_fast_evaluator(
    evaluator: Any,
    config: OptimizationConfig,
    mode: str,
    route: str | None = None,
    parameter_count: int | None = None,
) -> Any:
    """Build the established lead-time proposal stack without changing it."""

    method = getattr(evaluator, "stationary_copy", None)
    if not callable(method):
        return evaluator
    mc_config = resolved_mc_config(config, mode)
    guard = None
    if mc_config.enabled and mc_config.stationary_probe_enabled:
        guard = {
            "enabled": True,
            "project_period_limit": mode == "certify",
            "max_periods": mc_config.stationary_probe_max_periods,
            "max_active_states": mc_config.stationary_probe_max_active_states,
            "max_generated_branches": mc_config.stationary_probe_max_generated_branches,
            "max_seconds": mc_config.stationary_probe_max_seconds,
            "growth_guard_enabled": mc_config.stationary_probe_growth_guard_enabled,
            "growth_ratio": mc_config.stationary_probe_growth_ratio,
            "growth_hits": mc_config.stationary_probe_growth_hits,
            "growth_min_states": mc_config.stationary_probe_growth_min_states,
        }
    try:
        stationary = method(
            tolerance=1.0e-10,
            required_consecutive_hits=3,
            proposal_complexity_guard=guard,
        )
    except TypeError:
        stationary = method(tolerance=1.0e-10, required_consecutive_hits=3)
    if not mc_config.enabled or mc_config.routing_mode == "never":
        return stationary
    reference = getattr(stationary, "reference", getattr(evaluator, "reference", None))
    ast = getattr(stationary, "ast", getattr(evaluator, "ast", None))
    if reference is None or ast is None:
        return stationary
    horizon = int(getattr(stationary, "horizon", reference.horizon))
    final_mc_paths = int(mc_config.paths or 512)
    proposal_mc_paths = min(final_mc_paths, 512) if mode == "certify" else final_mc_paths

    def make_mc(mc_paths: int) -> MonteCarloPolicyCostEvaluator:
        mc_batch_size = max(16, min(128, 16_384 // max(1, mc_paths)))
        bank = DemandTrajectoryBank.create(
            reference.demand,
            horizon=horizon,
            num_paths=mc_paths,
            seed=int(mc_config.seed),
        )
        return MonteCarloPolicyCostEvaluator(
            ast,
            reference,
            bank,
            getattr(stationary, "motifs", None),
            rounding=getattr(stationary, "rounding", "round"),
            action_cap=getattr(stationary, "action_cap", reference.n),
            initial_state=getattr(stationary, "initial_state", None),
            use_exact_anchor_control_variate=mc_config.use_exact_anchor_control_variate,
            use_common_random_numbers=mc_config.use_common_random_numbers,
            collect_pathwise_costs=mc_config.collect_pathwise_costs,
            collect_mc_standard_error=mc_config.collect_mc_standard_error,
            action_evaluator=stationary,
            horizon=horizon,
            batch_size=mc_batch_size,
        )

    def make_proposal_mc() -> MonteCarloPolicyCostEvaluator:
        return make_mc(proposal_mc_paths)

    def make_rerank_mc() -> MonteCarloPolicyCostEvaluator:
        return make_mc(final_mc_paths)

    initial_mc = None if mc_config.lazy_init else make_proposal_mc()
    routing_mode = mc_config.routing_mode
    if (
        mode == "fast"
        and route == "R7_MIXED_JOINT"
        and int(parameter_count or 0) <= 4
        and int(reference.lead_time) <= 2
        and routing_mode == "auto"
    ):
        routing_mode = "probe_fallback"
    return AdaptiveProposalEvaluator(
        stationary,
        initial_mc,
        mc_factory=make_proposal_mc,
        rerank_mc_factory=(make_rerank_mc if proposal_mc_paths < final_mc_paths else None),
        routing_mode=routing_mode,
        warmup_candidates=mc_config.warmup_candidates,
        abort_rate_no_mc=mc_config.abort_rate_no_mc,
        abort_rate_mc_first=mc_config.abort_rate_mc_first,
        cost_switch_to_mc_ratio=mc_config.cost_switch_to_mc_ratio,
        cost_switch_to_probe_ratio=mc_config.cost_switch_to_probe_ratio,
        cost_switch_to_probe_confirmations=mc_config.cost_switch_to_probe_confirmations,
        sentinel_candidates=mc_config.sentinel_candidates,
        adaptive_sparse_sentinels=mode == "certify",
        min_mc_observations=mc_config.min_mc_observations,
    )


__all__ = ["make_fast_evaluator", "resolved_mc_config"]
