"""Lead-time constant-fitting API over the shared optimizer."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Sequence

import numpy as np

from evaluation.backends.lead_time.dp import DPResult
from evaluation.backends.lead_time.models import RollingDPReference
from evaluation.dp import ComposedDPResult
from evaluation.routing import select_evaluation_backend, solve_inventory_dp, validate_reference_for_case
from inventory.core import InventoryCase
from inventory.policy.ast import AST, ast_check, ast_to_formula
from inventory.policy.motifs import motifs_for_config

from optimization.batch.analysis import _BatchEvaluationContext
from optimization.batch.eval import _evaluate_ast_batch
from optimization.batch.features import _round_actions
from optimization.batch.samples import _batch_from_samples_cached
from optimization.models.types import FitResult, OptimizationConfig, ParameterSpec, SupervisedSample
from optimization.backends.lead_time.exact_evaluator import DenseExactPolicyCostEvaluator, SparseExactPolicyCostEvaluator
from optimization.evaluators.composed import (
    ComposedExactPolicyCostEvaluator,
    ComposedMonteCarloPolicyCostEvaluator,
    TwoFidelityComposedMCEvaluator,
    cost_features_for_case,
)
from optimization.objectives.objective import _Objective
from optimization.samples.dp import composed_dp_action_samples, dp_action_samples
from optimization.support.utils import _initial_state
from optimization.support.utils import _ordered_specs, _with_default_action_cap
from optimization.solvers.parameters import discrete_space_size, iter_discrete_params
from optimization.solvers.policy_cost import BoundPolicyCostEvaluator, optimize_policy_constants


def _fit_lead_time_constants(
    ast: AST,
    dp_result: DPResult | RollingDPReference,
    parameter_specs: Sequence[ParameterSpec],
    samples: Sequence[SupervisedSample] | None = None,
    metric_samples: Sequence[SupervisedSample] | None = None,
    motifs: Mapping[str, AST] | None = None,
    config: OptimizationConfig = OptimizationConfig(),
    cost_evaluator: BoundPolicyCostEvaluator | None = None,
    warm_starts: Sequence[Mapping[str, float]] = (),
    dp_optimal_cost: float | None = None,
    allowed_state_names: Sequence[str] | None = None,
) -> FitResult:
    """Fit constants through the canonical policy-cost optimization pipeline.

    Action samples report raw/action MSE diagnostics and, for a small fully
    discrete domain, choose one cheap sample-guided warm start.  That point is
    evaluated first, but no candidate is shortlisted or eliminated: R1 still
    visits the complete legal domain with the strict Exact evaluator.
    """

    samples = tuple(dp_action_samples(dp_result)) if samples is None else tuple(samples)
    if not samples:
        raise ValueError("At least one supervised sample is required for diagnostics")
    if config.rounding != "round":
        raise ValueError("The standard optimizer requires canonical round-and-clip actions")
    loss = (config.loss or config.objective).strip().lower()
    if loss not in {"policy_cost", "policy_reward"}:
        raise ValueError("The standard optimizer only accepts exact policy cost")
    diagnostics_samples = tuple(metric_samples) if metric_samples is not None else samples
    config = _with_default_action_cap(config, dp_result.n)
    check = ast_check(
        ast,
        motifs=motifs,
        allowed_state_names=allowed_state_names,
    )
    param_names = check.stats.parameters
    specs = _ordered_specs(param_names, parameter_specs)
    evaluator = cost_evaluator or SparseExactPolicyCostEvaluator(
        check.ast,
        dp_result,
        int(dp_result.horizon),
        motifs or {},
        rounding=config.rounding,
        action_cap=config.action_cap,
        initial_state=config.initial_state,
        max_cycle_period=0,
        probability_prune_threshold=0.0,
    )
    diagnostics = _Objective(
        check.ast,
        diagnostics_samples,
        _batch_from_samples_cached(diagnostics_samples),
        dp_result.demand.feature_dict(),
        {"h": dp_result.h, "b": dp_result.b},
        motifs or {},
        param_names,
        config,
        dp_result,
        dp_optimal_cost=dp_optimal_cost,
    )
    guided_warm_starts = tuple(warm_starts)
    if specs and all(spec.cardinality() is not None for spec in specs):
        # R1 contains at most 512 legal points in Certify.  Scoring that entire
        # small domain against the already-loaded DP samples is negligible next
        # to one horizon-1000 Exact evaluation and usually supplies a much
        # stronger first incumbent than lexicographic lower-bound ordering.
        space_size = discrete_space_size(specs)
        if space_size <= 512:
            sample_best = min(
                iter_discrete_params(specs),
                key=lambda params: diagnostics.action_mse(params),
            )
            guided_warm_starts = (sample_best, *guided_warm_starts)
    optimized = optimize_policy_constants(
        specs,
        evaluator,
        config=config,
        warm_starts=guided_warm_starts,
    )
    params = optimized.best_params
    initial_state = (
        tuple(0 for _ in range(dp_result.lead_time + 1))
        if config.initial_state is None
        else tuple(config.initial_state)
    )
    dp_cost = (
        max(float(dp_result.value(0, initial_state)), 1.0e-12)
        if dp_optimal_cost is None
        else max(float(dp_optimal_cost), 1.0e-12)
    )
    return FitResult(
        params=dict(params),
        objective_value=float(optimized.best_cost / dp_cost),
        action_mse=diagnostics.action_mse(params),
        raw_mse=diagnostics.raw_mse(params),
        policy_cost=float(optimized.best_cost),
        normalized_mse=diagnostics.normalized_mse(params),
        normalized_policy_cost=float(optimized.best_cost / dp_cost),
        optimizer=f"policy_cost/{optimized.route}",
        success=True,
        nfev=optimized.total_evaluations,
        elapsed_seconds=optimized.elapsed_seconds,
        formula=ast_to_formula(check.ast),
        param_names=tuple(param_names),
        batch_fallback_count=diagnostics.batch_fallback_count,
        batch_fallback_reasons=tuple(diagnostics.batch_fallback_reasons[:8]),
        status=optimized.status,
        route=optimized.route,
        optimizer_metadata={
            "completed_evaluations": optimized.completed_evaluations,
            "early_stopped_evaluations": optimized.early_stopped_evaluations,
            "unique_parameter_points": optimized.unique_parameter_points,
            "cache_hits": optimized.cache_hits,
            "best_found_at_evaluation": optimized.best_found_at_evaluation,
            "optimizer_runs": optimized.optimizer_runs,
            "convergence_report": optimized.convergence_report,
        },
    )


def fit_inventory_policy(
    ast: AST,
    case: InventoryCase,
    horizon: int,
    parameter_specs: Sequence[ParameterSpec],
    *,
    samples: Sequence[SupervisedSample] | None = None,
    metric_samples: Sequence[SupervisedSample] | None = None,
    motifs: Mapping[str, AST] | None = None,
    config: OptimizationConfig = OptimizationConfig(),
    warm_starts: Sequence[Mapping[str, float]] = (),
    max_states: int = 2_000_000,
    max_table_cells: int = 50_000_000,
    reference: DPResult | RollingDPReference | ComposedDPResult | None = None,
    cost_evaluator: BoundPolicyCostEvaluator | None = None,
    dp_optimal_cost: float | None = None,
) -> FitResult:
    """Fit one policy with an automatic optimized/composed backend switch.

    Callers supply the mathematical case, never a backend flag.  Exact
    deterministic lead-time lost-sales cases retain the established optimizer
    unchanged; every other component selection uses the shared controller
    with canonical composed Exact and Monte Carlo evaluators.  A routed
    ``reference`` may be supplied by a higher-level context to avoid repeating
    DP; its case signature, horizon, and concrete result type are validated.
    """

    if not isinstance(case, InventoryCase):
        raise TypeError("case must be an InventoryCase")
    decision = select_evaluation_backend(case)
    if reference is None:
        reference = solve_inventory_dp(
            case,
            horizon,
            max_states=max_states,
            max_table_cells=max_table_cells,
        )
    else:
        validate_reference_for_case(case, reference, horizon)
    resolved_motifs = (
        motifs_for_config(case.config) if motifs is None else dict(motifs)
    )
    config = _with_default_action_cap(config, case.n)

    if decision.optimized:
        assert isinstance(reference, (DPResult, RollingDPReference))
        if isinstance(reference, RollingDPReference):
            if samples is None:
                raise ValueError(
                    "RollingDPReference requires explicit Bellman action samples"
                )
            if dp_optimal_cost is None:
                raise ValueError(
                    "RollingDPReference requires an explicit optimal DP cost"
                )
            if not isinstance(cost_evaluator, SparseExactPolicyCostEvaluator):
                raise TypeError(
                    "RollingDPReference requires a bound SparseExactPolicyCostEvaluator"
                )
            if cost_evaluator.reference is not reference:
                raise ValueError(
                    "rolling cost evaluator must be bound to the supplied reference"
                )
            if cost_evaluator.horizon != int(horizon):
                raise ValueError(
                    "rolling cost evaluator horizon does not match the requested horizon"
                )
        # The lead-time evaluator defaults to an all-zero state.
        # Canonical cases may intentionally start with inventory or pipeline
        # stock, so bind the case initial state before entering that optimized
        # route unless the caller supplied an explicit override.
        optimized_config = config
        if config.initial_state is None:
            optimized_config = replace(
                config,
                initial_state=case.initial_flat_state,
            )
        result = _fit_lead_time_constants(
            ast,
            reference,
            parameter_specs,
            samples=samples,
            metric_samples=metric_samples,
            motifs=resolved_motifs,
            config=optimized_config,
            warm_starts=warm_starts,
            allowed_state_names=case.schema.allowed_state_names,
            cost_evaluator=cost_evaluator,
            dp_optimal_cost=dp_optimal_cost,
        )
        return replace(
            result,
            optimizer_metadata={
                **dict(result.optimizer_metadata),
                "evaluation_backend": decision.backend,
                "evaluation_backend_reasons": decision.reasons,
            },
        )

    assert isinstance(reference, ComposedDPResult)
    training_samples = (
        tuple(composed_dp_action_samples(reference))
        if samples is None
        else tuple(samples)
    )
    if not training_samples:
        raise ValueError("At least one supervised sample is required for diagnostics")
    diagnostics_samples = (
        training_samples if metric_samples is None else tuple(metric_samples)
    )
    check = ast_check(
        ast,
        motifs=resolved_motifs,
        allowed_state_names=case.schema.allowed_state_names,
    )
    param_names = check.stats.parameters
    specs = _ordered_specs(param_names, parameter_specs)
    evaluator = ComposedExactPolicyCostEvaluator(
        check.ast,
        case,
        horizon,
        resolved_motifs,
        rounding=config.rounding,
        action_cap=config.action_cap,
        initial_state=config.initial_state,
        compiled_transition=reference.compiled_transition,
    )
    if cost_evaluator is not None:
        raise ValueError("cost_evaluator is only supported by the optimized backend")
    optimized = optimize_policy_constants(
        specs,
        evaluator,
        config=config,
        warm_starts=warm_starts,
    )
    params = optimized.best_params
    batch = _batch_from_samples_cached(diagnostics_samples)
    context = _BatchEvaluationContext(batch, resolved_motifs)
    raw = _evaluate_ast_batch(
        check.ast,
        batch,
        params,
        case.demand.feature_dict(),
        cost_features_for_case(case),
        resolved_motifs,
        context=context,
    )
    finite_raw = np.maximum(
        np.nan_to_num(raw.astype(float), nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
    )
    actions = _round_actions(finite_raw, config.rounding, config.action_cap)
    weight_total = max(float(np.sum(batch.weights)), 1.0e-12)
    raw_mse = float(np.dot(batch.weights, (finite_raw - batch.targets) ** 2) / weight_total)
    action_mse = float(np.dot(batch.weights, (actions - batch.targets) ** 2) / weight_total)
    target_second_moment = float(
        np.dot(batch.weights, batch.targets**2) / weight_total + 1.0e-12
    )
    selected_mse = action_mse if config.mse_kind in {"action", "action_mse"} else raw_mse
    initial = case.initial_state if config.initial_state is None else config.initial_state
    dp_cost = max(float(reference.value(0, initial)), 1.0e-12)
    return FitResult(
        params=dict(params),
        objective_value=float(optimized.best_cost / dp_cost),
        action_mse=action_mse,
        raw_mse=raw_mse,
        policy_cost=float(optimized.best_cost),
        normalized_mse=float(selected_mse / target_second_moment),
        normalized_policy_cost=float(optimized.best_cost / dp_cost),
        optimizer=f"policy_cost/{optimized.route}",
        success=True,
        nfev=optimized.total_evaluations,
        elapsed_seconds=optimized.elapsed_seconds,
        formula=ast_to_formula(check.ast),
        param_names=tuple(param_names),
        status=optimized.status,
        route=optimized.route,
        optimizer_metadata={
            "evaluation_backend": decision.backend,
            "evaluation_backend_reasons": decision.reasons,
            "exact_evaluator": "composed_sparse_exact",
            "proposal_evaluator": "composed_vectorized_action_mc",
            "completed_evaluations": optimized.completed_evaluations,
            "early_stopped_evaluations": optimized.early_stopped_evaluations,
            "unique_parameter_points": optimized.unique_parameter_points,
            "cache_hits": optimized.cache_hits,
            "best_found_at_evaluation": optimized.best_found_at_evaluation,
            "optimizer_runs": optimized.optimizer_runs,
            "convergence_report": optimized.convergence_report,
        },
    )


def fit_inventory_constants(
    ast: AST,
    case: InventoryCase,
    horizon: int,
    parameter_specs: Sequence[ParameterSpec],
    *,
    samples: Sequence[SupervisedSample] | None = None,
    metric_samples: Sequence[SupervisedSample] | None = None,
    motifs: Mapping[str, AST] | None = None,
    config: OptimizationConfig = OptimizationConfig(),
    warm_starts: Sequence[Mapping[str, float]] = (),
    max_states: int = 2_000_000,
    max_table_cells: int = 50_000_000,
    reference: DPResult | RollingDPReference | ComposedDPResult | None = None,
    cost_evaluator: BoundPolicyCostEvaluator | None = None,
    dp_optimal_cost: float | None = None,
) -> FitResult:
    """Fit constants through the model-neutral inventory API.

    This is the exact/reference-backed public constant-fitting entry point.
    Backend selection, initial-state binding, state-schema validation, and
    reference validation all happen from the canonical ``InventoryCase``. Use
    ``fit_inventory_policy_mc`` for the separate empirical-MC-only route.
    """

    return fit_inventory_policy(
        ast,
        case,
        horizon,
        parameter_specs,
        samples=samples,
        metric_samples=metric_samples,
        motifs=motifs,
        config=config,
        warm_starts=warm_starts,
        max_states=max_states,
        max_table_cells=max_table_cells,
        reference=reference,
        cost_evaluator=cost_evaluator,
        dp_optimal_cost=dp_optimal_cost,
    )


def fit_inventory_policy_mc(
    ast: AST,
    case: InventoryCase,
    horizon: int,
    parameter_specs: Sequence[ParameterSpec],
    *,
    motifs: Mapping[str, AST] | None = None,
    config: OptimizationConfig = OptimizationConfig(),
    warm_starts: Sequence[Mapping[str, float]] = (),
    samples: Sequence[SupervisedSample] | None = None,
    proposal_paths: int = 128,
    strict_paths: int = 1_000,
    seed: int | None = None,
    reference_cost: float | None = None,
    reference_name: str | None = None,
) -> FitResult:
    """Fit a composed policy using empirical MC only.

    This entry point deliberately never calls ``solve_inventory_dp`` and never
    accepts a DP reference.  It mirrors the lead-time Stage-3 two-fidelity
    contract: low-path MC ranks proposals and an independent high-path MC
    evaluator arbitrates the strict tier.  ``reference_cost`` is optional and
    is only used to report a normalized empirical score; it is not computed or
    interpreted as an exact optimum here.

    ``samples`` are diagnostics only.  They may be omitted when no DP action
    labels are available; in that case the action-MSE fields are NaN and the
    policy-cost optimization remains fully defined.
    """

    if not isinstance(case, InventoryCase):
        raise TypeError("case must be an InventoryCase")
    if int(horizon) <= 0:
        raise ValueError("horizon must be positive")
    if int(proposal_paths) <= 0 or int(strict_paths) <= 0:
        raise ValueError("proposal_paths and strict_paths must be positive")
    if reference_cost is not None and float(reference_cost) <= 0.0:
        raise ValueError("reference_cost must be positive when supplied")

    resolved_motifs = motifs_for_config(case.config) if motifs is None else dict(motifs)
    config = _with_default_action_cap(config, case.n)
    check = ast_check(
        ast,
        motifs=resolved_motifs,
        allowed_state_names=case.schema.allowed_state_names,
    )
    param_names = check.stats.parameters
    specs = _ordered_specs(param_names, parameter_specs)
    base_seed = int(config.random_seed if seed is None else seed)
    from optimization.policies.parameters import apply_cap_modes
    specs = apply_cap_modes(specs, check.ast, resolved_motifs)
    common = {
        "ast": check.ast,
        "case": case,
        "horizon": int(horizon),
        "motifs": resolved_motifs,
        "rounding": config.rounding,
        "action_cap": config.action_cap,
        "initial_state": config.initial_state,
    }
    # Typed cyclic caps are time-dependent only through the cap node.  Route
    # them through the lead-time candidate×path kernel, which exposes the
    # explicit ``time`` feature and keeps the same vectorized simulation path
    # used by the ordinary lead-time evaluator.  Generic composed MC remains
    # the fallback for other inventory settings.
    from inventory.policy.ast import has_cyclic_cap
    from inventory.core.config import BacklogMode, LeadTimeMode, PerishabilityMode
    pure_lead_time = (
        case.config.lead_time.mode is LeadTimeMode.DETERMINISTIC
        and case.config.perishability.mode is PerishabilityMode.DISABLED
        and case.config.backlog.mode in (BacklogMode.DISABLED, BacklogMode.LOST_SALES)
    )
    if has_cyclic_cap(check.ast, resolved_motifs) and pure_lead_time:
        from optimization.backends.lead_time.mc_evaluator import (
            DemandTrajectoryBank, MonteCarloPolicyCostEvaluator,
        )
        from optimization.backends.lead_time.mc_adapters import TwoFidelityMCEvaluator
        from optimization.backends.lead_time.rolling import rolling_reference

        reference = rolling_reference(case)
        low_bank = DemandTrajectoryBank.create(case.demand, int(horizon), int(proposal_paths), base_seed)
        high_bank = DemandTrajectoryBank.create(case.demand, int(horizon), int(strict_paths), base_seed + 1)
        low = MonteCarloPolicyCostEvaluator(
            check.ast, reference, low_bank, resolved_motifs,
            horizon=int(horizon), initial_state=config.initial_state,
            use_exact_anchor_control_variate=False,
            use_common_random_numbers=True,
            batch_size=int(getattr(config, "candidate_batch_size", 64)),
        )
        high = MonteCarloPolicyCostEvaluator(
            check.ast, reference, high_bank, resolved_motifs,
            horizon=int(horizon), initial_state=config.initial_state,
            use_exact_anchor_control_variate=False,
            use_common_random_numbers=True,
            batch_size=int(getattr(config, "candidate_batch_size", 64)),
        )
        evaluator = TwoFidelityMCEvaluator(high=high, low=low)
    else:
        low = ComposedMonteCarloPolicyCostEvaluator.from_case(
            **common, paths=int(proposal_paths), seed=base_seed
        )
        high = ComposedMonteCarloPolicyCostEvaluator.from_case(
            **common, paths=int(strict_paths), seed=base_seed + 1
        )
        evaluator = TwoFidelityComposedMCEvaluator(high=high, low=low)
    optimized = optimize_policy_constants(
        specs,
        evaluator,
        config=config,
        warm_starts=warm_starts,
    )
    params = dict(optimized.best_params)

    if samples:
        diagnostics_samples = tuple(samples)
        batch = _batch_from_samples_cached(diagnostics_samples)
        context = _BatchEvaluationContext(batch, resolved_motifs)
        raw = _evaluate_ast_batch(
            check.ast,
            batch,
            params,
            case.demand.feature_dict(),
            cost_features_for_case(case),
            resolved_motifs,
            context=context,
        )
        finite_raw = np.maximum(
            np.nan_to_num(raw.astype(float), nan=0.0, posinf=0.0, neginf=0.0),
            0.0,
        )
        actions = _round_actions(finite_raw, config.rounding, config.action_cap)
        weight_total = max(float(np.sum(batch.weights)), 1.0e-12)
        raw_mse = float(np.dot(batch.weights, (finite_raw - batch.targets) ** 2) / weight_total)
        action_mse = float(np.dot(batch.weights, (actions - batch.targets) ** 2) / weight_total)
        target_second_moment = float(
            np.dot(batch.weights, batch.targets**2) / weight_total + 1.0e-12
        )
        normalized_mse = float(action_mse / target_second_moment)
    else:
        raw_mse = float("nan")
        action_mse = float("nan")
        normalized_mse = float("nan")

    policy_cost = float(optimized.best_cost)
    normalized_cost = (
        float(policy_cost / max(float(reference_cost), 1.0e-12))
        if reference_cost is not None
        else None
    )
    return FitResult(
        params=params,
        objective_value=(normalized_cost if normalized_cost is not None else policy_cost),
        action_mse=action_mse,
        raw_mse=raw_mse,
        policy_cost=policy_cost,
        normalized_mse=normalized_mse,
        normalized_policy_cost=normalized_cost,
        optimizer=f"policy_cost/{optimized.route}",
        success=True,
        nfev=optimized.total_evaluations,
        elapsed_seconds=optimized.elapsed_seconds,
        formula=ast_to_formula(check.ast),
        param_names=tuple(param_names),
        status=optimized.status,
        route=optimized.route,
        optimizer_metadata={
            "evaluation_backend": "composed_mc_only",
            "mc_only": True,
            "mc_proposal_paths": int(proposal_paths),
            "mc_strict_paths": int(strict_paths),
            "mc_proposal_seed": base_seed,
            "mc_strict_seed": base_seed + 1,
            "mc_common_random_numbers": True,
            "reference_name": reference_name,
            "reference_cost_supplied": reference_cost is not None,
            "completed_evaluations": optimized.completed_evaluations,
            "early_stopped_evaluations": optimized.early_stopped_evaluations,
            "unique_parameter_points": optimized.unique_parameter_points,
            "cache_hits": optimized.cache_hits,
            "best_found_at_evaluation": optimized.best_found_at_evaluation,
            "optimizer_runs": optimized.optimizer_runs,
            "convergence_report": optimized.convergence_report,
        },
    )


__all__ = ["fit_inventory_constants", "fit_inventory_policy", "fit_inventory_policy_mc"]
