"""Public parameter-optimization API with cycle-safe lazy exports."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "InventorySearchContext": ("optimization.search_context", "InventorySearchContext"),
    "EvaluationResult": ("optimization.models.types", "EvaluationResult"),
    "FitResult": ("optimization.models.types", "FitResult"),
    "MonteCarloProposalConfig": ("optimization.models.types", "MonteCarloProposalConfig"),
    "OptimizationConfig": ("optimization.models.types", "OptimizationConfig"),
    "ParameterSpec": ("optimization.models.types", "ParameterSpec"),
    "PolicyOptimizationResult": ("optimization.models.types", "PolicyOptimizationResult"),
    "SupervisedSample": ("optimization.models.types", "SupervisedSample"),
    "DenseExactPolicyCostEvaluator": (
        "optimization.backends.lead_time.exact_evaluator",
        "DenseExactPolicyCostEvaluator",
    ),
    "SparseExactPolicyCostEvaluator": (
        "optimization.backends.lead_time.exact_evaluator",
        "SparseExactPolicyCostEvaluator",
    ),
    "DemandTrajectoryBank": ("optimization.backends.lead_time.mc_evaluator", "DemandTrajectoryBank"),
    "MonteCarloEvaluationResult": (
        "optimization.backends.lead_time.mc_evaluator",
        "MonteCarloEvaluationResult",
    ),
    "MonteCarloPolicyCostEvaluator": (
        "optimization.backends.lead_time.mc_evaluator",
        "MonteCarloPolicyCostEvaluator",
    ),
    "evaluate_ast_policy_cost_exact": (
        "optimization.backends.lead_time.policy_cost",
        "evaluate_ast_policy_cost_exact",
    ),
    "evaluate_policy_cost_exact": (
        "optimization.backends.lead_time.policy_cost",
        "evaluate_policy_cost_exact",
    ),
    "dp_action_samples": ("optimization.samples.dp", "dp_action_samples"),
    "fit_inventory_constants": ("optimization.solvers.api", "fit_inventory_constants"),
    "fit_inventory_policy": (
        "optimization.solvers.api",
        "fit_inventory_policy",
    ),
    "fit_inventory_policy_mc": (
        "optimization.solvers.api",
        "fit_inventory_policy_mc",
    ),
    "canonical_parameter_key": (
        "optimization.solvers.parameters",
        "canonical_parameter_key",
    ),
    "canonicalize_params": ("optimization.solvers.parameters", "canonicalize_params"),
    "optimize_policy_constants": (
        "optimization.solvers.policy_cost",
        "optimize_policy_constants",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
