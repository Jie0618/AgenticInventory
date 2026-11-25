"""Opt-in lead-time cyclic experiment, retaining the existing SR controllers.

Pure policies retain their configured optimizer. Typed caps use explicit
two-fidelity MC. Both are ranked on the same independent final demand bank.
The DP remains a reporting reference, never a time-dependent cap evaluator.
"""
from __future__ import annotations

from runtime.artifacts import stable_key
import time
from dataclasses import dataclass, replace
from typing import Mapping

from inventory.core.config import BacklogMode, LeadTimeMode, PerishabilityMode
from inventory.policy.ast import ast_check, cap_parameter_modes, has_cyclic_cap
from optimization.evaluators.composed import ComposedMonteCarloPolicyCostEvaluator
from optimization.models.types import OptimizationConfig
from optimization.policies.parameters import parameter_specs_for_case, apply_cap_modes
from optimization.search_context import InventorySearchContext
from optimization.solvers.api import fit_inventory_policy_mc
from sr.inventory.mc import MonteCarloSearchContext


def bind_cyclic_runtime(runtime, settings):
    """Use identical numerical bindings and cache identity in both Stage-II phases."""
    import json
    inputs = {name: replace(row, context=CyclicLeadTimeSearchContext.from_inventory(row.context, settings))
              for name, row in runtime.case_inputs.items()}
    signature = stable_key(json.dumps({"base": runtime.initialization_signature,
        "cyclic": {name: row.context.identity_payload() for name, row in inputs.items()}},
        sort_keys=True).encode())
    return replace(runtime, case_inputs=inputs, initialization_signature=signature)


@dataclass(frozen=True)
class CyclicLeadTimeSearchContext(MonteCarloSearchContext):
    original_context: InventorySearchContext | None = None

    @classmethod
    def from_inventory(cls, context, settings):
        config = context.case.config
        if (config.lead_time.mode not in (LeadTimeMode.DISABLED, LeadTimeMode.DETERMINISTIC)
                or config.perishability.mode not in (PerishabilityMode.DISABLED, PerishabilityMode.DETERMINISTIC)
                or config.backlog.mode not in (BacklogMode.DISABLED, BacklogMode.LOST_SALES)):
            raise ValueError("DP-backed cyclic search requires deterministic lost-sales inventory")
        for name in ("proposal_paths", "strict_paths", "evaluation_paths",
                     "screen_paths", "screen_horizon", "screen_parameter_points"):
            value = settings[name]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        return cls(
            case=context.case, horizon=context.horizon,
            reference_cost=float(context.reference.value(0, context.case.initial_flat_state)),
            settings=dict(settings), reference_type="dp_optimal",
            seed_namespace="leadtime-cap-cyclic",
            samples=context.samples, original_context=context,
        )

    def identity_payload(self):
        return {**super().identity_payload(), "protocol": "leadtime-cap-cyclic",
                "pure_fitting": "historical_optimizer", "cyclic_fitting": "explicit_mc",
                "ranking": "shared_independent_final_mc", "time_origin": 0, "rounding": "ties_to_even"}

    def parameter_specs_for_ast(self, ast, motifs=None):
        names = ast_check(ast, motifs=self.motifs if motifs is None else motifs,
                          allowed_state_names=self.case.schema.allowed_state_names).stats.parameters
        specs = parameter_specs_for_case(names, self.case)
        return apply_cap_modes(specs, ast, self.motifs if motifs is None else motifs)

    def fit(self, ast, *, motifs=None, config=OptimizationConfig(), warm_starts=()):
        if warm_starts:
            raise ValueError("SR constants must be independently fitted")
        started = time.perf_counter()
        context = replace(self, fidelity=config.mode)
        active = context.motifs if motifs is None else motifs
        if cap_parameter_modes(ast, active):
            fit = fit_inventory_policy_mc(
                ast, context.case, context.horizon, context.parameter_specs_for_ast(ast, active),
                motifs=active, config=replace(config, total_evaluations=context.settings.get(
                    f"{config.mode}_proposal_budget", config.total_evaluations)),
                proposal_paths=context.paths("proposal"), strict_paths=context.paths("strict"),
                seed=context.bank_seed("tuning"), reference_cost=context.reference_cost,
                reference_name=context.reference_type,
            )
        else:
            if context.original_context is None:
                raise ValueError("pure optimizer context is missing")
            fit = context.original_context.fit(ast, motifs=active, config=config, warm_starts=())
        final = ComposedMonteCarloPolicyCostEvaluator.from_case(
            ast, context.case, context.horizon, motifs=active,
            paths=context.paths("evaluation"), seed=context.bank_seed("final"),
        ).evaluate_many((fit.params,))[0]
        ratio = float(final.cost) / context.reference_cost
        return replace(fit, policy_cost=float(final.cost), normalized_policy_cost=ratio,
                       objective_value=ratio, elapsed_seconds=time.perf_counter() - started,
                       optimizer_metadata={**fit.optimizer_metadata,
                           "action_execution": "cyclic_cap_round" if has_cyclic_cap(ast, active) else "round",
                           "fitting_cost": float(fit.policy_cost),
                           "ranking_fidelity": "independent_mc",
                           "final_bank_seed": context.bank_seed("final"),
                           "final_bank_paths": context.paths("evaluation"),
                           "final_mc_standard_error": final.metadata["mc_standard_error"],
                           "cyclic_protocol": context.identity_payload()})
