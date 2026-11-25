"""Explicit DP-free numerical binding for single/multi-distribution SR.

The controllers retain the ``bellman_regret_pct``/``dp_cost``
field names for artifact compatibility. In this binding they hold empirical
MC screening gaps and baseline reference costs, never Bellman/DP values.
"""
from __future__ import annotations

from runtime.artifacts import seed_from_text
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

import numpy as np

from evaluation.routing import BackendDecision
from inventory.core import InventoryCase
from inventory.policy.motifs import motifs_for_config
from optimization.evaluators.composed import ComposedMonteCarloPolicyCostEvaluator
from optimization.models.types import OptimizationConfig
from optimization.policies.parameters import parameter_specs_for_case
from optimization.solvers.api import fit_inventory_policy_mc
from sr.core.models import ScreenedCandidate


@dataclass(frozen=True)
class MonteCarloSearchContext:
    case: InventoryCase
    horizon: int
    reference_cost: float
    settings: Mapping[str, object]
    fidelity: str = "standard"
    reference_type: str = "best_formula_baseline_empirical_mc"
    seed_namespace: str = "ltb"
    samples: tuple = ()
    backend: BackendDecision = BackendDecision("composed_mc_only", False, ("explicit_mc_only",))

    def __post_init__(self):
        if self.horizon <= 0 or not np.isfinite(self.reference_cost) or self.reference_cost <= 0:
            raise ValueError("MC context requires a positive horizon and finite reference cost")
        if self.fidelity not in {"fast", "standard", "certify"}:
            raise ValueError("invalid MC fidelity")

    @property
    def motifs(self):
        return motifs_for_config(self.case.config)

    def identity_payload(self):
        payload = {"protocol": "distribution-sr-mc", "settings": dict(self.settings),
                "reference_cost": self.reference_cost, "reference_type": self.reference_type,
                "fidelity": self.fidelity}
        if self.seed_namespace != "ltb":
            payload["seed_namespace"] = self.seed_namespace
        return payload

    def parameter_specs(self, names):
        return parameter_specs_for_case(names, self.case)

    def parameter_specs_for_ast(self, ast, motifs=None):
        from inventory.policy.ast import ast_check
        from optimization.policies.parameters import apply_cap_modes
        active = self.motifs if motifs is None else motifs
        names = ast_check(ast, motifs=active,
                          allowed_state_names=self.case.schema.allowed_state_names).stats.parameters
        return apply_cap_modes(self.parameter_specs(names), ast, active)

    def paths(self, kind):
        return int(self.settings.get(f"{self.fidelity}_{kind}_paths", self.settings[f"{kind}_paths"]))

    def bank_seed(self, kind):
        # Match the traditional baseline reference's banks, independently of
        # candidate identity and the optimizer's point-generation seed.
        token = f"{self.seed_namespace}-{kind}|{self.case.name}|{self.fidelity}"
        return seed_from_text(token.encode())

    def fit(self, ast, *, motifs=None, config=OptimizationConfig(), warm_starts=()):
        if warm_starts:
            raise ValueError("distribution SR requires independent constant fitting")
        from inventory.policy.ast import ast_check
        active = self.motifs if motifs is None else motifs
        names = ast_check(ast, motifs=active, allowed_state_names=self.case.schema.allowed_state_names).stats.parameters
        fit = fit_inventory_policy_mc(
            ast, self.case, self.horizon, self.parameter_specs_for_ast(ast, active), motifs=active,
            config=replace(config, mode=self.fidelity,
                           total_evaluations=self.settings.get(f"{self.fidelity}_proposal_budget")),
            proposal_paths=self.paths("proposal"), strict_paths=self.paths("strict"),
            seed=self.bank_seed("tuning"), reference_cost=self.reference_cost,
            reference_name=self.reference_type,
        )
        final = ComposedMonteCarloPolicyCostEvaluator.from_case(
            ast, self.case, self.horizon, motifs=active,
            paths=self.paths("evaluation"), seed=self.bank_seed("final"),
        ).evaluate_many((fit.params,))[0]
        ratio = float(final.cost) / self.reference_cost
        return replace(fit, policy_cost=float(final.cost), normalized_policy_cost=ratio,
                       objective_value=ratio, optimizer_metadata={**fit.optimizer_metadata,
                       "final_bank_seed": self.bank_seed("final"),
                       "final_bank_paths": self.paths("evaluation"),
                       "reference_type": self.reference_type})


class MonteCarloScreen:
    """Small fixed-bank MC screen; screening constants are always discarded."""
    def __init__(self, context, motifs, *, diagnostic_json: Path | None = None):
        self.context = context
        self.motifs = dict(motifs)
        self._diagnostic_screen = None
        self._mc_diagnostics = None
        if diagnostic_json is not None:
            from sr.inventory.screening import BellmanScreen
            self._diagnostic_screen = BellmanScreen(
                diagnostic_json, context=context, motifs=self.motifs, random_seed=0,
            )

    def parent_diagnostics(self, parents):
        """Use DP evidence when available; otherwise keep simulated parent spread."""
        if self._diagnostic_screen is None:
            if not parents:
                return {"case": self.context.case.name, "available": False, "reason": "no_parents"}
            if self._mc_diagnostics is None:
                from sr.inventory.diagnostics import MonteCarloDiagnostics
                self._mc_diagnostics = MonteCarloDiagnostics(self.context.case, self.context.horizon)
            bank = self._mc_diagnostics
            actions = {parent.ast_hash: bank.actions(parent.candidate.ast, parent.params, self.motifs)
                       for parent in parents}
            return bank.records(actions)
        # Reflection can add motifs after this screen was constructed.
        self._diagnostic_screen.motifs = dict(self.motifs)
        return self._diagnostic_screen.parent_diagnostics(parents)

    def parent_behavior_signature(self, parent):
        """Compare final fitted actions on the same bank used for diagnostics."""
        if self._diagnostic_screen is not None:
            self._diagnostic_screen.motifs = dict(self.motifs)
            return self._diagnostic_screen.parent_behavior_signature(parent)
        if self._mc_diagnostics is None:
            from sr.inventory.diagnostics import MonteCarloDiagnostics
            self._mc_diagnostics = MonteCarloDiagnostics(self.context.case, self.context.horizon)
        from sr.core.diagnostics import action_signature
        bank = self._mc_diagnostics
        return action_signature([bank.signature,
                                 bank.actions(parent.candidate.ast, parent.params, self.motifs)])

    def score(self, candidate, config):
        context = self.context
        specs = (context.parameter_specs_for_ast(candidate.ast, self.motifs)
                 if hasattr(context, "parameter_specs_for_ast")
                 else context.parameter_specs(candidate.parameter_names))
        count = int(context.settings.get("screen_parameter_points", 16))
        paths = int(context.settings.get("screen_paths", 16))
        horizon = min(context.horizon, int(context.settings.get("screen_horizon", 64)))
        if min(count, paths, horizon) <= 0:
            raise ValueError("MC screen budgets must be positive")
        rng = np.random.default_rng(config.random_seed ^ int(candidate.ast_hash[:8], 16))
        points = [{spec.name: float(spec.initial_value()) for spec in specs}]
        for _ in range(count - 1 if specs else 0):
            points.append({spec.name: float(rng.integers(int(spec.lower), int(spec.upper) + 1))
                           if spec.kind == "integer" else float(rng.choice(spec.values))
                           if spec.kind == "grid" else float(rng.uniform(spec.lower, spec.upper))
                           for spec in specs})
        evaluator = ComposedMonteCarloPolicyCostEvaluator.from_case(
            candidate.ast, context.case, horizon, motifs=self.motifs,
            paths=paths, seed=context.bank_seed("screen"),
        )
        results = evaluator.evaluate_many(points)
        index = min(range(len(results)), key=lambda i: results[i].cost)
        gap = 100 * (results[index].cost / horizon / (context.reference_cost / context.horizon) - 1)
        # The screen is a ranking heuristic, not a Bellman certificate. Avoid
        # claiming behavior equivalence from one initial state or short rollout.
        return ScreenedCandidate(candidate, gap, points[index], candidate.ast_hash)
