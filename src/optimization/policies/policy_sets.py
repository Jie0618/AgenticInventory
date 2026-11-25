"""Resolve policy-set definitions for deterministic lead-time experiments."""

from __future__ import annotations

from optimization.policies.baselines.lead_time import baselines_for_lead_time
from inventory.policy.ast import ast_check, ast_to_formula
from inventory.policy.motifs.lead_time import lead_time_motifs
from optimization.policies.parameters import parameter_specs


def resolve_lead_time_policy(definition: object, lead_time: int, n: int):
    # Imported lazily to keep the shared JSON parser independent of this model.
    from optimization.policies.policy_sets_json import ResolvedPolicy

    if getattr(definition, "type") == "registered_baseline":
        available = {
            row.name: row for row in baselines_for_lead_time(int(lead_time), int(n))
        }
        registered_name = getattr(definition, "registered_name")
        if registered_name not in available:
            raise ValueError(f"unknown registered baseline: {registered_name}")
        spec = available[str(registered_name)]
        return ResolvedPolicy(
            getattr(definition, "id"),
            getattr(definition, "type"),
            spec.ast,
            spec.parameter_specs,
            spec.motifs,
            spec.formula,
        )

    motifs = dict(lead_time_motifs(int(lead_time)))
    motifs.update(dict(getattr(definition, "motifs") or {}))
    ast = getattr(definition, "ast")
    if ast is None:
        raise ValueError(f"formula policy {getattr(definition, 'id')} has no AST")
    checked = ast_check(ast, motifs=motifs)
    return ResolvedPolicy(
        getattr(definition, "id"),
        getattr(definition, "type"),
        checked.ast,
        parameter_specs(checked.stats.parameters, int(lead_time), int(n)),
        motifs,
        ast_to_formula(checked.ast),
    )


__all__ = ["resolve_lead_time_policy"]
