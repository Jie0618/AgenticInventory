"""Case-derived initial structures for inventory symbolic search."""

from __future__ import annotations

from inventory.core import InventoryCase
from inventory.policy.ast import AST, ast_check
from inventory.policy.motifs import motifs_for_config
from optimization.policies.seeds import policy_seed_asts_for_case
from optimization.policies.parameters import target_roles_for_motifs
from sr.core.gp import structural_signature, target_gap


def initial_seed_asts(case: InventoryCase) -> tuple[tuple[str, AST], ...]:
    """Build the default population from the case's registered vocabulary."""

    motifs = motifs_for_config(case.config)
    roles = target_roles_for_motifs(motifs)
    seeds: list[tuple[str, AST]] = [
        (f"primitive_gap:{name}", target_gap(roles[name], name))
        for name in sorted(motifs)
    ]
    seeds.extend(policy_seed_asts_for_case(case))
    unique: dict[str, tuple[str, AST]] = {}
    for source, ast in seeds:
        checked = ast_check(
            ast,
            motifs=motifs,
            allowed_state_names=case.schema.allowed_state_names,
        )
        unique.setdefault(checked.ast_hash, (source, checked.ast))
    return tuple(unique.values())


__all__ = ["initial_seed_asts", "structural_signature", "target_gap"]
