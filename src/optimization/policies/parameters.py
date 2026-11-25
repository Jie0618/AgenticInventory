"""Reproducible optimizer domains for symbolic-policy constants."""
from __future__ import annotations

from typing import Mapping, Sequence

from inventory.core import InventoryCase
from inventory.policy.ast import AST, cap_parameter_modes
from dataclasses import replace
from optimization.models.types import ParameterSpec


def parameter_spec(name: str, lead_time: int, n: int) -> ParameterSpec:
    token = str(name)
    upper_target = (int(lead_time) + 1) * int(n)
    upper_lag = int(n) * (int(lead_time) * (int(lead_time) + 1) // 2 + 1)
    if token == "rho" or token == "omega" or (
        token.startswith("omega") and token[5:].isdigit()
    ):
        return ParameterSpec(token, kind="continuous", lower=0.0, upper=1.0, default=1.0)
    if token.startswith("v") and token[1:].isdigit():
        return ParameterSpec(token, kind="continuous", lower=-1.0, upper=1.0, default=1.0)
    if token in {"s", "tau"}:
        return ParameterSpec(token, kind="integer", lower=0, upper=upper_target, default=0)
    if token in {"delta", "a_min"}:
        return ParameterSpec(token, kind="integer", lower=0, upper=n, default=0)
    if token.startswith("T"):
        return ParameterSpec(token, kind="integer", lower=0, upper=upper_lag, default=lead_time * n)
    if token.startswith(("S", "A", "P", "N", "F", "Q", "B", "E")):
        return ParameterSpec(token, kind="integer", lower=0, upper=upper_target, default=n)
    if token.startswith(("C", "cap", "r")):
        return ParameterSpec(token, kind="integer", lower=0, upper=n, default=max(1, n // 3))
    if token.startswith(("K", "alpha", "beta", "gamma", "w")):
        upper = 1.5 if token.startswith("K") else 1.0
        default = 1.0 if token.startswith(("K", "w")) else 0.25
        return ParameterSpec(token, kind="continuous", lower=0.0, upper=upper, default=default)
    if token.startswith("W"):
        return ParameterSpec(token, kind="continuous", lower=0.0, upper=10.0, default=1.0)
    raise ValueError(f"unknown SR parameter role: {token}")


def parameter_specs(names: Sequence[str], lead_time: int, n: int) -> tuple[ParameterSpec, ...]:
    return tuple(parameter_spec(name, lead_time, n) for name in names)


def parameter_specs_for_case(
    names: Sequence[str], case: InventoryCase
) -> tuple[ParameterSpec, ...]:
    """Build conservative role-based domains from a canonical state width."""

    if not isinstance(case, InventoryCase):
        raise TypeError("case must be an InventoryCase")
    # ``parameter_spec`` uses lead_time+1 as its aggregate-state scale.  A
    # general composition may aggregate every canonical coordinate, so using
    # dimension-1 preserves the established lead-time domains and safely
    # widens them for age/backlog coordinates.
    scale_width = max(0, case.schema.dimension - 1)
    return parameter_specs(names, scale_width, case.n)


def target_roles_for_motifs(motifs: Mapping[str, AST]) -> dict[str, str]:
    """Assign general optimizer target roles from registered motif semantics."""

    roles: dict[str, str] = {}
    for name in motifs:
        if "backlog" in name:
            role = "B"
        elif any(token in name for token in ("expir", "fresh", "surviv", "age", "life")):
            role = "E"
        elif "moment" in name or "lag" in name:
            role = "T"
        elif "pipeline" in name:
            role = "P"
        elif "available" in name:
            role = "A"
        elif "inventory" in name:
            role = "S"
        else:
            role = "Q"
        roles[name] = role
    return roles


def apply_cap_modes(specs, ast, motifs=None):
    """Cap mode, rather than a parameter-name heuristic, owns its domain."""
    modes = cap_parameter_modes(ast, motifs)
    return tuple(replace(p, kind="integer" if modes[p.name] == "integer" else "continuous")
                 if p.name in modes else p for p in specs)


__all__ = [
    "parameter_spec",
    "parameter_specs",
    "parameter_specs_for_case",
    "target_roles_for_motifs",
]
