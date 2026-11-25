"""Validated parameterized policy-set input."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from inventory.policy.ast import AST
from optimization.models.types import ParameterSpec


@dataclass(frozen=True)
class PolicyDefinition:
    id: str
    type: str
    registered_name: str | None = None
    ast: AST | None = None
    motifs: Mapping[str, AST] | None = None


@dataclass(frozen=True)
class ResolvedPolicy:
    id: str
    type: str
    ast: AST
    parameter_specs: tuple[ParameterSpec, ...]
    motifs: Mapping[str, AST]
    formula: str


def load_policy_set(path: Path) -> tuple[str, tuple[PolicyDefinition, ...]]:
    """Load registered baselines and/or arbitrary validated formula ASTs."""

    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    name = str(payload.get("policy_set", "")).strip()
    raw_policies = payload.get("policies")
    if not name or not isinstance(raw_policies, list) or not raw_policies:
        raise ValueError("policy_set and a nonempty policies list are required")
    output: list[PolicyDefinition] = []
    for raw in raw_policies:
        if not isinstance(raw, dict):
            raise ValueError("each policy must be an object")
        identifier = str(raw.get("id", "")).strip()
        policy_type = str(raw.get("type", "")).strip()
        if not identifier or policy_type not in {"registered_baseline", "formula_ast"}:
            raise ValueError("each policy needs an id and supported type")
        if policy_type == "registered_baseline":
            registered_name = str(raw.get("name", identifier)).strip()
            output.append(PolicyDefinition(identifier, policy_type, registered_name=registered_name))
            continue
        ast = _load_ast(raw, "ast", "ast_file", source.parent)
        motifs = _load_optional_mapping(raw, "motifs", "motifs_file", source.parent)
        output.append(PolicyDefinition(identifier, policy_type, ast=ast, motifs={} if motifs is None else motifs))
    if len({row.id for row in output}) != len(output):
        raise ValueError("policy ids must be unique")
    return name, tuple(output)


def resolve_policy(definition: PolicyDefinition, lead_time: int, n: int) -> ResolvedPolicy:
    """Resolve one policy-set definition for a lead-time case."""

    from optimization.policies.policy_sets import resolve_lead_time_policy

    return resolve_lead_time_policy(definition, lead_time, n)


def _load_ast(raw: Mapping[str, object], inline_key: str, file_key: str, base: Path) -> AST:
    inline = raw.get(inline_key)
    filename = raw.get(file_key)
    if (inline is None) == (filename is None):
        raise ValueError(f"provide exactly one of {inline_key} or {file_key}")
    value = inline if inline is not None else json.loads((base / str(filename)).read_text(encoding="utf-8"))
    if isinstance(value, dict) and "ast" in value and "op" not in value:
        value = value["ast"]
    if not isinstance(value, dict):
        raise ValueError("AST must be a JSON object")
    return dict(value)


def _load_optional_mapping(
    raw: Mapping[str, object], inline_key: str, file_key: str, base: Path
) -> dict[str, AST] | None:
    inline = raw.get(inline_key)
    filename = raw.get(file_key)
    if inline is not None and filename is not None:
        raise ValueError(f"provide at most one of {inline_key} or {file_key}")
    if inline is None and filename is None:
        return None
    value = inline if inline is not None else json.loads((base / str(filename)).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("motifs must be a JSON object mapping names to ASTs")
    if any(not isinstance(ast, dict) for ast in value.values()):
        raise ValueError("every motif value must be an AST object")
    return {str(name): dict(ast) for name, ast in value.items()}


__all__ = ["PolicyDefinition", "ResolvedPolicy", "load_policy_set", "resolve_policy"]
