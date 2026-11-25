"""Policy-set loader supporting lead-time computational baselines."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Union

from optimization.policies.policy_sets_json import PolicyDefinition, ResolvedPolicy, resolve_policy
from inventory.policy.ast import AST

from benchmarks.baselines.lead_time.catalog import (
    ComputationalBaselineSpec,
    computational_baseline,
)
from benchmarks.baselines.lead_time.evaluator import (
    dependency_sentinel_ast,
)


@dataclass(frozen=True)
class ComputationalPolicyDefinition:
    id: str
    type: str
    registered_name: str


@dataclass(frozen=True)
class ResolvedComputationalPolicy:
    id: str
    type: str
    ast: AST
    parameter_specs: tuple
    motifs: Mapping[str, AST]
    formula: str
    computational_spec: ComputationalBaselineSpec


BenchmarkPolicyDefinition = Union[PolicyDefinition, ComputationalPolicyDefinition]
BenchmarkResolvedPolicy = Union[ResolvedPolicy, ResolvedComputationalPolicy]


def load_benchmark_policy_set(path: Path) -> tuple[str, tuple[BenchmarkPolicyDefinition, ...]]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    name = str(payload.get("policy_set", "")).strip()
    raw_policies = payload.get("policies")
    if not name or not isinstance(raw_policies, list) or not raw_policies:
        raise ValueError("policy_set and a nonempty policies list are required")
    output: list[BenchmarkPolicyDefinition] = []
    for raw_value in raw_policies:
        if not isinstance(raw_value, dict):
            raise ValueError("each policy must be an object")
        raw = dict(raw_value)
        identifier = str(raw.get("id", "")).strip()
        policy_type = str(raw.get("type", "")).strip()
        if not identifier:
            raise ValueError("each policy needs an id")
        if policy_type == "computational_baseline":
            registered_name = str(raw.get("name", identifier)).strip()
            output.append(
                ComputationalPolicyDefinition(identifier, policy_type, registered_name)
            )
        elif policy_type == "registered_baseline":
            output.append(
                PolicyDefinition(
                    identifier,
                    policy_type,
                    registered_name=str(raw.get("name", identifier)).strip(),
                )
            )
        elif policy_type == "formula_ast":
            ast = _load_ast(raw, "ast", "ast_file", source.parent)
            motifs = _load_optional_mapping(raw, "motifs", "motifs_file", source.parent)
            output.append(
                PolicyDefinition(
                    identifier,
                    policy_type,
                    ast=ast,
                    motifs={} if motifs is None else motifs,
                )
            )
        else:
            raise ValueError(f"unsupported benchmark policy type: {policy_type}")
    if len({row.id for row in output}) != len(output):
        raise ValueError("policy ids must be unique")
    return name, tuple(output)


def resolve_benchmark_policy(
    definition: BenchmarkPolicyDefinition, lead_time: int, n: int
) -> BenchmarkResolvedPolicy:
    if isinstance(definition, ComputationalPolicyDefinition):
        spec = computational_baseline(definition.registered_name, lead_time, n)
        return ResolvedComputationalPolicy(
            id=definition.id,
            type=definition.type,
            ast=dependency_sentinel_ast(
                lead_time, horizon_dependent=spec.horizon_dependent
            ),
            parameter_specs=spec.parameter_specs,
            motifs={},
            formula=spec.formula,
            computational_spec=spec,
        )
    return resolve_policy(definition, lead_time, n)


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
    if not isinstance(value, dict) or any(not isinstance(ast, dict) for ast in value.values()):
        raise ValueError("motifs must map names to AST objects")
    return {str(key): dict(ast) for key, ast in value.items()}


__all__ = [
    "BenchmarkPolicyDefinition",
    "BenchmarkResolvedPolicy",
    "ComputationalPolicyDefinition",
    "ResolvedComputationalPolicy",
    "load_benchmark_policy_set",
    "resolve_benchmark_policy",
]
