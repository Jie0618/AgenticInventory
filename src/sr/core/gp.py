"""Domain-neutral candidate validation and genetic-programming operators."""

from __future__ import annotations

import json
import random
from copy import deepcopy
from typing import Callable, Mapping, Sequence

from inventory.policy.ast import (
    AST,
    ASTLimits,
    add,
    ast_check,
    ast_to_formula,
    const,
    maximum,
    minimum,
    motif,
    mul,
    neg,
    parameter,
    cap,
    cap_parameter_modes,
)

from sr.core.grammar import expanded_complexity, expand_motifs
from sr.core.models import SearchCandidate, SearchConfig, SearchIndividual
from sr.core.tree import iter_subtrees, replace_subtree


ParameterValidator = Callable[[Sequence[str]], object]
CandidateBuilder = Callable[..., SearchCandidate]


def target_gap(target: str, feature: str) -> AST:
    # The executable policy boundary already clips replenishment to zero.
    return add(parameter(target), neg(motif(feature)))


def make_candidate(
    ast: AST,
    *,
    motifs: Mapping[str, AST],
    config: SearchConfig,
    source: str,
    parameter_validator: ParameterValidator | None = None,
    allowed_state_names: Sequence[str] | None = None,
    parents: Sequence[str] = (),
    iteration: int = 0,
    island: int = 0,
    detail: Mapping[str, object] | None = None,
) -> SearchCandidate:
    """Validate a candidate without assuming a particular inventory component."""

    checked = ast_check(
        ast,
        motifs=motifs,
        limits=ASTLimits(
            max_depth=config.max_depth,
            max_nodes=config.max_nodes,
            max_parameters=config.max_parameters,
        ),
        allowed_state_names=allowed_state_names,
    )
    if cap_parameter_modes(checked.ast, motifs) and not config.cyclic_cap:
        raise ValueError("typed cap search is disabled")
    if parameter_validator is not None:
        parameter_validator(checked.stats.parameters)
    expanded = expanded_complexity(
        checked.ast, motifs, allowed_state_names=allowed_state_names
    )
    ast_check(expand_motifs(checked.ast, motifs), motifs={},
              limits=ASTLimits(max_depth=config.max_depth, max_nodes=config.max_nodes,
                               max_parameters=config.max_parameters),
              allowed_state_names=allowed_state_names)
    if expanded > float(config.max_expanded_complexity) + 1.0e-12:
        raise ValueError(
            f"expanded complexity {expanded:g} exceeds "
            f"{float(config.max_expanded_complexity):g}"
        )
    return SearchCandidate(
        ast=checked.ast,
        ast_hash=checked.ast_hash,
        formula=ast_to_formula(checked.ast),
        macro_complexity=float(checked.stats.complexity),
        expanded_complexity=expanded,
        parameter_names=tuple(checked.stats.parameters),
        source=source,
        parents=tuple(parents),
        iteration=int(iteration),
        island=int(island),
        detail=dict(detail or {}),
    )


def structural_signature(ast: AST) -> str:
    """Canonical structure signature that ignores parameter suffixes."""

    counters: dict[str, int] = {}
    names: dict[str, str] = {}

    def normalize(node: AST) -> AST:
        if node.get("op") == "parameter":
            name = str(node["name"])
            role = _role(name)
            if name not in names:
                index = counters.get(role, 0)
                counters[role] = index + 1
                names[name] = f"{role}{index}"
            return {"op": "parameter", "name": names[name]}
        output: AST = {}
        for key, value in node.items():
            if isinstance(value, dict):
                output[key] = normalize(value)
            elif isinstance(value, list):
                output[key] = [
                    normalize(item) if isinstance(item, dict) else item for item in value
                ]
            else:
                output[key] = value
        return output

    return json.dumps(
        normalize(deepcopy(ast)), sort_keys=True, separators=(",", ":")
    )


def generate_gp_descendants(
    parents: Sequence[SearchIndividual],
    *,
    count: int,
    motifs: Mapping[str, AST],
    config: SearchConfig,
    rng: random.Random,
    iteration: int,
    island: int,
    candidate_builder: CandidateBuilder,
    target_roles: Mapping[str, str],
    operator_weights: Mapping[str, float] | None = None,
    parent_weights: Mapping[str, float] | None = None,
) -> list[SearchCandidate]:
    if not parents:
        return []
    output: list[SearchCandidate] = []
    attempts = 0
    while len(output) < count and attempts < max(20, count * 8):
        attempts += 1
        weights = [
            max(0.0, float((parent_weights or {}).get(row.ast_hash, 1.0)))
            for row in parents
        ]
        left = rng.choices(parents, weights=weights, k=1)[0]
        left_behavior = str(left.candidate.detail.get("behavior_signature", ""))
        right_weights = [
            weight
            * (
                1.5
                if str(row.candidate.detail.get("behavior_signature", ""))
                != left_behavior
                else 1.0
            )
            for row, weight in zip(parents, weights)
        ]
        right = rng.choices(parents, weights=right_weights, k=1)[0]
        operators = (
            "minimum",
            "maximum",
            "subtree",
            "cap",
            "scale",
            "elastic",
            "feature",
            "unwrap",
            "immigrant",
        )
        defaults = dict(zip(operators, (18, 6, 12, 12, 8, 10, 14, 8, 12)))
        if config.cyclic_cap:
            operators += ("cap_mode",)
            defaults["cap_mode"] = 12
        operation_weights = operator_weights or defaults
        operation = rng.choices(
            operators,
            weights=[
                max(0.0, float(operation_weights.get(name, 0.0)))
                for name in operators
            ],
            k=1,
        )[0]
        try:
            ast = _apply_operation(
                operation,
                left.candidate.ast,
                right.candidate.ast,
                motifs,
                target_roles,
                rng,
            )
            output.append(
                candidate_builder(
                    ast,
                    source=f"gp:{operation}",
                    parents=(left.ast_hash, right.ast_hash)
                    if operation in {"minimum", "maximum", "subtree"}
                    else (left.ast_hash,),
                    iteration=iteration,
                    island=island,
                )
            )
        except (ValueError, KeyError, TypeError):
            continue
    return output


def _apply_operation(
    operation: str,
    left: AST,
    right: AST,
    motifs: Mapping[str, AST],
    target_roles: Mapping[str, str],
    rng: random.Random,
) -> AST:
    if operation == "cap_mode":
        choices = []
        for ref in iter_subtrees(left):
            node = ref.subtree
            if node["op"] == "cap":
                choices.append((ref.path, node["arg"], node["limit"], node["mode"]))
            elif node["op"] == "minimum":
                for side, other in (("left", "right"), ("right", "left")):
                    limit = node[side]
                    if limit.get("op") == "parameter" and str(limit["name"]).startswith(("C", "cap")):
                        choices.append((ref.path, node[other], limit, "integer"))
        if not choices:
            raise ValueError("no cap slot")
        path, arg, limit, old = rng.choice(choices)
        mode = rng.choice([m for m in ("integer", "continuous", "cyclic_integer") if m != old])
        return replace_subtree(left, path, cap(arg, limit, mode))
    return _apply_expression_operation(operation, left, right, motifs, target_roles, rng)


def _apply_expression_operation(
    operation: str, left: AST, right: AST, motifs: Mapping[str, AST],
    target_roles: Mapping[str, str], rng: random.Random,
) -> AST:
    if operation == "minimum":
        return minimum(left, right)
    if operation == "maximum":
        return maximum(left, right)
    if operation == "subtree":
        return replace_subtree(
            left,
            rng.choice(iter_subtrees(left)).path,
            rng.choice(iter_subtrees(right)).subtree,
        )
    if operation == "cap":
        return minimum(left, parameter(_fresh_name(left, "C")))
    if operation == "scale":
        return mul(parameter(_fresh_name(left, "K")), left)
    if operation == "elastic":
        cap = parameter(_fresh_name(left, "C"))
        slope = parameter(_fresh_name(left, "alpha"))
        return minimum(left, add(cap, mul(slope, left)))
    if operation == "feature":
        refs = [
            ref for ref in iter_subtrees(left) if ref.subtree.get("op") == "motif"
        ]
        if not refs:
            raise ValueError("no motif leaf")
        ref = rng.choice(refs)
        choices = [name for name in motifs if name != ref.subtree.get("name")]
        if not choices:
            raise ValueError("no alternative motif")
        return replace_subtree(left, ref.path, motif(rng.choice(choices)))
    if operation == "unwrap":
        branches = [
            ref
            for ref in iter_subtrees(left)
            if ref.subtree.get("op") in {"minimum", "maximum"}
        ]
        if not branches:
            raise ValueError("no wrapper")
        ref = rng.choice(branches)
        return replace_subtree(
            left, ref.path, ref.subtree[rng.choice(("left", "right"))]
        )
    if operation == "immigrant":
        available = [name for name in target_roles if name in motifs]
        if not available:
            raise ValueError("no target motif")
        feature = rng.choice(available)
        gap = target_gap(_fresh_name(left, target_roles[feature]), feature)
        return rng.choice((gap, minimum(left, gap), maximum(left, gap)))
    raise ValueError(f"unknown GP operation: {operation}")


def _fresh_name(ast: AST, prefix: str) -> str:
    used = {
        str(ref.subtree["name"])
        for ref in iter_subtrees(ast)
        if ref.subtree.get("op") == "parameter"
    }
    if prefix not in used:
        return prefix
    index = 1
    while f"{prefix}{index}" in used:
        index += 1
    return f"{prefix}{index}"


def _role(name: str) -> str:
    for prefix in (
        "alpha",
        "beta",
        "gamma",
        "cap",
        "S",
        "T",
        "A",
        "P",
        "N",
        "F",
        "Q",
        "C",
        "r",
        "K",
        "W",
        "w",
    ):
        if name.startswith(prefix):
            return prefix
    return "u"


__all__ = [
    "generate_gp_descendants",
    "make_candidate",
    "structural_signature",
    "target_gap",
]
