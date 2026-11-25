"""Validated representation for an ML-MD policy-family proposal."""
from __future__ import annotations

from runtime.artifacts import stable_key
from dataclasses import dataclass
import json
from typing import Any, Mapping, Sequence

from inventory.policy.ast import AST, ASTLimits, EXECUTABLE_AST_LIMITS, ast_check
from inventory.core import InventoryCase
from inventory.policy.motifs import motifs_for_config
from inventory.policy.motifs.lead_time import lead_time_motifs
from inventory.policy.features import (
    BUILTIN_FEATURE_NAMES,
    FeatureSpec,
    legacy_sum_feature,
    materialize_feature,
    resolve_window_rule,
)


TARGET_LEAD_TIMES = tuple(range(2, 11))
# Lead times used by the numerical Stage-III run.  Structural validation is
# intentionally aligned with the evaluated tensor rather than requiring an
# additional cross-L pass over every possible lead time.
ACTIVE_LEAD_TIMES = (3, 5, 8, 10)
FEATURE_NAMES = set(BUILTIN_FEATURE_NAMES)


@dataclass(frozen=True)
class FamilyParameter:
    name: str
    kind: str
    lower: float
    upper: float
    index_rule: str = "scalar"
    context_dependence: str = "independent_per_case"

    def __post_init__(self) -> None:
        if self.lower > self.upper:
            raise ValueError(f"invalid bounds for {self.name}")
        if self.kind not in {"target", "scale", "cap", "weight", "lag"}:
            raise ValueError(f"unsupported parameter kind: {self.kind}")
        if self.context_dependence != "independent_per_case":
            raise ValueError("family search requires independent per-case parameters")


def materialize_family(family: "PolicyFamily", lead_time: int) -> tuple[AST, dict[str, AST]]:
    """Expand feature nodes and built-in features for one concrete L."""
    custom = {f.name: f for f in family.features}
    motifs = dict(lead_time_motifs(lead_time))
    motifs.update(family.motifs)
    # Fixed-window aliases are executable, lead-time-aware motifs;
    # install them before walking the policy AST to resolve named window references.
    for name in _feature_names(family.ast):
        alias = legacy_sum_feature(name, int(lead_time))
        if alias is not None:
            motifs[name] = alias
    for name in list(custom):
        motifs[name] = materialize_feature(name, lead_time, custom[name])
    def walk(node: Any) -> Any:
        if isinstance(node, list):
            return [walk(v) for v in node]
        if not isinstance(node, dict):
            return node
        if node.get("op") == "feature":
            name = str(node.get("name"))
            if name not in motifs:
                raise ValueError(f"unknown feature: {name}")
            return {"op": "motif", "name": name}
        return {k: walk(v) for k, v in node.items()}
    return walk(family.ast), {k: walk(v) for k, v in motifs.items()}


def materialize_family_for_case(
    family: "PolicyFamily", case: InventoryCase
) -> tuple[AST, dict[str, AST]]:
    """Materialize a family against the components enabled by ``case``."""

    if not isinstance(case, InventoryCase):
        raise TypeError("case must be an InventoryCase")
    motifs = dict(motifs_for_config(case.config))
    shadowed = set(motifs).intersection(family.motifs)
    if shadowed:
        raise ValueError(
            "custom motifs shadow built-in motif(s): "
            + ", ".join(sorted(shadowed))
        )
    motifs.update(family.motifs)
    custom = {feature.name: feature for feature in family.features}
    lead_time = int(case.config.lead_time.state_dimension)
    referenced_features = _feature_names(family.ast)
    for motif_ast in family.motifs.values():
        referenced_features.update(_feature_names(motif_ast))
    for name in referenced_features:
        if name in motifs:
            continue
        spec = custom.get(name)
        if spec is None:
            alias = legacy_sum_feature(name, lead_time) if lead_time > 0 else None
            if alias is None:
                raise ValueError(f"unknown feature for case {case.name}: {name}")
            motifs[name] = alias
            continue
        if lead_time <= 0:
            raise ValueError(
                f"feature {name!r} requires a lead-time pipeline, but case "
                f"{case.name!r} has none"
            )
        motifs[name] = materialize_feature(name, lead_time, spec)

    def walk(node: Any) -> Any:
        if isinstance(node, list):
            return [walk(value) for value in node]
        if not isinstance(node, dict):
            return node
        if node.get("op") == "feature":
            name = str(node.get("name"))
            if name not in motifs:
                raise ValueError(f"unknown feature for case {case.name}: {name}")
            return {"op": "motif", "name": name}
        return {key: walk(value) for key, value in node.items()}

    return walk(family.ast), {name: walk(ast) for name, ast in motifs.items()}


def _contains_raw_pipeline_state(node: Any) -> bool:
    if isinstance(node, dict):
        if node.get("op") == "state" and str(node.get("name", "")).startswith("p"):
            return True
        return any(_contains_raw_pipeline_state(v) for v in node.values())
    if isinstance(node, list):
        return any(_contains_raw_pipeline_state(v) for v in node)
    return False


def _feature_names(node: Any) -> set[str]:
    if isinstance(node, dict):
        found = {str(node["name"])} if node.get("op") == "feature" and node.get("name") is not None else set()
        for value in node.values():
            found.update(_feature_names(value))
        return found
    if isinstance(node, list):
        found: set[str] = set()
        for value in node:
            found.update(_feature_names(value))
        return found
    return set()


def _expanded_stats(node: AST, motifs: Mapping[str, AST], stack: tuple[str, ...] = ()) -> tuple[int, int, int, set[str]]:
    """Count the executable tree, expanding motif/feature bodies recursively."""
    op = str(node.get("op", ""))
    if op == "motif":
        name = str(node["name"])
        if name in stack:
            raise ValueError(f"cyclic motif: {name}")
        return _expanded_stats(motifs[name], motifs, stack + (name,))
    children: list[AST] = []
    for key in ("arg", "left", "right", "lower", "upper", "limit"):
        value = node.get(key)
        if isinstance(value, dict):
            children.append(value)
    value = node.get("args")
    if isinstance(value, list):
        children.extend(item for item in value if isinstance(item, dict))
    nodes = 1
    depth = 1
    nonlinear = int(op in {"absolute", "minimum", "maximum", "indicator_gt", "safe_ratio", "clip", "cap"})
    params = {str(node["name"])} if op == "parameter" else set()
    for child in children:
        c_nodes, c_depth, c_nonlinear, c_params = _expanded_stats(child, motifs, stack)
        nodes += c_nodes
        depth = max(depth, 1 + c_depth)
        nonlinear += c_nonlinear
        params.update(c_params)
    return nodes, depth, nonlinear, params


@dataclass(frozen=True)
class PolicyFamily:
    family_id: str
    ast: AST
    motifs: Mapping[str, AST]
    parameters: tuple[FamilyParameter, ...]
    description: str = ""
    features: tuple[FeatureSpec, ...] = ()
    parent_family_id: str | None = None
    mutation_action: str | None = None
    parent_family_hash: str | None = None
    parent_family_hashes: tuple[str, ...] = ()

    def validate(self, lead_times: tuple[int, ...] = ACTIVE_LEAD_TIMES) -> None:
        if not lead_times:
            raise ValueError("lead_times must be nonempty")
        declared = {p.name for p in self.parameters}
        by_name = {p.name: p for p in self.parameters}
        # Built-in motifs are lead-time aware and must not be shadowed by a
        # proposal's fixed-state definition (e.g. redefining total_pipeline
        # with p1,...,p10 makes the family invalid for L=2).
        reserved_motifs = set(lead_time_motifs(max(lead_times)))
        shadowed = reserved_motifs.intersection(self.motifs)
        if shadowed:
            raise ValueError(f"custom motifs shadow built-in motif(s): {', '.join(sorted(shadowed))}")
        if _contains_raw_pipeline_state(self.ast) or any(_contains_raw_pipeline_state(node) for node in self.motifs.values()):
            raise ValueError("custom motifs may not contain raw state nodes; declare an L-aware feature instead")
        feature_names = set(FEATURE_NAMES) | {f.name for f in self.features}
        if len(feature_names) != len(FEATURE_NAMES) + len(self.features):
            raise ValueError("duplicate or reserved custom feature name")
        for lead_time in lead_times:
            ast, motifs = materialize_family(self, lead_time)
            allowed_states = {"x"} | {f"p{i}" for i in range(1, int(lead_time) + 1)}
            try:
                checked = ast_check(ast, motifs=motifs, allowed_state_names=allowed_states, limits=EXECUTABLE_AST_LIMITS)
                for motif_ast in self.motifs.values():
                    ast_check(motif_ast, motifs=motifs, allowed_state_names=allowed_states, limits=EXECUTABLE_AST_LIMITS)
            except Exception as exc:
                raise ValueError(f"L={lead_time}: {exc}") from exc
            used = set(checked.stats.parameters)
            for feature in self.features:
                if feature.name in _feature_names(self.ast):
                    used.update(feature.parameter_names)
                    for name in feature.parameter_names:
                        parameter = by_name.get(name)
                        if parameter is None or parameter.kind != "weight" or parameter.lower < 0 or parameter.upper > 1:
                            raise ValueError(f"weight parameter {name!r} must be declared kind=weight with bounds inside [0,1]")
            if used != declared:
                raise ValueError(f"L={lead_time}: declared parameters {sorted(declared)} != used {sorted(used)}")

    def complexity_stats(self, lead_times: tuple[int, ...] = ACTIVE_LEAD_TIMES) -> dict[str, object]:
        """Report both template and expanded complexity without hiding motifs.

        ``ast_check`` expands motif references when computing stats, so the
        expanded values include builtin/custom motif bodies and materialized
        feature bodies.  Parameter count remains the declared unique family
        parameter count and is checked against every expanded AST.
        """
        self.validate(lead_times)
        rows = []
        for lead_time in lead_times:
            ast, motifs = materialize_family(self, lead_time)
            checked = ast_check(ast, motifs=motifs, allowed_state_names={"x"} | {f"p{i}" for i in range(1, lead_time + 1)}, limits=ASTLimits(max_depth=16, max_nodes=256, max_parameters=128))
            expanded_nodes, expanded_depth, expanded_nonlinear, expanded_params = _expanded_stats(ast, motifs)
            rows.append({"lead_time": int(lead_time), "folded_nodes": checked.stats.nodes, "folded_depth": checked.stats.depth, "expanded_nodes": expanded_nodes, "expanded_depth": expanded_depth, "expanded_nonlinear_ops": expanded_nonlinear, "complexity": expanded_nodes + expanded_depth + 2.0 * len(expanded_params) + 1.5 * expanded_nonlinear, "parameter_count": len(expanded_params)})
        complexities = [float(row["complexity"]) for row in rows]
        template_ast, template_motifs = materialize_family(self, max(lead_times))
        template_checked = ast_check(template_ast, motifs=template_motifs, allowed_state_names={"x"} | {f"p{i}" for i in range(1, max(lead_times) + 1)}, limits=ASTLimits(max_depth=64, max_nodes=10000, max_parameters=128))
        return {"template_parameter_count": len(self.parameters), "expanded_parameter_count": max(int(row["parameter_count"]) for row in rows), "template_ast_nodes": template_checked.stats.nodes, "expanded": rows, "expanded_complexity_mean": sum(complexities) / len(complexities), "expanded_complexity_max": max(complexities)}


def validate_family_for_cases(
    family: PolicyFamily, cases: Sequence[InventoryCase]
) -> None:
    """Validate one family against every concrete inventory case."""

    if not cases:
        raise ValueError("family validation requires at least one case")
    declared = {parameter.name for parameter in family.parameters}
    if len(declared) != len(family.parameters):
        raise ValueError("family parameter names must be unique")
    by_name = {parameter.name: parameter for parameter in family.parameters}
    feature_names = {feature.name for feature in family.features}
    if len(feature_names) != len(family.features):
        raise ValueError("family feature names must be unique")
    for case in cases:
        ast, motifs = materialize_family_for_case(family, case)
        try:
            checked = ast_check(
                ast,
                motifs=motifs,
                allowed_state_names=case.schema.names,
                limits=EXECUTABLE_AST_LIMITS,
            )
            # Features inside custom motifs must use the same materialized
            # bodies as the executable AST, not the raw template vocabulary.
            for motif_name in family.motifs:
                ast_check(
                    motifs[motif_name],
                    motifs=motifs,
                    allowed_state_names=case.schema.names,
                    limits=EXECUTABLE_AST_LIMITS,
                )
        except Exception as exc:
            raise ValueError(f"context case {case.name}: {exc}") from exc
        from inventory.policy.ast import cap_parameter_modes
        for name in cap_parameter_modes(checked.ast, motifs):
            if by_name[name].kind != "cap" or by_name[name].lower < 0:
                raise ValueError("typed cap parameters must be declared kind=cap with nonnegative bounds")
        used = set(checked.stats.parameters)
        referenced = _feature_names(family.ast)
        for motif_ast in family.motifs.values():
            referenced.update(_feature_names(motif_ast))
        for feature in family.features:
            if feature.name not in referenced:
                continue
            used.update(feature.parameter_names)
            for name in feature.parameter_names:
                parameter = by_name.get(name)
                if (
                    parameter is None
                    or parameter.kind != "weight"
                    or parameter.lower < 0
                    or parameter.upper > 1
                ):
                    raise ValueError(
                        f"weight parameter {name!r} must be declared kind=weight "
                        "with bounds inside [0,1]"
                    )
        if used != declared:
            raise ValueError(
                f"context case {case.name}: declared parameters "
                f"{sorted(declared)} != used {sorted(used)}"
            )


def family_complexity_for_cases(
    family: PolicyFamily,
    contexts: Sequence[tuple[str, InventoryCase]],
) -> dict[str, object]:
    """Return executable complexity for an arbitrary set of case contexts."""

    validate_family_for_cases(family, [case for _, case in contexts])
    rows: list[dict[str, object]] = []
    for context_id, case in contexts:
        ast, motifs = materialize_family_for_case(family, case)
        checked = ast_check(
            ast,
            motifs=motifs,
            allowed_state_names=case.schema.names,
            limits=ASTLimits(max_depth=16, max_nodes=256, max_parameters=128),
        )
        nodes, depth, nonlinear, parameters = _expanded_stats(ast, motifs)
        rows.append({
            "context_id": str(context_id),
            "folded_nodes": checked.stats.nodes,
            "folded_depth": checked.stats.depth,
            "expanded_nodes": nodes,
            "expanded_depth": depth,
            "expanded_nonlinear_ops": nonlinear,
            "complexity": nodes + depth + 2.0 * len(parameters) + 1.5 * nonlinear,
            "parameter_count": len(parameters),
        })
    complexities = [float(row["complexity"]) for row in rows]
    return {
        "template_parameter_count": len(family.parameters),
        "expanded_parameter_count": max(int(row["parameter_count"]) for row in rows),
        "expanded": rows,
        "expanded_complexity_mean": sum(complexities) / len(complexities),
        "expanded_complexity_max": max(complexities),
    }


def _expand_semantic_motifs(node: AST, motifs: Mapping[str, AST], stack: tuple[str, ...] = ()) -> AST:
    """Inline motifs for executable, name-independent family comparison."""

    if node.get("op") == "motif":
        name = str(node["name"])
        if name in stack:
            raise ValueError(f"cyclic motif while canonicalizing family: {name}")
        if name not in motifs:
            raise ValueError(f"unknown motif while canonicalizing family: {name}")
        return _expand_semantic_motifs(motifs[name], motifs, stack + (name,))
    out: AST = {"op": str(node["op"])}
    for key, value in node.items():
        if key == "op":
            continue
        if isinstance(value, Mapping):
            out[key] = _expand_semantic_motifs(dict(value), motifs, stack)
        elif isinstance(value, list):
            out[key] = [
                _expand_semantic_motifs(dict(item), motifs, stack)
                if isinstance(item, Mapping) else item
                for item in value
            ]
        else:
            out[key] = value
    return out


def _parameter_descriptor(parameter: FamilyParameter) -> dict[str, object]:
    """Return the optimizer-relevant part of a parameter declaration."""

    return {
        "kind": parameter.kind,
        "lower": float(parameter.lower),
        "upper": float(parameter.upper),
        "index_rule": parameter.index_rule,
        "context_dependence": parameter.context_dependence,
    }


def _semantic_sort_key(node: AST, descriptors: Mapping[str, Mapping[str, object]]) -> str:
    """A parameter-name-free structural key used before alpha renaming."""

    op = str(node["op"])
    if op == "parameter":
        return json.dumps({"op": op, "descriptor": descriptors[str(node["name"])]}, sort_keys=True, separators=(",", ":"))
    payload: dict[str, object] = {"op": op}
    for key, value in node.items():
        if key == "op":
            continue
        if isinstance(value, Mapping):
            payload[key] = _semantic_sort_key(dict(value), descriptors)
        elif isinstance(value, list):
            payload[key] = [
                _semantic_sort_key(dict(item), descriptors) if isinstance(item, Mapping) else item
                for item in value
            ]
        else:
            payload[key] = value
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _semantic_alpha_ast(node: AST, descriptors: Mapping[str, Mapping[str, object]]) -> tuple[AST, tuple[dict[str, object], ...]]:
    """Canonicalize commutative operators and alpha-rename parameters.

    This deliberately compares the executable family rather than proposal
    labels.  Thus renaming ``K`` to ``K2`` or declaring an identical typed
    feature under a new name cannot create a new numerical candidate.
    """

    names: dict[str, str] = {}
    ordered_descriptors: list[dict[str, object]] = []

    def rename_parameter(name: str) -> AST:
        if name not in names:
            names[name] = f"p{len(names)}"
            ordered_descriptors.append(dict(descriptors[name]))
        return {"op": "parameter", "name": names[name]}

    def flatten(op: str, raw: AST) -> list[AST]:
        if raw.get("op") != op:
            return [raw]
        if "args" in raw:
            return [item for child in raw["args"] for item in flatten(op, child)]
        return flatten(op, raw["left"]) + flatten(op, raw["right"])

    def constant_value(raw: AST) -> float | None:
        if raw.get("op") != "const":
            return None
        value = raw.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)

    def walk(raw: AST) -> AST:
        op = str(raw["op"])
        if op == "const":
            value = constant_value(raw)
            if value is not None:
                return {"op": "const", "value": 0.0 if value == 0.0 else value}
        if op == "parameter":
            return rename_parameter(str(raw["name"]))
        if op in {"minimum", "maximum"}:
            operands = sorted(flatten(op, raw), key=lambda item: _semantic_sort_key(item, descriptors))
            normalized: list[AST] = []
            seen: set[str] = set()
            for operand in operands:
                item = walk(operand)
                item_key = json.dumps(item, sort_keys=True, separators=(",", ":"))
                if item_key in seen:
                    continue
                seen.add(item_key)
                normalized.append(item)
            if not normalized:
                raise ValueError(f"empty {op} while canonicalizing family")
            result = normalized[0]
            for operand in normalized[1:]:
                result = {"op": op, "left": result, "right": operand}
            return result
        if op in {"add", "mul"}:
            # Sort *before* alpha-renaming.  Renaming while traversing the
            # model's original operand order would make ``a+b`` and ``b+a``
            # receive different parameter labels and therefore different
            # hashes, defeating commutative semantic deduplication.
            raw_operands = sorted(flatten(op, raw), key=lambda item: _semantic_sort_key(item, descriptors))
            operands = [walk(operand) for operand in raw_operands]
            constants = [constant_value(operand) for operand in operands]
            non_constants = [operand for operand, value in zip(operands, constants) if value is None]
            numeric = [value for value in constants if value is not None]
            if op == "add":
                total = sum(numeric)
                if total != 0.0:
                    non_constants.append({"op": "const", "value": total})
                identity = {"op": "const", "value": 0.0}
            else:
                product = 1.0
                for value in numeric:
                    product *= value
                if product == 0.0:
                    return {"op": "const", "value": 0.0}
                if product != 1.0:
                    non_constants.append({"op": "const", "value": product})
                identity = {"op": "const", "value": 1.0}
            non_constants.sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
            if not non_constants:
                return identity
            if len(non_constants) == 1:
                return non_constants[0]
            return {"op": op, "args": non_constants}
        if op == "sub":
            left, right = walk(dict(raw["left"])), walk(dict(raw["right"]))
            if constant_value(right) == 0.0:
                return left
            return {"op": "sub", "left": left, "right": right}
        if op == "neg":
            argument = walk(dict(raw["arg"]))
            if argument.get("op") == "neg":
                return argument["arg"]
            return {"op": "neg", "arg": argument}
        out: AST = {"op": op}
        for key, value in raw.items():
            if key == "op":
                continue
            if isinstance(value, Mapping):
                out[key] = walk(dict(value))
            elif isinstance(value, list):
                values = [walk(dict(item)) if isinstance(item, Mapping) else item for item in value]
                out[key] = values
            else:
                out[key] = value
        return out

    return walk(node), tuple(ordered_descriptors)


def _strip_root_nonnegative_wrapper(ast: Mapping[str, Any]) -> dict[str, Any]:
    """Remove a root maximum whose constant arm is redundant after action clipping."""
    semantic_ast = dict(ast)
    if semantic_ast.get("op") != "maximum":
        return semantic_ast

    left = dict(semantic_ast["left"])
    right = dict(semantic_ast["right"])
    left_value = left.get("value") if left.get("op") == "const" else None
    right_value = right.get("value") if right.get("op") == "const" else None
    if isinstance(left_value, (int, float)) and not isinstance(left_value, bool) and float(left_value) <= 0.0:
        return right
    if isinstance(right_value, (int, float)) and not isinstance(right_value, bool) and float(right_value) <= 0.0:
        return left
    return semantic_ast


def semantic_family_payload(family: PolicyFamily, lead_times: Sequence[int] = ACTIVE_LEAD_TIMES) -> dict[str, object]:
    """Return a cross-L executable representation independent of proposal labels.

    The payload inlines motifs and typed features for every evaluated lead
    time, normalizes algebraic AST aliases, and alpha-renames parameters while
    retaining their optimizer domains.  It is intentionally separate from the
    task/cache hash: changing cache identity mid-run would invalidate durable
    results, whereas semantic identity is the correct key for deduplication.
    """

    leads = tuple(dict.fromkeys(int(lead) for lead in lead_times))
    family.validate(leads)
    descriptors = {parameter.name: _parameter_descriptor(parameter) for parameter in family.parameters}
    signatures: list[dict[str, object]] = []
    for lead_time in leads:
        ast, motifs = materialize_family(family, lead_time)
        executable = _expand_semantic_motifs(ast, motifs)
        checked = ast_check(
            executable,
            motifs={},
            allowed_state_names={"x"} | {f"p{i}" for i in range(1, lead_time + 1)},
            limits=ASTLimits(max_depth=64, max_nodes=10000, max_parameters=128),
        )
        semantic_ast = _strip_root_nonnegative_wrapper(checked.ast)
        canonical_ast, alpha_parameters = _semantic_alpha_ast(semantic_ast, descriptors)
        signatures.append({
            "lead_time": lead_time,
            "ast": canonical_ast,
            "parameters": list(alpha_parameters),
        })
    return {"schema_version": 2, "lead_time_signatures": signatures}


def semantic_family_hash(family: PolicyFamily, lead_times: Sequence[int] = ACTIVE_LEAD_TIMES) -> str:
    payload = semantic_family_payload(family, lead_times)
    return stable_key(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


def semantic_family_payload_for_cases(
    family: PolicyFamily,
    contexts: Sequence[tuple[str, InventoryCase]],
) -> dict[str, object]:
    """Return a component-agnostic executable family representation."""

    validate_family_for_cases(family, [case for _, case in contexts])
    descriptors = {
        parameter.name: _parameter_descriptor(parameter)
        for parameter in family.parameters
    }
    signatures: list[dict[str, object]] = []
    for context_id, case in contexts:
        ast, motifs = materialize_family_for_case(family, case)
        executable = _expand_semantic_motifs(ast, motifs)
        checked = ast_check(
            executable,
            motifs={},
            allowed_state_names=case.schema.names,
            limits=ASTLimits(max_depth=64, max_nodes=10000, max_parameters=128),
        )
        semantic_ast = _strip_root_nonnegative_wrapper(checked.ast)
        canonical_ast, alpha_parameters = _semantic_alpha_ast(
            semantic_ast, descriptors
        )
        signatures.append({
            "context_id": str(context_id),
            "model_id": case.model_id,
            "ast": canonical_ast,
            "parameters": list(alpha_parameters),
        })
    return {"schema_version": 3, "context_signatures": signatures}


def semantic_family_hash_for_cases(
    family: PolicyFamily,
    contexts: Sequence[tuple[str, InventoryCase]],
) -> str:
    payload = semantic_family_payload_for_cases(family, contexts)
    return stable_key(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
