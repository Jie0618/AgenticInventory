"""Stage-III ML-MD proposal protocol and strict proposal repair loop."""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Mapping
from typing import Sequence

from inventory.policy.family import FeatureSpec, PolicyFamily, FamilyParameter
from inventory.policy.family_domain import InventoryFamilyDomain
from sr.core.tree import contains_subtree, walk_nodes


FEATURE_INIT_VARIANTS_PER_FAMILY = 8
FEATURE_INIT_PROTOCOL = "curated-single-point-max8"

_FEATURE_INIT_TYPED_PRIORITY = (
    "tail_linear_half_normalized",
    "head_linear_half_normalized",
    "tail_sum_ceil_l_div_2",
    "head_sum_ceil_l_div_2",
)

_FEATURE_INIT_MOTIF_PRIORITY = (
    "effective_inventory_position",
    "effective_net_inventory_position",
    "net_inventory_position",
    "inventory_position",
    "decision_inventory",
    "inventory_age_moment",
    "decision_mean_remaining_life",
    "expiring_inventory",
    "imminent_abandonment_if_unserved",
    "backlog_age_moment",
    "oldest_backlog",
    "pipeline_imbalance",
    "pipeline_lag_moment",
    "far_pipeline",
    "near_pipeline",
    "demand_mean",
    "demand_cv",
    "cost_ratio",
)


def initial_population(seeds: Sequence[PolicyFamily] = ()) -> tuple[PolicyFamily, ...]:
    """Normalize experiment-provided seeds for the generic SR engine.

    The engine deliberately has no inventory policy seed of its own.  A seed
    is an experiment choice and must be supplied by the caller.
    """
    return tuple(seeds)


def feature_initialization_population(
    domain: InventoryFamilyDomain,
    parents: Sequence[PolicyFamily] | None = None,
    *,
    variants_per_family: int = FEATURE_INIT_VARIANTS_PER_FAMILY,
) -> tuple[PolicyFamily, ...]:
    """Create a small, deterministic set of common one-step variants per parent.

    Every supplied seed participates, but initialization is deliberately
    bounded: at most ``variants_per_family`` semantically distinct children
    are retained for each seed.  Candidates use a short shared priority list,
    alternate across eligible AST slots, and never combine multiple edits.
    This keeps iteration zero interpretable and linear in the seed count.
    """
    if int(variants_per_family) < 0:
        raise ValueError("variants_per_family must be nonnegative")
    base_parents = tuple(parents or ())
    core = domain.common_motif_names
    typed_by_name = domain.common_features

    typed = [
        typed_by_name[name]
        for name in _FEATURE_INIT_TYPED_PRIORITY
        if name in typed_by_name
    ]
    motifs = [name for name in _FEATURE_INIT_MOTIF_PRIORITY if name in core]
    def walk(node: Any, path: tuple[object, ...] = ()) -> list[tuple[list[object], Mapping[str, Any]]]:
        found: list[tuple[list[object], Mapping[str, Any]]] = []
        if isinstance(node, Mapping):
            if node.get("op") == "motif" and str(node.get("name")) in core:
                found.append((list(path), node))
            elif node.get("op") == "feature":
                found.append((list(path), node))
            for key, value in node.items():
                if isinstance(value, (Mapping, list)):
                    found.extend(walk(value, path + (key,)))
        elif isinstance(node, list):
            for i, value in enumerate(node):
                if isinstance(value, (Mapping, list)):
                    found.extend(walk(value, path + (i,)))
        return found
    out: list[PolicyFamily] = []
    for parent in base_parents:
        slots = walk(parent.ast)
        if not slots or variants_per_family == 0:
            continue
        candidates = [
            (feature.name, {"op": "feature", "name": feature.name}, (feature,))
            for feature in typed
        ] + [
            (name, {"op": "motif", "name": name}, ())
            for name in motifs
        ]
        parent_hash = domain.semantic_hash(parent)
        selected: dict[str, PolicyFamily] = {}
        for candidate_index, (name, fragment, features) in enumerate(candidates):
            start = candidate_index % len(slots)
            for offset in range(len(slots)):
                path, old = slots[(start + offset) % len(slots)]
                slot = "_".join(str(x) for x in path)
                row = {
                    "family_id": f"{parent.family_id}__init__slot_{slot}__{name}",
                    "parent_family_id": parent.family_id,
                    "action": "replace_subtree",
                    "path": path,
                    "old_subtree": old,
                    "fragment": fragment,
                    "new_parameters": [
                        {"name": parameter, "kind": "weight", "lower": 0.0, "upper": 1.0}
                        for feature in features
                        for parameter in feature.parameter_names
                    ],
                }
                try:
                    family = _apply_family_mutation(
                        parent, row, features, domain=domain
                    )
                except Exception:
                    continue
                semantic_hash = domain.semantic_hash(family)
                if semantic_hash == parent_hash or semantic_hash in selected:
                    continue
                selected[semantic_hash] = family
                break
            if len(selected) >= int(variants_per_family):
                break
        out.extend(selected.values())
    return tuple(out)


def feature_registry(
    domain: InventoryFamilyDomain,
    parents: Sequence[PolicyFamily],
) -> dict[str, object]:
    """Build the compact active registry shown to the next mutation call."""
    core = list(domain.common_motif_names)
    candidates: dict[str, dict[str, object]] = {}
    for family in parents:
        for feature in family.features:
            row = candidates.setdefault(feature.name, {"kind": feature.kind, "window": feature.window, "window_selected_by": "LLM_structure_search", "window_optimized": False, "definition": _feature_definition(feature), "weights": list(feature.weights), "weight_rule": feature.weight_rule, "used_by": []})
            row["used_by"] = sorted(set(row["used_by"]) | {family.family_id})
    defaults = {
        name: asdict(spec) for name, spec in domain.common_features.items()
    }
    return {
        "core": core,
        "core_definitions_by_context": domain.common_motifs,
        "default_typed_features": defaults,
        "candidates": candidates,
        "pruning": "only referenced candidates remain active; historical definitions stay in family archives",
    }


def _feature_definition(feature: FeatureSpec) -> str:
    return json.dumps(asdict(feature), sort_keys=True)


def build_family_prompt(
    parents: list[Mapping[str, Any]],
    *,
    domain: InventoryFamilyDomain,
) -> str:
    """Build a context-neutral family prompt from an inventory domain."""
    parent_context = []
    for parent in parents:
        # Scores are deliberately included as diagnostics, never as schema
        # inputs.  This gives the model an evolutionary signal while keeping
        # the family AST independent of any particular training case.
        parent_context.append({
            "family_id": str(parent.get("family_id", "")),
            "family_hash": str(parent.get("family_hash", "")),
            "description": str(parent.get("description", "")),
            "cross_context_score": parent.get("cross_context_score"),
            "status": parent.get("status"),
            "per_context": parent.get(
                "per_context", []
            ),
            "ast": parent.get("ast"),
            "motifs": parent.get("motifs", {}),
            "parameters": parent.get("parameters", []),
            "features": parent.get("features", []),
            "case_evaluations": parent.get("case_evaluations", []),
            "diagnostics": parent.get("diagnostics", {"available": False, "reason": "no_dp_reference"}),
        })
    performance_context = ""
    if any(
        row["cross_context_score"] is not None or row["per_context"]
        for row in parent_context
    ):
        performance_context = "PARENT PERFORMANCE CONTEXT (diagnostic only; never copy scores into the AST): " + json.dumps([
            {
                "family_id": row["family_id"],
                "cross_context_score": row["cross_context_score"],
                "status": row["status"],
                "per_context": row["per_context"],
            }
            for row in parent_context
        ], sort_keys=True) + "\n"
    return (
        "Propose an inventory policy family by applying ONE local structural mutation to a listed parent. Return a mutation object, not a complete replacement AST. "
        f"The same AST must validate in every context {list(domain.context_ids)} and use {domain.training_case_count} fixed training cases per context. "
        "Parameters are fitted independently for each (context, case); never branch on a context identifier.\n"
        "Search limits after motif expansion: at most 8 fitted parameters, depth 10, 64 nodes, complexity 150.\n"
        "For two-parent crossover use action=crossover, other_parent_family_hash, and combination=minimum or maximum; combine both complete parent expressions and retain compatible shared declarations.\n"
        "DOMAIN VOCABULARY (authoritative; do not invent state, motif, feature, distribution, or cost names): "
        + json.dumps(domain.prompt_payload(), sort_keys=True)
        + ("\nTYPED CAPS: cap={op:'cap',arg:expression,limit:{op:'parameter',name:'C'},mode:'integer'|'continuous'|'cyclic_integer'}. cyclic_integer emits round(C*(t+1))-round(C*t) with continuous C, global t starting at zero and ties-to-even rounding. Only the cap changes; no residual or final-order wrapper. Action set_cap_mode requires path (empty list for root) and mode; target an existing cap or minimum with one declared cap parameter. Caps may be internal; cap parameters cannot be reused outside cap limits. Never put cyclic caps in motifs/features. Ordinary initial seeds remain unchanged.\n" if domain.cyclic_cap else "")
        + "\nThe JSON schema is strict: optionally declare features using the exact registered feature definitions, then return one mutation with family_id, parent_family_hash, action, fragment/path, and optional new_parameters. family_id is only a display label; parent_family_hash is the authoritative parent identity. "
        "Parameters use objects with name, kind, lower, upper. AST operators are only const, "
        "parameter, feature, state, distribution, cost, motif, add, mul, neg, sub, absolute, minimum, "
        "maximum, indicator_gt, safe_ratio, and clip (plus typed cap when enabled above). Use exact fields: clip={op,arg,lower,upper}; "
        "sub/minimum/maximum/indicator_gt/safe_ratio={op,left,right}; add/mul={op,args}; "
        "neg/absolute={op,arg}; const={op,value}; parameter/state/distribution/motif={op,name}. "
        "Valid parameter kinds are target, scale, cap, weight, lag. Use registered motifs and features for context-varying summaries; do not shadow registered names or hard-code state coordinates that are absent from any context. "
        "Feature declarations use exactly {name,kind,window,weights,weight_rule} and must match a registered feature or be accepted by the inventory feature schema. "
        "Do not use ast_template, parameter_declarations, "
        "natural-language formulas, let bindings, reduce_sum, sqrt, max, clamp, or extra wrapper keys.\n"
        + performance_context
        + "PARENT AST CONTEXT (use these exact trees and canonical paths when an internal replacement is selected): "
        + json.dumps(parent_context, sort_keys=True)
        + "\nFor replace_subtree/replace_feature, path is a list of AST field tokens, e.g. [\"left\",\"args\",0,\"right\",\"args\",1,\"arg\"]. Include old_subtree exactly and replace only that node. "
        + "MUTATION CHOICE: choose the action for each proposal according to its scientific value and the selected parent's structure. Valid operations are replacement, additive, and safe deletion. Replacement should normally be the dominant strategy because it preserves simplicity and supports controlled local comparisons; in a typical batch, more than half of the mutations should use replace_subtree or replace_feature. This is a soft prior, not a quota: use a different mixture when the evaluation evidence or parent structures justify it. Give meaningful opportunities to add_term, minimum_term, maximum_term, add_cap, delete_term, and delete_wrapper, and do not choose replacement merely because its JSON schema is easier. For replacement actions, change exactly one internal subtree. For additive or wrapper actions, preserve the complete parent AST as a subtree and introduce exactly one simple structural layer. The fragment for add_term/minimum_term/maximum_term must contain ONLY that new term or wrapper threshold; never embed, copy, or repeat the parent AST inside the fragment. For deletion actions, remove exactly one eligible term or wrapper while keeping a valid nonempty AST. Use delete_term only on an add.args operand and delete_wrapper only for safe local simplification.\n"
        + "SEMANTIC NOVELTY: a proposal must change the executable policy family, not only its labels. Do not rename parameters, reorder declarations, rename/redeclare an already-equivalent typed feature, wrap a policy in an additional cap with the same optimizer domain, or submit an algebraically equivalent form. In particular, do not replace a parameter with a differently named parameter having the same kind and bounds. Replenishment actions are already made nonnegative after policy evaluation, so NEVER wrap the outermost policy AST in maximum(policy, 0) or maximum(0, policy): that wrapper is semantically ineffective and is removed by semantic hashing.\n"
        + "Use the parent AST, registry definitions, and evaluation evidence to make one concise structural mutation that tests a clear inventory-control hypothesis.\n"
        + "PARENT DIVERSIFICATION REQUIREMENT: when at least three distinct parents are listed, use at least three different parent_family_hash values in the mutation batch and do not use any single parent for more than half of the returned mutations. Include at least one mutation from a non-canonical parent whenever one is listed.\n"
    )


def family_from_json(
    payload: Mapping[str, Any],
    *,
    domain: InventoryFamilyDomain,
) -> PolicyFamily:
    """Validate and materialize one LLM JSON proposal."""
    if not isinstance(payload, Mapping):
        raise ValueError("proposal must be a JSON object")
    # Accept the canonical schema only.  Earlier models occasionally returned
    # a prose-oriented schema (family_name/ast_template/parameter_declarations)
    # that cannot be evaluated by the benchmark AST kernel.  Failing clearly
    # here prevents one malformed proposal from aborting the complete search.
    required = {"family_id", "ast", "parameters"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"proposal missing canonical keys: {', '.join(missing)}")
    if not isinstance(payload.get("ast"), Mapping):
        raise ValueError("proposal ast must be an AST object")
    params = tuple(FamilyParameter(
        _normalize_param_name(str(row["name"])), _normalize_kind(str(row.get("kind", "scale"))), float(row["lower"]), float(row["upper"]),
        str(row.get("index_rule", "scalar")), str(row.get("context_dependence", "independent_per_case")),
    ) for row in payload.get("parameters", ()))
    features = tuple(
        FeatureSpec(
            name=str(row["name"]), kind=str(row["kind"]), window=_parse_window(row.get("window", 1)),
            weights=tuple(float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else str(x) for x in row.get("weights", ())),
            weight_rule=None if row.get("weight_rule") is None else str(row["weight_rule"]),
        )
        for row in payload.get("features", ())
    )
    family = PolicyFamily(
        str(payload["family_id"]), _normalize_ast(payload["ast"]),
        {str(k): _normalize_ast(v) for k, v in dict(payload.get("motifs", {})).items()},
        params, str(payload.get("description", "")), features,
        None if payload.get("parent_family_id") is None else str(payload["parent_family_id"]),
        None if payload.get("mutation_action") is None else str(payload["mutation_action"]),
        None if payload.get("parent_family_hash") is None else str(payload["parent_family_hash"]),
        tuple(str(value) for value in payload.get("parent_family_hashes", ())),
    )
    domain.validate_family(family)
    return family


def _parse_feature_payload(raw: str) -> tuple[FeatureSpec, ...]:
    payload = json.loads(raw)
    # Some otherwise valid clients double-wrap the requested envelope as
    # {"features": {"features": [...]}}.  Unwrap only this harmless shape;
    # semantic validation still happens through FeatureSpec/PolicyFamily.
    if isinstance(payload, Mapping) and isinstance(payload.get("features"), Mapping):
        payload = payload["features"]
    rows = payload.get("features", []) if isinstance(payload, Mapping) else []
    # Invalid optional declarations are discarded independently so a valid
    # mutation elsewhere in the response can still be evaluated.
    parsed: list[FeatureSpec] = []
    for row in rows:
        try:
            parsed.append(FeatureSpec(
                name=str(row["name"]), kind=str(row["kind"]),
                window=_parse_window(row.get("window", 1)),
                weights=tuple(float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else str(x) for x in row.get("weights", ())),
                weight_rule=None if row.get("weight_rule") is None else str(row["weight_rule"]),
            ))
        except (KeyError, TypeError, ValueError):
            continue
    return tuple(parsed)


def _normalize_kind(kind: str) -> str:
    value = kind.strip().lower()
    return {"real": "scale", "continuous": "scale", "float": "scale"}.get(value, value)


def _parse_window(value: Any) -> object:
    if isinstance(value, (int, str)) and not isinstance(value, bool):
        return value
    if isinstance(value, Mapping):
        return dict(value)
    raise ValueError("feature window must be integer, string rule, or rule object")


def _normalize_param_name(name: str) -> str:
    return name.replace("[L,case]", "").replace("[L, case]", "")


def _normalize_ast(node: Any) -> Any:
    """Convert harmless LLM syntax variants to the canonical AST schema."""
    if not isinstance(node, Mapping):
        if isinstance(node, list):
            return [_normalize_ast(v) for v in node]
        return node
    op = str(node.get("op", ""))
    # OpenAI-compatible models occasionally serialize binary add/mul nodes
    # with the left/right fields used by minimum/sub.  Accept this variant
    # only when both operands are present and the canonical args field is not;
    # the resulting tree still goes through the normal strict validation.
    if op in {"add", "mul"} and "args" not in node and "left" in node and "right" in node:
        return {
            "op": op,
            "args": [_normalize_ast(node["left"]), _normalize_ast(node["right"])],
        }
    if op in {"sub", "minimum", "maximum", "indicator_gt", "safe_ratio"} and "args" in node and len(node.get("args", ())) == 2:
        args = list(node["args"])
        return {"op": op, "left": _normalize_ast(args[0]), "right": _normalize_ast(args[1])}
    if op == "clip" and "args" in node and "arg" not in node:
        args = list(node["args"])
        if len(args) == 2:
            return {"op": "clip", "arg": _normalize_ast(args[0]), "lower": {"op": "const", "value": 0}, "upper": _normalize_ast(args[1])}
        if len(args) == 3:
            return {"op": "clip", "arg": _normalize_ast(args[0]), "lower": _normalize_ast(args[1]), "upper": _normalize_ast(args[2])}
    if op == "clip":
        # Models often emit JSON numbers for clip bounds even though the
        # canonical AST grammar represents every operand as a node.  Convert
        # only these harmless literal bounds; all other malformed operands
        # remain validation errors and are sent through the refill loop.
        node = dict(node)
        for bound in ("lower", "upper"):
            if bound in node and isinstance(node[bound], (int, float)) and not isinstance(node[bound], bool):
                node[bound] = {"op": "const", "value": float(node[bound])}
    if op == "neg" and "args" in node and "arg" not in node and len(node["args"]) == 1:
        return {"op": "neg", "arg": _normalize_ast(node["args"][0])}
    out = {str(k): _normalize_ast(v) for k, v in node.items() if k != "args" or op in {"add", "mul"}}
    if op == "motif" and "args" in out:
        out.pop("args", None)
    if op in {"sub", "minimum", "maximum", "indicator_gt", "safe_ratio"} and "args" in node and "left" not in node and len(node["args"]) == 2:
        args = list(node["args"]); out = {"op": op, "left": _normalize_ast(args[0]), "right": _normalize_ast(args[1])}
    if out.get("op") == "parameter" and "name" in out:
        out["name"] = _normalize_param_name(str(out["name"]))
    return out


_FAMILY_MUTATION_ACTIONS = {
    "set_cap_mode",
    "add_term",
    "minimum_term",
    "maximum_term",
    "replace_subtree",
    "replace_feature",
    "add_cap",
    "delete_term",
    "delete_wrapper",
}


def _walk_ast(node: Any):
    """Local name for the shared AST walker."""
    yield from walk_nodes(node)


def _used_parameters(node: Any) -> set[str]:
    return {
        str(row.get("name"))
        for row in _walk_ast(node)
        if row.get("op") == "parameter" and row.get("name") is not None
    }


def _motif_names(node: Any) -> set[str]:
    return {
        str(row.get("name"))
        for row in _walk_ast(node)
        if row.get("op") == "motif" and row.get("name") is not None
    }


def _family_definition_key(
    family: PolicyFamily,
    domain: InventoryFamilyDomain,
) -> str:
    """Stable semantic key excluding proposal labels and alpha-renaming."""
    return domain.semantic_hash(family)


def _contains_exact_subtree(node: Any, target: Any) -> bool:
    return contains_subtree(node, target)


def _replace_path(node: Any, path: Sequence[Any], replacement: Any) -> Any:
    """Replace one AST child by positional path without guessing semantics."""
    if not path:
        return replacement
    if not path:
        return replacement
    # Canonical paths use AST field names and list indices.  Integer-only
    # paths remain accepted for archived proposals.
    if isinstance(path[0], str):
        key = path[0]
        if isinstance(node, dict) and key in node:
            copied = dict(node); copied[key] = _replace_path(node[key], path[1:], replacement); return copied
        if isinstance(node, list):
            try: idx = int(key)
            except ValueError: raise ValueError("list path token must be an integer")
            copied = list(node); copied[idx] = _replace_path(node[idx], path[1:], replacement); return copied
        raise ValueError("replace_subtree field path is invalid")
    if isinstance(node, list):
        idx = int(path[0]); copied = list(node); copied[idx] = _replace_path(node[idx], path[1:], replacement); return copied
    if not isinstance(node, dict):
        raise ValueError("replace_subtree path enters a non-object AST node")
    key_order = {
        "args": lambda i: ("args", i),
        "left": lambda i: ("left", None),
        "right": lambda i: ("right", None),
        "arg": lambda i: ("arg", None),
        "lower": lambda i: ("lower", None),
        "upper": lambda i: ("upper", None),
    }
    # Paths are encoded as child indices over the canonical AST traversal.
    children: list[tuple[str, int | None, Any]] = []
    for key in ("arg", "left", "right", "lower", "upper", "limit"):
        if key in node:
            children.append((key, None, node[key]))
    if "args" in node:
        children.extend(("args", i, value) for i, value in enumerate(node["args"]))
    index = int(path[0])
    if index < 0 or index >= len(children):
        raise ValueError("replace_subtree path is outside the AST")
    key, item, value = children[index]
    copied = dict(node)
    if item is None:
        copied[key] = _replace_path(value, path[1:], replacement)
    else:
        args = list(copied[key])
        args[item] = _replace_path(value, path[1:], replacement)
        copied[key] = args
    return copied


def _get_path(node: Any, path: Sequence[Any]) -> Any:
    cur = node
    for token in path:
        if isinstance(token, str) and isinstance(cur, dict):
            if token not in cur: raise ValueError("path field not found")
            cur = cur[token]
        elif isinstance(token, int) and isinstance(cur, list):
            cur = cur[token]
        elif isinstance(token, int) and isinstance(cur, dict):
            children = []
            for key in ("arg", "left", "right", "lower", "upper", "limit"):
                if key in cur: children.append(cur[key])
            if "args" in cur: children.extend(cur["args"])
            if token < 0 or token >= len(children):
                raise ValueError("path index outside AST")
            cur = children[token]
        else:
            raise ValueError("invalid canonical AST path")
    return cur


def _find_matching_paths(node: Any, target: Any, prefix: tuple[Any, ...] = ()) -> list[list[Any]]:
    """Find canonical field-token paths whose node equals target."""
    out: list[list[Any]] = []
    if node == target:
        out.append(list(prefix))
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "op":
                continue
            if isinstance(value, list):
                for i, child in enumerate(value):
                    out.extend(_find_matching_paths(child, target, prefix + (key, i)))
            elif isinstance(value, dict):
                out.extend(_find_matching_paths(value, target, prefix + (key,)))
    return out


def _resolve_old_subtree_path(node: Any, raw_path: Any, old_subtree: Any) -> list[Any]:
    """Validate a supplied path or recover it only when the target is unique."""
    if isinstance(raw_path, str):
        raw_path = [int(x) if x.isdigit() else x for x in raw_path.strip("/").replace(".", "/").split("/") if x]
    expected = _normalize_ast(old_subtree)
    try:
        if isinstance(raw_path, list) and _get_path(node, raw_path) == expected:
            return list(raw_path)
    except Exception:
        pass
    matches = _find_matching_paths(node, expected)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError("ambiguous old_subtree: matches multiple AST paths; valid path required")
    raise ValueError("old_subtree does not match target path")


def _apply_family_mutation(
    parent: PolicyFamily,
    row: Mapping[str, Any],
    features: tuple[FeatureSpec, ...],
    reflected_motifs: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    domain: InventoryFamilyDomain,
) -> PolicyFamily:
    """Apply exactly one local mutation to a validated parent family."""
    action = str(row.get("action", "")).strip().lower()
    # Common LLM aliases for the canonical local mutation vocabulary.
    action = {"insert_add_term": "add_term", "insert_addend": "add_term", "add": "add_term", "insert_minimum_term": "minimum_term", "insert_maximum_term": "maximum_term", "cap": "add_cap", "replacement": "replace_subtree", "replace": "replace_subtree", "delete": "delete_term", "remove_term": "delete_term", "remove_wrapper": "delete_wrapper", "simplify": "delete_wrapper"}.get(action, action)
    if action not in _FAMILY_MUTATION_ACTIONS:
        raise ValueError(f"unsupported family mutation action: {action}")
    # ``new_subtree`` is the natural spelling in the prompt/example; accept it
    # as an alias for the canonical internal field ``fragment``.
    fragment_raw = row.get("fragment", row.get("new_subtree"))
    # Some model responses wrap the replacement inside fragment as
    # {old_subtree: ..., new_subtree: ...}.  Unwrap that transport envelope;
    # old_subtree is still checked independently below.
    if isinstance(fragment_raw, Mapping) and "new_subtree" in fragment_raw and "op" not in fragment_raw:
        if "old_subtree" in fragment_raw and "old_subtree" not in row:
            row = dict(row)
            row["old_subtree"] = fragment_raw["old_subtree"]
        fragment_raw = fragment_raw["new_subtree"]
    if fragment_raw is not None and not isinstance(fragment_raw, Mapping):
        raise ValueError("mutation fragment must be an AST object")
    fragment = _normalize_ast(fragment_raw) if fragment_raw is not None else None
    ast = parent.ast
    base_ast = parent.ast
    if action == "set_cap_mode":
        if not domain.cyclic_cap:
            raise ValueError("typed cap search is not enabled")
        from inventory.policy.ast import cap, CAP_MODES
        path = row.get("path", [])
        if not isinstance(path, list):
            raise ValueError("set_cap_mode requires a list path (empty for root)")
        target = _get_path(parent.ast, path) if path else parent.ast
        mode = row.get("mode")
        if mode not in CAP_MODES:
            raise ValueError("set_cap_mode requires integer, continuous, or cyclic_integer")
        if target.get("op") == "cap":
            replacement = cap(target["arg"], target["limit"], mode)
        elif target.get("op") == "minimum":
            sides = [(target[a], target[b]) for a, b in (("left", "right"), ("right", "left"))]
            slots = [(limit, arg) for limit, arg in sides if limit.get("op") == "parameter" and any(p.name == limit["name"] and p.kind == "cap" for p in parent.parameters)]
            if len(slots) != 1:
                raise ValueError("set_cap_mode requires exactly one declared cap parameter")
            limit, arg = slots[0]
            replacement = cap(arg, limit, mode)
        else:
            raise ValueError("set_cap_mode target must be a cap")
        ast = _replace_path(parent.ast, path, replacement)
    elif action == "add_term":
        if fragment is None:
            raise ValueError("add_term requires fragment")
        if _contains_exact_subtree(fragment, parent.ast):
            raise ValueError("add_term fragment must contain only the new term, not a copy of the parent AST")
        ast = {"op": "add", "args": [base_ast, fragment]}
    elif action == "minimum_term":
        if fragment is None:
            raise ValueError("minimum_term requires fragment")
        if _contains_exact_subtree(fragment, parent.ast):
            raise ValueError("minimum_term fragment must contain only the new threshold, not a copy of the parent AST")
        ast = {"op": "minimum", "left": base_ast, "right": fragment}
    elif action == "maximum_term":
        if fragment is None:
            raise ValueError("maximum_term requires fragment")
        if _contains_exact_subtree(fragment, parent.ast):
            raise ValueError("maximum_term fragment must contain only the new threshold, not a copy of the parent AST")
        ast = {"op": "maximum", "left": base_ast, "right": fragment}
    elif action == "add_cap":
        raw_caps = row.get("new_parameters", row.get("parameters", ())) or ()
        inferred_cap = next((r.get("name") for r in raw_caps if isinstance(r, Mapping) and str(r.get("kind", "")).lower() == "cap"), None)
        cap_name = _normalize_param_name(str(row.get("cap_parameter") or inferred_cap or "C"))
        ast = {"op": "minimum", "left": base_ast, "right": {"op": "parameter", "name": cap_name}}
    elif action in {"replace_subtree", "replace_feature"}:
        if fragment is None:
            raise ValueError(f"{action} requires fragment")
        raw_path = row.get("path")
        if isinstance(raw_path, str):
            raw_path = [int(x) if x.isdigit() else x for x in raw_path.strip("/").replace(".", "/").split("/") if x]
        if not isinstance(raw_path, list) or not raw_path:
            raise ValueError(f"{action} requires a nonempty integer path")
        if "old_subtree" in row:
            raw_path = _resolve_old_subtree_path(parent.ast, raw_path, row["old_subtree"])
        if "old_subtree" in row and _get_path(parent.ast, raw_path) != _normalize_ast(row["old_subtree"]):
            raise ValueError("old_subtree does not match target path")
        if action == "replace_feature":
            probe = _get_path(parent.ast, raw_path)
            if not isinstance(probe, dict) or probe.get("op") != "feature":
                action = "replace_subtree"
            old = _replace_path(parent.ast, raw_path, fragment)
            ast = old
        else:
            ast = _replace_path(parent.ast, raw_path, fragment)
    elif action in {"delete_term", "delete_wrapper"}:
        raw_path = row.get("path")
        if isinstance(raw_path, str):
            raw_path = [int(x) if x.isdigit() else x for x in raw_path.strip("/").replace(".", "/").split("/") if x]
        if not isinstance(raw_path, list) or (not raw_path and action == "delete_term"):
            raise ValueError(f"{action} requires a nonempty path")
        if "old_subtree" in row:
            raw_path = _resolve_old_subtree_path(parent.ast, raw_path, row["old_subtree"])
        target = _get_path(parent.ast, raw_path) if raw_path else parent.ast
        if action == "delete_term":
            if len(raw_path) < 2 or raw_path[-2] != "args":
                raise ValueError("delete_term path must select an operand of add.args")
            parent_path = raw_path[:-2]
            add_node = _get_path(parent.ast, parent_path)
            if not isinstance(add_node, Mapping) or add_node.get("op") != "add" or not isinstance(add_node.get("args"), list):
                raise ValueError("delete_term target parent must be an add node")
            args = list(add_node["args"]); idx = int(raw_path[-1])
            if idx < 0 or idx >= len(args) or len(args) <= 1:
                raise ValueError("delete_term would create an empty add expression")
            replacement = args[1 - idx] if len(args) == 2 else {"op": "add", "args": args[:idx] + args[idx+1:]}
            ast = _replace_path(parent.ast, parent_path, replacement)
        else:
            op = target.get("op") if isinstance(target, Mapping) else None
            if op in {"neg", "absolute", "clip", "cap"} and "arg" in target:
                replacement = target["arg"]
            elif op in {"minimum", "maximum", "indicator_gt", "safe_ratio", "sub"} and "left" in target:
                replacement = target["left"]
            else:
                raise ValueError("delete_wrapper supports only safe unary/binary wrappers")
            ast = _replace_path(parent.ast, raw_path, replacement)
    new_params = list(parent.parameters)
    # Some otherwise schema-valid model responses use the family-level
    # ``parameters`` spelling inside a mutation.  Treat it as an alias only
    # for newly introduced parameters; the parent schema is still preserved.
    raw_new_parameters = row.get("new_parameters", row.get("parameters", ()))
    if raw_new_parameters is None:
        raw_new_parameters = ()
    for raw in raw_new_parameters:
        if not isinstance(raw, Mapping):
            raise ValueError("new_parameters entries must be objects")
        if not isinstance(raw, Mapping) or "name" not in raw or "lower" not in raw or "upper" not in raw:
            raise ValueError("new parameter must declare name, lower, and upper")
        new_params.append(FamilyParameter(
            _normalize_param_name(str(raw["name"])), _normalize_kind(str(raw.get("kind", "scale"))),
            float(raw["lower"]), float(raw["upper"]), str(raw.get("index_rule", "scalar")),
            str(raw.get("context_dependence", "independent_per_case")),
        ))
    used = _used_parameters(ast)
    by_name = {p.name: p for p in new_params}
    missing = used - set(by_name)
    if missing:
        raise ValueError("mutation uses undeclared parameters: " + ", ".join(sorted(missing)))
    family_id = str(row.get("family_id", "")).strip()
    if not family_id:
        raise ValueError("mutation requires family_id")
    description = str(row.get("description") or row.get("rationale") or "")
    # Keep only declarations that are actually referenced by the resulting
    # AST.  Carrying every batch-level feature into every child polluted the
    # registry and made semantically identical families look different.
    def used_feature_names(node: Any) -> set[str]:
        if isinstance(node, dict):
            found = {str(node["name"])} if node.get("op") == "feature" and node.get("name") is not None else set()
            for value in node.values():
                found |= used_feature_names(value)
            return found
        if isinstance(node, list):
            found: set[str] = set()
            for value in node:
                found |= used_feature_names(value)
            return found
        return set()
    child_motifs = dict(parent.motifs)
    available_reflections = reflected_motifs or {}
    referenced_motifs = _motif_names(ast)
    pending = list(referenced_motifs & set(available_reflections))
    while pending:
        motif_name = pending.pop()
        if motif_name in child_motifs:
            reflected_ast = _normalize_ast(available_reflections[motif_name]["ast"])
            if _normalize_ast(child_motifs[motif_name]) != reflected_ast:
                raise ValueError(f"reflected motif name collision: {motif_name}")
            continue
        definition = available_reflections[motif_name]
        motif_ast = _normalize_ast(definition["ast"])
        child_motifs[motif_name] = motif_ast
        pending.extend((_motif_names(motif_ast) & set(available_reflections)) - set(child_motifs))
    referenced = used_feature_names(ast)
    for motif_ast in child_motifs.values():
        referenced |= used_feature_names(motif_ast)
    merged_features: list[FeatureSpec] = [f for f in parent.features if f.name in referenced]
    known_features = {feature.name for feature in merged_features}
    for feature in features:
        if feature.name in referenced and feature.name not in known_features:
            merged_features.append(feature)
            known_features.add(feature.name)
    # Weighted feature coefficients may themselves be parameter names.  They
    # must be present in the family parameter schema just like explicit AST
    # parameter nodes.
    feature_parameter_names = {name for feature in merged_features for name in feature.parameter_names}
    missing_feature_params = feature_parameter_names - set(by_name)
    if missing_feature_params:
        raise ValueError("weighted feature uses undeclared parameters: " + ", ".join(sorted(missing_feature_params)))
    params = tuple(by_name[name] for name in sorted(used | feature_parameter_names))
    family = PolicyFamily(
        family_id, ast, child_motifs, params, description, tuple(merged_features),
        parent.family_id, action, domain.semantic_hash(parent),
    )
    # Validate at the mutation boundary, not only after the whole batch has
    # been assembled.  Invalid model rows are then discarded and replaced by
    # the bounded refill loop instead of poisoning a generation.
    from .validation import validate_search_family
    validate_search_family(family, domain)
    if domain.semantic_hash(family) == domain.semantic_hash(parent):
        raise ValueError("mutation is semantically identical to its parent")
    return family


def _apply_family_crossover(parent, other, row, *, domain):
    """Combine two complete expressions and preserve both lineage identities."""
    if domain.semantic_hash(parent) == domain.semantic_hash(other):
        raise ValueError("crossover requires two distinct parents")
    mode = str(row.get("combination", "minimum"))
    if mode not in {"minimum", "maximum"}:
        raise ValueError("crossover combination must be minimum or maximum")
    def merge(left, right, label):
        output = dict(left)
        for name, value in right.items():
            if name in output and output[name] != value:
                raise ValueError(f"crossover has conflicting {label}: {name}")
            output[name] = value
        return output
    params = merge({p.name: p for p in parent.parameters},
                   {p.name: p for p in other.parameters}, "parameter")
    motifs = merge(parent.motifs, other.motifs, "motif")
    features = merge({f.name: f for f in parent.features},
                     {f.name: f for f in other.features}, "summary")
    family = PolicyFamily(
        str(row["family_id"]), {"op": mode, "left": parent.ast, "right": other.ast},
        motifs, tuple(params.values()), str(row.get("rationale", "")), tuple(features.values()),
        parent.family_id, "crossover", domain.semantic_hash(parent),
        (domain.semantic_hash(parent), domain.semantic_hash(other)),
    )
    from .validation import validate_search_family
    validate_search_family(family, domain)
    return family


def llm_template_initialization(
    client: Any,
    parents: Sequence[PolicyFamily],
    *,
    domain: InventoryFamilyDomain,
    count: int = 6,
    existing_templates: Sequence[PolicyFamily] = (),
    previous_rejections: Sequence[Mapping[str, Any]] = (),
) -> tuple[tuple[PolicyFamily, ...], dict[str, Any]]:
    """Generate executable summary templates from supplied Stage-II seeds.

    Invalid children are rejected independently. Transport/schema failure
    returns no candidates; the initialization coordinator retries or stops.
    """
    if count < 1:
        raise ValueError("template initialization count must be positive")
    from .template_init import PROTOCOL
    parent_by_hash = {domain.semantic_hash(parent): parent for parent in parents}
    audit: dict[str, Any] = {"protocol": PROTOCOL,
                             "requested_count": count, "accepted_count": 0,
                             "rejected": []}
    if not parents:
        return (), {**audit, "status": "no_parents"}
    prompt = (
        "Initialize portable inventory policy summary templates from the supplied rule seeds. "
        "These templates form the Stage-III initial population; the supplied seeds "
        "provide structural guidance and are not retained automatically. "
        "Generalize seed state summaries into context-dependent, executable typed features "
        "or registered motifs. Preserve a clear parent control hypothesis and prefer its "
        "parameter declarations and formula skeleton. Do not merely rename labels. "
        "For a single seed, propose five or six distinct structural generalizations: "
        "vary executable state summaries, windows or weighted aggregates while retaining "
        "the seed control hypothesis. Parameter-value changes alone are not templates. "
        "The same family must validate in EVERY listed context. Never branch on context IDs "
        "or hard-code unavailable state coordinates. Parameters are fitted per case.\n"
        "Return only JSON {\"families\":[...]}, at most " + str(count) + " families. "
        "Each complete family object requires family_id, parent_family_hash (exactly one "
        "supplied hash), ast, parameters, and optionally motifs, features, description. "
        "Parameters use {name,kind,lower,upper}; kinds: target,scale,cap,weight,lag. "
        "Features use {name,kind,window,weights,weight_rule} accepted by the registry/schema. "
        "AST operators: const={op,value}; parameter/state/distribution/cost/motif/feature={op,name}; "
        "add/mul={op,args}; neg/absolute={op,arg}; "
        "sub/minimum/maximum/indicator_gt/safe_ratio={op,left,right}; clip={op,arg,lower,upper}. "
        + ("Typed cap={op:'cap',arg:expression,limit:{op:'parameter',name:'C'},"
           "mode:'integer'|'continuous'|'cyclic_integer'}. cyclic_integer means "
           "round(C*(t+1))-round(C*t), global t=0 and ties-to-even. Cap parameters may "
           "only occur in cap limits; no cyclic wrappers or cyclic features. " if domain.cyclic_cap else "")
        + "\nDOMAIN: " + json.dumps(domain.prompt_payload(), sort_keys=True)
        + "\nFEATURE REGISTRY: " + json.dumps(feature_registry(domain, parents), sort_keys=True)
        + "\nRULE SEEDS: " + json.dumps([
            {**asdict(parent), "family_hash": key} for key, parent in parent_by_hash.items()
        ], sort_keys=True)
    )
    if existing_templates:
        prompt += ("\nALREADY ACCEPTED TEMPLATES: " + json.dumps([
            asdict(family) for family in existing_templates], sort_keys=True)
            + "\nPropose different executable summary structures; do not repeat these templates.")
    if previous_rejections:
        prompt += ("\nPREVIOUS VALIDATION ERRORS (correct these; return complete family objects, "
                   "including parameters, not patches): " + json.dumps(list(previous_rejections)[-20:]))
    try:
        envelope = json.loads(client.complete(prompt))
        if not isinstance(envelope, Mapping) or not isinstance(envelope.get("families"), list):
            raise ValueError("initialization requires a families array")
        rows = envelope["families"]
    except Exception as exc:
        return (), {**audit, "status": "failed", "error": str(exc)}
    seen = set(parent_by_hash)
    accepted = []
    for index, row in enumerate(rows[:count]):
        try:
            if not isinstance(row, Mapping):
                raise ValueError("family must be an object")
            parent = parent_by_hash.get(str(row.get("parent_family_hash", "")))
            if parent is None:
                raise ValueError("unknown parent_family_hash")
            family = family_from_json({**row, "parent_family_id": parent.family_id,
                                       "mutation_action": "template_initialization"}, domain=domain)
            from .validation import validate_search_family
            validate_search_family(family, domain)
            key = domain.semantic_hash(family)
            if key in seen:
                raise ValueError("duplicate executable family")
            seen.add(key)
            accepted.append(family)
        except Exception as exc:
            audit["rejected"].append({"index": index, "reason": str(exc)})
    audit.update(status="complete", accepted_count=len(accepted),
                 returned_count=len(rows), ignored_excess=max(0, len(rows) - count))
    return tuple(accepted), audit


def llm_proposal_batch_fn(
    client: Any,
    parents: Sequence[PolicyFamily],
    iteration: int,
    reflection_client: Any | None = None,
    count: int = 20,
    parent_evaluations: Mapping[str, Mapping[str, Any]] | None = None,
    reflected_motifs: Mapping[str, Mapping[str, Any]] | None = None,
    known_definition_keys: Sequence[str] = (),
    *,
    domain: InventoryFamilyDomain,
) -> tuple[PolicyFamily, ...]:
    """Generate a bounded batch of distinct families.

    Models sometimes return fewer objects than requested (or a single object
    despite the batch schema).  The initial call and every repair call request
    up to ``count`` mutations. Valid results are merged and the accepted list
    is capped at ``count``. At most ten repair calls are made, stopping early
    once the batch is full; a zero-valid initial response follows the same
    bounded repair budget.
    """
    parent_rows = []
    for p in parents:
        parent_hash = domain.semantic_hash(p)
        row = {"family_id": p.family_id, "family_hash": parent_hash, "description": p.description}
        if parent_evaluations and parent_hash in parent_evaluations:
            row.update(dict(parent_evaluations[parent_hash]))
        elif parent_evaluations and p.family_id in parent_evaluations:
            row.update(dict(parent_evaluations[p.family_id]))
        parent_rows.append(row)
    base = build_family_prompt(parent_rows, domain=domain) + "\nACTIVE_FEATURE_REGISTRY:\n" + json.dumps(feature_registry(domain, parents), sort_keys=True)
    from .diagnostics import SHARED_DIAGNOSTICS
    if parent_evaluations and SHARED_DIAGNOSTICS in parent_evaluations:
        base += "\nSHARED PARENT-ACTION DIAGNOSTICS:\n" + json.dumps(
            parent_evaluations[SHARED_DIAGNOSTICS], sort_keys=True)
    if reflected_motifs:
        base += "\nVALIDATED_REFLECTION_MOTIFS (optional, reusable):\n" + json.dumps(dict(reflected_motifs), sort_keys=True)

    # One-call production protocol: feature declarations and all mutations are
    # returned in the same envelope.  This matches Stage I/II and avoids an
    # unbounded sequence of slow LLM refill calls per iteration.
    single_prompt = base + f"\nReturn exactly one JSON object with optional features and exactly up to {int(count)} mutations: {{\"features\":[...],\"mutations\":[...]}}. Return at least {int(count)} distinct valid mutations when possible. Features are structural LLM choices; mutations must use canonical actions add_term, minimum_term, maximum_term, add_cap, replace_subtree, replace_feature, delete_term, delete_wrapper. Do not return prose."
    if domain.cyclic_cap:
        single_prompt += " Also allow set_cap_mode, as described above."
    # Keep Stage 3 local, while leaving operation choice to the model.
    # Iteration zero evaluates the LLM templates derived from Stage-II seeds;
    # no per-operation quota is imposed on later mutation batches.
    if int(count) >= 8:
        single_prompt += (
            "\nTYPED FEATURES: if you introduce a new typed feature, declare it exactly once in features and reference its declared name in the mutation fragment. "
            "Never reference historical aliases unless you also declare them canonically; every proposal must still validate for every L."
        )
    used_legacy_second = False

    def mutation_rows(payload: Any) -> list[Any]:
        """Read the canonical envelope while tolerating a bare mutation array."""

        if isinstance(payload, list):
            return payload
        if not isinstance(payload, Mapping):
            return []
        rows = payload.get("mutations", payload.get("families", []))
        return rows if isinstance(rows, list) else []

    try:
        envelope = json.loads(client.complete(single_prompt))
        # Alternate-format repair for clients that
        # return features and mutations in separate responses. Production prompts request
        # the combined envelope, so compliant iterations use exactly one call.
        if isinstance(envelope, Mapping) and "mutations" not in envelope and "families" not in envelope:
            used_legacy_second = True
            second = json.loads(client.complete(base + f"\nReturn exactly JSON {{\"mutations\":[...]}} with up to {int(count)} valid one-step mutations using canonical actions."))
            if isinstance(second, Mapping):
                envelope = {"features": envelope.get("features", []), **second}
        features = _parse_feature_payload(json.dumps(envelope))
    except Exception:
        # Treat transport/schema failure as an empty response so the bounded
        # LLM rescue loop below can retry with the same parent context.
        envelope = {"features": [], "mutations": []}
        features = ()
        rows = []
    feature_rows = [{"name": f.name, "kind": f.kind, "window": f.window, "weights": list(f.weights), "weight_rule": f.weight_rule, "definition": _feature_definition(f)} for f in features]
    builtin_features = set(feature_registry(domain, parents).get("core", ()))
    parent_by_hash = {domain.semantic_hash(p): p for p in parents}
    parents_by_id: dict[str, list[PolicyFamily]] = {}
    for parent in parents:
        parents_by_id.setdefault(parent.family_id, []).append(parent)

    def resolve_parent(row: Mapping[str, Any]) -> PolicyFamily | None:
        parent_hash = str(row.get("parent_family_hash", ""))
        if parent_hash:
            return parent_by_hash.get(parent_hash)
        # Compatibility for archived clients. A display ID is accepted only
        # when it resolves unambiguously; duplicate labels are never guessed.
        matches = parents_by_id.get(str(row.get("parent_family_id", "")), [])
        return matches[0] if len(matches) == 1 else None

    reflected_feature_rows = []
    for definition in (reflected_motifs or {}).values():
        if isinstance(definition, Mapping):
            reflected_feature_rows.extend(definition.get("features", ()))
    reflected_features = tuple(
        FeatureSpec(
            name=str(row["name"]), kind=str(row["kind"]), window=_parse_window(row.get("window", 1)),
            weights=tuple(row.get("weights", ())),
            weight_rule=None if row.get("weight_rule") is None else str(row["weight_rule"]),
        )
        for row in reflected_feature_rows
        if isinstance(row, Mapping)
    )
    features_by_name = {feature.name: feature for feature in (*features, *reflected_features)}
    features = tuple(features_by_name.values())
    accepted: list[PolicyFamily] = []
    seen = {p.family_id for p in parents}
    # The controller supplies semantic keys from the complete archive and
    # population, not just the handful of parents shown in this prompt.
    seen_definition_keys = {
        _family_definition_key(p, domain) for p in parents
    } | {str(key) for key in known_definition_keys}
    rows = mutation_rows(envelope)
    def rewrite_builtin_feature_aliases(node: Any) -> Any:
        if isinstance(node, list): return [rewrite_builtin_feature_aliases(x) for x in node]
        if not isinstance(node, Mapping): return node
        out = {k: rewrite_builtin_feature_aliases(v) for k, v in node.items()}
        if out.get("op") == "feature" and str(out.get("name")) in builtin_features:
            return {"op": "motif", "name": str(out["name"])}
        return out
    if not isinstance(rows, list):
        return ()
    additive_actions = {"add_term", "minimum_term", "maximum_term", "add_cap"}
    replacement_actions = {"replace_subtree", "replace_feature"}
    deletion_actions = {"delete_term", "delete_wrapper"}
    additive_count = 0
    replacement_count = 0
    rejection_reasons: list[str] = []

    def accept_rows(rows: Any) -> None:
      nonlocal additive_count, replacement_count
      if not isinstance(rows, list):
        return
      for row in rows:
        if len(accepted) >= int(count) or not isinstance(row, Mapping):
            break
        try:
            row = dict(row)
            if "fragment" in row:
                row["fragment"] = rewrite_builtin_feature_aliases(row["fragment"])
            if "new_subtree" in row:
                row["new_subtree"] = rewrite_builtin_feature_aliases(row["new_subtree"])
            normalized_action = {"insert_add_term": "add_term", "insert_addend": "add_term", "add": "add_term", "insert_minimum_term": "minimum_term", "insert_maximum_term": "maximum_term", "cap": "add_cap"}.get(str(row.get("action", "")).strip().lower(), str(row.get("action", "")).strip().lower())
            # LLMs sometimes use replace_feature for a built-in motif target.
            # If the selected node is not a feature node, preserve the
            # intended local replacement rather than rejecting a valid AST
            # mutation solely due to the action label.
            if normalized_action == "replace_feature" and isinstance(row.get("path"), list):
                try:
                    target = _get_path(resolve_parent(row).ast, row["path"])
                    if isinstance(target, Mapping) and target.get("op") != "feature":
                        normalized_action = "replace_subtree"
                        row["action"] = "replace_subtree"
                except Exception:
                    pass
            # Some responses put the canonical path/old/new fields inside
            # ``fragment`` rather than at the mutation top level.
            if isinstance(row.get("fragment"), Mapping) and "new_subtree" in row["fragment"] and "path" in row["fragment"]:
                wrapped = dict(row["fragment"])
                row["path"] = wrapped["path"]
                row["old_subtree"] = wrapped.get("old_subtree", row.get("old_subtree"))
                row["fragment"] = wrapped["new_subtree"]
            if isinstance(row.get("fragment"), Mapping) and "new_subtree" in row["fragment"] and "path" not in row["fragment"] and "op" not in row["fragment"]:
                wrapped = row["fragment"]
                if "old_subtree" in wrapped and "old_subtree" not in row:
                    row["old_subtree"] = wrapped["old_subtree"]
                row["fragment"] = wrapped["new_subtree"]
            # Recover common off-by-one/alternate-root paths by locating the
            # explicitly supplied old_subtree in the parent AST.
            parent_for_path = resolve_parent(row)
            if parent_for_path is not None and isinstance(row.get("old_subtree"), Mapping):
                row["path"] = _resolve_old_subtree_path(parent_for_path.ast, row.get("path", []), row["old_subtree"])
            # Re-evaluate replace_feature after path recovery; the corrected
            # target may be a motif node and should then use subtree semantics.
            if normalized_action == "replace_feature" and parent_for_path is not None:
                try:
                    target = _get_path(parent_for_path.ast, row.get("path", []))
                    if isinstance(target, Mapping) and target.get("op") != "feature":
                        normalized_action = "replace_subtree"
                        row["action"] = "replace_subtree"
                except Exception:
                    pass
            parent = resolve_parent(row)
            if parent is None:
                if "ast" not in row or "parameters" not in row:
                    continue
                payload = dict(row); payload["features"] = feature_rows
                family = family_from_json(payload, domain=domain)
            elif normalized_action == "crossover":
                other = resolve_parent({"parent_family_hash": row.get("other_parent_family_hash")})
                if other is None:
                    raise ValueError("crossover requires other_parent_family_hash")
                family = _apply_family_crossover(parent, other, row, domain=domain)
            else:
                family = _apply_family_mutation(
                    parent,
                    row,
                    features,
                    reflected_motifs,
                    domain=domain,
                )
            if family.family_id in seen:
                continue
            definition_key = _family_definition_key(family, domain)
            if parent is not None and definition_key in seen_definition_keys:
                continue
            # Reflection is only used for complete-family responses;
            # mutation envelopes are already structurally constrained and are
            # intentionally kept to one generation call in production.
            if parent is None and reflection_client is not None and (int(iteration) + 1) % 5 == 0:
                review = json.loads(reflection_client.complete(
                    'Review this validated cross-context policy family. Return JSON {"accept":true|false,"reason":"..."}. '
                    + json.dumps({"family_id": family.family_id, "description": family.description,
                                  "ast": family.ast, "motifs": family.motifs,
                                  "parameters": [asdict(p) for p in family.parameters],
                                  "features": [asdict(f) for f in family.features]}, sort_keys=True)))
                if isinstance(review, Mapping) and review.get("accept") is False:
                    continue
            accepted.append(family); seen.add(family.family_id)
            if parent is not None:
                seen_definition_keys.add(definition_key)
            if family.mutation_action in additive_actions:
                additive_count += 1
            elif family.mutation_action in replacement_actions:
                replacement_count += 1
            elif family.mutation_action in deletion_actions:
                pass
        except Exception as exc:
            rejection_reasons.append(f"{row.get('family_id', '<missing>')}: {exc}")
            continue
    accept_rows(rows)
    # Never conceal a weak LLM response by filling it with the deterministic
    # iter0 replacement grid. Later batches receive at most ten repair calls;
    # every call requests a full batch of up to ``count`` mutations and the
    # accepted results are merged and capped at ``count``.
    if accepted and used_legacy_second:
        # Archived two-stage clients have already spent their repair call on
        # the Stage-B request.  Returning here keeps compatibility tests and
        # old runners at two calls while compliant production envelopes still
        # receive the single semantic repair pass below.
        return tuple(accepted)
    if accepted:
        for repair_attempt in range(1, 11):
            if len(accepted) >= int(count):
                break
            try:
                repair = json.loads(client.complete(
                    base + f"\nREPAIR PASS {repair_attempt}/10: the current response has {len(accepted)} semantically novel valid mutations; produce exactly {int(count)} additional semantically novel local mutations when possible (otherwise return every valid one). "
                    "Use the same top-level envelope as the initial call: return exactly one JSON object {\"features\":[],\"mutations\":[...]}; do not return a bare array. "
                    "Do not repeat accepted family IDs, executable definitions, parameter renames, feature aliases, or equivalent cap wrappers. Previous rejection reasons: "
                    + json.dumps(rejection_reasons[-20:]) + "\nAlready accepted IDs: " + json.dumps(sorted(seen))
                ))
                accept_rows(mutation_rows(repair))
            except Exception:
                pass
        return tuple(accepted)
    # A malformed/fully redundant first response gets a bounded rescue loop.
    # Each rescue has the full batch budget; stop immediately once at least one
    # valid mutation is recovered, while preventing an unbounded API loop.
    for rescue_attempt in range(1, 11):
        if accepted:
            break
        try:
            refill = json.loads(client.complete(
                base + f"\nRESCUE PASS {rescue_attempt}/10: return exactly {int(count)} semantically novel mutations when possible (otherwise return every valid one). "
                "Use the same top-level envelope as the initial call: return exactly one JSON object {\"features\":[],\"mutations\":[...]}; do not return a bare array. "
                "Do not use parameter renames, typed-feature aliases, or redundant cap wrappers. "
                "For add/minimum/maximum actions, fragment contains only the new local term."
            ))
            accept_rows(mutation_rows(refill))
        except Exception:
            continue
    return tuple(accepted)


def llm_reflect_motifs(
    client: Any,
    families: Sequence[PolicyFamily],
    iteration: int,
    *,
    domain: InventoryFamilyDomain,
    parent_evaluations: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the periodic Stage-III reflection pass.

    Reflection is intentionally separate from per-iteration mutation.  Every
    fifth completed iteration, the model may extract reusable, context-aware
    motif definitions from the evaluated archive.  Definitions are validated
    against all target contexts before being returned; invalid or duplicate motifs
    are discarded and never silently inserted into a family.
    """
    recent = tuple(families)
    from .diagnostics import SHARED_DIAGNOSTICS
    context = [{"family_id": f.family_id, "family_hash": domain.semantic_hash(f),
                "ast": f.ast, "motifs": f.motifs, "parameters": [asdict(p) for p in f.parameters],
                "features": [asdict(x) for x in f.features],
                "evaluation": dict((parent_evaluations or {}).get(domain.semantic_hash(f), {}))}
               for f in recent]
    prompt = (
        "You are the periodic motif-reflection pass for an inventory policy-family search. "
        "Return exactly JSON {\"motifs\":[{\"name\":string,\"ast\":AST,\"description\":string}]}. "
        "A feature is a low-level domain quantity; a motif is a higher-level reusable AST that may combine multiple features, built-in motifs, distribution leaves, cost leaves, and arithmetic operators. "
        "Use {\"op\":\"feature\",\"name\":...} for feature references and {\"op\":\"motif\",\"name\":...} only for built-in motif references. Never relabel a feature as a motif. "
        "Extract at most two parameter-free subtrees of the selected parents. Use their fitted parameters, costs and diagnostic evidence to justify each choice. Do not reference family-private custom motifs or parameters; every dependency must be a built-in motif or a feature declaration present in these parents. "
        "Do not emit raw context-specific state coordinates, context branches, or built-in motif names. Return an empty list if nothing is reusable. "
        "Validate conceptually for every context in DOMAIN. DOMAIN:\n"
        + json.dumps(domain.prompt_payload(), sort_keys=True)
        + "\nSELECTED PARENTS AND EVALUATION RECORDS:\n" + json.dumps(context, sort_keys=True)
        + "\nSHARED PARENT-ACTION DIAGNOSTICS:\n"
        + json.dumps((parent_evaluations or {}).get(SHARED_DIAGNOSTICS, {}), sort_keys=True)
    )
    try:
        raw = json.loads(client.complete(prompt))
    except Exception:
        return {"iteration": int(iteration), "accepted": [], "rejected": ["reflection_call_failed"]}
    feature_definitions: dict[str, dict[str, FeatureSpec]] = {}
    for family in recent:
        for feature in family.features:
            signature = json.dumps(asdict(feature), sort_keys=True, separators=(",", ":"))
            feature_definitions.setdefault(feature.name, {})[signature] = feature

    accepted: list[dict[str, Any]] = []; rejected: list[str] = []
    accepted_names: set[str] = set()
    for row in raw.get("motifs", []) if isinstance(raw, Mapping) else []:
        if len(accepted) >= 2:
            break
        try:
            name = str(row["name"]).strip(); ast = _normalize_ast(row["ast"])
            if not name or name in accepted_names:
                raise ValueError("empty or duplicate reflected motif name")
            reserved_names = set(domain.common_motif_names) | set(domain.common_features)
            if name in reserved_names:
                raise ValueError("built-in motif name")
            if _used_parameters(ast):
                raise ValueError("reflected motifs must be parameter-free")
            from sr.core.grammar import expand_motifs
            from sr.core.tree import contains_subtree
            if not any(contains_subtree(expand_motifs(parent.ast, {**next(iter(domain.common_motifs.values())), **parent.motifs}),
                                       expand_motifs(ast, {**next(iter(domain.common_motifs.values())), **parent.motifs}))
                       for parent in recent):
                raise ValueError("reflected motif must be a subtree of a selected parent")
            non_builtin_motifs = _motif_names(ast) - set(domain.common_motif_names)
            if non_builtin_motifs:
                raise ValueError("family-private motif dependency: " + ", ".join(sorted(non_builtin_motifs)))

            dependencies: list[FeatureSpec] = []
            for feature_name in sorted({
                str(node["name"]) for node in _walk_ast(ast)
                if node.get("op") == "feature" and node.get("name") is not None
            } - set(domain.common_features)):
                definitions = feature_definitions.get(feature_name, {})
                if not definitions:
                    raise ValueError(f"unknown feature: {feature_name}")
                if len(definitions) != 1:
                    raise ValueError(f"ambiguous feature definition: {feature_name}")
                dependencies.append(next(iter(definitions.values())))

            # PolicyFamily owns the actual namespace/materialization contract:
            # feature leaves are expanded first for each L, then the high-level
            # motif is checked as an ordinary reusable AST.
            probe = PolicyFamily(
                f"reflection_probe_{name}", ast, {}, (), features=tuple(dependencies),
            )
            domain.validate_family(probe)
            accepted.append({
                "name": name, "ast": ast, "description": str(row.get("description", "")),
                "features": [asdict(feature) for feature in dependencies],
            })
            accepted_names.add(name)
        except Exception as exc:
            rejected.append(str(exc)[:300])
    return {"iteration": int(iteration), "accepted": accepted, "rejected": rejected}
