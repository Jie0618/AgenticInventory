from __future__ import annotations

from runtime.artifacts import stable_key
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Collection, Mapping, Sequence, Union

import numpy as np


AST = dict[str, Any]
State = Union[Mapping[str, Any], Sequence[float], int, float]


class ASTValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ASTLimits:
    max_depth: int = 10
    max_nodes: int = 256
    max_parameters: int = 64
    max_abs_value: float = 1.0e9


# One shared contract for every policy that can reach an evaluator.  Family
# proposal validation and optimization must use this exact object; otherwise
# a structurally valid proposal can be accepted by Stage 3 and fail only after
# its numerical tasks have been launched.
EXECUTABLE_AST_LIMITS = ASTLimits()


@dataclass(frozen=True)
class ASTStats:
    nodes: int
    depth: int
    parameters: tuple[str, ...]
    nonlinear_ops: int

    @property
    def complexity(self) -> float:
        return self.nodes + self.depth + 2.0 * len(self.parameters) + 1.5 * self.nonlinear_ops


@dataclass(frozen=True)
class ASTCheck:
    ast: AST
    ast_hash: str
    stats: ASTStats
    output_type: str


OPERATOR_SCHEMA: dict[str, tuple[str, ...]] = {
    "const": ("op", "value"),
    "parameter": ("op", "name"),
    "state": ("op", "name"),
    "distribution": ("op", "name"),
    "cost": ("op", "name"),
    "motif": ("op", "name"),
    "add": ("op", "args"),
    "mul": ("op", "args"),
    "neg": ("op", "arg"),
    "sub": ("op", "left", "right"),
    "absolute": ("op", "arg"),
    "minimum": ("op", "left", "right"),
    "maximum": ("op", "left", "right"),
    "indicator_gt": ("op", "left", "right"),
    "safe_ratio": ("op", "left", "right"),
    "clip": ("op", "arg", "lower", "upper"),
    "cap": ("op", "arg", "limit", "mode"),
}

NONLINEAR_OPS = {
    "cap",
    "absolute",
    "minimum",
    "maximum",
    "indicator_gt",
    "safe_ratio",
    "clip",
}
COMPLEXITY_PENALIZED_OPS = NONLINEAR_OPS
# Probability summaries only; support endpoints are numerical demand metadata.
DISTRIBUTION_FEATURES = {
    "mu",
    "sigma",
    "variance",
    "cv",
    "p_zero",
}
COST_FEATURES = {
    "h",
    "b",
    "w",
    "holding",
    "shortage",
    "expiry",
    "backlog",
    "abandonment",
    "ordering",
    "critical_fractile",
    "cost_ratio",
}


def const(value: float) -> AST:
    return {"op": "const", "value": float(value)}


def parameter(name: str) -> AST:
    return {"op": "parameter", "name": str(name)}


def state(name: str) -> AST:
    return {"op": "state", "name": str(name)}


def dist(name: str) -> AST:
    return {"op": "distribution", "name": str(name)}


def cost(name: str) -> AST:
    return {"op": "cost", "name": str(name)}


def motif(name: str) -> AST:
    return {"op": "motif", "name": str(name)}


def add(*args: AST) -> AST:
    return {"op": "add", "args": list(args)}


def mul(*args: AST) -> AST:
    return {"op": "mul", "args": list(args)}


def neg(arg: AST) -> AST:
    return {"op": "neg", "arg": arg}


def sub(left: AST, right: AST) -> AST:
    return {"op": "sub", "left": left, "right": right}


def absolute(arg: AST) -> AST:
    return {"op": "absolute", "arg": arg}


def minimum(left: AST, right: AST) -> AST:
    return {"op": "minimum", "left": left, "right": right}


def maximum(left: AST, right: AST) -> AST:
    return {"op": "maximum", "left": left, "right": right}


def indicator_gt(left: AST, right: AST) -> AST:
    """Return the scalar indicator ``1{left > right}``."""

    return {"op": "indicator_gt", "left": left, "right": right}


def safe_ratio(left: AST, right: AST) -> AST:
    return {"op": "safe_ratio", "left": left, "right": right}


def clip(arg: AST, lower: AST, upper: AST) -> AST:
    return {"op": "clip", "arg": arg, "lower": lower, "upper": upper}


CAP_MODES = frozenset({"integer", "continuous", "cyclic_integer"})


def cap(arg: AST, limit: AST, mode: str = "integer") -> AST:
    """Cap an expression; cyclic_integer changes only the cap constant."""
    return {"op": "cap", "arg": arg, "limit": limit, "mode": mode}


def cap_parameter_modes(ast: AST, motifs: Mapping[str, AST] | None = None) -> dict[str, str]:
    modes: dict[str, str] = {}
    def visit(node):
        if isinstance(node, dict):
            if node.get("op") == "cap":
                name, mode = node["limit"]["name"], node["mode"]
                if name in modes and modes[name] != mode:
                    raise ASTValidationError(f"conflicting cap modes for {name}")
                modes[name] = mode
            if node.get("op") == "motif" and node["name"] in (motifs or {}):
                visit((motifs or {})[node["name"]])
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)
    visit(ast)
    return modes


def has_cyclic_cap(ast: AST, motifs: Mapping[str, AST] | None = None) -> bool:
    return "cyclic_integer" in cap_parameter_modes(ast, motifs).values()


def ast_check(
    ast: AST,
    motifs: Mapping[str, AST] | None = None,
    limits: ASTLimits = EXECUTABLE_AST_LIMITS,
    allowed_state_names: Collection[str] | None = None,
) -> ASTCheck:
    motif_map = {} if motifs is None else dict(motifs)
    allowed = None if allowed_state_names is None else frozenset(str(name) for name in allowed_state_names)
    if allowed is not None and (not allowed or any(not name for name in allowed)):
        raise ASTValidationError("allowed_state_names must contain nonempty names")
    _validate(ast, motif_map, (), allowed, root=True)
    modes = cap_parameter_modes(ast, motif_map)
    def check_uses(node, limit=False):
        if isinstance(node, dict):
            if node.get("op") == "parameter" and node["name"] in modes and not limit:
                raise ASTValidationError("typed cap parameters may only occur as cap limits")
            if node.get("op") == "motif":
                check_uses(motif_map[node["name"]])
            for key, value in node.items():
                check_uses(value, node.get("op") == "cap" and key == "limit")
        elif isinstance(node, list):
            for value in node:
                check_uses(value)
    check_uses(ast)
    canonical = _strip_root_nonnegative_wrapper(_canonicalize(ast))
    stats = _stats(canonical, motif_map, ())
    if stats.depth > limits.max_depth:
        raise ASTValidationError(f"AST depth {stats.depth} exceeds {limits.max_depth}")
    if stats.nodes > limits.max_nodes:
        raise ASTValidationError(f"AST nodes {stats.nodes} exceeds {limits.max_nodes}")
    if len(stats.parameters) > limits.max_parameters:
        raise ASTValidationError(
            f"AST parameters {len(stats.parameters)} exceeds {limits.max_parameters}"
        )
    return ASTCheck(canonical, ast_hash(canonical), stats, "Scalar")


def _strip_root_nonnegative_wrapper(ast: AST) -> AST:
    """Remove a root lower bound already enforced by the policy boundary.

    This is deliberately a policy-root rule.  A nested ``maximum`` remains part
    of the formula because removing it can change later arithmetic.
    """

    if ast.get("op") != "maximum":
        return ast
    left = ast["left"]
    right = ast["right"]
    if left.get("op") == "const" and float(left["value"]) <= 0.0:
        return right
    if right.get("op") == "const" and float(right["value"]) <= 0.0:
        return left
    return ast


def order_ast_fields(node: AST) -> AST:
    op = _op(node)
    ordered: AST = {}
    for field in OPERATOR_SCHEMA[op]:
        value = node[field]
        if isinstance(value, dict):
            value = order_ast_fields(value)
        elif isinstance(value, list):
            value = [order_ast_fields(item) for item in value]
        ordered[field] = value
    return ordered


def evaluate_ast(
    node: AST,
    state_value: State,
    params: Mapping[str, float] | None = None,
    dist_features: Mapping[str, Any] | None = None,
    cost_features: Mapping[str, float] | None = None,
    motifs: Mapping[str, AST] | None = None,
    _local_vars: Mapping[str, float] | None = None,
) -> float:
    params = {} if params is None else params
    dist_features = {} if dist_features is None else dist_features
    cost_features = {} if cost_features is None else cost_features
    motifs = {} if motifs is None else motifs
    state_map = _state_mapping(state_value)
    op = _op(node)
    if op == "cap":
        limit = float(params[node["limit"]["name"]])
        if node["mode"] == "integer":
            limit = round(limit)
        elif node["mode"] == "cyclic_integer":
            if "time" not in state_map:
                raise ASTValidationError("cyclic cap requires explicit time starting at zero")
            t = float(state_map["time"])
            limit = round(limit * (t + 1)) - round(limit * t)
        return min(evaluate_ast(node["arg"], state_map, params, dist_features, cost_features, motifs), limit)
    if op == "const":
        return float(node["value"])
    if op == "parameter":
        return float(params[str(node["name"])])
    if op == "state":
        return _state_feature(str(node["name"]), state_map)
    if op == "distribution":
        return _distribution_feature(str(node["name"]), dist_features)
    if op == "cost":
        return _cost_feature(str(node["name"]), cost_features)
    if op == "motif":
        return evaluate_ast(motifs[str(node["name"])], state_map, params, dist_features, cost_features, motifs)
    if op == "add":
        return sum(evaluate_ast(x, state_map, params, dist_features, cost_features, motifs) for x in node["args"])
    if op == "mul":
        return math.prod(evaluate_ast(x, state_map, params, dist_features, cost_features, motifs) for x in node["args"])
    if op == "neg":
        return -evaluate_ast(node["arg"], state_map, params, dist_features, cost_features, motifs)
    if op == "sub":
        return evaluate_ast(node["left"], state_map, params, dist_features, cost_features, motifs) - evaluate_ast(
            node["right"], state_map, params, dist_features, cost_features, motifs
        )
    if op == "absolute":
        return abs(evaluate_ast(node["arg"], state_map, params, dist_features, cost_features, motifs))
    if op in {"minimum", "maximum", "safe_ratio"}:
        left = evaluate_ast(node["left"], state_map, params, dist_features, cost_features, motifs)
        right = evaluate_ast(node["right"], state_map, params, dist_features, cost_features, motifs)
        if op == "minimum":
            return min(left, right)
        if op == "maximum":
            return max(left, right)
        return left / (1.0 + abs(right))
    if op == "indicator_gt":
        return float(
            evaluate_ast(node["left"], state_map, params, dist_features, cost_features, motifs)
            > evaluate_ast(node["right"], state_map, params, dist_features, cost_features, motifs)
        )
    if op == "clip":
        value = evaluate_ast(node["arg"], state_map, params, dist_features, cost_features, motifs)
        lower = evaluate_ast(node["lower"], state_map, params, dist_features, cost_features, motifs)
        upper = evaluate_ast(node["upper"], state_map, params, dist_features, cost_features, motifs)
        return min(max(value, lower), upper)
    raise ASTValidationError(f"unknown SOF operator: {op}")


def compile_policy(
    ast: AST,
    params: Mapping[str, float],
    motifs: Mapping[str, AST] | None = None,
    rounding: str = "round",
    action_cap: int | None = None,
    allowed_state_names: Collection[str] | None = None,
) -> Callable[[State, Mapping[str, Any], Mapping[str, float]], int]:
    checked = ast_check(ast, motifs=motifs, allowed_state_names=allowed_state_names)

    def policy(
        state_value: State,
        dist_features: Mapping[str, Any],
        cost_features: Mapping[str, float],
    ) -> int:
        value = evaluate_ast(
            checked.ast, state_value, params, dist_features, cost_features, motifs
        )
        value = max(float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)), 0.0)
        action = math.floor(value) if rounding == "floor" else math.ceil(value) if rounding == "ceil" else round(value)
        if action_cap is not None:
            action = min(action, int(action_cap))
        return max(int(action), 0)

    return policy


def ast_hash(node: AST) -> str:
    canonical = _strip_root_nonnegative_wrapper(_canonicalize(node))
    return stable_key(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode())


def ast_to_formula(node: AST) -> str:
    op = _op(node)
    if op == "cap":
        name = ast_to_formula(node["limit"])
        limit = f"(round({name}*(t+1))-round({name}*t))" if node["mode"] == "cyclic_integer" else name
        return f"cap[{node['mode']}]({ast_to_formula(node['arg'])}, {limit})"
    if op == "const":
        return f"{float(node['value']):g}"
    if op in {"parameter", "state", "distribution", "cost", "motif"}:
        return str(node["name"])
    if op == "add":
        return "(" + " + ".join(ast_to_formula(x) for x in node["args"]) + ")"
    if op == "mul":
        return "(" + " * ".join(ast_to_formula(x) for x in node["args"]) + ")"
    if op == "neg":
        return f"(-{ast_to_formula(node['arg'])})"
    if op == "sub":
        return f"({ast_to_formula(node['left'])} - {ast_to_formula(node['right'])})"
    if op == "absolute":
        return f"abs({ast_to_formula(node['arg'])})"
    if op in {"minimum", "maximum"}:
        return f"{op}({ast_to_formula(node['left'])}, {ast_to_formula(node['right'])})"
    if op == "indicator_gt":
        return f"1{{{ast_to_formula(node['left'])} > {ast_to_formula(node['right'])}}}"
    if op == "safe_ratio":
        return f"({ast_to_formula(node['left'])}/(1+abs({ast_to_formula(node['right'])})))"
    if op == "clip":
        return f"clip({ast_to_formula(node['arg'])}, {ast_to_formula(node['lower'])}, {ast_to_formula(node['upper'])})"
    raise ASTValidationError(f"unknown SOF operator: {op}")


def _validate(
    node: Any,
    motifs: Mapping[str, AST],
    stack: tuple[str, ...],
    allowed_state_names: frozenset[str] | None,
    *,
    root: bool = False,
) -> None:
    if not isinstance(node, dict):
        raise ASTValidationError("AST node must be a dictionary")
    op = _op(node)
    expected = set(OPERATOR_SCHEMA[op])
    if set(node) != expected:
        raise ASTValidationError(f"{op} expects fields {sorted(expected)}, got {sorted(node)}")
    if op == "cap":
        if node["mode"] not in CAP_MODES:
            raise ASTValidationError("unknown cap mode")
        if not isinstance(node["limit"], dict) or node["limit"].get("op") != "parameter":
            raise ASTValidationError("cap limit must be a direct parameter")
        if stack and node["mode"] == "cyclic_integer":
            raise ASTValidationError("cyclic caps cannot be hidden inside motifs")
        _validate(node["arg"], motifs, stack, allowed_state_names)
        _validate(node["limit"], motifs, stack, allowed_state_names)
    elif op == "const":
        if not math.isfinite(float(node["value"])):
            raise ASTValidationError("const must be finite")
    elif op in {"parameter", "state", "distribution", "cost", "motif"}:
        name = str(node["name"])
        if not name:
            raise ASTValidationError(f"{op} name must be nonempty")
        if op == "state":
            if allowed_state_names is None:
                known = name in {"x", "remaining_horizon"} or re.fullmatch(
                    r"p[1-9][0-9]*", name
                )
            else:
                known = name in allowed_state_names
            if not known:
                raise ASTValidationError(f"unknown state feature: {name}")
        if op == "distribution" and name not in DISTRIBUTION_FEATURES:
            raise ASTValidationError(f"unknown distribution summary: {name}")
        if op == "cost" and name not in COST_FEATURES:
            raise ASTValidationError(f"unknown cost feature: {name}")
        if op == "motif":
            if name not in motifs:
                raise ASTValidationError(f"unknown motif: {name}")
            if name in stack:
                raise ASTValidationError(f"cyclic motif: {name}")
            _validate(motifs[name], motifs, stack + (name,), allowed_state_names)
    elif op in {"add", "mul"}:
        if not isinstance(node["args"], list) or len(node["args"]) < 2:
            raise ASTValidationError(f"{op} expects at least two args")
        for child in node["args"]:
            _validate(child, motifs, stack, allowed_state_names)
    else:
        for field in OPERATOR_SCHEMA[op][1:]:
            _validate(node[field], motifs, stack, allowed_state_names)


def _canonicalize(node: AST) -> AST:
    op = _op(node)
    if op == "cap":
        return cap(_canonicalize(node["arg"]), _canonicalize(node["limit"]), node["mode"])
    if op == "sub":
        return _canonicalize(add(node["left"], neg(node["right"])))
    if op in {"const", "parameter", "state", "distribution", "cost", "motif"}:
        return order_ast_fields(dict(node))
    if op in {"add", "mul"}:
        children = [_canonicalize(child) for child in node["args"]]
        flattened: list[AST] = []
        for child in children:
            flattened.extend(child["args"] if child["op"] == op else [child])
        if op == "add":
            simplified = _simplify_add(flattened)
        else:
            simplified = _simplify_mul(flattened)
            simplified.sort(key=_ast_sort_key)
        if len(simplified) == 1:
            return simplified[0]
        return order_ast_fields({"op": op, "args": simplified})
    if op == "neg":
        child = _canonicalize(node["arg"])
        if child["op"] == "const":
            return const(-float(child["value"]))
        if child["op"] == "neg":
            return child["arg"]
        return order_ast_fields({"op": "neg", "arg": child})
    if op == "absolute":
        child = _canonicalize(node["arg"])
        if child["op"] == "const":
            return const(abs(float(child["value"])))
        if child["op"] == "absolute":
            return child
        if child["op"] == "neg":
            child = child["arg"]
        return order_ast_fields({"op": "absolute", "arg": child})
    if op in {"minimum", "maximum"}:
        left = _canonicalize(node["left"])
        right = _canonicalize(node["right"])
        if left == right:
            return left
        if left["op"] == right["op"] == "const":
            values = (float(left["value"]), float(right["value"]))
            return const(min(values) if op == "minimum" else max(values))
        if _ast_sort_key(right) < _ast_sort_key(left):
            left, right = right, left
        return order_ast_fields({"op": op, "left": left, "right": right})
    canonical = {"op": op}
    for field in OPERATOR_SCHEMA[op][1:]:
        canonical[field] = _canonicalize(node[field])
    return order_ast_fields(canonical)


def _simplify_add(children: Sequence[AST]) -> list[AST]:
    """Apply cheap exact identities to one already-flattened addition."""

    positive: dict[str, list[AST]] = {}
    negative: dict[str, list[AST]] = {}
    for child in children:
        if child["op"] == "const" and float(child["value"]) == 0.0:
            continue
        if child["op"] == "neg":
            target = child["arg"]
            negative.setdefault(_ast_sort_key(target), []).append(child)
        else:
            positive.setdefault(_ast_sort_key(child), []).append(child)

    # Exact structural inverse pairs cancel without invoking a symbolic
    # algebra engine. Multiplicity is retained, e.g. x + x - x -> x.
    for key in positive.keys() & negative.keys():
        count = min(len(positive[key]), len(negative[key]))
        if count:
            del positive[key][-count:]
            del negative[key][-count:]

    output = [
        child
        for _key, group in sorted(
            (*positive.items(), *negative.items()), key=lambda item: item[0]
        )
        for child in group
    ]
    return output or [const(0.0)]


def _simplify_mul(children: Sequence[AST]) -> list[AST]:
    """Apply cheap exact identities to one already-flattened product."""

    output: list[AST] = []
    for child in children:
        if child["op"] == "const":
            value = float(child["value"])
            if value == 0.0:
                return [const(0.0)]
            if value == 1.0:
                continue
        output.append(child)
    return output or [const(1.0)]


def _ast_sort_key(node: AST) -> str:
    return json.dumps(node, sort_keys=True, separators=(",", ":"))


def _stats(node: AST, motifs: Mapping[str, AST], stack: tuple[str, ...]) -> ASTStats:
    op = _op(node)
    if op == "motif":
        name = str(node["name"])
        child = _stats(motifs[name], motifs, stack + (name,))
        # A declared motif is one reusable operational building block. Its
        # parameters still belong to the formula, but repeated references do
        # not repeatedly charge the expanded implementation tree.
        return ASTStats(1, 1, child.parameters, 0)
    children: list[AST] = []
    for field in OPERATOR_SCHEMA[op][1:]:
        value = node[field]
        children.extend(value if isinstance(value, list) else [value] if isinstance(value, dict) else [])
    child_stats = [_stats(child, motifs, stack) for child in children]
    parameters = {str(node["name"])} if op == "parameter" else set()
    for child in child_stats:
        parameters.update(child.parameters)
    return ASTStats(
        1 + sum(child.nodes for child in child_stats),
        1 + max((child.depth for child in child_stats), default=0),
        tuple(sorted(parameters)),
        int(op in NONLINEAR_OPS) + sum(child.nonlinear_ops for child in child_stats),
    )


def _state_mapping(value: State) -> dict[str, float]:
    if isinstance(value, Mapping):
        result = {"x": float(value.get("x", 0.0))}
        pipeline = value.get("p", [])
        for index, item in enumerate(pipeline, start=1):
            result[f"p{index}"] = float(item)
        for name, item in value.items():
            if name != "p":
                result[str(name)] = float(item)
        return result
    if isinstance(value, (int, float)):
        return {"x": float(value)}
    values = list(value)
    return {"x": float(values[0]), **{f"p{i}": float(v) for i, v in enumerate(values[1:], 1)}}


def _state_feature(name: str, values: Mapping[str, float]) -> float:
    if name not in values:
        raise ASTValidationError(f"state feature missing at evaluation: {name}")
    return float(values[name])


def _distribution_feature(name: str, features: Mapping[str, Any]) -> float:
    if name not in DISTRIBUTION_FEATURES:
        raise ASTValidationError(f"unknown distribution summary: {name}")
    if name in features:
        return float(features[name])
    values = np.asarray(features.get("values", []), dtype=float)
    probs = np.asarray(features.get("probs", []), dtype=float)
    if values.size == 0 or probs.size != values.size or probs.sum() <= 0:
        raise ASTValidationError(f"distribution summary missing: {name}")
    probs = probs / probs.sum()
    mean = float(np.dot(values, probs))
    variance = float(np.dot((values - mean) ** 2, probs))
    derived = {
        "mu": mean,
        "variance": variance,
        "sigma": math.sqrt(max(variance, 0.0)),
        "cv": math.sqrt(max(variance, 0.0)) / mean if mean > 0 else 0.0,
        "p_zero": float(probs[values == 0].sum()),
    }
    return derived[name]


def _cost_feature(name: str, features: Mapping[str, float]) -> float:
    if name in features:
        return float(features[name])
    h, b = float(features["h"]), float(features["b"])
    if name == "critical_fractile":
        return b / (b + h)
    if name == "cost_ratio":
        return b / h
    raise ASTValidationError(f"cost feature missing: {name}")


def _op(node: Any) -> str:
    if not isinstance(node, dict) or not isinstance(node.get("op"), str):
        raise ASTValidationError("AST node must contain string op")
    op = str(node["op"])
    if op not in OPERATOR_SCHEMA:
        raise ASTValidationError(f"operator is outside Simple Operational Formula grammar: {op}")
    return op
