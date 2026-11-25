"""Exact structural reductions for lead-time vector base-stock policies."""

from __future__ import annotations

from functools import lru_cache
from typing import Mapping, Sequence

import numpy as np

from inventory.policy.ast import AST


VBS_MONOTONE_GROUP = "vector_base_stock_targets"


def is_vbs_parameterization(specs: Sequence[object]) -> bool:
    """Return whether ``specs`` are exactly one ordered VBS target vector."""

    if len(specs) < 2:
        return False
    indexes: list[int] = []
    for spec in specs:
        if getattr(spec, "monotone_group", None) != VBS_MONOTONE_GROUP:
            return False
        index = getattr(spec, "monotone_index", None)
        name = str(getattr(spec, "name", ""))
        if index is None or name != f"S{int(index)}":
            return False
        indexes.append(int(index))
    return sorted(indexes) == list(range(len(specs)))


def detect_vbs_ast(ast: AST, lead_time: int) -> bool:
    """Recognize the registered ``[min_l(S_l-u_l)]^+`` VBS structure."""

    root = _positive_part_child(ast)
    if root is None:
        return False
    gaps = _flatten_minimum(root)
    parsed = [_parse_gap(node) for node in gaps]
    if any(item is None for item in parsed):
        return False
    return sorted(item for item in parsed if item is not None) == list(
        range(int(lead_time) + 1)
    )


def effective_vbs_constraint_indexes(
    params: Mapping[str, float],
    lead_time: int,
    n: int,
) -> tuple[int, ...]:
    """Return constraints that can attain the VBS pointwise minimum.

    Write ``g_i = S_i - u_i`` and, for ``i < j``, note that
    ``u_j-u_i`` lies in ``[0, (j-i)n]`` on the finite state grid.

    * If ``S_j <= S_i``, then ``g_j <= g_i`` for every state, so ``i`` is
      dominated by ``j``.
    * If ``S_j-S_i >= (j-i)n``, then ``g_i <= g_j`` for every state, so
      ``j`` is dominated by ``i``.

    Removing either constraint is therefore pointwise exact, including before
    rounding and action clipping.  The pairwise test also works for raw target
    vectors; canonical nondecreasing targets are not required for correctness.
    """

    targets = tuple(float(params[f"S{index}"]) for index in range(int(lead_time) + 1))
    return _effective_indexes_cached(targets, int(n))


def evaluate_vbs_actions(
    states: np.ndarray,
    params: Mapping[str, float],
    lead_time: int,
    n: int,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Evaluate VBS actions using only pointwise-undominated constraints."""

    active = effective_vbs_constraint_indexes(params, lead_time, n)
    cumulative = states[:, 0].astype(float, copy=True)
    raw = np.full(len(states), np.inf, dtype=float)
    active_set = set(active)
    for index in range(int(lead_time) + 1):
        if index > 0:
            cumulative += states[:, index]
        if index in active_set:
            np.minimum(raw, float(params[f"S{index}"]) - cumulative, out=raw)
    actions = np.rint(raw)
    np.clip(actions, 0.0, float(n), out=actions)
    return actions.astype(np.int32), active


@lru_cache(maxsize=16_384)
def _effective_indexes_cached(targets: tuple[float, ...], n: int) -> tuple[int, ...]:
    active = [True] * len(targets)
    for left in range(len(targets)):
        for right in range(left + 1, len(targets)):
            difference = targets[right] - targets[left]
            if difference <= 0.0:
                active[left] = False
            if difference >= float((right - left) * n):
                active[right] = False
    indexes = tuple(index for index, keep in enumerate(active) if keep)
    # The n=0 degenerate grid can make every constraint mutually equivalent.
    return indexes or (len(targets) - 1,)


def _positive_part_child(ast: AST) -> AST | None:
    if ast.get("op") != "maximum":
        return None
    left, right = ast.get("left"), ast.get("right")
    if _is_zero(left) and isinstance(right, dict):
        return right
    if _is_zero(right) and isinstance(left, dict):
        return left
    return None


def _flatten_minimum(node: AST) -> list[AST]:
    if node.get("op") != "minimum":
        return [node]
    left, right = node.get("left"), node.get("right")
    if not isinstance(left, dict) or not isinstance(right, dict):
        return [node]
    return _flatten_minimum(left) + _flatten_minimum(right)


def _parse_gap(node: AST) -> int | None:
    if node.get("op") != "add" or len(node.get("args", ())) != 2:
        return None
    parameter_node, negative_node = node["args"]
    if parameter_node.get("op") != "parameter" or negative_node.get("op") != "neg":
        return None
    name = str(parameter_node.get("name", ""))
    if not name.startswith("S") or not name[1:].isdigit():
        return None
    index = int(name[1:])
    source = negative_node.get("arg", {})
    if index == 0:
        return 0 if source == {"op": "state", "name": "x"} else None
    return (
        index
        if source == {"op": "motif", "name": f"cumulative_inventory_{index}"}
        else None
    )


def _is_zero(node: object) -> bool:
    return isinstance(node, dict) and node.get("op") == "const" and float(node.get("value", 1.0)) == 0.0


__all__ = [
    "VBS_MONOTONE_GROUP",
    "detect_vbs_ast",
    "effective_vbs_constraint_indexes",
    "evaluate_vbs_actions",
    "is_vbs_parameterization",
]
