from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from inventory.policy.ast import AST

from optimization.batch.analysis import _BatchEvaluationContext
from optimization.batch.features import (
    _BatchEvaluationUnsupported,
    _batch_cost_feature,
    _batch_distribution_feature,
    _batch_scalar,
)
from optimization.batch.samples import _BatchSamples


def _reduce_add(
    children: list[AST],
    evaluate: Any,
    size: int,
) -> np.ndarray:
    """Add AST children with one output allocation.

    ``sum`` allocates one full candidate×path array for every intermediate
    addition.  Proposal MC repeats this operation at every horizon period, so
    copying the first child once and accumulating in place is materially
    cheaper while preserving NumPy's left-to-right arithmetic.
    """

    if not children:
        return np.zeros(size, dtype=float)
    result = np.array(evaluate(children[0]), dtype=float, copy=True)
    for child in children[1:]:
        np.add(result, evaluate(child), out=result)
    return result


def _reduce_multiply(
    children: list[AST],
    evaluate: Any,
    size: int,
) -> np.ndarray:
    """Multiply AST children without stacking all operands."""

    if not children:
        return np.ones(size, dtype=float)
    result = np.array(evaluate(children[0]), dtype=float, copy=True)
    for child in children[1:]:
        np.multiply(result, evaluate(child), out=result)
    return result


def _evaluate_ast_batch(
    node: AST,
    batch: _BatchSamples,
    params: Mapping[str, Any],
    dist_features: Mapping[str, Any],
    cost_features: Mapping[str, float],
    motifs: Mapping[str, AST],
    local_vars: Mapping[str, np.ndarray | float] | None = None,
    context: _BatchEvaluationContext | None = None,
) -> np.ndarray:
    """Vectorized evaluator for the deliberately small SOF grammar."""

    cache_key = None if context is None else context.cache_key(node, local_vars)
    if cache_key is not None:
        cached = context.get(cache_key)
        if cached is not None:
            return cached

    op = str(node.get("op", ""))
    evaluate = lambda child: _evaluate_ast_batch(
        child,
        batch,
        params,
        dist_features,
        cost_features,
        motifs,
        local_vars=local_vars,
        context=context,
    )
    if op == "const":
        result = _batch_scalar(float(node["value"]), batch.size)
    elif op == "parameter":
        result = _batch_scalar(params[str(node["name"])], batch.size)
    elif op == "state":
        name = str(node["name"])
        if name not in batch.features:
            raise _BatchEvaluationUnsupported(f"state feature missing: {name}")
        result = np.asarray(batch.features[name], dtype=float)
    elif op == "distribution":
        result = _batch_scalar(
            _batch_distribution_feature(str(node["name"]), dist_features), batch.size
        )
    elif op == "cost":
        result = _batch_scalar(_batch_cost_feature(str(node["name"]), cost_features), batch.size)
    elif op == "motif":
        name = str(node["name"])
        if name not in motifs:
            raise _BatchEvaluationUnsupported(f"unknown motif: {name}")
        result = evaluate(motifs[name])
    elif op == "add":
        result = _reduce_add(node["args"], evaluate, batch.size)
    elif op == "mul":
        result = _reduce_multiply(node["args"], evaluate, batch.size)
    elif op == "neg":
        result = -evaluate(node["arg"])
    elif op == "sub":
        result = evaluate(node["left"]) - evaluate(node["right"])
    elif op == "absolute":
        result = np.abs(evaluate(node["arg"]))
    elif op == "cap":
        limit = evaluate(node["limit"])
        if node["mode"] == "integer":
            limit = np.rint(limit)
        elif node["mode"] == "cyclic_integer":
            if "time" not in batch.features:
                raise ValueError("cyclic cap requires explicit time")
            t = batch.features["time"]
            limit = np.rint(limit * (t + 1)) - np.rint(limit * t)
        result = np.minimum(evaluate(node["arg"]), limit)
    elif op == "minimum":
        result = np.minimum(evaluate(node["left"]), evaluate(node["right"]))
    elif op == "maximum":
        result = np.maximum(evaluate(node["left"]), evaluate(node["right"]))
    elif op == "indicator_gt":
        result = np.greater(evaluate(node["left"]), evaluate(node["right"])).astype(float)
    elif op == "safe_ratio":
        result = evaluate(node["left"]) / (1.0 + np.abs(evaluate(node["right"])))
    elif op == "clip":
        result = np.minimum(
            np.maximum(evaluate(node["arg"]), evaluate(node["lower"])),
            evaluate(node["upper"]),
        )
    else:
        raise _BatchEvaluationUnsupported(f"operator is outside SOF batch grammar: {op}")
    if cache_key is not None:
        context.put(cache_key, result)
    return result
