from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from inventory.policy.ast.core import DISTRIBUTION_FEATURES


class _BatchEvaluationUnsupported(Exception):
    pass


def _round_actions(
    raw: np.ndarray, rounding: str, action_cap: int | None
) -> np.ndarray:
    safe = np.maximum(
        np.nan_to_num(raw.astype(float), nan=0.0, posinf=0.0, neginf=0.0), 0.0
    )
    if rounding == "floor":
        action = np.floor(safe)
    elif rounding == "ceil":
        action = np.ceil(safe)
    elif rounding == "none":
        action = safe.astype(int).astype(float)
    else:
        action = np.rint(safe)
    if action_cap is not None:
        action = np.minimum(action, int(action_cap))
    return np.maximum(action, 0.0)


def _batch_scalar(value: Any, size: int) -> np.ndarray:
    """Return a scalar or candidate-batched value with the requested length.

    Proposal MC uses one flattened row per ``candidate × trajectory``.  AST
    parameter leaves therefore may already be arrays, while constants and
    distribution/cost features remain scalars.  Keeping this broadcast in a
    single helper preserves the scalar evaluator behavior and avoids creating
    a new full array for every scalar leaf.
    """

    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return np.full(size, float(array), dtype=float)
    flat = array.reshape(-1)
    if flat.size == size:
        return flat
    if flat.size == 1:
        return np.full(size, float(flat[0]), dtype=float)
    raise ValueError(f"batched value has {flat.size} entries, expected {size}")


def _batch_distribution_feature(name: str, features: Mapping[str, Any]) -> float:
    if name not in DISTRIBUTION_FEATURES:
        raise _BatchEvaluationUnsupported(f"unknown distribution summary: {name}")
    if name in features:
        return float(features[name])
    values = np.asarray(features.get("values", []), dtype=float)
    probs = np.asarray(features.get("probs", []), dtype=float)
    if values.size == 0 or values.size != probs.size or probs.sum() <= 0:
        raise _BatchEvaluationUnsupported(f"distribution summary missing: {name}")
    probs = probs / probs.sum()
    mean = float(np.dot(values, probs))
    variance = float(np.dot((values - mean) ** 2, probs))
    summaries = {
        "mu": mean,
        "variance": variance,
        "sigma": float(np.sqrt(max(variance, 0.0))),
        "cv": float(np.sqrt(max(variance, 0.0))) / mean if mean > 0 else 0.0,
        "p_zero": float(probs[values == 0].sum()),
    }
    if name not in summaries:
        raise _BatchEvaluationUnsupported(f"unknown distribution summary: {name}")
    return summaries[name]


def _batch_cost_feature(name: str, features: Mapping[str, float]) -> float:
    if name in features:
        return float(features[name])
    h, b = float(features["h"]), float(features["b"])
    if name == "critical_fractile":
        return b / (b + h)
    if name == "cost_ratio":
        return b / h
    raise _BatchEvaluationUnsupported(f"unknown cost feature: {name}")
