"""Reusable schema-aware inventory policy features.

Features are templates over the configured pipeline branch.  They are
materialized only after a concrete case/context supplies its pipeline length;
no experiment owns the window vocabulary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from inventory.policy.ast import AST
from inventory.core.config import InventoryModelConfig
from inventory.policy.motifs.lead_time import lead_time_motifs


BUILTIN_FEATURE_NAMES = frozenset(
    {
        "inventory_position",
        "available_inventory",
        "total_pipeline",
        "near_pipeline",
        "far_pipeline",
        "pipeline_imbalance",
        "pipeline_lag_moment",
        "pipeline_mean_arrival_lag",
        "demand_mean",
        "demand_cv",
        "cost_ratio",
    }
)


def validate_window_rule(rule: object) -> None:
    if isinstance(rule, bool):
        raise ValueError("feature window must be a positive integer or supported L rule")
    if isinstance(rule, int):
        if not 1 <= rule <= 10:
            raise ValueError("feature window must be in [1, 10]")
        return
    if isinstance(rule, str):
        value = rule.strip().replace(" ", "")
        allowed = {"L", "floor(L/2)", "ceil(L/2)", "floor(L/3)", "ceil(L/3)", "floor(L/4)", "ceil(L/4)", "floor(L/5)", "ceil(L/5)"}
        if value in allowed:
            return
        if value.startswith("fixed(") and value.endswith(")"):
            try:
                width = int(value[6:-1])
            except ValueError:
                width = 0
            if 1 <= width <= 10:
                return
    if isinstance(rule, Mapping):
        op = rule.get("op")
        if op == "lead_time" or (op == "const" and isinstance(rule.get("value"), int) and 1 <= rule["value"] <= 10):
            return
        if op in {"floor_div", "ceil_div"} and rule.get("left") == {"op": "lead_time"}:
            right = rule.get("right")
            if isinstance(right, Mapping) and right.get("value") in {1, 2, 3, 4}:
                return
    raise ValueError("unsupported feature window rule")


def resolve_window_rule(rule: object, lead_time: int) -> int:
    if isinstance(rule, int):
        return min(rule, lead_time)
    if isinstance(rule, str):
        value = rule.strip().replace(" ", "")
        if value == "L":
            return lead_time
        if value == "floor(L/2)":
            return max(1, lead_time // 2)
        if value == "ceil(L/2)":
            return max(1, (lead_time + 1) // 2)
        for operator in ("floor", "ceil"):
            prefix = f"{operator}(L/"
            if value.startswith(prefix) and value.endswith(")"):
                divisor = int(value[len(prefix) : -1])
                return max(1, lead_time // divisor) if operator == "floor" else max(1, (lead_time + divisor - 1) // divisor)
        if value.startswith("fixed("):
            return min(int(value[6:-1]), lead_time)
    if isinstance(rule, Mapping):
        op = rule.get("op")
        if op == "lead_time":
            return lead_time
        if op == "const":
            return min(int(rule["value"]), lead_time)
        divisor = int(rule["right"]["value"])
        return max(1, lead_time // divisor) if op == "floor_div" else max(1, (lead_time + divisor - 1) // divisor)
    raise ValueError("unsupported feature window rule")


@dataclass(frozen=True)
class FeatureSpec:
    """A typed, case-aware feature template over the pipeline branch."""

    name: str
    kind: str
    window: object = 1
    weights: tuple[object, ...] = ()
    weight_rule: str | None = None

    def __post_init__(self) -> None:
        if not self.name or self.name in BUILTIN_FEATURE_NAMES or re.fullmatch(r"(?:tail|head)_sum_\d+", self.name):
            raise ValueError(f"feature name is reserved or empty: {self.name!r}")
        if self.kind not in {"tail_sum", "head_sum", "tail_weighted", "head_weighted"}:
            raise ValueError(f"unsupported feature kind: {self.kind}")
        validate_window_rule(self.window)
        allowed_rule = {"tail_weighted": "linear_desc_normalized", "head_weighted": "linear_asc_normalized"}.get(self.kind)
        if self.weight_rule is not None:
            if self.weight_rule != allowed_rule:
                raise ValueError(f"unsupported weight_rule {self.weight_rule!r} for {self.kind}")
            if self.weights:
                raise ValueError("weight_rule and explicit weights are mutually exclusive")
        elif self.kind in {"tail_weighted", "head_weighted"}:
            if not self.weights or len(self.weights) > 10:
                raise ValueError(f"{self.kind} requires 1..10 weights or a weight_rule")
            if isinstance(self.window, int) and len(self.weights) != self.window:
                raise ValueError(f"{self.kind} requires one weight per fixed window position")
            numeric = any(isinstance(weight, (int, float)) and not isinstance(weight, bool) for weight in self.weights)
            symbolic = any(isinstance(weight, str) for weight in self.weights)
            if numeric and symbolic:
                raise ValueError("feature weights must be all numeric or all parameter names")
        elif self.weights:
            raise ValueError(f"{self.kind} must not declare weights")
        for weight in self.weights:
            if not isinstance(weight, (int, float, str)) or isinstance(weight, bool) or (isinstance(weight, str) and not weight.strip()):
                raise ValueError("feature weights must be numeric coefficients or parameter names")

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(str(weight) for weight in self.weights if isinstance(weight, str))

    def resolve_window(self, lead_time: int) -> int:
        return resolve_window_rule(self.window, int(lead_time))


def materialize_feature(name: str, lead_time: int, spec: FeatureSpec | None = None) -> AST:
    motifs = lead_time_motifs(lead_time)
    if name in motifs:
        return motifs[name]
    if spec is None:
        raise ValueError(f"unknown feature: {name}")
    width = spec.resolve_window(lead_time)
    indices = list(range(1, width + 1)) if spec.kind in {"head_sum", "head_weighted"} else list(range(lead_time, lead_time - width, -1))
    weights = spec.weights
    if spec.weight_rule == "linear_desc_normalized":
        denominator = width * (width + 1) / 2
        weights = tuple((width - position) / denominator for position in range(width))
    elif spec.weight_rule == "linear_asc_normalized":
        denominator = width * (width + 1) / 2
        weights = tuple((position + 1) / denominator for position in range(width))
    terms: list[AST] = []
    for position, index in enumerate(indices):
        term: AST = {"op": "state", "name": f"p{index}"}
        if spec.kind in {"tail_weighted", "head_weighted"}:
            if position >= len(weights):
                raise ValueError(
                    f"{spec.kind} does not provide a weight for resolved "
                    f"window position {position + 1}"
                )
            weight = weights[position]
            weight_ast: AST = {"op": "parameter", "name": str(weight)} if isinstance(weight, str) else {"op": "const", "value": float(weight)}
            term = {"op": "mul", "args": [weight_ast, term]}
        terms.append(term)
    return {"op": "add", "args": terms} if len(terms) > 1 else terms[0]


def legacy_sum_feature(name: str, lead_time: int) -> AST | None:
    match = re.fullmatch(r"(head|tail)_sum_(\d+)", str(name))
    if match is None:
        return None
    width = int(match.group(2))
    if not 1 <= width <= 10:
        return None
    indices = list(range(1, min(width, lead_time) + 1)) if match.group(1) == "head" else list(range(lead_time, max(lead_time - width, 0), -1))
    terms: list[AST] = [{"op": "state", "name": f"p{index}"} for index in indices]
    return {"op": "add", "args": terms} if len(terms) > 1 else terms[0]


def feature_specs_for_config(config: InventoryModelConfig) -> dict[str, FeatureSpec]:
    """Return the reusable feature templates supported by one composition."""

    if not isinstance(config, InventoryModelConfig):
        raise TypeError("config must be an InventoryModelConfig")
    if config.lead_time.state_dimension == 0:
        return {}
    output: dict[str, FeatureSpec] = {}
    for divisor in (2, 3, 4, 5):
        for rounding in ("floor", "ceil"):
            rule = f"{rounding}(L/{divisor})"
            suffix = f"{rounding}_l_div_{divisor}"
            output[f"head_sum_{suffix}"] = FeatureSpec(
                f"head_sum_{suffix}", "head_sum", rule
            )
            output[f"tail_sum_{suffix}"] = FeatureSpec(
                f"tail_sum_{suffix}", "tail_sum", rule
            )
    output["head_linear_half_normalized"] = FeatureSpec(
        "head_linear_half_normalized",
        "head_weighted",
        "floor(L/2)",
        weight_rule="linear_asc_normalized",
    )
    output["tail_linear_half_normalized"] = FeatureSpec(
        "tail_linear_half_normalized",
        "tail_weighted",
        "floor(L/2)",
        weight_rule="linear_desc_normalized",
    )
    return output


__all__ = ["BUILTIN_FEATURE_NAMES", "FeatureSpec", "feature_specs_for_config", "legacy_sum_feature", "materialize_feature", "resolve_window_rule", "validate_window_rule"]
