"""Canonical parameter encoding and discrete-domain enumeration."""

from __future__ import annotations

import itertools
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from optimization.models.types import ParameterSpec


def canonicalize_params(params: Mapping[str, float], specs: Sequence[ParameterSpec]) -> dict[str, float]:
    result: dict[str, float] = {}
    for spec in specs:
        raw = float(params[spec.name])
        if not np.isfinite(raw):
            raise ValueError(f"Non-finite parameter {spec.name}")
        clipped = float(np.clip(raw, spec.lower, spec.upper))
        if spec.kind == "integer":
            value = float(np.rint(clipped))
        elif spec.kind == "grid":
            assert spec.values is not None
            value = float(min(spec.values, key=lambda item: abs(float(item) - clipped)))
        else:
            value = clipped
        result[spec.name] = value

    # A marked group has a unique nondecreasing representative obtained by
    # suffix minima. Unmarked parameters retain independent canonicalization.
    groups: dict[str, list[ParameterSpec]] = {}
    for spec in specs:
        if spec.monotone_group is not None:
            groups.setdefault(spec.monotone_group, []).append(spec)
    for group_specs in groups.values():
        ordered = sorted(group_specs, key=lambda item: int(item.monotone_index or 0))
        if len({spec.monotone_index for spec in ordered}) != len(ordered):
            raise ValueError("monotone group indexes must be unique")
        suffix_min = float("inf")
        for spec in reversed(ordered):
            suffix_min = min(suffix_min, result[spec.name])
            result[spec.name] = suffix_min
    return result


def canonical_parameter_key(params: Mapping[str, float], specs: Sequence[ParameterSpec]) -> tuple[Any, ...]:
    values: list[Any] = []
    for spec in specs:
        value = float(params[spec.name])
        values.append(value if spec.kind in {"integer", "grid"} else round(value, 12))
    return tuple(values)


def decode_params(z: Sequence[float], specs: Sequence[ParameterSpec]) -> dict[str, float]:
    result: dict[str, float] = {}
    for unit_raw, spec in zip(z, specs):
        unit = float(np.clip(unit_raw, 0.0, 1.0))
        if spec.kind == "integer":
            raw = spec.lower + unit * (spec.upper - spec.lower)
            value = float(np.clip(np.rint(raw), spec.lower, spec.upper))
        elif spec.kind == "grid":
            assert spec.values is not None
            index = min(int(math.floor(unit * len(spec.values))), len(spec.values) - 1)
            value = float(spec.values[index])
        elif spec.scale == "log":
            value = float(np.exp(np.log(spec.lower) + unit * (np.log(spec.upper) - np.log(spec.lower))))
        else:
            value = float(spec.lower + unit * (spec.upper - spec.lower))
        result[spec.name] = value
    return canonicalize_params(result, specs)


def encode_params(params: Mapping[str, float], specs: Sequence[ParameterSpec]) -> np.ndarray:
    encoded: list[float] = []
    canonical = canonicalize_params(params, specs)
    for spec in specs:
        value = canonical[spec.name]
        if spec.kind == "grid":
            assert spec.values is not None
            index = min(range(len(spec.values)), key=lambda idx: abs(spec.values[idx] - value))
            unit = (index + 0.5) / len(spec.values)
        elif spec.scale == "log":
            unit = (np.log(value) - np.log(spec.lower)) / max(np.log(spec.upper) - np.log(spec.lower), 1.0e-15)
        else:
            unit = (value - spec.lower) / max(spec.upper - spec.lower, 1.0e-15)
        encoded.append(float(np.clip(unit, 0.0, 1.0)))
    return np.asarray(encoded, dtype=float)


def discrete_space_size(specs: Sequence[ParameterSpec]) -> int:
    total = 1
    for spec in specs:
        cardinality = spec.cardinality()
        if cardinality is not None:
            total *= cardinality
    return int(total)


def iter_discrete_params(specs: Sequence[ParameterSpec]) -> Iterable[dict[str, float]]:
    values = [spec.grid_values(spec.cardinality() or 1) for spec in specs]
    for combination in itertools.product(*values):
        yield {spec.name: float(value) for spec, value in zip(specs, combination)}
