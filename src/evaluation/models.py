"""Model-neutral evaluation data contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class ExactEvaluationResult:
    cost: float
    completed: bool
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class MonteCarloEvaluationResult:
    mean_cost: float
    standard_error: float
    paths: int
    seed: int
    metadata: Mapping[str, object] = field(default_factory=dict)

    def confidence_interval(self, z: float = 1.96) -> tuple[float, float]:
        radius = float(z) * float(self.standard_error)
        return self.mean_cost - radius, self.mean_cost + radius
