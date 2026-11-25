"""Generic contracts for evaluating one policy family across contexts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class FamilyTask:
    family_id: str
    context_id: str
    case_id: str
    evaluator: str
    fidelity: str
    seed: int


@dataclass(frozen=True)
class CaseMetric:
    context_id: str
    case_id: str
    gap_pct: float
    runtime_seconds: float = 0.0
    status: str = "complete"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScenarioScore:
    context_id: str
    mean_gap: float
    cvar80_gap: float
    robust_score: float


@dataclass(frozen=True)
class FamilyEvaluation:
    family: Any
    tasks: tuple[FamilyTask, ...]
    metrics: tuple[CaseMetric, ...]
    per_context: tuple[ScenarioScore, ...]
    cross_context_score: float
    status: str


__all__ = ["CaseMetric", "FamilyEvaluation", "FamilyTask", "ScenarioScore"]
