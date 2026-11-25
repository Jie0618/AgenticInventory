"""Typed results and objective semantics for multi-case SR."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from sr.core.models import SearchCandidate, SearchIndividual
from inventory.common.fitness import ALPHA, TAIL_WEIGHT, empirical_cvar


@dataclass(frozen=True)
class MultiObjective:
    mean_weight: float = 0.6
    mean_top_k_weight: float = 0.4
    cvar_alpha: float = ALPHA

    def to_payload(self) -> dict[str, object]:
        return {"mean_weight": self.mean_weight,
                "mean_top_k_weight": self.mean_top_k_weight, "cvar_alpha": self.cvar_alpha}

    def __post_init__(self) -> None:
        weights = (float(self.mean_weight), float(self.mean_top_k_weight))
        if not all(math.isfinite(value) and value >= 0 for value in weights):
            raise ValueError("objective weights must be finite and nonnegative")
        if not math.isclose(sum(weights), 1.0, abs_tol=1.0e-12):
            raise ValueError("objective weights must sum to one")
        object.__setattr__(self, "mean_weight", weights[0])
        object.__setattr__(self, "mean_top_k_weight", weights[1])
        if self.cvar_alpha != ALPHA or weights != (1 - TAIL_WEIGHT, TAIL_WEIGHT):
            raise ValueError("search scoring requires 0.60 Mean + 0.40 CVaR80")

    def aggregate(
        self, gaps: Sequence[float]
    ) -> tuple[float, float, float, tuple[int, ...]]:
        values = tuple(float(value) for value in gaps)
        if not values:
            raise ValueError("need at least one case gap")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("case gaps must be finite")
        mean = sum(values) / len(values)
        indices = tuple(
            sorted(range(len(values)), key=lambda index: (-values[index], index))[
                : max(1, math.ceil((1 - self.cvar_alpha) * len(values) - 1e-12))
            ]
        )
        mean_top_k = empirical_cvar(values, self.cvar_alpha)
        score = self.mean_weight * mean + self.mean_top_k_weight * mean_top_k
        return mean, mean_top_k, score, indices


@dataclass(frozen=True)
class MultiSearchIndividual:
    candidate: SearchCandidate
    case_results: Mapping[str, SearchIndividual]
    case_order: tuple[str, ...]
    mean_gap_pct: float
    mean_top_k_gap_pct: float
    worst_gap_pct: float
    worst_case: str
    top_k_cases: tuple[str, ...]
    score: float
    completed_at: float
    result_path: str = ""
    provenance: tuple[str, ...] = ()
    optimizer_metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def ast_hash(self) -> str:
        return self.candidate.ast_hash

    @property
    def num_parameters(self) -> int:
        return self.candidate.num_parameters

    @property
    def expanded_complexity(self) -> float:
        return self.candidate.expanded_complexity

    @property
    def gap_pct(self) -> float:
        return self.score

    @property
    def params(self) -> Mapping[str, float]:
        return {}

    @property
    def policy_cost(self) -> float:
        return self.score

    @property
    def dp_cost(self) -> float:
        return 100.0

    @property
    def status(self) -> str:
        return "multi_complete"

    @property
    def route(self) -> str:
        return "independent_standard_per_case"

    @property
    def optimizer_seconds(self) -> float:
        return sum(row.optimizer_seconds for row in self.case_results.values())


def aggregate_case_results(
    candidate: SearchCandidate,
    case_results: Mapping[str, SearchIndividual],
    case_order: Sequence[str],
    objective: MultiObjective,
    *,
    result_path: str = "",
    provenance: Sequence[str] = (),
) -> MultiSearchIndividual:
    names = tuple(str(name) for name in case_order)
    missing = [name for name in names if name not in case_results]
    extras = sorted(set(case_results) - set(names))
    if missing or extras:
        raise ValueError(f"incomplete case results: missing={missing}, extras={extras}")
    for name in names:
        if case_results[name].ast_hash != candidate.ast_hash:
            raise ValueError(f"AST mismatch for case {name}")
    gaps = tuple(case_results[name].gap_pct for name in names)
    mean, mean_top_k, score, top_indices = objective.aggregate(gaps)
    worst_index = top_indices[0]
    completed_at = max(case_results[name].completed_at for name in names)
    return MultiSearchIndividual(
        candidate=candidate,
        case_results={name: case_results[name] for name in names},
        case_order=names,
        mean_gap_pct=mean,
        mean_top_k_gap_pct=mean_top_k,
        worst_gap_pct=gaps[worst_index],
        worst_case=names[worst_index],
        top_k_cases=tuple(names[index] for index in top_indices),
        score=score,
        completed_at=completed_at,
        result_path=str(result_path),
        provenance=tuple(dict.fromkeys(str(value) for value in provenance)),
    )


__all__ = ["MultiObjective", "MultiSearchIndividual", "aggregate_case_results"]
