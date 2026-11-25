"""Scenario-neutral aggregation of candidate screens across cases."""

from __future__ import annotations

from runtime.artifacts import stable_key
import json
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

from inventory.policy.ast import AST
from sr.core.models import SearchCandidate, SearchConfig
from sr.multi.models import MultiObjective


class CandidateScreenResult(Protocol):
    bellman_regret_pct: float
    screen_params: Mapping[str, float]
    behavior_signature: str


class CandidateScreen(Protocol):
    motifs: dict[str, AST]

    def score(
        self, candidate: SearchCandidate, config: SearchConfig
    ) -> CandidateScreenResult: ...


@dataclass(frozen=True)
class MultiScreenedCandidate:
    candidate: SearchCandidate
    score: float
    mean_regret_pct: float
    mean_top_k_regret_pct: float
    case_regrets_pct: Mapping[str, float]
    case_screen_params: Mapping[str, Mapping[str, float]]
    case_behavior_signatures: Mapping[str, str]
    behavior_signature: str

    @property
    def bellman_regret_pct(self) -> float:
        return self.score


class MultiCandidateScreen:
    def __init__(
        self,
        screens: Mapping[str, CandidateScreen],
        objective: MultiObjective,
    ) -> None:
        self.screens = dict(screens)
        if not self.screens:
            raise ValueError("at least one candidate screen is required")
        self.case_names = tuple(self.screens)
        self.objective = objective

    @property
    def motifs(self) -> dict[str, AST]:
        return next(iter(self.screens.values())).motifs

    def register_motif(self, name: str, ast: AST) -> None:
        for screen in self.screens.values():
            screen.motifs[name] = ast

    def score(
        self, candidate: SearchCandidate, config: SearchConfig
    ) -> MultiScreenedCandidate:
        rows = {
            name: self.screens[name].score(candidate, config)
            for name in self.case_names
        }
        regrets = tuple(rows[name].bellman_regret_pct for name in self.case_names)
        mean, mean_top_k, score, _ = self.objective.aggregate(regrets)
        behaviors = {
            name: rows[name].behavior_signature for name in self.case_names
        }
        encoded = json.dumps(
            behaviors, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return MultiScreenedCandidate(
            candidate=candidate,
            score=score,
            mean_regret_pct=mean,
            mean_top_k_regret_pct=mean_top_k,
            case_regrets_pct={
                name: rows[name].bellman_regret_pct for name in self.case_names
            },
            case_screen_params={
                name: dict(rows[name].screen_params) for name in self.case_names
            },
            case_behavior_signatures=behaviors,
            behavior_signature=stable_key(encoded),
        )


def select_multi_promotions(
    rows: Sequence[MultiScreenedCandidate],
    limit: int,
) -> tuple[MultiScreenedCandidate, ...]:
    if limit <= 0:
        return ()
    best_by_behavior: dict[str, MultiScreenedCandidate] = {}
    for row in rows:
        current = best_by_behavior.get(row.behavior_signature)
        key = (row.score, row.candidate.expanded_complexity, row.candidate.ast_hash)
        if current is None or key < (
            current.score,
            current.candidate.expanded_complexity,
            current.candidate.ast_hash,
        ):
            best_by_behavior[row.behavior_signature] = row
    buckets: dict[int, list[MultiScreenedCandidate]] = {}
    for row in best_by_behavior.values():
        buckets.setdefault(row.candidate.num_parameters, []).append(row)
    selected: dict[str, MultiScreenedCandidate] = {}
    quota = max(1, int(limit) // max(1, len(buckets)))
    for parameter_count in sorted(buckets):
        ranked = sorted(
            buckets[parameter_count],
            key=lambda row: (
                row.score,
                row.mean_top_k_regret_pct,
                row.candidate.expanded_complexity,
                row.candidate.ast_hash,
            ),
        )
        for row in ranked[:quota]:
            selected[row.candidate.ast_hash] = row
    for row in sorted(
        rows,
        key=lambda row: (
            row.score,
            row.candidate.num_parameters,
            row.candidate.expanded_complexity,
            row.candidate.ast_hash,
        ),
    ):
        if len(selected) >= int(limit):
            break
        selected.setdefault(row.candidate.ast_hash, row)
    return tuple(
        sorted(selected.values(), key=lambda row: (row.score, row.candidate.ast_hash))[
            : int(limit)
        ]
    )


__all__ = [
    "CandidateScreen",
    "CandidateScreenResult",
    "MultiCandidateScreen",
    "MultiScreenedCandidate",
    "select_multi_promotions",
]
