"""Inventory-context binding for generic multi-case SR evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from inventory.policy.ast import AST
from optimization.models.types import SupervisedSample
from optimization.search_context import InventorySearchContext
from sr.core.models import SearchCandidate, SearchConfig
from sr.evaluation.multi import MultiCaseEvaluationScheduler
from sr.evaluation.standard import load_standard_result
from sr.multi.models import MultiObjective

from sr.inventory.evaluation import (
    EvaluationTask,
    _evaluation_signature,
    _standard_worker,
)


@dataclass(frozen=True)
class CaseEvaluationInput:
    case_name: str
    context: InventorySearchContext | None
    samples: tuple[SupervisedSample, ...]
    random_seed: int


class MultiStandardScheduler(MultiCaseEvaluationScheduler):
    def __init__(
        self,
        config: SearchConfig,
        *,
        case_names: Sequence[str],
        objective: MultiObjective,
        motifs: Mapping[str, AST],
        result_dir: Path,
        event_log: Path | None,
        provenance: Mapping[str, Sequence[str]] | None = None,
        max_pair_retries: int = 5,
        worker_target: Callable[[int, EvaluationTask, Any], None] = _standard_worker,
    ) -> None:
        super().__init__(
            config,
            case_names=case_names,
            objective=objective,
            motifs=motifs,
            result_dir=result_dir,
            event_log=event_log,
            signature_factory=_case_signature,
            task_factory=_make_task,
            result_loader=load_standard_result,
            worker_target=worker_target,
            provenance=provenance,
            max_pair_retries=max_pair_retries,
        )


def _case_signature(
    case: CaseEvaluationInput, motifs: Mapping[str, AST], random_seed: int
) -> str:
    return _evaluation_signature(
        case.context, case.samples, motifs, random_seed=random_seed
    )


def _make_task(
    candidate: SearchCandidate,
    case_name: str,
    case: CaseEvaluationInput,
    motifs: Mapping[str, AST],
    result_path: Path,
    random_seed: int,
    evaluation_signature: str,
) -> EvaluationTask:
    return EvaluationTask(
        candidate=candidate,
        case_name=case_name,
        context=case.context,
        samples=case.samples,
        motifs=motifs,
        result_path=result_path,
        random_seed=random_seed,
        evaluation_signature=evaluation_signature,
    )


__all__ = ["CaseEvaluationInput", "MultiStandardScheduler"]
