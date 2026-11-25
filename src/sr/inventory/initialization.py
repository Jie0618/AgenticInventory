"""Lead-time binding for generic multi-case frontier initialization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from sr.analysis.frontier import FrontierSeedBundle
from sr.core.models import SearchConfig
from sr.multi.initialization import FrontierInitializationResult
from sr.multi.initialization import run_frontier_initialization as run_generic_initialization
from sr.multi.models import MultiObjective

from sr.inventory.multi_evaluation import (
    CaseEvaluationInput,
    MultiStandardScheduler,
)
from sr.inventory.evaluation import (
    EvaluationTask,
    _standard_worker,
)


def run_frontier_initialization(
    bundle: FrontierSeedBundle,
    case_inputs: Mapping[str, CaseEvaluationInput],
    *,
    config: SearchConfig,
    objective: MultiObjective,
    cache_dir: Path,
    initialization_signature: str,
    max_pair_retries: int = 5,
    progress_interval_seconds: float = 5.0,
    worker_target: Callable[[int, EvaluationTask, Any], None] = _standard_worker,
) -> FrontierInitializationResult:
    def scheduler_factory(
        result_dir: Path,
        event_log: Path,
        provenance: Mapping[str, tuple[str, ...]],
    ) -> MultiStandardScheduler:
        return MultiStandardScheduler(
            config,
            case_names=tuple(case_inputs),
            objective=objective,
            motifs=bundle.motifs,
            result_dir=result_dir,
            event_log=event_log,
            provenance=provenance,
            max_pair_retries=max_pair_retries,
            worker_target=worker_target,
        )

    return run_generic_initialization(
        bundle,
        case_inputs,
        cache_dir=cache_dir,
        initialization_signature=initialization_signature,
        scheduler_factory=scheduler_factory,
        progress_interval_seconds=progress_interval_seconds,
    )


__all__ = ["FrontierInitializationResult", "run_frontier_initialization"]
