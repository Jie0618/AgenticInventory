"""Asynchronous Standard evaluation for canonical inventory search contexts."""

from __future__ import annotations

from runtime.artifacts import stable_key
import json
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from inventory.policy.ast import AST
from optimization.models.types import SupervisedSample
from optimization.search_context import InventorySearchContext
from sr.core.models import SearchCandidate, SearchConfig, SearchIndividual
from sr.evaluation.scheduler import AsyncEvaluationScheduler
from sr.evaluation.standard import (
    load_standard_result,
    standard_optimization_config,
)


@dataclass(frozen=True)
class EvaluationTask:
    candidate: SearchCandidate
    case_name: str
    context: InventorySearchContext | None
    samples: tuple[SupervisedSample, ...]
    motifs: Mapping[str, AST]
    result_path: Path
    random_seed: int
    evaluation_signature: str


def _standard_worker(task_id: int, task: EvaluationTask, output_queue: Any) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        if task.context is None:
            raise ValueError("Standard inventory evaluation requires a search context")
        context = replace(task.context, samples=task.samples)
        fit = context.fit(
            task.candidate.ast,
            motifs=task.motifs,
            config=standard_optimization_config(task.random_seed),
            warm_starts=(),
        )
        from sr.inventory.mc import MonteCarloSearchContext
        reference_cost = (
            context.reference_cost if isinstance(context, MonteCarloSearchContext)
            else float(context.reference.value(0, context.case.initial_flat_state))
        )
        payload = {
            "schema_version": 2,
            "case": task.case_name,
            "mode": "standard",
            "ast_hash": task.candidate.ast_hash,
            "ast": task.candidate.ast,
            "formula": task.candidate.formula,
            "source": task.candidate.source,
            "parents": list(task.candidate.parents),
            "iteration": task.candidate.iteration,
            "island": task.candidate.island,
            "macro_complexity": task.candidate.macro_complexity,
            "expanded_complexity": task.candidate.expanded_complexity,
            "parameter_names": list(task.candidate.parameter_names),
            "candidate_detail": dict(task.candidate.detail),
            "params": dict(fit.params),
            "policy_cost": float(fit.policy_cost),
            "dp_cost": reference_cost,
            "gap_pct": 100.0 * (float(fit.policy_cost) / reference_cost - 1.0),
            "status": fit.status,
            "route": fit.route,
            "optimizer_seconds": float(fit.elapsed_seconds),
            "optimizer_metadata": fit.optimizer_metadata,
            "completed_at": time.time(),
            "random_seed": int(task.random_seed),
            "optimizer_config_version": "standard-independent",
            "evaluation_signature": task.evaluation_signature,
            "independent_warm_starts": 0,
            "reference_type": getattr(context, "reference_type", "dp_optimal"),
        }
        task.result_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = task.result_path.with_suffix(
            task.result_path.suffix + f".tmp-{os.getpid()}"
        )
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(task.result_path)
        output_queue.put((task_id, "completed", str(task.result_path)))
    except BaseException as exc:
        output_queue.put((task_id, "failed", f"{type(exc).__name__}: {exc}"))


class AsyncStandardScheduler(AsyncEvaluationScheduler[EvaluationTask]):
    def __init__(
        self,
        config: SearchConfig,
        *,
        result_dir: Path,
        event_log: Path | None,
        worker_target: Callable[[int, EvaluationTask, Any], None] = _standard_worker,
    ) -> None:
        super().__init__(
            config,
            result_dir=result_dir,
            event_log=event_log,
            worker_target=worker_target,
            result_loader=self._load_task_result,
        )

    def submit(
        self,
        candidate: SearchCandidate,
        *,
        case_name: str,
        context: InventorySearchContext | None,
        samples: Sequence[SupervisedSample],
        motifs: Mapping[str, AST],
        random_seed: int,
    ) -> str:
        signature = _evaluation_signature(
            context, samples, motifs, random_seed=random_seed
        )
        previous = self.evaluation_signatures.get(candidate.ast_hash)
        if previous is not None and previous != signature:
            raise ValueError(
                "the same AST hash was submitted with different Standard evaluation inputs"
            )
        return self.submit_task(
            EvaluationTask(
                candidate=candidate,
                case_name=case_name,
                context=context,
                samples=tuple(samples),
                motifs=dict(motifs),
                result_path=self.result_dir / f"{candidate.ast_hash}.json",
                random_seed=int(random_seed),
                evaluation_signature=signature,
            )
        )

    @staticmethod
    def _load_task_result(task: EvaluationTask) -> SearchIndividual | None:
        return load_standard_result(
            task.result_path,
            task.candidate,
            task.case_name,
            task.random_seed,
            task.evaluation_signature,
        )


def _evaluation_signature(
    context: InventorySearchContext | None,
    samples: Sequence[SupervisedSample],
    motifs: Mapping[str, AST],
    *,
    random_seed: int,
) -> str:
    context_payload = None
    if context is not None:
        context_payload = {
            "case_signature": context.case.case_signature,
            "horizon": int(context.horizon),
            "backend": asdict(context.backend),
        }
        from sr.inventory.mc import MonteCarloSearchContext
        if isinstance(context, MonteCarloSearchContext):
            context_payload["mc_protocol"] = context.identity_payload()
    payload = {
        "protocol": "standard-independent",
        "optimizer": asdict(standard_optimization_config(random_seed)),
        "context": context_payload,
        "samples": [
            {
                "state": dict(sample.state),
                "target_action": float(sample.target_action),
                "weight": float(sample.weight),
            }
            for sample in samples
        ],
        "motifs": dict(motifs),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return stable_key(encoded)


__all__ = ["AsyncStandardScheduler", "EvaluationTask"]
