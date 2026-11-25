"""Generic grouped multi-case evaluation scheduler."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from runtime.processes import (
    available_memory_gb,
    close_queue,
    process_context,
    read_worker_message,
    run_single_threaded_worker,
    terminate_process,
)
from sr.core.models import SearchCandidate, SearchConfig, SearchIndividual
from sr.core.randomness import derived_seed
from sr.multi.models import (
    MultiObjective,
    MultiSearchIndividual,
    aggregate_case_results,
)


SignatureFactory = Callable[[Any, Mapping[str, Any], int], str]
TaskFactory = Callable[
    [SearchCandidate, str, Any, Mapping[str, Any], Path, int, str], Any
]
ResultLoader = Callable[
    [Path, SearchCandidate, str, int, str], Optional[SearchIndividual]
]


@dataclass
class PendingPair:
    pair_key: str
    task: Any
    attempt: int


@dataclass
class ActivePair:
    pair: PendingPair
    process: Any
    output_queue: Any
    started_at: float


class MultiCaseEvaluationScheduler:
    """Evaluate every candidate independently on every configured case."""

    def __init__(
        self,
        config: SearchConfig,
        *,
        case_names: Sequence[str],
        objective: MultiObjective,
        motifs: Mapping[str, Any],
        result_dir: Path,
        event_log: Path | None,
        signature_factory: SignatureFactory,
        task_factory: TaskFactory,
        result_loader: ResultLoader,
        worker_target: Callable[[int, Any, Any], None],
        provenance: Mapping[str, Sequence[str]] | None = None,
        max_pair_retries: int = 5,
    ) -> None:
        self.config = config
        self.case_names = tuple(str(name) for name in case_names)
        if len(set(self.case_names)) != len(self.case_names) or not self.case_names:
            raise ValueError("case_names must be unique and nonempty")
        self.objective = objective
        self.motifs = dict(motifs)
        self.result_dir = Path(result_dir)
        self.event_log = None if event_log is None else Path(event_log)
        self.signature_factory = signature_factory
        self.task_factory = task_factory
        self.result_loader = result_loader
        self.worker_target = worker_target
        self.provenance = {
            key: tuple(value) for key, value in (provenance or {}).items()
        }
        self.max_pair_retries = max(0, int(max_pair_retries))
        self.pending: list[PendingPair] = []
        self.active: dict[int, ActivePair] = {}
        self.pair_states: dict[str, str] = {}
        self.pair_failures: dict[str, str] = {}
        self.group_candidates: dict[str, SearchCandidate] = {}
        self.group_results: dict[str, dict[str, SearchIndividual]] = {}
        self.group_states: dict[str, str] = {}
        self.completed: dict[str, MultiSearchIndividual] = {}
        self.cache_hits = 0
        self._next_task_id = 0
        self.result_dir.mkdir(parents=True, exist_ok=True)
        if self.event_log is not None:
            self.event_log.parent.mkdir(parents=True, exist_ok=True)

    def submit_group(
        self, candidate: SearchCandidate, case_inputs: Mapping[str, Any]
    ) -> str:
        ast_hash = candidate.ast_hash
        previous = self.group_candidates.get(ast_hash)
        if previous is not None and previous.ast != candidate.ast:
            raise ValueError(f"candidate hash reused for a different AST: {ast_hash}")
        if self.group_states.get(ast_hash) in {"queued", "running", "completed"}:
            return self.group_states[ast_hash]
        if set(case_inputs) != set(self.case_names):
            raise ValueError("case inputs do not match scheduler case set")
        self.group_candidates[ast_hash] = candidate
        self.group_results.setdefault(ast_hash, {})
        queued = cache_hits = 0
        for case_name in self.case_names:
            case = case_inputs[case_name]
            base_seed = int(getattr(case, "random_seed"))
            seed = derived_seed(base_seed, ast_hash, case_name)
            pair_key = self._pair_key(ast_hash, case_name)
            path = self.result_dir / ast_hash / f"{case_name}.json"
            signature = self.signature_factory(case, self.motifs, seed)
            cached = self.result_loader(
                path, candidate, case_name, seed, signature
            )
            if cached is not None:
                self.group_results[ast_hash][case_name] = cached
                self.pair_states[pair_key] = "completed"
                cache_hits += 1
                self.cache_hits += 1
                self._log("cache_hit", candidate, case_name=case_name, pair_key=pair_key)
                continue
            task = self.task_factory(
                candidate,
                case_name,
                case,
                self.motifs,
                path,
                seed,
                signature,
            )
            self.pending.append(PendingPair(pair_key, task, 0))
            self.pair_states[pair_key] = "queued"
            queued += 1
            self._log("submitted", candidate, case_name=case_name, pair_key=pair_key, attempt=0)
        if self._complete_group_if_ready(ast_hash):
            return "completed"
        self.group_states[ast_hash] = "queued" if queued else "failed"
        self._log(
            "group_submitted",
            candidate,
            queued_pairs=queued,
            cache_hits=cache_hits,
            total_cases=len(self.case_names),
        )
        self.poll()
        return self.group_states[ast_hash]

    def register_motif(self, name: str, ast: Any) -> None:
        previous = self.motifs.get(str(name))
        if previous is not None and previous != ast:
            raise ValueError(f"motif {name!r} already has a different definition")
        self.motifs[str(name)] = dict(ast)

    def poll(self) -> tuple[MultiSearchIndividual, ...]:
        self._start_available()
        ready: list[MultiSearchIndividual] = []
        now = time.monotonic()
        for task_id, active in list(self.active.items()):
            pair, task = active.pair, active.pair.task
            elapsed = now - active.started_at
            if (
                active.process.is_alive()
                and elapsed >= self.config.evaluation_hard_timeout_seconds
            ):
                terminate_process(active.process)
                close_queue(active.output_queue)
                self.active.pop(task_id, None)
                self._fail_timed_out_group(
                    pair,
                    f"exceeded total timeout {self.config.evaluation_hard_timeout_seconds:g}s",
                )
                continue
            if task_id not in self.active:
                # A previous pair timeout may have cancelled this group.
                continue
            if active.process.is_alive():
                continue
            active.process.join(timeout=0.2)
            status, detail = read_worker_message(
                task_id, active.output_queue, active.process.exitcode
            )
            close_queue(active.output_queue)
            self.active.pop(task_id, None)
            if status == "completed":
                individual = self.result_loader(
                    Path(detail),
                    task.candidate,
                    task.case_name,
                    task.random_seed,
                    task.evaluation_signature,
                )
                if individual is not None:
                    self.pair_states[pair.pair_key] = "completed"
                    self.group_results[task.candidate.ast_hash][task.case_name] = individual
                    self._log(
                        "completed",
                        task.candidate,
                        case_name=task.case_name,
                        pair_key=pair.pair_key,
                        elapsed_s=elapsed,
                        gap_pct=individual.gap_pct,
                    )
                    if self._complete_group_if_ready(task.candidate.ast_hash):
                        ready.append(self.completed[task.candidate.ast_hash])
                    continue
                detail = "worker result failed canonical validation"
            self._retry_or_fail(pair, detail)
        self._refresh_group_states()
        self._start_available()
        return tuple(ready)

    def wait_for_groups(
        self,
        hashes: Sequence[str],
        *,
        wait_seconds: float | None = None,
        idle_active_workers: int | None = None,
    ) -> tuple[MultiSearchIndividual, ...]:
        targets = tuple(dict.fromkeys(str(value) for value in hashes))
        started = time.monotonic()
        while True:
            self.poll()
            if all(self.group_states.get(value) in {"completed", "failed"} for value in targets):
                return tuple(self.completed[value] for value in targets if value in self.completed)
            if idle_active_workers is not None and not self.pending and len(self.active) <= int(idle_active_workers):
                return tuple(self.completed[value] for value in targets if value in self.completed)
            if wait_seconds is not None and time.monotonic() - started >= max(0.0, float(wait_seconds)):
                return tuple(self.completed[value] for value in targets if value in self.completed)
            time.sleep(0.05)

    def wait_for_batch(
        self,
        hashes: Sequence[str],
        *,
        wait_seconds: float | None = None,
        idle_active_workers: int | None = None,
    ) -> tuple[MultiSearchIndividual, ...]:
        return self.wait_for_groups(
            hashes,
            wait_seconds=wait_seconds,
            idle_active_workers=idle_active_workers,
        )

    def drain(self) -> tuple[MultiSearchIndividual, ...]:
        while self.pending or self.active:
            self.poll()
            if self.pending or self.active:
                time.sleep(0.1)
        self._refresh_group_states()
        return tuple(self.completed[key] for key in sorted(self.completed))

    def wait_for_capacity(self, additional_groups: int, max_outstanding_groups: int) -> None:
        while self.outstanding_group_count() + int(additional_groups) > int(max_outstanding_groups):
            self.poll()
            if self.outstanding_group_count() + int(additional_groups) <= int(max_outstanding_groups):
                return
            time.sleep(0.05)

    def outstanding_group_count(self) -> int:
        return sum(state in {"queued", "running"} for state in self.group_states.values())

    def pending_candidates(self) -> tuple[SearchCandidate, ...]:
        hashes = {pair.task.candidate.ast_hash for pair in self.pending} | {
            active.pair.task.candidate.ast_hash for active in self.active.values()
        }
        return tuple(self.group_candidates[key] for key in sorted(hashes))

    def counts(self) -> dict[str, int]:
        output: dict[str, int] = {}
        for state in self.pair_states.values():
            output[f"pair_{state}"] = output.get(f"pair_{state}", 0) + 1
        for state in self.group_states.values():
            output[f"group_{state}"] = output.get(f"group_{state}", 0) + 1
        output.update(active_workers=len(self.active), pending_pairs=len(self.pending), cache_hits=self.cache_hits)
        return output

    def shutdown(self) -> None:
        self.pending.clear()
        for task_id, active in list(self.active.items()):
            terminate_process(active.process)
            close_queue(active.output_queue)
            self.active.pop(task_id, None)

    # Auxiliary SR work such as LLM calls must use the same portable process
    # policy as numerical workers. Single- and multi-case controllers share
    # this scheduler contract.
    _context = staticmethod(process_context)

    def _complete_group_if_ready(self, ast_hash: str) -> bool:
        if ast_hash in self.completed:
            return True
        rows = self.group_results.get(ast_hash, {})
        if set(rows) != set(self.case_names):
            return False
        individual = aggregate_case_results(
            self.group_candidates[ast_hash],
            rows,
            self.case_names,
            self.objective,
            result_path=str(self.result_dir / ast_hash / "aggregate.json"),
            provenance=self.provenance.get(ast_hash, ()),
        )
        _write_aggregate(individual, Path(individual.result_path))
        self.completed[ast_hash] = individual
        self.group_states[ast_hash] = "completed"
        self._log(
            "group_completed",
            individual.candidate,
            score=individual.score,
            mean_gap_pct=individual.mean_gap_pct,
            mean_top_k_gap_pct=individual.mean_top_k_gap_pct,
            worst_case=individual.worst_case,
        )
        return True

    def _retry_or_fail(self, pair: PendingPair, error: str) -> None:
        task = pair.task
        if pair.attempt < self.max_pair_retries:
            retry = PendingPair(pair.pair_key, task, pair.attempt + 1)
            self.pending.append(retry)
            self.pair_states[pair.pair_key] = "queued"
            self._log("retry_scheduled", task.candidate, case_name=task.case_name, pair_key=pair.pair_key, attempt=retry.attempt, error=str(error))
            return
        self.pair_states[pair.pair_key] = "failed"
        self.pair_failures[pair.pair_key] = str(error)
        self.group_states[task.candidate.ast_hash] = "failed"
        self._log("failed", task.candidate, case_name=task.case_name, pair_key=pair.pair_key, attempts=pair.attempt + 1, error=str(error))

    def _fail_timed_out_group(self, pair: PendingPair, error: str) -> None:
        """A deterministic hard timeout makes the entire group ineligible."""
        candidate = pair.task.candidate
        ast_hash = candidate.ast_hash
        self.pair_states[pair.pair_key] = "failed"
        self.pair_failures[pair.pair_key] = error
        self.group_states[ast_hash] = "failed"
        self._log(
            "failed",
            candidate,
            case_name=pair.task.case_name,
            pair_key=pair.pair_key,
            attempts=pair.attempt + 1,
            error=error,
        )

        retained = []
        for queued in self.pending:
            if queued.task.candidate.ast_hash != ast_hash:
                retained.append(queued)
                continue
            self.pair_states[queued.pair_key] = "cancelled"
            self._log(
                "cancelled",
                candidate,
                case_name=queued.task.case_name,
                pair_key=queued.pair_key,
                reason="group_hard_timeout",
            )
        self.pending = retained
        for task_id, active in list(self.active.items()):
            other = active.pair
            if other.task.candidate.ast_hash != ast_hash:
                continue
            terminate_process(active.process)
            close_queue(active.output_queue)
            self.active.pop(task_id, None)
            self.pair_states[other.pair_key] = "cancelled"
            self._log(
                "cancelled",
                candidate,
                case_name=other.task.case_name,
                pair_key=other.pair_key,
                reason="group_hard_timeout",
            )

    def _refresh_group_states(self) -> None:
        for ast_hash in self.group_candidates:
            if ast_hash in self.completed or self.group_states.get(ast_hash) == "failed":
                continue
            states = [self.pair_states.get(self._pair_key(ast_hash, name), "missing") for name in self.case_names]
            self.group_states[ast_hash] = "failed" if any(state == "failed" for state in states) else "running" if any(state == "running" for state in states) else "queued"

    def _start_available(self) -> None:
        context = process_context()
        while self.pending and len(self.active) < self.config.optimizer_workers:
            if self.active and self.config.memory_reserve_gb > 0 and available_memory_gb() < self.config.memory_reserve_gb:
                break
            pair = self.pending.pop(0)
            task_id = self._next_task_id
            self._next_task_id += 1
            queue = context.Queue(maxsize=1)
            process = context.Process(target=run_single_threaded_worker, args=(self.worker_target, task_id, pair.task, queue))
            process.daemon = False
            process.start()
            self.active[task_id] = ActivePair(pair, process, queue, time.monotonic())
            self.pair_states[pair.pair_key] = "running"
            self.group_states[pair.task.candidate.ast_hash] = "running"
            self._log("started", pair.task.candidate, case_name=pair.task.case_name, pair_key=pair.pair_key, task_id=task_id, attempt=pair.attempt, active_workers=len(self.active))

    def _log(self, event: str, candidate: SearchCandidate, **detail: object) -> None:
        if self.event_log is None:
            return
        row = {"time": time.time(), "event": event, "ast_hash": candidate.ast_hash, "iteration": candidate.iteration, "island": candidate.island, "source": candidate.source, **detail}
        with self.event_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    @staticmethod
    def _pair_key(ast_hash: str, case_name: str) -> str:
        return f"{ast_hash}::{case_name}"


def _write_aggregate(individual: MultiSearchIndividual, path: Path) -> None:
    payload = {
        "schema_version": 1,
        "mode": "multi_standard_aggregate",
        "ast_hash": individual.ast_hash,
        "ast": individual.candidate.ast,
        "formula": individual.candidate.formula,
        "case_order": list(individual.case_order),
        "case_results": {name: {"policy_cost": row.policy_cost, "dp_cost": row.dp_cost, "gap_pct": row.gap_pct, "params": dict(row.params), "status": row.status, "route": row.route, "optimizer_seconds": row.optimizer_seconds, "result_path": row.result_path} for name, row in individual.case_results.items()},
        "mean_gap_pct": individual.mean_gap_pct,
        "mean_top_k_gap_pct": individual.mean_top_k_gap_pct,
        "worst_gap_pct": individual.worst_gap_pct,
        "worst_case": individual.worst_case,
        "top_k_cases": list(individual.top_k_cases),
        "score": individual.score,
        "provenance": list(individual.provenance),
        "completed_at": individual.completed_at,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


__all__ = ["MultiCaseEvaluationScheduler"]
