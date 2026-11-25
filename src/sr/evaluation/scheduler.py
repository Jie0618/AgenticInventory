"""Generic process scheduler for independent SR candidate evaluations."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generic, Optional, Protocol, Sequence, TypeVar

from runtime.processes import (
    available_memory_gb,
    close_queue,
    process_context,
    read_worker_message,
    run_single_threaded_worker,
    terminate_process,
)
from sr.core.models import SearchCandidate, SearchConfig, SearchIndividual


class EvaluationTaskProtocol(Protocol):
    candidate: SearchCandidate
    result_path: Path
    evaluation_signature: str


TaskT = TypeVar("TaskT", bound=EvaluationTaskProtocol)
WorkerTarget = Callable[[int, Any, Any], None]
ResultLoader = Callable[[TaskT], Optional[SearchIndividual]]


@dataclass
class ActiveEvaluation(Generic[TaskT]):
    task: TaskT
    process: Any
    output_queue: Any
    started_at: float


class AsyncEvaluationScheduler(Generic[TaskT]):
    """Queue, execute, validate, and audit independent evaluation tasks."""

    def __init__(
        self,
        config: SearchConfig,
        *,
        result_dir: Path,
        event_log: Path | None,
        worker_target: WorkerTarget,
        result_loader: Callable[[TaskT], Optional[SearchIndividual]],
    ) -> None:
        self.config = config
        self.result_dir = Path(result_dir)
        self.event_log = None if event_log is None else Path(event_log)
        self.worker_target = worker_target
        self.result_loader = result_loader
        self.pending: list[TaskT] = []
        self.active: dict[int, ActiveEvaluation[TaskT]] = {}
        self.states: dict[str, str] = {}
        self.evaluation_signatures: dict[str, str] = {}
        self.completed: dict[str, SearchIndividual] = {}
        self.failures: dict[str, str] = {}
        self._next_task_id = 0
        self.result_dir.mkdir(parents=True, exist_ok=True)
        if self.event_log is not None:
            self.event_log.parent.mkdir(parents=True, exist_ok=True)

    def submit_task(self, task: TaskT) -> str:
        candidate = task.candidate
        previous = self.evaluation_signatures.get(candidate.ast_hash)
        if previous is not None and previous != task.evaluation_signature:
            raise ValueError(
                "the same AST hash was submitted with different evaluation inputs"
            )
        self.evaluation_signatures[candidate.ast_hash] = task.evaluation_signature
        cached = self.result_loader(task)
        if cached is not None:
            self.completed[candidate.ast_hash] = cached
            self.states[candidate.ast_hash] = "completed"
            self._log("cache_hit", candidate, result_path=str(task.result_path))
            return "completed"
        state = self.states.get(candidate.ast_hash)
        if state in {"queued", "running", "completed"}:
            return state
        if not self._admit_pending(task):
            return self.states[candidate.ast_hash]
        self.states[candidate.ast_hash] = "queued"
        self._log("submitted", candidate)
        self.poll()
        return self.states[candidate.ast_hash]

    def _admit_pending(self, task: TaskT) -> bool:
        limit = int(self.config.max_outstanding_tasks)
        capacity = limit - len(self.active)
        if capacity <= 0:
            self.states[task.candidate.ast_hash] = "queue_rejected"
            self._log(
                "queue_rejected",
                task.candidate,
                reason="max_outstanding_tasks",
                max_outstanding_tasks=limit,
                outstanding_tasks=self.outstanding_count(),
            )
            return False
        if len(self.pending) < capacity:
            self.pending.append(task)
            self.pending.sort(key=self._pending_priority)
            return True
        worst = max(self.pending, key=self._pending_priority)
        if self._pending_priority(task) >= self._pending_priority(worst):
            self.states[task.candidate.ast_hash] = "queue_rejected"
            self._log("queue_rejected", task.candidate, pending_capacity=capacity)
            return False
        self.pending.remove(worst)
        self.states[worst.candidate.ast_hash] = "queue_evicted"
        self._log(
            "queue_evicted",
            worst.candidate,
            replacement_ast_hash=task.candidate.ast_hash,
            pending_capacity=capacity,
        )
        self.pending.append(task)
        self.pending.sort(key=self._pending_priority)
        return True

    @staticmethod
    def _pending_priority(
        task: EvaluationTaskProtocol,
    ) -> tuple[float, int, float, int, str]:
        candidate = task.candidate
        return (
            float(candidate.detail.get("bellman_regret_pct", float("inf"))),
            candidate.num_parameters,
            float(candidate.expanded_complexity),
            -int(candidate.iteration),
            candidate.ast_hash,
        )

    def poll(self) -> tuple[SearchIndividual, ...]:
        self._start_available()
        now = time.monotonic()
        ready: list[SearchIndividual] = []
        for task_id, active in list(self.active.items()):
            candidate = active.task.candidate
            elapsed = now - active.started_at
            hard_limit = self.config.evaluation_hard_timeout_seconds
            if active.process.is_alive() and elapsed >= hard_limit:
                terminate_process(active.process)
                close_queue(active.output_queue)
                self.active.pop(task_id, None)
                self.states[candidate.ast_hash] = "hard_timeout"
                self.failures[candidate.ast_hash] = (
                    f"exceeded total timeout {hard_limit:g}s"
                )
                self._log("hard_timeout", candidate, elapsed_s=elapsed)
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
                individual = self.result_loader(active.task)
                if individual is not None:
                    self.completed[candidate.ast_hash] = individual
                    self.states[candidate.ast_hash] = "completed"
                    ready.append(individual)
                    self._log(
                        "completed",
                        candidate,
                        elapsed_s=elapsed,
                        result_path=detail,
                        gap_pct=individual.gap_pct,
                    )
                    continue
                status, detail = (
                    "failed",
                    "worker reported completion but result was invalid",
                )
            self.states[candidate.ast_hash] = "failed"
            self.failures[candidate.ast_hash] = detail
            self._log("failed", candidate, elapsed_s=elapsed, error=detail)
        self._start_available()
        return tuple(ready)

    def wait_for_batch(
        self,
        hashes: Sequence[str],
        *,
        wait_seconds: float | None = None,
        idle_active_workers: int | None = None,
    ) -> tuple[SearchIndividual, ...]:
        targets = tuple(dict.fromkeys(hashes))
        if not targets:
            return ()
        started = time.monotonic()
        collected: dict[str, SearchIndividual] = {}
        terminal = {
            "completed",
            "failed",
            "hard_timeout",
            "queue_rejected",
            "queue_evicted",
        }
        while True:
            for row in self.poll():
                if row.ast_hash in targets:
                    collected[row.ast_hash] = row
            for ast_hash in targets:
                if ast_hash in self.completed:
                    collected[ast_hash] = self.completed[ast_hash]
            states = [self.states.get(ast_hash, "missing") for ast_hash in targets]
            if states and all(state in terminal for state in states):
                self._log_batch_gate("all_terminal", started, states)
                return tuple(collected.values())
            if idle_active_workers is not None and not self.pending and len(
                self.active
            ) <= max(0, int(idle_active_workers)):
                self._log_batch_gate("idle_workers", started, states)
                return tuple(collected.values())
            if wait_seconds is not None and time.monotonic() - started >= max(
                0.0, float(wait_seconds)
            ):
                self._log_batch_gate("wait_limit", started, states)
                return tuple(collected.values())
            time.sleep(0.05)

    def wait_for_capacity(
        self, additional_tasks: int, max_outstanding_tasks: int
    ) -> None:
        additional = max(0, int(additional_tasks))
        limit = max(0, int(max_outstanding_tasks))
        started = time.monotonic()
        blocked = False
        while self.outstanding_count() + additional > limit:
            blocked = True
            self.poll()
            if self.outstanding_count() + additional <= limit:
                break
            time.sleep(0.05)
        if blocked:
            self._log_scheduler(
                "capacity_gate_finished",
                waited_seconds=time.monotonic() - started,
                additional_tasks=additional,
                max_outstanding_tasks=limit,
                outstanding_tasks=self.outstanding_count(),
            )

    def outstanding_count(self) -> int:
        return len(self.pending) + len(self.active)

    def drain(self) -> tuple[SearchIndividual, ...]:
        ready: dict[str, SearchIndividual] = {}
        while self.pending or self.active:
            for row in self.poll():
                ready[row.ast_hash] = row
            if self.pending or self.active:
                time.sleep(0.1)
        return tuple(ready.values())

    def shutdown(self) -> None:
        for task in self.pending:
            self.states[task.candidate.ast_hash] = "cancelled"
            self._log(
                "cancelled", task.candidate, reason="scheduler_shutdown_before_start"
            )
        self.pending.clear()
        for task_id, active in list(self.active.items()):
            terminate_process(active.process)
            close_queue(active.output_queue)
            self.states[active.task.candidate.ast_hash] = "cancelled"
            self._log(
                "cancelled",
                active.task.candidate,
                reason="scheduler_shutdown",
                task_id=task_id,
            )
        self.active.clear()

    def counts(self) -> dict[str, int]:
        output: dict[str, int] = {}
        for state in self.states.values():
            output[state] = output.get(state, 0) + 1
        return output

    def _start_available(self) -> None:
        context = process_context()
        while self.pending and len(self.active) < self.config.optimizer_workers:
            if (
                self.active
                and self.config.memory_reserve_gb > 0
                and available_memory_gb() < self.config.memory_reserve_gb
            ):
                break
            task = self.pending.pop(0)
            task_id = self._next_task_id
            self._next_task_id += 1
            output_queue = context.Queue(maxsize=1)
            process = context.Process(
                target=run_single_threaded_worker,
                args=(self.worker_target, task_id, task, output_queue),
            )
            process.daemon = False
            process.start()
            self.active[task_id] = ActiveEvaluation(
                task, process, output_queue, time.monotonic()
            )
            self.states[task.candidate.ast_hash] = "running"
            self._log(
                "started",
                task.candidate,
                task_id=task_id,
                active_count=len(self.active),
            )

    def _log_batch_gate(
        self, reason: str, started: float, states: Sequence[str]
    ) -> None:
        self._append_log(
            {
                "time": time.time(),
                "event": "batch_wait_finished",
                "reason": reason,
                "waited_seconds": time.monotonic() - started,
                "batch_states": {
                    state: states.count(state) for state in sorted(set(states))
                },
                "scheduler": self.counts(),
            }
        )

    def _log(
        self, event: str, candidate: SearchCandidate, **detail: object
    ) -> None:
        self._append_log(
            {
                "time": time.time(),
                "event": event,
                "ast_hash": candidate.ast_hash,
                "iteration": candidate.iteration,
                "island": candidate.island,
                "source": candidate.source,
                **detail,
            }
        )

    def _log_scheduler(self, event: str, **detail: object) -> None:
        self._append_log({"time": time.time(), "event": event, **detail})

    def _append_log(self, row: dict[str, object]) -> None:
        if self.event_log is None:
            return
        with self.event_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    _context = staticmethod(process_context)
    _read_message = staticmethod(read_worker_message)
    _terminate = staticmethod(terminate_process)
    _close_queue = staticmethod(close_queue)
    _available_memory_gb = staticmethod(available_memory_gb)


__all__ = [
    "ActiveEvaluation",
    "AsyncEvaluationScheduler",
    "EvaluationTaskProtocol",
]
