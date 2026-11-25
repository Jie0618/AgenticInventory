"""Reusable adaptive subprocess queue with retry and memory admission."""

from __future__ import annotations

import signal
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Hashable, IO, Sequence


Task = tuple[object, ...]

DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_COOLDOWN_SECONDS = 30.0
DEFAULT_MEMORY_SAFETY_RESERVE_GB = 100.0
DEFAULT_ESTIMATED_TASK_MEMORY_GB = 8.0
DEFAULT_MEMORY_EWMA_ALPHA = 0.2
DEFAULT_MEMORY_SAFETY_MARGIN = 0.2
DEFAULT_POLL_SECONDS = 0.2


@dataclass(frozen=True)
class QueuedTask:
    task: Task
    retries: int = 0


@dataclass
class RunningTask:
    queued: QueuedTask
    process: subprocess.Popen
    log_stream: IO[str] | None
    destination: Path
    memory_estimate_bytes: int = int(DEFAULT_ESTIMATED_TASK_MEMORY_GB * 1024**3)
    peak_rss_bytes: int = 0
    deadline: float | None = None


@dataclass(frozen=True)
class QueueRunSummary:
    total_tasks: int
    completed_tasks: int
    skipped_tasks: int
    retry_count: int
    max_active_workers: int
    failures: tuple[tuple[str, str, int], ...]


def _default_group_key(task: Task) -> Hashable:
    return task[5], task[0]


def _always_ready(_task: Task) -> bool:
    return True


def _zero_reservation(_task: Task) -> int:
    return 0


def _no_cleanup(_task: Task) -> None:
    return None


@dataclass(frozen=True)
class TaskResourceHooks:
    """Optional case-level cache hooks used by memory-aware schedulers."""

    group_key: Callable[[Task], Hashable] = _default_group_key
    cache_ready: Callable[[Task], bool] = _always_ready
    uncached_reservation_bytes: Callable[[Task], int] = _zero_reservation
    cleanup_group: Callable[[Task], None] = _no_cleanup


def _is_retryable_returncode(returncode: int) -> bool:
    """Retry only SIGKILL, including the shell-style 128+signal encoding."""

    return int(returncode) in {-int(signal.SIGKILL), 128 + int(signal.SIGKILL)}


def _memory_available_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _process_rss_bytes(pid: int) -> int | None:
    """Prefer PSS so shared mmap pages are not charged once per worker."""

    try:
        rollup = Path(f"/proc/{pid}/smaps_rollup")
        if rollup.exists():
            for line in rollup.read_text(encoding="ascii").splitlines():
                if line.startswith("Pss:"):
                    return int(line.split()[1]) * 1024
        for line in Path(f"/proc/{pid}/status").read_text(encoding="ascii").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _ewma(previous: float, observation: float, alpha: float) -> float:
    return (1.0 - alpha) * previous + alpha * observation


def _close_log(running: RunningTask) -> None:
    if running.log_stream is not None:
        running.log_stream.close()
        running.log_stream = None


def run_task_queue(
    tasks: Sequence[Task],
    *,
    workers: int,
    launcher: Callable[[QueuedTask], RunningTask | None],
    result_validator: Callable[[RunningTask], bool],
    retryable_failure: Callable[[RunningTask, int], bool] | None = None,
    retry_launch_errors: bool = False,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_cooldown_seconds: float = DEFAULT_RETRY_COOLDOWN_SECONDS,
    memory_safety_reserve_gb: float = DEFAULT_MEMORY_SAFETY_RESERVE_GB,
    estimated_task_memory_gb: float = DEFAULT_ESTIMATED_TASK_MEMORY_GB,
    memory_ewma_alpha: float = DEFAULT_MEMORY_EWMA_ALPHA,
    memory_safety_margin: float = DEFAULT_MEMORY_SAFETY_MARGIN,
    memory_available_bytes: Callable[[], int | None] = _memory_available_bytes,
    process_rss_bytes: Callable[[int], int | None] = _process_rss_bytes,
    resource_hooks: TaskResourceHooks = TaskResourceHooks(),
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    emit: Callable[[str], None] = print,
) -> QueueRunSummary:
    """Run tasks under a hard worker cap and adaptive memory admission.

    The queue is agnostic to benchmark/gap-study files and numerical code.
    Callers supply process launch/result validation and optional case-level
    cache hooks. By default only SIGKILL tasks are returned to the queue up to
    ``max_retries``. Callers whose subprocess protocol is idempotent may opt in
    to retry other invalid results and transient launch errors.
    """

    if workers <= 0:
        raise ValueError("workers must be positive")
    if max_retries < 0:
        raise ValueError("max_retries must be nonnegative")
    if retry_cooldown_seconds < 0.0:
        raise ValueError("retry cooldown seconds must be nonnegative")
    if memory_safety_reserve_gb < 0.0 or estimated_task_memory_gb <= 0.0:
        raise ValueError("memory reserve must be nonnegative and task estimate positive")
    if not 0.0 < memory_ewma_alpha <= 1.0:
        raise ValueError("memory_ewma_alpha must be in (0, 1]")
    if memory_safety_margin < 0.0:
        raise ValueError("memory_safety_margin must be nonnegative")
    if poll_seconds <= 0.0:
        raise ValueError("poll_seconds must be positive")

    pending = deque(QueuedTask(task) for task in tasks)
    active: dict[int, RunningTask] = {}
    failures: list[tuple[str, str, int]] = []
    completed_tasks = 0
    skipped_tasks = 0
    retry_count = 0
    settled_tasks = 0
    max_active_workers = 0
    admission_blocked_until = 0.0
    safety_reserve_bytes = int(memory_safety_reserve_gb * 1024**3)
    estimated_task_bytes = int(estimated_task_memory_gb * 1024**3)
    memory_ewma_bytes = float(estimated_task_bytes)
    memory_admission_paused = False
    reserved_cache_bytes: dict[Hashable, int] = {}
    reserved_cache_tasks: dict[Hashable, Task] = {}

    def sample_memory() -> None:
        for running in active.values():
            rss = process_rss_bytes(running.process.pid)
            if rss is not None:
                running.memory_estimate_bytes = max(running.memory_estimate_bytes, rss)
                running.peak_rss_bytes = max(running.peak_rss_bytes, rss)

    def update_memory_estimate(running: RunningTask, *, killed: bool = False) -> None:
        nonlocal memory_ewma_bytes
        if running.peak_rss_bytes > 0:
            observation = float(running.peak_rss_bytes)
        elif killed:
            observation = memory_ewma_bytes * 1.5
        else:
            return
        memory_ewma_bytes = _ewma(memory_ewma_bytes, observation, memory_ewma_alpha)
        if killed:
            memory_ewma_bytes = max(memory_ewma_bytes, observation)

    def admission_estimate_bytes() -> int:
        return int(memory_ewma_bytes * (1.0 + memory_safety_margin))

    def refresh_cache_reservations() -> None:
        for key, task in list(reserved_cache_tasks.items()):
            if resource_hooks.cache_ready(task):
                reserved_cache_tasks.pop(key, None)
                reserved_cache_bytes.pop(key, None)

    def cache_reservation_for(task: Task) -> tuple[Hashable, int]:
        key = resource_hooks.group_key(task)
        if resource_hooks.cache_ready(task):
            return key, 0
        if key in reserved_cache_tasks:
            return key, -1
        return key, max(0, int(resource_hooks.uncached_reservation_bytes(task)))

    def memory_allows_admission(extra_reserved_bytes: int = 0) -> bool:
        available = memory_available_bytes()
        if available is None:
            return True
        return available >= (
            safety_reserve_bytes
            + admission_estimate_bytes()
            + sum(reserved_cache_bytes.values())
            + int(extra_reserved_bytes)
        )

    def cleanup_if_group_settled(task: Task) -> None:
        key = resource_hooks.group_key(task)
        if any(resource_hooks.group_key(queued.task) == key for queued in pending):
            return
        if any(resource_hooks.group_key(running.queued.task) == key for running in active.values()):
            return
        resource_hooks.cleanup_group(task)

    def emit_settled(running: RunningTask, returncode: int, *, skipped: bool = False) -> None:
        nonlocal settled_tasks, completed_tasks, skipped_tasks
        settled_tasks += 1
        completed_tasks += 1
        skipped_tasks += int(skipped)
        case, policy = running.queued.task[:2]
        emit(
            f"completed {settled_tasks}/{len(tasks)} {case}/{policy} "
            f"returncode={returncode} skipped={str(skipped).lower()} "
            f"active={len(active)} target={workers}"
        )

    while pending or active:
        now = monotonic()
        sample_memory()
        refresh_cache_reservations()

        if now >= admission_blocked_until:
            while pending and len(active) < workers:
                queued = None
                cache_key = None
                cache_reservation = 0
                memory_blocked = False
                # Policies are normally adjacent by instance.  Scan one full
                # queue turn so an uncached case whose reference is already
                # in flight cannot occupy the remaining worker slots with
                # file-lock waiters.  Once its cache appears, all remaining
                # policies for that case become independently admissible.
                for _ in range(len(pending)):
                    candidate = pending.popleft()
                    candidate_key, candidate_reservation = cache_reservation_for(candidate.task)
                    if candidate_reservation < 0:
                        pending.append(candidate)
                        continue
                    if not memory_allows_admission(candidate_reservation):
                        memory_blocked = True
                        pending.append(candidate)
                        continue
                    queued = candidate
                    cache_key = candidate_key
                    cache_reservation = candidate_reservation
                    break
                if queued is None:
                    if not memory_blocked:
                        break
                    if not memory_admission_paused:
                        available = memory_available_bytes()
                        available_gb = "unknown" if available is None else f"{available / 1024**3:.1f}"
                        emit(
                            f"memory admission paused available_gb={available_gb} "
                            f"reserve_gb={memory_safety_reserve_gb:g} "
                            f"estimate_gb={admission_estimate_bytes() / 1024**3:.1f} "
                            f"reserved_cache_gb={sum(reserved_cache_bytes.values()) / 1024**3:.1f} "
                            f"target={workers}"
                        )
                        memory_admission_paused = True
                    break
                assert cache_key is not None
                if memory_admission_paused:
                    emit(
                        f"memory admission resumed target={workers} "
                        f"estimate_gb={admission_estimate_bytes() / 1024**3:.1f}"
                    )
                    memory_admission_paused = False
                if cache_reservation > 0:
                    reserved_cache_tasks[cache_key] = queued.task
                    reserved_cache_bytes[cache_key] = cache_reservation
                try:
                    running = launcher(queued)
                except OSError as exc:
                    reserved_cache_tasks.pop(cache_key, None)
                    reserved_cache_bytes.pop(cache_key, None)
                    case, policy = queued.task[:2]
                    if retry_launch_errors and queued.retries < max_retries:
                        retry_count += 1
                        pending.append(QueuedTask(queued.task, queued.retries + 1))
                        admission_blocked_until = max(
                            admission_blocked_until,
                            monotonic() + retry_cooldown_seconds,
                        )
                        emit(
                            f"retry {retry_count} {case}/{policy} launch_error=true "
                            f"error={type(exc).__name__}:{exc.errno} "
                            f"attempt={queued.retries + 1}/{max_retries + 1} "
                            f"active={len(active)} target={workers}"
                        )
                        continue
                    failures.append((case, policy, 1))
                    settled_tasks += 1
                    emit(
                        f"failed {settled_tasks}/{len(tasks)} {case}/{policy} "
                        f"launch_error=true error={type(exc).__name__}:{exc.errno} "
                        f"retries={queued.retries}"
                    )
                    cleanup_if_group_settled(queued.task)
                    continue
                if running is None:
                    reserved_cache_tasks.pop(cache_key, None)
                    reserved_cache_bytes.pop(cache_key, None)
                    completed_tasks += 1
                    skipped_tasks += 1
                    settled_tasks += 1
                    case, policy = queued.task[:2]
                    emit(
                        f"completed {settled_tasks}/{len(tasks)} {case}/{policy} "
                        f"returncode=0 skipped=true active={len(active)} target={workers}"
                    )
                    cleanup_if_group_settled(queued.task)
                    continue
                active[running.process.pid] = running
                max_active_workers = max(max_active_workers, len(active))

        finished: list[tuple[int, RunningTask, int]] = []
        for pid, running in list(active.items()):
            returncode = running.process.poll()
            if returncode is None and running.deadline is not None and monotonic() >= running.deadline:
                running.process.kill()
                running.process.wait()
                returncode = 124
                emit(f"timeout {running.queued.task[0]}/{running.queued.task[1]}")
            if returncode is not None:
                finished.append((pid, running, int(returncode)))
        if not finished:
            if active or pending:
                sleep(poll_seconds)
            continue

        for pid, running, returncode in finished:
            active.pop(pid, None)
            _close_log(running)
            task = running.queued.task
            if result_validator(running):
                update_memory_estimate(running)
                emit_settled(running, returncode)
                cleanup_if_group_settled(task)
                continue
            should_retry = (
                _is_retryable_returncode(returncode)
                if retryable_failure is None
                else bool(retryable_failure(running, returncode))
            )
            if should_retry and running.queued.retries < max_retries:
                update_memory_estimate(
                    running,
                    killed=_is_retryable_returncode(returncode),
                )
                cache_key = resource_hooks.group_key(task)
                if not resource_hooks.cache_ready(task):
                    reserved_cache_tasks.pop(cache_key, None)
                    reserved_cache_bytes.pop(cache_key, None)
                retry_count += 1
                pending.append(QueuedTask(task, running.queued.retries + 1))
                admission_blocked_until = max(
                    admission_blocked_until, monotonic() + retry_cooldown_seconds
                )
                emit(
                    f"retry {retry_count} {task[0]}/{task[1]} returncode={returncode} "
                    f"attempt={running.queued.retries + 1}/{max_retries + 1} "
                    f"active={len(active)} target={workers} "
                    f"estimate_gb={admission_estimate_bytes() / 1024**3:.1f}"
                )
                continue
            update_memory_estimate(running)
            cache_key = resource_hooks.group_key(task)
            reserved_cache_tasks.pop(cache_key, None)
            reserved_cache_bytes.pop(cache_key, None)
            failures.append((task[0], task[1], returncode))
            settled_tasks += 1
            emit(
                f"failed {settled_tasks}/{len(tasks)} {task[0]}/{task[1]} "
                f"returncode={returncode} retries={running.queued.retries}"
            )
            cleanup_if_group_settled(task)

    return QueueRunSummary(
        total_tasks=len(tasks),
        completed_tasks=completed_tasks,
        skipped_tasks=skipped_tasks,
        retry_count=retry_count,
        max_active_workers=max_active_workers,
        failures=tuple(failures),
    )


__all__ = [
    "DEFAULT_ESTIMATED_TASK_MEMORY_GB",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_MEMORY_EWMA_ALPHA",
    "DEFAULT_MEMORY_SAFETY_MARGIN",
    "DEFAULT_MEMORY_SAFETY_RESERVE_GB",
    "DEFAULT_POLL_SECONDS",
    "DEFAULT_RETRY_COOLDOWN_SECONDS",
    "QueuedTask",
    "QueueRunSummary",
    "RunningTask",
    "Task",
    "TaskResourceHooks",
    "run_task_queue",
]
