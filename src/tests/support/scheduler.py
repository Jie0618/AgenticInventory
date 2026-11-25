"""Deterministic worker transport for scheduler state-machine tests.

Only the process and clock are replaced. Production scheduling, cache loading,
result validation, cancellation and audit logging still run normally.
"""
from contextlib import ExitStack, contextmanager
from queue import Queue
from unittest.mock import patch

from runtime.processes import run_single_threaded_worker


class Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    perf_counter = monotonic

    def time(self):
        return 1_700_000_000 + self.now

    def sleep(self, seconds):
        self.now += max(0.0, seconds)
        if self.now > 120:
            raise AssertionError("scheduler did not settle within 120 simulated seconds")


class _Queue(Queue):
    def close(self):
        pass


class _Process:
    def __init__(self, clock, delay, *, target, args):
        assert target is run_single_threaded_worker
        self.clock = clock
        self.worker, self.task_id, self.task, self.queue = args
        self.delay = delay(self.worker, self.task)
        self.exitcode = None
        self.started = False

    def start(self):
        self.started = True
        self.ready_at = None if self.delay is None else self.clock.now + self.delay

    def is_alive(self):
        if not self.started or self.exitcode is not None:
            return False
        if self.ready_at is not None and self.clock.now >= self.ready_at:
            self.worker(self.task_id, self.task, self.queue)
            self.exitcode = 0
            return False
        return True

    def join(self, timeout=None):
        # Scheduler joins completed or terminated workers; no wall-clock wait.
        assert not self.is_alive(), "test must terminate a live worker before joining"

    def terminate(self):
        self.exitcode = -15

    kill = terminate


class _Context:
    Queue = _Queue

    def __init__(self, clock, delay):
        self.clock = clock
        self.delay = delay
        self.processes = []

    def Process(self, **kwargs):
        process = _Process(self.clock, self.delay, **kwargs)
        self.processes.append(process)
        return process


@contextmanager
def scheduler_workers(delay):
    """delay(worker, task) returns runtime, or None for a hung worker."""
    clock = Clock()
    context = _Context(clock, delay)
    with ExitStack() as stack:
        for module in ("sr.evaluation.scheduler", "sr.evaluation.multi"):
            stack.enter_context(patch(f"{module}.process_context", return_value=context))
            stack.enter_context(patch(f"{module}.time", clock))
        stack.enter_context(patch("sr.multi.initialization.time", clock))
        yield clock
