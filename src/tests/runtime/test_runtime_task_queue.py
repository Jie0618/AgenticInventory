from __future__ import annotations

import unittest
from pathlib import Path

from runtime.task_queue import (
    DEFAULT_MEMORY_EWMA_ALPHA,
    QueuedTask,
    RunningTask,
    TaskResourceHooks,
    _ewma,
    _is_retryable_returncode,
    run_task_queue,
)


def test_runtime_public_exports_are_available():
    import runtime
    from runtime.artifacts import atomic_json, load_result_records

    assert all(hasattr(runtime, name) for name in runtime.__all__)
    assert runtime.atomic_json is atomic_json
    assert runtime.load_result_records is load_result_records


class _FakeProcess:
    _next_pid = 10_000

    def __init__(self, returncode: int, polls_before_exit: int = 1) -> None:
        self.pid = _FakeProcess._next_pid
        _FakeProcess._next_pid += 1
        self._returncode = returncode
        self._polls_left = polls_before_exit
        self.returncode: int | None = None

    def poll(self) -> int | None:
        if self._polls_left:
            self._polls_left -= 1
            return None
        self.returncode = self._returncode
        return self.returncode


def _task(index: int) -> tuple[str, str, str, int, str, str]:
    return (f"case{index}", "constant_order", "/tmp/task-queue-tests", 1, "fast", "test")


class TaskQueueTests(unittest.TestCase):
    def test_timeout_kills_and_reaps_worker_without_retrying(self):
        class HungProcess(_FakeProcess):
            def kill(self):
                self.killed = True
                self._polls_left = 0
                self._returncode = -9

            def wait(self):
                self.reaped = True
                return self.poll()

        process = HungProcess(0, polls_before_exit=100)
        summary = run_task_queue(
            [_task(0)], workers=1,
            launcher=lambda task: RunningTask(task, process, None, Path("/tmp/unused"), deadline=0),
            result_validator=lambda task: False, monotonic=lambda: 1,
            sleep=lambda _: None, **self._memory_kwargs())
        self.assertTrue(process.killed)
        self.assertTrue(process.reaped)
        self.assertEqual(summary.failures, (("case0", "constant_order", 124),))
        self.assertEqual(summary.retry_count, 0)

    def test_memory_estimate_uses_ewma_not_historical_maximum(self) -> None:
        gib = 1024**3
        estimate = 8.0 * gib
        estimate = _ewma(estimate, 20.0 * gib, DEFAULT_MEMORY_EWMA_ALPHA)
        self.assertAlmostEqual(estimate / gib, 10.4, places=6)
        estimate = _ewma(estimate, 5.0 * gib, DEFAULT_MEMORY_EWMA_ALPHA)
        self.assertAlmostEqual(estimate / gib, 9.32, places=6)
        self.assertLess(estimate, 20.0 * gib)

    @staticmethod
    def _memory_kwargs() -> dict[str, object]:
        return {
            "memory_available_bytes": lambda: 10**15,
            "estimated_task_memory_gb": 1.0,
            "memory_safety_reserve_gb": 0.0,
        }

    def test_only_sigkill_is_retryable(self) -> None:
        self.assertTrue(_is_retryable_returncode(-9))
        self.assertTrue(_is_retryable_returncode(137))
        self.assertFalse(_is_retryable_returncode(-11))
        self.assertFalse(_is_retryable_returncode(1))

    def test_killed_task_is_requeued_at_most_five_times(self) -> None:
        attempts = 0

        def launcher(queued: QueuedTask) -> RunningTask:
            nonlocal attempts
            attempts += 1
            code = -9 if attempts <= 5 else 0
            process = _FakeProcess(code)
            return RunningTask(queued, process, None, Path("/tmp/unused.json"))

        summary = run_task_queue(
            [_task(0)],
            workers=60,
            max_retries=5,
            retry_cooldown_seconds=0.0,
            poll_seconds=0.001,
            launcher=launcher,
            result_validator=lambda running: running.process.returncode == 0,
            sleep=lambda _seconds: None,
            **self._memory_kwargs(),
        )
        self.assertEqual(attempts, 6)
        self.assertEqual(summary.retry_count, 5)
        self.assertEqual(summary.completed_tasks, 1)
        self.assertEqual(summary.failures, ())

    def test_retry_limit_reports_a_persistent_kill(self) -> None:
        attempts = 0

        def launcher(queued: QueuedTask) -> RunningTask:
            nonlocal attempts
            attempts += 1
            return RunningTask(
                queued,
                _FakeProcess(-9),
                None,
                Path("/tmp/unused.json"),
            )

        summary = run_task_queue(
            [_task(1)],
            workers=60,
            max_retries=5,
            retry_cooldown_seconds=0.0,
            poll_seconds=0.001,
            launcher=launcher,
            result_validator=lambda running: running.process.returncode == 0,
            sleep=lambda _seconds: None,
            **self._memory_kwargs(),
        )
        self.assertEqual(attempts, 6)
        self.assertEqual(summary.retry_count, 5)
        self.assertEqual(summary.completed_tasks, 0)
        self.assertEqual(summary.failures, (("case1", "constant_order", -9),))

    def test_opt_in_retries_nonzero_exit(self) -> None:
        attempts = 0

        def launcher(queued: QueuedTask) -> RunningTask:
            nonlocal attempts
            attempts += 1
            return RunningTask(
                queued,
                _FakeProcess(1 if attempts == 1 else 0),
                None,
                Path("/tmp/unused.json"),
            )

        summary = run_task_queue(
            [_task(2)],
            workers=1,
            max_retries=2,
            retry_cooldown_seconds=0.0,
            poll_seconds=0.001,
            launcher=launcher,
            result_validator=lambda running: running.process.returncode == 0,
            retryable_failure=lambda _running, _returncode: True,
            sleep=lambda _seconds: None,
            **self._memory_kwargs(),
        )
        self.assertEqual(attempts, 2)
        self.assertEqual(summary.retry_count, 1)
        self.assertEqual(summary.completed_tasks, 1)
        self.assertEqual(summary.failures, ())

    def test_opt_in_retries_launch_error(self) -> None:
        attempts = 0

        def launcher(queued: QueuedTask) -> RunningTask:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise OSError(11, "temporarily unavailable")
            return RunningTask(
                queued,
                _FakeProcess(0),
                None,
                Path("/tmp/unused.json"),
            )

        summary = run_task_queue(
            [_task(3)],
            workers=1,
            max_retries=2,
            retry_cooldown_seconds=0.0,
            poll_seconds=0.001,
            launcher=launcher,
            result_validator=lambda running: running.process.returncode == 0,
            retry_launch_errors=True,
            sleep=lambda _seconds: None,
            **self._memory_kwargs(),
        )
        self.assertEqual(attempts, 2)
        self.assertEqual(summary.retry_count, 1)
        self.assertEqual(summary.completed_tasks, 1)
        self.assertEqual(summary.failures, ())

    def test_worker_cap_reaches_requested_sixty(self) -> None:
        def launcher(queued: QueuedTask) -> RunningTask:
            process = _FakeProcess(0, polls_before_exit=10)
            # The scheduler owns the authoritative count; this assertion is
            # retained here to ensure all tasks are real child-like entries.
            self.assertIsNotNone(process.pid)
            return RunningTask(queued, process, None, Path("/tmp/unused.json"))

        summary = run_task_queue(
            [_task(index) for index in range(120)],
            workers=60,
            max_retries=5,
            retry_cooldown_seconds=0.0,
            poll_seconds=0.001,
            launcher=launcher,
            result_validator=lambda running: running.process.returncode == 0,
            sleep=lambda _seconds: None,
            **self._memory_kwargs(),
        )
        self.assertEqual(summary.completed_tasks, 120)
        self.assertEqual(summary.failures, ())
        self.assertEqual(summary.max_active_workers, 60)

    def test_uncached_instances_are_admitted_single_flight_across_workers(self) -> None:
        cache_ready: set[str] = set()
        launch_order: list[tuple[str, str]] = []
        tasks = [
            (case, policy, "/tmp/gap-runner-tests", 5, "certify", "benchmark")
            for case in ("case0", "case1", "case2")
            for policy in ("p0", "p1")
        ]

        def launcher(queued: QueuedTask) -> RunningTask:
            launch_order.append(queued.task[:2])
            return RunningTask(
                queued,
                _FakeProcess(0, polls_before_exit=3),
                None,
                Path("/tmp/unused.json"),
            )

        def validate(running: RunningTask) -> bool:
            cache_ready.add(running.queued.task[0])
            return running.process.returncode == 0

        summary = run_task_queue(
            tasks,
            workers=3,
            poll_seconds=0.001,
            launcher=launcher,
            result_validator=validate,
            sleep=lambda _seconds: None,
            resource_hooks=TaskResourceHooks(
                cache_ready=lambda task: task[0] in cache_ready,
                uncached_reservation_bytes=lambda _task: 1024,
            ),
            **self._memory_kwargs(),
        )
        self.assertEqual(launch_order[:3], [("case0", "p0"), ("case1", "p0"), ("case2", "p0")])
        self.assertEqual(summary.completed_tasks, 6)
        self.assertEqual(summary.max_active_workers, 3)

    def test_memory_admission_limits_active_workers(self) -> None:
        reserve = 100 * 1024**3
        estimate = 8 * 1024**3
        admission_cost = int(estimate * 1.2)
        available = reserve + 2 * admission_cost

        def memory_available() -> int:
            return available

        def launcher(queued: QueuedTask) -> RunningTask:
            nonlocal available
            available -= admission_cost
            return RunningTask(
                queued,
                _FakeProcess(0, polls_before_exit=1),
                None,
                Path("/tmp/unused.json"),
            )

        def validate(running: RunningTask) -> bool:
            nonlocal available
            available += admission_cost
            return running.process.returncode == 0

        summary = run_task_queue(
            [_task(index) for index in range(8)],
            workers=60,
            poll_seconds=0.001,
            launcher=launcher,
            result_validator=validate,
            sleep=lambda _seconds: None,
            memory_available_bytes=memory_available,
            memory_safety_reserve_gb=100.0,
            estimated_task_memory_gb=8.0,
        )
        self.assertEqual(summary.completed_tasks, 8)
        self.assertEqual(summary.max_active_workers, 2)

    def test_retry_keeps_full_cap_and_inflates_memory_estimate(self) -> None:
        messages: list[str] = []
        attempts = 0

        def launcher(queued: QueuedTask) -> RunningTask:
            nonlocal attempts
            attempts += 1
            return RunningTask(
                queued,
                _FakeProcess(-9 if attempts == 1 else 0),
                None,
                Path("/tmp/unused.json"),
            )

        summary = run_task_queue(
            [_task(8)],
            workers=60,
            max_retries=1,
            retry_cooldown_seconds=0.0,
            poll_seconds=0.001,
            launcher=launcher,
            result_validator=lambda running: running.process.returncode == 0,
            sleep=lambda _seconds: None,
            emit=messages.append,
            **self._memory_kwargs(),
        )
        self.assertEqual(summary.completed_tasks, 1)
        self.assertEqual(summary.retry_count, 1)
        self.assertTrue(any(message.startswith("retry ") and "target=60" in message for message in messages))
        self.assertTrue(any(message.startswith("retry ") and "estimate_gb=" in message for message in messages))


if __name__ == "__main__":
    unittest.main()
