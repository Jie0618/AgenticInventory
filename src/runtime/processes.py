"""Process helpers shared by asynchronous evaluators."""

from __future__ import annotations

import multiprocessing as mp
import os
import queue as queue_module
from typing import Any, Callable


SINGLE_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}


def run_single_threaded_worker(
    worker_target: Callable[[int, Any, Any], None],
    task_id: int,
    task: Any,
    output_queue: Any,
) -> None:
    os.environ.update(SINGLE_THREAD_ENV)
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        output_queue.put(
            (
                task_id,
                "failed",
                "ImportError: threadpoolctl is required to enforce one CPU thread per optimizer worker",
            )
        )
        return
    with threadpool_limits(limits=1):
        worker_target(task_id, task, output_queue)


def process_context() -> mp.context.BaseContext:
    try:
        if "fork" in mp.get_all_start_methods():
            return mp.get_context("fork")
    except RuntimeError:
        pass
    return mp.get_context()


def read_worker_message(
    task_id: int, output_queue: Any, exitcode: int | None
) -> tuple[str, str]:
    try:
        result_id, status, detail = output_queue.get(timeout=0.5)
        if int(result_id) != int(task_id):
            return "failed", f"mismatched task id {result_id}"
        return str(status), str(detail)
    except queue_module.Empty:
        return "failed", f"worker exited without message; exitcode={exitcode}"


def terminate_process(process: Any) -> None:
    process.terminate()
    process.join(timeout=1.0)
    if process.is_alive() and hasattr(process, "kill"):
        process.kill()
        process.join(timeout=1.0)


def close_queue(output_queue: Any) -> None:
    try:
        output_queue.close()
    except Exception:
        pass


def available_memory_gb() -> float:
    try:
        import psutil

        return float(psutil.virtual_memory().available) / 1.0e9
    except (ImportError, AttributeError):
        return float("inf")


__all__ = [
    "SINGLE_THREAD_ENV",
    "available_memory_gb",
    "close_queue",
    "process_context",
    "read_worker_message",
    "run_single_threaded_worker",
    "terminate_process",
]
