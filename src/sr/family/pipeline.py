"""Bounded family coordination over a shared pool of numerical task slots."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import math
import time
import threading

from .evaluator import evaluate_families


class FamilyPipeline:
    def __init__(self, *, worker, domain, case_ids, output_dir, workers,
                 global_seed, fidelity, score_protocol, max_pending_families=20):
        if max_pending_families <= 0:
            raise ValueError("max_pending_families must be positive")
        self.limit = max_pending_families
        self.domain = domain
        self.expected = len(domain.context_ids) * len(case_ids)
        self.tasks = ThreadPoolExecutor(max_workers=workers)
        self.tasks.cancel_event = threading.Event()
        self.families = ThreadPoolExecutor(max_workers=max_pending_families)
        self.pending = {}
        self.completed = set()
        self.kwargs = dict(worker=worker, domain=domain, case_ids=case_ids,
                           output_dir=output_dir, workers=workers,
                           global_seed=global_seed, fidelity=fidelity,
                           score_protocol=score_protocol, task_executor=self.tasks)

    def poll(self):
        rows = []
        for key, future in list(self.pending.items()):
            if not future.done():
                continue
            results = future.result()
            if not results:
                # Evaluator durably audited a budget timeout. This family is
                # settled but cannot enter the scored archive or parent pool.
                self.completed.add(key)
                del self.pending[key]
                continue
            result, = results
            if (result.status != "complete" or len(result.metrics) != self.expected
                    or not math.isfinite(result.cross_context_score)):
                raise RuntimeError(f"incomplete family evaluation: {key}")
            rows.append(result)
            self.completed.add(key)
            del self.pending[key]
        return rows

    def submit(self, candidates):
        rows = self.poll()
        for family in candidates:
            key = self.domain.semantic_hash(family)
            if key in self.pending or key in self.completed:
                continue
            while len(self.pending) >= self.limit:
                wait(self.pending.values(), return_when=FIRST_COMPLETED)
                rows.extend(self.poll())
            self.pending[key] = self.families.submit(
                evaluate_families, (family,), **self.kwargs)
        return rows

    def collect(self, seconds=None):
        deadline = None if seconds is None else time.monotonic() + seconds
        rows = self.poll()
        while self.pending:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                break
            wait(self.pending.values(), timeout=remaining, return_when=FIRST_COMPLETED)
            rows.extend(self.poll())
        return rows

    def capacity(self, incoming=10):
        rows = self.poll()
        target = max(0, self.limit - incoming)
        while len(self.pending) > target:
            wait(self.pending.values(), return_when=FIRST_COMPLETED)
            rows.extend(self.poll())
        return rows

    def close(self):
        # Running numerical jobs retain their per-task watchdog. Cancel jobs
        # that have not started; never leave detached compute after an error.
        self.tasks.cancel_event.set()
        self.families.shutdown(wait=True, cancel_futures=True)
        self.tasks.shutdown(wait=True, cancel_futures=True)
