from __future__ import annotations

import json
import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from inventory.common.distributions import DiscreteDemand
from inventory.core import DeterministicLeadTime, LostSales, TransitionCosts
from inventory.core.factory import inventory_case
from evaluation.backends.lead_time.dp import solve_dp
from inventory.policy.ast import add, const, parameter
from inventory.policy.motifs.lead_time import lead_time_motifs
from sr.core.models import SearchConfig
from sr.core.inventory import InventorySearchBindings
from sr.inventory.evaluation import AsyncStandardScheduler
from sr.inventory.single import SymbolicSearch
from optimization import InventorySearchContext
from tests.support.scheduler import scheduler_workers


def make_candidate(ast, *, motifs, lead_time, n, config, source, **kwargs):
    case = inventory_case(
        name=f"test-L{lead_time}-n{n}",
        demand=DiscreteDemand.from_pmf({0: 0.5, 1: 0.5}, name="test"),
        n=n,
        costs=TransitionCosts(holding=1.0, shortage=2.0),
        lead_time=DeterministicLeadTime(lead_time),
        backlog=LostSales(),
        capacity=n,
    )
    return InventorySearchBindings.build_context_free(case, config).candidate(
        ast, motifs=motifs, source=source, **kwargs
    )


def search_context(reference, demand):
    case = inventory_case(
        name="signature-test",
        demand=demand,
        n=int(reference.n),
        costs=TransitionCosts(holding=float(reference.h), shortage=float(reference.b)),
        lead_time=DeterministicLeadTime(int(reference.lead_time)),
        backlog=LostSales(),
        capacity=int(reference.n),
    )
    return InventorySearchContext.from_reference(case, int(reference.horizon), reference)


def _write_fake_result(task, *, delay: float) -> None:
    starts = task.result_path.with_suffix(".starts")
    with starts.open("a", encoding="utf-8") as handle:
        handle.write("start\n")
    time.sleep(delay)
    payload = {
        "schema_version": 2,
        "case": task.case_name,
        "mode": "standard",
        "ast_hash": task.candidate.ast_hash,
        "ast": task.candidate.ast,
        "params": {"r": 1.0},
        "policy_cost": 101.0,
        "dp_cost": 100.0,
        "gap_pct": 1.0,
        "status": "fake",
        "route": "fake",
        "optimizer_seconds": delay,
        "completed_at": time.time(),
        "optimizer_metadata": {},
        "random_seed": task.random_seed,
        "optimizer_config_version": "standard-independent",
        "evaluation_signature": task.evaluation_signature,
        "independent_warm_starts": 0,
    }
    task.result_path.write_text(json.dumps(payload), encoding="utf-8")


def soft_worker(task_id, task, output_queue) -> None:
    _write_fake_result(task, delay=0.0)
    output_queue.put((task_id, "completed", str(task.result_path)))


def hard_worker(task_id, task, output_queue) -> None:
    time.sleep(1.0)


def thread_probe_worker(task_id, task, output_queue) -> None:
    import numpy as np
    from threadpoolctl import threadpool_info

    values = np.ones((64, 64))
    _ = values @ values
    probe = {
        "environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "BLIS_NUM_THREADS",
            )
        },
        "native_pools": [
            {
                "internal_api": row.get("internal_api"),
                "num_threads": row.get("num_threads"),
            }
            for row in threadpool_info()
        ],
    }
    task.result_path.with_suffix(".threads.json").write_text(
        json.dumps(probe), encoding="utf-8"
    )
    _write_fake_result(task, delay=0.0)
    output_queue.put((task_id, "completed", str(task.result_path)))


def wait_for_started_events(
    event_log: str, expected: int, timeout_seconds: float
) -> int:
    """Block until the scheduler has dispatched the expected worker count."""

    deadline = time.monotonic() + timeout_seconds
    path = Path(event_log)
    while time.monotonic() < deadline:
        if path.is_file():
            starts = sum(
                json.loads(line).get("event") == "started"
                for line in path.read_text(encoding="utf-8").splitlines()
            )
            if starts >= expected:
                return starts
        time.sleep(0.02)
    return 0


class AsyncEvaluationTests(unittest.TestCase):
    def setUp(self):
        # Keep real processes only where OS isolation/concurrency is the subject.
        real_process_tests = {
            "test_worker_limits_environment_and_loaded_native_pools_to_one_thread",
            "test_hard_timeout_terminates_worker",
            "test_shutdown_terminates_active_and_cancels_pending_tasks",
            "test_blocking_auxiliary_keeps_replenishing_dynamic_worker_pool",
        }
        if self._testMethodName not in real_process_tests:
            clock = self.enterContext(scheduler_workers(
                lambda worker, task: None if worker is hard_worker else 0.12,
            ))
            self.enterContext(patch(f"{__name__}.time", clock))

    def _candidate(self, config: SearchConfig):
        return make_candidate(parameter("r"), motifs=lead_time_motifs(1), lead_time=1, n=4, config=config, source="test")

    def _ranked_candidate(self, config: SearchConfig, value: float, regret: float):
        candidate = make_candidate(
            add(parameter("r"), const(value)),
            motifs=lead_time_motifs(1),
            lead_time=1,
            n=4,
            config=config,
            source="test",
            detail={"bellman_regret_pct": regret},
        )
        return candidate

    def test_worker_limits_environment_and_loaded_native_pools_to_one_thread(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = SearchConfig(
                iterations=1,
                optimizer_workers=1,
                evaluation_hard_timeout_seconds=5,
                memory_reserve_gb=0,
            )
            scheduler = AsyncStandardScheduler(
                config,
                result_dir=root / "results",
                event_log=root / "events.jsonl",
                worker_target=thread_probe_worker,
            )
            candidate = self._candidate(config)
            scheduler.submit(
                candidate,
                case_name="case",
                context=None,
                samples=(),
                motifs={},
                random_seed=17,
            )
            deadline = time.monotonic() + 10
            while (
                scheduler.states[candidate.ast_hash] not in {"completed", "failed"}
                and time.monotonic() < deadline
            ):
                scheduler.poll()
                time.sleep(0.01)
            self.assertEqual(scheduler.states[candidate.ast_hash], "completed")
            probe = json.loads(
                (root / "results" / f"{candidate.ast_hash}.threads.json").read_text()
            )
            self.assertEqual(set(probe["environment"].values()), {"1"})
            self.assertTrue(probe["native_pools"])
            self.assertTrue(
                all(row["num_threads"] == 1 for row in probe["native_pools"]),
                probe["native_pools"],
            )

    def test_event_log_can_be_disabled(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = SearchConfig(
                iterations=1,
                optimizer_workers=1,
                evaluation_hard_timeout_seconds=5,
                memory_reserve_gb=0,
            )
            scheduler = AsyncStandardScheduler(
                config,
                result_dir=root / "results",
                event_log=None,
                worker_target=soft_worker,
            )
            candidate = self._candidate(config)
            scheduler.submit(
                candidate,
                case_name="case",
                context=None,
                samples=(),
                motifs={},
                random_seed=17,
            )
            scheduler.wait_for_batch((candidate.ast_hash,))
            self.assertFalse((root / "events.jsonl").exists())
            self.assertEqual(scheduler.states[candidate.ast_hash], "completed")
            self.assertNotIn("deferred_running", scheduler.states.values())
            self.assertEqual((root / "results" / f"{candidate.ast_hash}.starts").read_text().count("start"), 1)
            self.assertEqual(scheduler.completed[candidate.ast_hash].gap_pct, 1.0)

    def test_hard_timeout_terminates_worker(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = SearchConfig(iterations=1, optimizer_workers=1, evaluation_hard_timeout_seconds=0.05, memory_reserve_gb=0)
            scheduler = AsyncStandardScheduler(config, result_dir=root / "results", event_log=root / "events.jsonl", worker_target=hard_worker)
            candidate = self._candidate(config)
            scheduler.submit(candidate, case_name="case", context=None, samples=(), motifs={}, random_seed=23)
            deadline = time.monotonic() + 2
            while scheduler.states[candidate.ast_hash] not in {"hard_timeout", "failed"} and time.monotonic() < deadline:
                scheduler.poll()
                time.sleep(0.005)
            self.assertEqual(scheduler.states[candidate.ast_hash], "hard_timeout")
            self.assertFalse(scheduler.active)

    def test_shutdown_terminates_active_and_cancels_pending_tasks(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = SearchConfig(iterations=1, optimizer_workers=1, evaluation_hard_timeout_seconds=10, memory_reserve_gb=0)
            scheduler = AsyncStandardScheduler(config, result_dir=root / "results", event_log=root / "events.jsonl", worker_target=hard_worker)
            candidates = [self._ranked_candidate(config, float(index), float(index)) for index in range(1, 3)]
            for index, candidate in enumerate(candidates):
                scheduler.submit(candidate, case_name="case", context=None, samples=(), motifs={}, random_seed=index)
            processes = [active.process for active in scheduler.active.values()]
            scheduler.shutdown()
            scheduler.shutdown()
            self.assertFalse(scheduler.active)
            self.assertFalse(scheduler.pending)
            self.assertTrue(all(not process.is_alive() for process in processes))
            self.assertEqual({scheduler.states[row.ast_hash] for row in candidates}, {"cancelled"})

    def test_cache_seed_is_part_of_identity(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = SearchConfig(iterations=1, optimizer_workers=1, evaluation_hard_timeout_seconds=0.5, memory_reserve_gb=0)
            candidate = self._candidate(config)
            first = AsyncStandardScheduler(config, result_dir=root / "results", event_log=root / "a.jsonl", worker_target=soft_worker)
            first.submit(candidate, case_name="case", context=None, samples=(), motifs={}, random_seed=29)
            first.drain()
            second = AsyncStandardScheduler(config, result_dir=root / "results", event_log=root / "b.jsonl", worker_target=soft_worker)
            self.assertEqual(second.submit(candidate, case_name="case", context=None, samples=(), motifs={}, random_seed=29), "completed")
            third = AsyncStandardScheduler(config, result_dir=root / "results", event_log=root / "c.jsonl", worker_target=soft_worker)
            self.assertNotEqual(third.submit(candidate, case_name="case", context=None, samples=(), motifs={}, random_seed=31), "completed")
            third.drain()
            self.assertEqual((root / "results" / f"{candidate.ast_hash}.starts").read_text().count("start"), 2)

    def test_cache_semantic_inputs_are_part_of_identity(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = SearchConfig(iterations=1, optimizer_workers=1, evaluation_hard_timeout_seconds=0.5, memory_reserve_gb=0)
            candidate = self._candidate(config)
            demand_a = DiscreteDemand.from_pmf({0: 0.5, 1: 0.5}, name="same-name")
            demand_b = DiscreteDemand.from_pmf({0: 0.75, 1: 0.25}, name="same-name")
            dp_a = solve_dp(1, 3, 2, demand_a, 1.0, 2.0)
            dp_b = solve_dp(1, 3, 2, demand_b, 1.0, 2.0)
            first = AsyncStandardScheduler(config, result_dir=root / "results", event_log=root / "a.jsonl", worker_target=soft_worker)
            first.submit(candidate, case_name="case", context=search_context(dp_a, demand_a), samples=(), motifs={}, random_seed=29)
            first.drain()
            second = AsyncStandardScheduler(config, result_dir=root / "results", event_log=root / "b.jsonl", worker_target=soft_worker)
            self.assertNotEqual(
                second.submit(candidate, case_name="case", context=search_context(dp_b, demand_b), samples=(), motifs={}, random_seed=29),
                "completed",
            )
            second.drain()
            self.assertEqual((root / "results" / f"{candidate.ast_hash}.starts").read_text().count("start"), 2)

    def test_scheduler_rejects_same_ast_with_different_semantics(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = SearchConfig(iterations=1, optimizer_workers=1, evaluation_hard_timeout_seconds=0.5, memory_reserve_gb=0)
            candidate = self._candidate(config)
            scheduler = AsyncStandardScheduler(config, result_dir=root / "results", event_log=root / "events.jsonl", worker_target=soft_worker)
            scheduler.submit(candidate, case_name="case", context=None, samples=(), motifs={}, random_seed=29)
            with self.assertRaisesRegex(ValueError, "different Standard evaluation inputs"):
                scheduler.submit(candidate, case_name="case", context=None, samples=(), motifs={}, random_seed=31)
            scheduler.drain()

    def test_corrupt_cache_is_recomputed_instead_of_crashing(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = SearchConfig(iterations=1, optimizer_workers=1, evaluation_hard_timeout_seconds=0.5, memory_reserve_gb=0)
            candidate = self._candidate(config)
            result = root / "results" / f"{candidate.ast_hash}.json"
            result.parent.mkdir(parents=True)
            result.write_text('{"schema_version": 2, "case": "case", "mode": "standard"}', encoding="utf-8")
            scheduler = AsyncStandardScheduler(config, result_dir=root / "results", event_log=root / "events.jsonl", worker_target=soft_worker)
            self.assertNotEqual(
                scheduler.submit(candidate, case_name="case", context=None, samples=(), motifs={}, random_seed=29),
                "completed",
            )
            scheduler.drain()
            self.assertEqual(scheduler.states[candidate.ast_hash], "completed")

    def test_cache_with_inconsistent_gap_is_recomputed(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = SearchConfig(iterations=1, optimizer_workers=1, evaluation_hard_timeout_seconds=0.5, memory_reserve_gb=0)
            candidate = self._candidate(config)
            first = AsyncStandardScheduler(config, result_dir=root / "results", event_log=root / "a.jsonl", worker_target=soft_worker)
            first.submit(candidate, case_name="case", context=None, samples=(), motifs={}, random_seed=29)
            first.drain()
            path = root / "results" / f"{candidate.ast_hash}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["gap_pct"] = 999.0
            path.write_text(json.dumps(payload), encoding="utf-8")
            second = AsyncStandardScheduler(config, result_dir=root / "results", event_log=root / "b.jsonl", worker_target=soft_worker)
            self.assertNotEqual(
                second.submit(candidate, case_name="case", context=None, samples=(), motifs={}, random_seed=29),
                "completed",
            )
            second.drain()
            self.assertEqual(second.completed[candidate.ast_hash].gap_pct, 1.0)

    def test_bounded_pending_pool_replaces_worst_unstarted_candidate(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = SearchConfig(
                iterations=1,
                optimizer_workers=1,
                standard_promotions_per_iteration=1,
                max_outstanding_tasks=3,
                evaluation_hard_timeout_seconds=10,
                memory_reserve_gb=0,
            )
            scheduler = AsyncStandardScheduler(
                config,
                result_dir=root / "results",
                event_log=root / "events.jsonl",
                worker_target=hard_worker,
            )
            running = self._ranked_candidate(config, 1.0, 0.1)
            scheduler.submit(running, case_name="case", context=None, samples=(), motifs={}, random_seed=1)
            candidates = [
                self._ranked_candidate(config, 2.0, 3.0),
                self._ranked_candidate(config, 3.0, 2.0),
                self._ranked_candidate(config, 4.0, 1.0),
            ]
            for index, candidate in enumerate(candidates):
                scheduler.submit(candidate, case_name="case", context=None, samples=(), motifs={}, random_seed=10 + index)
            self.assertEqual(len(scheduler.pending), 2)
            self.assertEqual(
                {task.candidate.ast_hash for task in scheduler.pending},
                {candidates[1].ast_hash, candidates[2].ast_hash},
            )
            self.assertEqual(scheduler.states[candidates[0].ast_hash], "queue_evicted")
            for active in scheduler.active.values():
                scheduler._terminate(active.process)
                scheduler._close_queue(active.output_queue)
            scheduler.active.clear()

    def test_blocking_auxiliary_keeps_replenishing_dynamic_worker_pool(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = SearchConfig(
                iterations=1,
                optimizer_workers=1,
                evaluation_hard_timeout_seconds=10,
                memory_reserve_gb=0,
            )
            scheduler = AsyncStandardScheduler(
                config,
                result_dir=root / "results",
                event_log=root / "events.jsonl",
                worker_target=soft_worker,
            )
            candidates = [self._ranked_candidate(config, float(index), float(index)) for index in range(1, 3)]
            for index, candidate in enumerate(candidates):
                scheduler.submit(candidate, case_name="case", context=None, samples=(), motifs={}, random_seed=index)
            search = SymbolicSearch.__new__(SymbolicSearch)
            search.scheduler = scheduler
            try:
                self.assertEqual(
                    search._run_while_scheduling(
                        wait_for_started_events,
                        str(root / "events.jsonl"),
                        2,
                        10.0,
                    ),
                    2,
                )
                scheduler.poll()
                events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
                starts = [row for row in events if row["event"] == "started"]
                self.assertEqual(len(starts), 2)
            finally:
                scheduler.drain()
            self.assertEqual(len(scheduler.completed), 2)

    def test_batch_wait_stops_at_hard_iteration_limit_while_dispatching(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = SearchConfig(
                iterations=1,
                optimizer_workers=1,
                evaluation_hard_timeout_seconds=10,
                memory_reserve_gb=0,
            )
            scheduler = AsyncStandardScheduler(
                config,
                result_dir=root / "results",
                event_log=root / "events.jsonl",
                worker_target=soft_worker,
            )
            candidates = [self._ranked_candidate(config, float(index), float(index)) for index in range(1, 4)]
            hashes = [
                scheduler.submit(candidate, case_name="case", context=None, samples=(), motifs={}, random_seed=index)
                for index, candidate in enumerate(candidates)
            ]
            # submit() returns state names; wait_for_batch consumes AST hashes.
            self.assertTrue(all(state in {"running", "queued"} for state in hashes))
            try:
                started_at = time.monotonic()
                scheduler.wait_for_batch(
                    [candidate.ast_hash for candidate in candidates],
                    wait_seconds=0.1,
                    idle_active_workers=0,
                )
                elapsed = time.monotonic() - started_at
                self.assertGreaterEqual(elapsed, 0.1)
                self.assertLess(elapsed, 0.12)
                self.assertLess(len(scheduler.completed), 3)
                self.assertEqual(scheduler.states[candidates[-1].ast_hash], "queued")
                events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
                gates = [row for row in events if row["event"] == "batch_wait_finished"]
                self.assertEqual(gates[-1]["reason"], "wait_limit")
                self.assertGreaterEqual(gates[-1]["waited_seconds"], 0.1)
            finally:
                scheduler.drain()
            self.assertEqual(len(scheduler.completed), 3)

    def test_batch_can_finish_before_iteration_limit_when_all_tasks_complete(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = SearchConfig(
                iterations=1,
                optimizer_workers=2,
                evaluation_hard_timeout_seconds=10,
                memory_reserve_gb=0,
            )
            scheduler = AsyncStandardScheduler(
                config,
                result_dir=root / "results",
                event_log=root / "events.jsonl",
                worker_target=soft_worker,
            )
            candidates = [self._ranked_candidate(config, float(index), float(index)) for index in range(1, 3)]
            for index, candidate in enumerate(candidates):
                scheduler.submit(candidate, case_name="case", context=None, samples=(), motifs={}, random_seed=index)
            started_at = time.monotonic()
            scheduler.wait_for_batch(
                [candidate.ast_hash for candidate in candidates],
                wait_seconds=2.0,
                idle_active_workers=0,
            )
            self.assertLess(time.monotonic() - started_at, 1.0)
            self.assertEqual(len(scheduler.completed), 2)

    def test_capacity_gate_waits_before_admitting_next_batch(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = SearchConfig(
                iterations=1,
                optimizer_workers=1,
                evaluation_hard_timeout_seconds=10,
                memory_reserve_gb=0,
            )
            scheduler = AsyncStandardScheduler(
                config,
                result_dir=root / "results",
                event_log=root / "events.jsonl",
                worker_target=soft_worker,
            )
            candidates = [self._ranked_candidate(config, float(index), float(index)) for index in range(1, 3)]
            for index, candidate in enumerate(candidates):
                scheduler.submit(candidate, case_name="case", context=None, samples=(), motifs={}, random_seed=index)
            started_at = time.monotonic()
            scheduler.wait_for_capacity(additional_tasks=1, max_outstanding_tasks=2)
            elapsed = time.monotonic() - started_at
            self.assertGreaterEqual(elapsed, 0.08)
            self.assertLessEqual(scheduler.outstanding_count() + 1, 2)
            scheduler.shutdown()

    def test_submit_enforces_global_outstanding_ceiling(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = SearchConfig(
                iterations=1,
                standard_promotions_per_iteration=1,
                max_outstanding_tasks=1,
                optimizer_workers=1,
                evaluation_hard_timeout_seconds=10,
                memory_reserve_gb=0,
            )
            scheduler = AsyncStandardScheduler(
                config,
                result_dir=root / "results",
                event_log=root / "events.jsonl",
                worker_target=hard_worker,
            )
            candidates = [self._ranked_candidate(config, float(index), float(index)) for index in range(1, 3)]
            first = scheduler.submit(candidates[0], case_name="case", context=None, samples=(), motifs={}, random_seed=1)
            second = scheduler.submit(candidates[1], case_name="case", context=None, samples=(), motifs={}, random_seed=2)
            self.assertEqual(first, "running")
            self.assertEqual(second, "queue_rejected")
            self.assertEqual(scheduler.outstanding_count(), 1)
            scheduler.shutdown()


if __name__ == "__main__":
    unittest.main()
