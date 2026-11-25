from sr.family.template_init import PROTOCOL as TEMPLATE_PROTOCOL
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import subprocess
import sys
import threading
import unittest
from unittest.mock import patch

from src.tests.support.family import family_domain as _domain_for
from src.tests.support.family import canonical_dual_shortfall_seed
from sr.family.models import CaseMetric
from sr.family.pipeline import FamilyPipeline
from sr.family.search import MultiContextFamilySearch
from sr.family.evaluator import evaluate_families, record_task_timeout


class FamilyPipelineTests(unittest.TestCase):
    def setUp(self):
        self.domain = _domain_for((3,))
        self.seed = canonical_dual_shortfall_seed()
        self.cases = tuple(str(i) for i in range(8))

    def child(self, index):
        return replace(self.seed, family_id=f"child{index}",
                       ast={"op": "add", "args": [self.seed.ast,
                            {"op": "const", "value": index}]},
                       parent_family_id=self.seed.family_id,
                       parent_family_hash=self.domain.semantic_hash(self.seed),
                       mutation_action="add_term")

    def test_next_llm_overlaps_tail_and_final_drain_preserves_full_scores(self):
        tail_started = threading.Event()
        release = threading.Event()
        calls = []
        active = 0
        maximum = 0
        lock = threading.Lock()

        def worker(family, task):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            try:
                if family.family_id == "child1" and task.case_id == "7":
                    tail_started.set()
                    self.assertTrue(release.wait(5))
                return CaseMetric(task.context_id, task.case_id, 1.0)
            finally:
                with lock:
                    active -= 1

        def propose(parents, iteration, context):
            calls.append(iteration)
            self.assertTrue(all(row["status"] == "complete" for row in context.values()))
            if iteration == 1:
                self.assertTrue(tail_started.wait(5))
                self.assertFalse(release.is_set())
                self.assertNotIn("child1", [family.family_id for family in parents])
                release.set()
            return (self.child(iteration + 1),)

        with TemporaryDirectory() as temporary:
            search = MultiContextFamilySearch(
                domain=self.domain, initial_families=(self.seed,), case_ids=self.cases,
                worker=worker, output_dir=Path(temporary), workers=2, iterations=2,
                feature_init_pass=False, proposal_fn=propose,
                scheduling={"async_evaluation": True, "iteration_wait_seconds": 0,
                            "max_pending_families": 20})
            try:
                archive = search.run()
            finally:
                release.set()
            self.assertEqual(calls, [0, 1])
            self.assertEqual(len(archive), 3)
            self.assertTrue(all(len(row.metrics) == 8 and row.status == "complete" for row in archive))
            self.assertLessEqual(maximum, 2)
            checkpoint = json.loads((Path(temporary) / "CHECKPOINT.json").read_text())
            self.assertEqual(checkpoint["pending_family_hashes"], [])
            visits = sum(search._operator_visits.values())
            self.assertEqual(visits, 2)
            resumed = MultiContextFamilySearch(
                domain=self.domain, initial_families=(self.seed,), case_ids=self.cases,
                worker=worker, output_dir=Path(temporary), workers=2, iterations=2,
                feature_init_pass=False, proposal_fn=propose, resume=True,
                scheduling=search.scheduling)
            self.assertEqual(len(resumed.run()), 3)
            self.assertEqual(sum(resumed._operator_visits.values()), 2)

    def test_pending_checkpoint_restores_compute_without_another_llm_call(self):
        release = threading.Event()
        def worker(family, task):
            if family.family_id == "child1":
                self.assertTrue(release.wait(5))
            return CaseMetric(task.context_id, task.case_id, 1.0)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            kwargs = dict(domain=self.domain, initial_families=(self.seed,),
                          case_ids=self.cases, worker=worker, output_dir=root,
                          workers=2, iterations=1, feature_init_pass=False,
                          scheduling={"async_evaluation": True, "iteration_wait_seconds": 0})
            # Fail during the final drain, after the pending checkpoint is durable.
            original = FamilyPipeline.collect
            def interrupted(pipeline, seconds=None):
                if seconds is None:
                    release.set()
                    raise RuntimeError("simulated interruption")
                return original(pipeline, seconds)
            with patch.object(FamilyPipeline, "collect", interrupted):
                with self.assertRaisesRegex(RuntimeError, "simulated"):
                    MultiContextFamilySearch(**kwargs, proposal_fn=lambda *_: (self.child(1),)).run()
            cp = json.loads((root / "CHECKPOINT.json").read_text())
            self.assertEqual(cp["pending_family_hashes"], [self.domain.semantic_hash(self.child(1))])
            archive = MultiContextFamilySearch(**kwargs, resume=True,
                          proposal_fn=lambda *_: self.fail("resume called LLM")).run()
            self.assertEqual(len(archive), 2)
            self.assertEqual(json.loads((root / "CHECKPOINT.json").read_text())["pending_family_hashes"], [])

    def test_subprocess_timeout_is_audited_and_not_retried(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_file = root / "config.json"
            config_file.write_text("{}")
            class Worker:
                subprocess_module = "unused"
                config_path = str(config_file)
                config = {"evaluation_hard_timeout_seconds": .1,
                          "memory_safety_reserve_gb": 0, "max_retries": 2}
            real_popen = subprocess.Popen
            spawned = []
            def sleepy(command, **kwargs):
                process = real_popen([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
                spawned.append(process)
                return process
            with patch("sr.family.evaluator.subprocess.Popen", side_effect=sleepy):
                result = evaluate_families((self.seed,), self.cases, Worker(),
                            domain=self.domain, output_dir=root, workers=8)
                # Resume must not spend the same budget again or create a
                # scored family from the incomplete tensor.
                resumed = evaluate_families((self.seed,), self.cases, Worker(),
                            domain=self.domain, output_dir=root, workers=8)
            self.assertEqual(result, ())
            self.assertEqual(resumed, ())
            self.assertEqual(len(spawned), 8)
            self.assertTrue(all(process.poll() is not None for process in spawned))
            self.assertEqual(len(list(root.glob("task_audit/*/*.timeout.json"))), 8)
            self.assertEqual(len(list(root.glob("task_cache/*/*.timeout.json"))), 8)
            rejected, = root.glob("rejected_families/*.json")
            self.assertEqual(json.loads(rejected.read_text())["completed_tasks"], 0)

    def test_timeout_during_initialization_and_pipeline_keeps_healthy_families(self):
        class Worker:
            subprocess_module = "unused"
        seen = []
        def transport(pending, *, metrics, **kwargs):
            for key, family, task, cache in pending:
                seen.append((family.family_id, task.case_id))
                if family.family_id == "child1" and task.case_id == "7":
                    metrics[key].append(record_task_timeout(key, task, cache, seconds=1500))
                else:
                    metrics[key].append(CaseMetric(task.context_id, task.case_id, 1.0))

        for initialization in (True, False):
            with self.subTest(initialization=initialization), TemporaryDirectory() as temporary:
                root = Path(temporary)
                kwargs = dict(domain=self.domain, initial_families=(self.seed,),
                    case_ids=self.cases, worker=Worker(), output_dir=root, workers=2,
                    iterations=2, feature_init_pass=False,
                    scheduling={"async_evaluation": True, "iteration_wait_seconds": .01})
                if initialization:
                    def initialize(_):
                        return (self.child(10), self.child(1)), {"status": "complete"}
                    initialize.protocol = TEMPLATE_PROTOCOL
                    kwargs["initialization_fn"] = initialize
                def propose(parents, iteration, context):
                    self.assertNotIn("child1", [f.family_id for f in parents])
                    child = self.child(iteration + (2 if initialization else 1))
                    if initialization:
                        parent = parents[0]
                        child = replace(child,
                            ast={"op": "add", "args": [parent.ast,
                                 {"op": "const", "value": iteration + 2}]},
                            parent_family_id=parent.family_id,
                            parent_family_hash=self.domain.semantic_hash(parent))
                    return (child,)
                with patch("sr.family.evaluator._run_subprocess_tasks", side_effect=transport):
                    archive = MultiContextFamilySearch(**kwargs, proposal_fn=propose).run()
                expected = {self.seed.family_id, "child2"}
                if initialization:
                    expected = {"child10", "child2", "child3"}
                self.assertEqual({row.family.family_id for row in archive}, expected)
                self.assertTrue(all(len(row.metrics) == 8 for row in archive))
                rejected, = root.glob("rejected_families/*.json")
                self.assertEqual(json.loads(rejected.read_text())["completed_tasks"], 7)
                checkpoint = json.loads((root / "CHECKPOINT.json").read_text())
                self.assertEqual(checkpoint["pending_family_hashes"], [])

    def test_bounded_queue_deduplicates_completed_and_pending_families(self):
        counts = {}
        def worker(family, task):
            key = (family.family_id, task.case_id)
            counts[key] = counts.get(key, 0) + 1
            return CaseMetric(task.context_id, task.case_id, 1.0)
        with TemporaryDirectory() as temporary:
            pipeline = FamilyPipeline(worker=worker, domain=self.domain,
                case_ids=self.cases, output_dir=Path(temporary), workers=2,
                global_seed=42, fidelity="standard", score_protocol="mean-cvar80-60-40",
                max_pending_families=1)
            try:
                families = [self.child(1), self.child(2), self.child(3)]
                rows = pipeline.submit(families)
                self.assertLessEqual(len(pipeline.pending), 1)
                rows.extend(pipeline.collect())
                self.assertEqual(pipeline.submit(families), [])
                self.assertEqual(len(rows), 3)
                self.assertEqual(set(counts.values()), {1})
            finally:
                pipeline.close()

    def test_watchdog_change_reuses_success_but_retries_timeouts(self):
        from runtime.artifacts import atomic_json
        class Worker:
            subprocess_module = "unused"
        calls = []
        def transport(pending, *, worker, metrics, **kwargs):
            for key, family, task, cache in pending:
                calls.append((worker.budget, task.case_id))
                if worker.budget == 1500 and task.case_id == "7":
                    result = record_task_timeout(key, task, cache, seconds=1500)
                else:
                    result = CaseMetric(task.context_id, task.case_id, 1.0)
                    atomic_json({"family_hash": key, "context_id": task.context_id,
                        "case_id": task.case_id, "fidelity": task.fidelity,
                        "gap_pct": 1.0, "status": "complete"}, cache)
                metrics[key].append(result)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.json"
            worker = Worker(); worker.config_path = config
            kwargs = dict(domain=self.domain, output_dir=root, workers=2)
            with patch("sr.family.evaluator._run_subprocess_tasks", side_effect=transport):
                for budget in (1500, 2400):
                    worker.budget = budget
                    config.write_text(json.dumps({"evaluation_hard_timeout_seconds": budget, "paths": 128}))
                    rows = evaluate_families((self.seed,), self.cases, worker, **kwargs)
                    self.assertEqual(len(rows), int(budget == 2400))
                self.assertEqual(calls, [(1500, str(i)) for i in range(8)] + [(2400, "7")])
                config.write_text(json.dumps({"evaluation_hard_timeout_seconds": 2400, "paths": 256}))
                evaluate_families((self.seed,), self.cases, worker, **kwargs)
                self.assertEqual(len(calls), 17)  # Actual numerical changes invalidate every task.
            # Old timeout audit remains visible, but a complete re-evaluation
            # under the new budget restores parent eligibility.
            search = MultiContextFamilySearch(domain=self.domain, initial_families=(self.seed,),
                case_ids=self.cases, worker=worker, output_dir=root, feature_init_pass=False)
            self.assertEqual(search._select_parents((self.seed,), rows, 1), [self.seed])

    def test_worker_failure_propagates_without_partial_family_score(self):
        def worker(family, task):
            if task.case_id == "7":
                raise RuntimeError("broken numerical task")
            return CaseMetric(task.context_id, task.case_id, 1.0)
        with TemporaryDirectory() as temporary:
            pipeline = FamilyPipeline(worker=worker, domain=self.domain,
                case_ids=self.cases, output_dir=Path(temporary), workers=2,
                global_seed=42, fidelity="standard", score_protocol="mean-cvar80-60-40")
            try:
                with self.assertRaisesRegex(RuntimeError, "broken numerical task"):
                    pipeline.submit((self.seed,))
                    pipeline.collect()
                self.assertFalse(pipeline.completed)
            finally:
                pipeline.close()

    def test_resume_allows_verified_watchdog_increase_and_iteration_extension(self):
        class Worker:
            def __call__(self, family, task):
                return CaseMetric(task.context_id, task.case_id, 1.0)
        with TemporaryDirectory() as temporary:
            root = Path(temporary); config = root / "config.json"
            worker = Worker(); worker.config_path = config
            config.write_text(json.dumps({"evaluation_hard_timeout_seconds": 1500, "paths": 128}))
            kwargs = dict(domain=self.domain, initial_families=(self.seed,), case_ids=self.cases,
                          output_dir=root, worker=worker, feature_init_pass=False)
            MultiContextFamilySearch(**kwargs, iterations=0).run()
            config.write_text(json.dumps({"evaluation_hard_timeout_seconds": 2400, "paths": 128}))
            resumed = MultiContextFamilySearch(**kwargs, iterations=1, resume=True,
                                               proposal_fn=lambda *_: (self.child(1),))
            self.assertEqual(len(resumed.run()), 2)
            manifest = json.loads((root / "RUN_MANIFEST.json").read_text())
            self.assertEqual(manifest["iterations"], 1)
            self.assertEqual(manifest["execution_changes"][0]["previous"], 1500)
            self.assertEqual(manifest["execution_changes"][0]["current"], 2400)
            config.write_text(json.dumps({"evaluation_hard_timeout_seconds": 2400, "paths": 256}))
            with self.assertRaisesRegex(ValueError, "evaluation_config"):
                MultiContextFamilySearch(**kwargs, iterations=1, resume=True).run()


if __name__ == "__main__":
    unittest.main()
