from __future__ import annotations

import json
import tempfile
import time
import unittest
from argparse import Namespace
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

from inventory.common.distributions import DiscreteDemand
from inventory.core import DeterministicLeadTime, LostSales, TransitionCosts
from inventory.core.factory import inventory_case
from inventory.policy.ast import add, const, motif, neg, parameter, state
from sr.inventory.certify_multi import _certify_seed, _frontier_union, _load_pair
from runtime.sr_workflow import WorkflowCliDefaults, build_commands
from sr.inventory.runtime import _case_signature_payload
from sr.core.models import SearchConfig, SearchIndividual
from sr.core.inventory import InventorySearchBindings
from sr.inventory.multi_evaluation import CaseEvaluationInput, MultiStandardScheduler
from sr.inventory.frontier import import_frontier_seeds
from sr.analysis.frontier import FrontierSeedBundle
from sr.inventory.initialization import run_frontier_initialization
from sr.analysis.multi_report import multi_report_row, search_evaluation_dir, table_header
from sr.multi.models import MultiObjective, aggregate_case_results
from sr.multi.population import MultiIslandPopulation
from sr.inventory.evaluation import EvaluationTask
from tests.support.scheduler import scheduler_workers


def _case(lead_time: int, n: int):
    return inventory_case(
        name=f"test-L{lead_time}-n{n}",
        demand=DiscreteDemand.from_pmf({0: 0.5, 1: 0.5}, name="test"),
        n=n,
        costs=TransitionCosts(holding=1.0, shortage=2.0),
        lead_time=DeterministicLeadTime(lead_time),
        backlog=LostSales(),
        capacity=n,
    )


def make_candidate(ast, *, motifs, lead_time, n, config, source, **kwargs):
    bindings = InventorySearchBindings.build_context_free(_case(lead_time, n), config)
    return bindings.candidate(
        ast, motifs={**bindings.motifs, **motifs}, source=source, **kwargs
    )


def _fake_standard_worker(task_id: int, task: EvaluationTask, output_queue: Any) -> None:
    gap = 1.0 if task.case_name == "case_a" else 3.0
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
        "params": {},
        "policy_cost": 100.0 + gap,
        "dp_cost": 100.0,
        "gap_pct": gap,
        "status": "ok",
        "route": "fake",
        "optimizer_seconds": 0.01,
        "optimizer_metadata": {},
        "completed_at": time.time(),
        "random_seed": task.random_seed,
        "optimizer_config_version": "standard-independent",
        "evaluation_signature": task.evaluation_signature,
        "independent_warm_starts": 0,
    }
    task.result_path.parent.mkdir(parents=True, exist_ok=True)
    task.result_path.write_text(json.dumps(payload), encoding="utf-8")
    output_queue.put((task_id, "completed", str(task.result_path)))


def _always_fail_worker(task_id: int, task: EvaluationTask, output_queue: Any) -> None:
    output_queue.put((task_id, "failed", "intentional"))


def _hanging_worker(task_id: int, task: EvaluationTask, output_queue: Any) -> None:
    del task_id, task, output_queue
    time.sleep(10)


def _selective_timeout_worker(
    task_id: int, task: EvaluationTask, output_queue: Any
) -> None:
    assert task.candidate.source != "timeout", "hung worker should never finish"
    _fake_standard_worker(task_id, task, output_queue)


class MultiDistributionSRTests(unittest.TestCase):
    def setUp(self):
        def delay(worker, task):
            if worker is _hanging_worker or (
                worker is _selective_timeout_worker and task.candidate.source == "timeout"
            ):
                return None
            return 0.01

        self.clock = self.enterContext(scheduler_workers(delay))
        self.enterContext(patch(f"{__name__}.time", self.clock))

    def test_formal_search_evaluations_are_run_local_not_initialization_cached(self) -> None:
        first = search_evaluation_dir(Path("runs/first"))
        second = search_evaluation_dir(Path("runs/second"))
        self.assertEqual(first, Path("runs/first/standard_evaluations"))
        self.assertEqual(second, Path("runs/second/standard_evaluations"))
        self.assertNotEqual(first, second)
        self.assertNotIn("initialization", first.parts)

    def test_runtime_case_signature_is_json_serializable_with_numpy_scalars(self) -> None:
        import numpy as np

        class Demand:
            values = np.asarray([0, 4], dtype=np.int64)
            probs = np.asarray([0.5, 0.5], dtype=np.float64)

        class CaseLike:
            name = "test"
            n = np.int64(20)
            h = np.float64(1.0)
            b = np.float64(3.0)
            demand = Demand()

            def identity_payload(self):
                return {
                    "case_name": self.name,
                    "n": int(self.n),
                    "costs": {"holding": float(self.h), "shortage": float(self.b)},
                    "demand": {
                        "values": [int(value) for value in self.demand.values],
                        "probs": [float(value) for value in self.demand.probs],
                    },
                }

            case_signature = "canonical-test-signature"

        with tempfile.TemporaryDirectory() as directory:
            diagnostic = Path(directory) / "diagnostic.json"
            diagnostic.write_text("{}", encoding="utf-8")
            encoded = json.dumps(_case_signature_payload(CaseLike(), diagnostic))  # type: ignore[arg-type]
        self.assertIn('"case_signature": "canonical-test-signature"', encoded)

    def test_unified_launcher_orders_initialization_before_search_and_passes_workers(self) -> None:
        args = Namespace(
            phase="all",
            experiment_config=Path("experiment.json"),
            search_config=Path("search.json"),
            checkpoint_root=Path("checkpoints"),
            diagnostic_root=Path("diagnostics"),
            dp_cache_dir=Path("dp-cache"),
            horizon=12,
            cache_root=Path("multi-cache"),
            workers=60,
            reference_suite="custom-suite",
            output_dir=Path("output"),
            mutation_config=Path("mutation.json"),
            reflection_config=Path("reflection.json"),
        )
        defaults = WorkflowCliDefaults(
            repository_root=Path("."),
            initialization_entrypoint=Path("initialize.py"),
            search_entrypoint=Path("search.py"),
            experiment_config=Path("experiment.json"),
            search_config=Path("search.json"),
            checkpoint_root=Path("checkpoints"),
            diagnostic_root=Path("diagnostics"),
            cache_root=Path("multi-cache"),
            reference_suite="custom-suite",
        )
        commands = build_commands(args, defaults=defaults)
        self.assertEqual([label for label, _ in commands], ["initialization", "search"])
        for _, command in commands:
            self.assertEqual(command[command.index("--workers") + 1], "60")
            self.assertEqual(command[command.index("--dp-cache-dir") + 1], "dp-cache")
            self.assertEqual(command[command.index("--horizon") + 1], "12")
            self.assertEqual(
                command[command.index("--reference-suite") + 1], "custom-suite"
            )
        search = commands[1][1]
        self.assertEqual(search[search.index("--output-dir") + 1], "output")
        self.assertEqual(search[search.index("--mutation-config") + 1], "mutation.json")
        self.assertEqual(search[search.index("--reflection-config") + 1], "reflection.json")

    def test_objective_uses_mean_and_fractional_cvar80(self) -> None:
        objective = MultiObjective()
        mean, top2, score, indices = objective.aggregate((1, 2, 3, 4, 5, 6, 7, 8))
        self.assertEqual(mean, 4.5)
        self.assertAlmostEqual(top2, 7.625)
        self.assertAlmostEqual(score, 5.75)
        self.assertEqual(indices, (7, 6))

    def test_objective_rejects_invalid_weights_and_incomplete_top_k(self) -> None:
        with self.assertRaises(ValueError):
            MultiObjective(mean_weight=0.5, mean_top_k_weight=0.4)
        self.assertEqual(MultiObjective().aggregate((1.0,))[:3], (1., 1., 1.))
        with self.assertRaises(ValueError):
            MultiObjective(cvar_alpha=.9)

    def test_frontier_report_rows_include_every_case_gap(self) -> None:
        candidate = self._candidate()
        rows = {
            "case_a": SearchIndividual(candidate, {"S": 1}, 101, 100, 1, "ok", "x", 1, 1, "a.json"),
            "case_b": SearchIndividual(candidate, {"S": 2}, 103, 100, 3, "ok", "x", 2, 2, "b.json"),
        }
        individual = aggregate_case_results(candidate, rows, ("case_a", "case_b"), MultiObjective())

        header = table_header(("case_a", "case_b"))
        report_row = multi_report_row(individual, ("case_a", "case_b"))

        self.assertIn("case_a gap", header)
        self.assertIn("case_b gap", header)
        self.assertIn("1.0000%", report_row)
        self.assertIn("3.0000%", report_row)
        self.assertIn(candidate.ast_hash[:12], report_row)

    def test_certify_frontier_union_is_deduplicated_and_seed_is_order_independent(self) -> None:
        first = {
            "ast_hash": "a",
            "ast": const(1),
            "score": 2.0,
            "num_parameters": 0,
            "expanded_complexity": 1.0,
        }
        second = {
            "ast_hash": "b",
            "ast": const(2),
            "score": 1.0,
            "num_parameters": 0,
            "expanded_complexity": 1.0,
        }
        rows = _frontier_union(
            {"parameter_frontier": [first, second], "complexity_frontier": [first]}
        )
        self.assertEqual([row["ast_hash"] for row in rows], ["b", "a"])
        self.assertEqual(_certify_seed("a", "case"), _certify_seed("a", "case"))
        self.assertNotEqual(_certify_seed("a", "case"), _certify_seed("b", "case"))

    def test_certify_pair_cache_requires_matching_signature_and_no_warm_start(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pair.json"
            path.write_text(
                json.dumps(
                    {
                        "mode": "multi_independent_certify_pair",
                        "signature": "expected",
                        "independent_warm_starts": 0,
                    }
                ),
                encoding="utf-8",
            )
            self.assertIsNotNone(_load_pair(path, "expected"))
            self.assertIsNone(_load_pair(path, "other"))

    def test_aggregate_requires_every_case_and_keeps_independent_params(self) -> None:
        candidate = self._candidate()
        rows = {
            "a": SearchIndividual(candidate, {"S": 1}, 101, 100, 1, "ok", "x", 1, 1, "a.json"),
            "b": SearchIndividual(candidate, {"S": 2}, 103, 100, 3, "ok", "x", 2, 2, "b.json"),
        }
        individual = aggregate_case_results(candidate, rows, ("a", "b"), MultiObjective())
        self.assertEqual(individual.mean_gap_pct, 2.0)
        self.assertEqual(individual.mean_top_k_gap_pct, 3.0)
        self.assertAlmostEqual(individual.score, 2.4)
        self.assertEqual(individual.case_results["a"].params, {"S": 1})
        self.assertEqual(individual.case_results["b"].params, {"S": 2})
        self.assertEqual(individual.params, {})
        with self.assertRaises(ValueError):
            aggregate_case_results(candidate, {"a": rows["a"]}, ("a", "b"), MultiObjective())

    def test_frontier_import_namespaces_conflicting_motifs_and_deduplicates_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ast = add(parameter("S"), neg(motif("reflection_0_same")))
            self._checkpoint(root / "a.json", "a", ast, state("x"))
            self._checkpoint(root / "b.json", "b", ast, state("p1"))
            # A differently named source motif with the same content as a's
            # motif must collapse to the same expanded structure.
            ast_c = add(parameter("S"), neg(motif("reflection_v9_0_other")))
            self._checkpoint(root / "c.json", "c", ast_c, state("x"), motif_name="reflection_v9_0_other")
            bundle = import_frontier_seeds(
                {"a": root / "a.json", "b": root / "b.json", "c": root / "c.json"},
                case=_case(3, 20),
                config=replace(SearchConfig(), island_count=3),
            )
            self.assertEqual(len(bundle.candidates), 2)
            custom = sorted(name for name in bundle.motifs if name.startswith("seed_motif_"))
            self.assertEqual(len(custom), 2)
            self.assertFalse(any(name.startswith("reflection_") for name in bundle.motifs))
            shared = [values for values in bundle.provenance.values() if len(values) == 2]
            self.assertEqual(shared, [("a", "c")])

    def test_multi_scheduler_completes_group_and_reuses_pair_cache(self) -> None:
        candidate = make_candidate(
            const(1),
            motifs={},
            lead_time=1,
            n=5,
            config=self._config(),
            source="test",
        )
        inputs = {
            "case_a": CaseEvaluationInput("case_a", None, (), 11),  # type: ignore[arg-type]
            "case_b": CaseEvaluationInput("case_b", None, (), 13),  # type: ignore[arg-type]
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scheduler = MultiStandardScheduler(
                self._config(),
                case_names=("case_a", "case_b"),
                objective=MultiObjective(),
                motifs={},
                result_dir=root / "results",
                event_log=root / "events.jsonl",
                worker_target=_fake_standard_worker,
            )
            scheduler.submit_group(candidate, inputs)
            scheduler.drain()
            row = scheduler.completed[candidate.ast_hash]
            self.assertEqual(row.mean_gap_pct, 2.0)
            self.assertEqual(row.mean_top_k_gap_pct, 3.0)
            self.assertAlmostEqual(row.score, 2.4)
            self.assertEqual(scheduler.counts()["group_completed"], 1)

            cached = MultiStandardScheduler(
                self._config(),
                case_names=("case_a", "case_b"),
                objective=MultiObjective(),
                motifs={},
                result_dir=root / "results",
                event_log=root / "events-cache.jsonl",
                worker_target=_always_fail_worker,
            )
            self.assertEqual(cached.submit_group(candidate, inputs), "completed")
            self.assertFalse(cached.pending)
            self.assertFalse(cached.active)
            self.assertIn(candidate.ast_hash, cached.completed)

    def test_multi_scheduler_does_not_aggregate_failed_partial_group(self) -> None:
        candidate = make_candidate(
            const(1), motifs={}, lead_time=1, n=5, config=self._config(), source="test"
        )
        inputs = {
            "case_a": CaseEvaluationInput("case_a", None, (), 11),  # type: ignore[arg-type]
            "case_b": CaseEvaluationInput("case_b", None, (), 13),  # type: ignore[arg-type]
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scheduler = MultiStandardScheduler(
                self._config(),
                case_names=("case_a", "case_b"),
                objective=MultiObjective(),
                motifs={},
                result_dir=root / "results",
                event_log=root / "events.jsonl",
                max_pair_retries=1,
                worker_target=_always_fail_worker,
            )
            scheduler.submit_group(candidate, inputs)
            scheduler.drain()
            self.assertNotIn(candidate.ast_hash, scheduler.completed)
            self.assertEqual(scheduler.group_states[candidate.ast_hash], "failed")
            self.assertEqual(scheduler.counts()["pair_failed"], 2)

    def test_multi_scheduler_timeout_cancellation_is_scoped_to_one_group(self) -> None:
        timed_out = make_candidate(
            const(1), motifs={}, lead_time=1, n=5, config=self._config(), source="timeout"
        )
        successful = make_candidate(
            const(2), motifs={}, lead_time=1, n=5, config=self._config(), source="success"
        )
        inputs = {
            "case_a": CaseEvaluationInput("case_a", None, (), 11),  # type: ignore[arg-type]
            "case_b": CaseEvaluationInput("case_b", None, (), 13),  # type: ignore[arg-type]
        }
        config = replace(
            self._config(),
            optimizer_workers=2,
            evaluation_hard_timeout_seconds=0.1,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scheduler = MultiStandardScheduler(
                config,
                case_names=("case_a", "case_b"),
                objective=MultiObjective(),
                motifs={},
                result_dir=root / "results",
                event_log=root / "events.jsonl",
                max_pair_retries=5,
                worker_target=_selective_timeout_worker,
            )
            scheduler.submit_group(timed_out, inputs)
            scheduler.submit_group(successful, inputs)
            self.clock.sleep(0.099)
            scheduler.poll()
            self.assertEqual(scheduler.group_states[timed_out.ast_hash], "running")
            self.clock.sleep(0.002)
            scheduler.poll()
            self.assertEqual(scheduler.group_states[timed_out.ast_hash], "failed")
            scheduler.drain()

            self.assertEqual(scheduler.group_states[timed_out.ast_hash], "failed")
            self.assertEqual(scheduler.group_states[successful.ast_hash], "completed")
            self.assertNotIn(timed_out.ast_hash, scheduler.completed)
            self.assertIn(successful.ast_hash, scheduler.completed)
            self.assertEqual(scheduler.counts()["pair_failed"], 1)
            self.assertEqual(scheduler.counts()["pair_cancelled"], 1)
            events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
            self.assertFalse(any(row["event"] == "retry_scheduled" for row in events))
            cancelled = [row for row in events if row["event"] == "cancelled"]
            self.assertEqual([row["ast_hash"] for row in cancelled], [timed_out.ast_hash])

    def test_ten_source_cases_share_six_islands_without_losing_evaluations(self) -> None:
        config = replace(self._config(), island_count=6)
        candidate = make_candidate(
            const(1), motifs={}, lead_time=1, n=5, config=config,
            source="seed", island=9,
        )
        rows = {
            f"case_{index}": SearchIndividual(
                candidate, {}, 101, 100, 1, "ok", "test", 1, 1, f"{index}.json"
            )
            for index in range(10)
        }
        individual = aggregate_case_results(candidate, rows, tuple(rows), MultiObjective())
        population = MultiIslandPopulation(config)
        self.assertTrue(population.add(individual, initial_islands=tuple(range(10))))
        self.assertEqual(len(individual.case_results), 10)
        self.assertEqual(len(population.islands), 6)
        for island in population.islands:
            self.assertIn(candidate.ast_hash, island)

    def test_frontier_initialization_persists_and_skips_dropped_candidates(self) -> None:
        timed_out = make_candidate(
            const(1), motifs={}, lead_time=1, n=5, config=self._config(), source="timeout"
        )
        successful = make_candidate(
            const(2), motifs={}, lead_time=1, n=5, config=self._config(), source="success"
        )
        bundle = FrontierSeedBundle(
            candidates=(timed_out, successful),
            motifs={},
            provenance={},
            initial_islands={timed_out.ast_hash: (0,), successful.ast_hash: (1,)},
            source_checkpoints={"source_case": "checkpoint-sha"},
        )
        inputs = {
            "case_a": CaseEvaluationInput("case_a", None, (), 11),  # type: ignore[arg-type]
            "case_b": CaseEvaluationInput("case_b", None, (), 13),  # type: ignore[arg-type]
        }
        config = replace(
            self._config(),
            optimizer_workers=2,
            evaluation_hard_timeout_seconds=0.1,
        )
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "initialization"
            first = run_frontier_initialization(
                bundle,
                inputs,
                config=config,
                objective=MultiObjective(),
                cache_dir=cache,
                initialization_signature="test-signature",
                progress_interval_seconds=0.01,
                worker_target=_selective_timeout_worker,
            )
            self.assertEqual([row.ast_hash for row in first.individuals], [successful.ast_hash])
            self.assertEqual(first.dropped_candidate_hashes, (timed_out.ast_hash,))
            complete = json.loads((cache / "INITIALIZATION_COMPLETE.json").read_text())
            self.assertEqual(complete["requested_candidate_count"], 2)
            self.assertEqual(complete["candidate_count"], 1)
            self.assertEqual(complete["dropped_candidate_hashes"], [timed_out.ast_hash])

            started = time.monotonic()
            second = run_frontier_initialization(
                bundle,
                inputs,
                config=config,
                objective=MultiObjective(),
                cache_dir=cache,
                initialization_signature="test-signature",
                progress_interval_seconds=0.01,
                worker_target=_hanging_worker,
            )
            self.assertLess(time.monotonic() - started, 1.0)
            self.assertEqual([row.ast_hash for row in second.individuals], [successful.ast_hash])
            self.assertEqual(second.dropped_candidate_hashes, (timed_out.ast_hash,))
            self.assertEqual(second.scheduler_counts["cache_hits"], 2)

    @staticmethod
    def _candidate():
        return make_candidate(
            parameter("S"),
            motifs={},
            lead_time=1,
            n=5,
            config=MultiDistributionSRTests._config(),
            source="test",
        )

    @staticmethod
    def _config() -> SearchConfig:
        return replace(
            SearchConfig(),
            iterations=1,
            island_count=2,
            optimizer_workers=2,
            standard_promotions_per_iteration=1,
            max_outstanding_tasks=2,
            memory_reserve_gb=0,
            evaluation_hard_timeout_seconds=5,
        )

    @staticmethod
    def _checkpoint(
        path: Path,
        case_name: str,
        ast: dict,
        motif_ast: dict,
        *,
        motif_name: str = "reflection_0_same",
    ) -> None:
        row = {
            "ast_hash": f"source-{case_name}",
            "ast": ast,
            "expanded_ast": ast,
            "expanded_complexity": 10,
        }
        payload = {
            "schema_version": 2,
            "case": case_name,
            "parameter_frontier": [row],
            "complexity_frontier": [row],
            "reflection_motifs": [
                {"name": motif_name, "ast": motif_ast, "rationale": "test", "version": 1, "status": "promoted"}
            ],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
