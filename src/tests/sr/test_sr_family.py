from __future__ import annotations
from sr.family.template_init import PROTOCOL as TEMPLATE_PROTOCOL

import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from sr.family.proposals import (
    _apply_family_mutation as _apply_family_mutation_impl,
    build_family_prompt as build_family_prompt_impl,
    family_from_json as family_from_json_impl,
    feature_initialization_population as feature_initialization_population_impl,
    feature_registry as feature_registry_impl,
    initial_population,
    llm_proposal_batch_fn as llm_proposal_batch_fn_impl,
    llm_reflect_motifs as llm_reflect_motifs_impl,
)
from sr.family.evaluator import family_hash, screen_families
from sr.family.planner import plan_family_tasks
from inventory.policy.family import FeatureSpec, FamilyParameter, PolicyFamily, materialize_family, semantic_family_hash
from inventory.policy.ast import evaluate_ast
from src.tests.support.family import canonical_dual_shortfall_seed
from sr.family.models import CaseMetric, FamilyEvaluation, ScenarioScore
from sr.family.scoring import cross_context_score, scenario_score
from sr.family.search import MultiContextFamilySearch as MultiContextFamilySearchImpl

from src.tests.support.family import family_domain as _domain_for


def MultiContextFamilySearch(**kwargs):
    lead_times = tuple(kwargs.pop("lead_times", (3, 5, 8, 10)))
    kwargs.setdefault("initial_families", (canonical_dual_shortfall_seed(),))
    return MultiContextFamilySearchImpl(domain=_domain_for(lead_times), **kwargs)


def family_from_json(payload):
    return family_from_json_impl(payload, domain=_domain_for())


def build_family_prompt(parents):
    return build_family_prompt_impl(parents, domain=_domain_for())


def feature_initialization_population(parents=None):
    return feature_initialization_population_impl(
        _domain_for(), parents or (canonical_dual_shortfall_seed(),)
    )


def feature_registry(parents):
    return feature_registry_impl(_domain_for(), parents)


def _apply_family_mutation(parent, row, features, reflected_motifs=None):
    return _apply_family_mutation_impl(
        parent, row, features, reflected_motifs, domain=_domain_for()
    )


def llm_proposal_batch_fn(*args, **kwargs):
    kwargs.setdefault("domain", _domain_for())
    return llm_proposal_batch_fn_impl(*args, **kwargs)


def llm_reflect_motifs(*args, **kwargs):
    kwargs.setdefault("domain", _domain_for())
    return llm_reflect_motifs_impl(*args, **kwargs)


def _state_names(node):
    if isinstance(node, dict):
        if node.get("op") == "state":
            yield node["name"]
        for value in node.values():
            yield from _state_names(value)
    elif isinstance(node, list):
        for value in node:
            yield from _state_names(value)


class UnifiedSearchProtocolTests(unittest.TestCase):
    def test_fractional_cvar_and_signed_scores_match_reporting(self):
        from inventory.common.fitness import SCORE_PROTOCOL, metrics
        from sr.multi.models import MultiObjective
        values = [-8, -4, 0, 2, 4, 10]
        expected = metrics(values)
        self.assertAlmostEqual(expected["cvar80_gap_pct"], 9.0)
        self.assertAlmostEqual(MultiObjective(cvar_alpha=.8).aggregate(values)[2], expected["fitness"])
        score = scenario_score([CaseMetric("3", str(i), v) for i, v in enumerate(values)], "3", protocol=SCORE_PROTOCOL)
        self.assertAlmostEqual(score.robust_score, expected["fitness"])
        self.assertAlmostEqual(cross_context_score([score], protocol=SCORE_PROTOCOL), expected["fitness"])
        self.assertEqual(metrics([-2])["fitness"], -2)

    def test_nested_context_aggregation_and_protocol_resume_guard(self):
        from inventory.common.fitness import SCORE_PROTOCOL, metrics
        scores = [scenario_score([CaseMetric(str(c), str(i), v) for i, v in enumerate(values)], str(c), protocol=SCORE_PROTOCOL)
                  for c, values in enumerate(([0, 10], [-5, -3, -1, 0]))]
        self.assertAlmostEqual(cross_context_score(scores, protocol=SCORE_PROTOCOL),
                               metrics(s.robust_score for s in scores)["fitness"])
        with TemporaryDirectory() as temporary:
            kwargs = dict(case_ids=tuple(str(i) for i in range(8)), worker=lambda *_: None, output_dir=Path(temporary),
                          iterations=0, lead_times=(3,), feature_init_pass=False)
            search = MultiContextFamilySearch(**kwargs, score_protocol=SCORE_PROTOCOL)
            search.run()
            with self.assertRaisesRegex(ValueError, "score"):
                MultiContextFamilySearch(**kwargs, resume=True, score_protocol="direct-robust-gap").run()

    def test_llm_template_validation_dedup_and_resume(self):
        from dataclasses import asdict
        from sr.family.proposals import llm_template_initialization
        domain = _domain_for()
        seed = canonical_dual_shortfall_seed()
        valid = {**asdict(seed), "family_id": "summary_child",
                 "parent_family_hash": domain.semantic_hash(seed),
                 "parameters": [], "motifs": {}}
        # Use an existing validated, context-dependent feature definition.
        feature = domain.common_features["tail_sum_ceil_l_div_2"]
        valid["features"] = [asdict(feature)]
        valid["ast"] = {"op": "feature", "name": feature.name}
        class Client:
            def complete(self, prompt):
                self.prompt = prompt
                return json.dumps({"families": [valid, valid, {**valid, "parent_family_hash": "unknown"},
                                                {**valid, "ast": {"op": "feature", "name": "absent"}}]})
        client = Client()
        families, audit = llm_template_initialization(client, (seed,), domain=domain)
        self.assertEqual(len(families), 1)
        self.assertEqual(len(audit["rejected"]), 3)
        self.assertIn('"ast"', client.prompt)
        self.assertEqual(families[0].mutation_action, "template_initialization")
        class Failed:
            def complete(self, _):
                raise RuntimeError("unavailable")
        self.assertEqual(llm_template_initialization(Failed(), (seed,), domain=domain)[1]["status"], "failed")
        with TemporaryDirectory() as temporary:
            calls = []
            def initialize(parents):
                calls.append(parents)
                return families, audit
            initialize.protocol = TEMPLATE_PROTOCOL
            kwargs = dict(case_ids=tuple(str(i) for i in range(8)), worker=lambda *_: None, output_dir=Path(temporary),
                          iterations=0, initialization_fn=initialize)
            search = MultiContextFamilySearch(**kwargs)
            self.assertEqual(search._llm_initial_candidates(), families)
            restored = MultiContextFamilySearch(**kwargs)._llm_initial_candidates()
            self.assertEqual(restored, families)
            self.assertEqual(len(calls), 1)
            numerical_calls = []
            def worker(family, task):
                numerical_calls.append((family.family_id, task.context_id, task.case_id))
                return CaseMetric(task.context_id, task.case_id, 1.0)
            kwargs.update(worker=worker, workers=1, feature_init_pass=False)
            result = MultiContextFamilySearch(**kwargs).run()
            self.assertEqual({row.family.family_id for row in result}, {families[0].family_id})
            self.assertEqual(len(numerical_calls), 1 * 4 * 8)
            MultiContextFamilySearch(**kwargs, resume=True).run()
            self.assertEqual(len(calls), 1)
            self.assertEqual(len(numerical_calls), 1 * 4 * 8)

    def test_initialization_failure_aborts_without_evaluating_rule_seed(self):
        def initialize(_):
            raise RuntimeError("temporary API failure")
        initialize.protocol = TEMPLATE_PROTOCOL
        with TemporaryDirectory() as temporary:
            calls = []
            with self.assertRaisesRegex(RuntimeError, "temporary API failure"):
                MultiContextFamilySearch(
                    case_ids=tuple(str(i) for i in range(8)),
                    worker=lambda *args: calls.append(args),
                    output_dir=Path(temporary), workers=1, iterations=0,
                    feature_init_pass=False, initialization_fn=initialize,
                ).run()
            self.assertEqual(calls, [])

    def test_resume_after_initial_audit_before_checkpoint_keeps_llm_candidates(self):
        from runtime.artifacts import atomic_json
        child = PolicyFamily("llm-child", {"op": "const", "value": 1}, {}, ())
        initializations = []
        numerical_calls = []

        def initialize(parents):
            initializations.append(parents)
            return (child,), {"status": "complete"}
        initialize.protocol = TEMPLATE_PROTOCOL

        def worker(family, task):
            numerical_calls.append((family.family_id, task.context_id, task.case_id))
            return CaseMetric(task.context_id, task.case_id, 1.0)

        def interrupted_write(payload, path):
            if path.name == "CHECKPOINT.json":
                raise OSError("simulated interruption before checkpoint commit")
            atomic_json(payload, path)

        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            kwargs = dict(case_ids=tuple(str(i) for i in range(8)), worker=worker,
                          output_dir=output, workers=1, iterations=0,
                          feature_init_pass=False, initialization_fn=initialize)
            with patch("sr.family.search.atomic_json", side_effect=interrupted_write):
                with self.assertRaisesRegex(OSError, "simulated interruption"):
                    MultiContextFamilySearch(**kwargs).run()
            self.assertTrue((output / "audit/iterations/ITERATION_AUDIT_0.json").exists())
            self.assertFalse((output / "CHECKPOINT.json").exists())
            count = len(numerical_calls)
            result = MultiContextFamilySearch(**kwargs, resume=True).run()
            self.assertEqual({r.family.family_id for r in result},
                             {child.family_id})
            self.assertEqual(len(initializations), 1)
            self.assertEqual(len(numerical_calls), count)
            self.assertTrue((output / "CHECKPOINT.json").exists())

    def test_failed_initial_evaluation_cannot_commit_checkpoint(self):
        seed = canonical_dual_shortfall_seed()
        child = PolicyFamily("init-child", {"op": "const", "value": 1}, {}, ())
        failure = FamilyEvaluation(child, (), (), (), float("inf"), "failed:1")
        def initialize(_):
            return (child,), {"status": "complete"}
        initialize.protocol = TEMPLATE_PROTOCOL
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            with patch("sr.family.search.evaluate_families", return_value=(failure,)):
                with self.assertRaisesRegex(RuntimeError, "initial family.*failed evaluation"):
                    MultiContextFamilySearch(
                        case_ids=tuple(str(i) for i in range(8)), worker=lambda *_: None,
                        output_dir=output, iterations=0, feature_init_pass=False,
                        initialization_fn=initialize,
                    ).run()
            self.assertFalse((output / "CHECKPOINT.json").exists())
            self.assertFalse((output / "audit/iterations/ITERATION_AUDIT_0.json").exists())

    def test_executable_large_family_is_rejected_by_search_limits(self):
        seed = canonical_dual_shortfall_seed()
        nodes = [{"op": "parameter", "name": f"a{i}"} for i in range(33)]
        while len(nodes) > 1:
            nodes = [{"op": "add", "args": nodes[i:i + 2]} if i + 1 < len(nodes)
                     else nodes[i] for i in range(0, len(nodes), 2)]
        child = PolicyFamily("33-parameters", nodes[0], {},
                             tuple(FamilyParameter(f"a{i}", "scale", 0.5, 1) for i in range(33)),
                             parent_family_id=seed.family_id, mutation_action="replace_subtree")
        _domain_for().validate_family(child)
        with TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "eight fitted parameters"):
                MultiContextFamilySearch(
                    case_ids=tuple(str(i) for i in range(8)),
                    worker=lambda family, task: CaseMetric(task.context_id, task.case_id, 1),
                    output_dir=Path(temporary), workers=1, iterations=1,
                    feature_init_pass=False, proposal_fn=lambda *_: (child,),
                ).run()

    def test_no_config_worker_cache_still_binds_reference_protocol(self):
        from sr.family.evaluator import evaluate_families
        class Worker:
            def __init__(self, protocol, gap):
                self.evaluation_protocol_signature = protocol
                self.gap = gap
                self.calls = 0
            def __call__(self, family, task):
                self.calls += 1
                return CaseMetric(task.context_id, task.case_id, self.gap)
        with TemporaryDirectory() as temporary:
            first, second = Worker("reference-A", 1), Worker("reference-B", 2)
            kwargs = dict(domain=_domain_for(), output_dir=Path(temporary), workers=1)
            args = ((canonical_dual_shortfall_seed(),), tuple(str(i) for i in range(8)))
            evaluate_families(*args, first, **kwargs)
            result = evaluate_families(*args, second, **kwargs)
            self.assertEqual(second.calls, 32)
            self.assertTrue(all(m.gap_pct == 2 for m in result[0].metrics))
            evaluate_families(*args, second, **kwargs)
            self.assertEqual(second.calls, 32)

    def test_missing_worker_config_fails_instead_of_using_generic_cache(self):
        from sr.family.evaluator import evaluate_families
        class Worker:
            def __call__(self, *_):
                raise AssertionError("must reject before evaluation")
        with TemporaryDirectory() as temporary:
            worker = Worker()
            worker.config_path = Path(temporary) / "missing.json"
            with self.assertRaises(FileNotFoundError):
                evaluate_families((canonical_dual_shortfall_seed(),),
                                  tuple(str(i) for i in range(8)), worker,
                                  domain=_domain_for(), output_dir=Path(temporary))

    def test_subprocess_invalid_results_are_never_cached(self):
        from types import SimpleNamespace
        from runtime.task_queue import QueueRunSummary
        from sr.family.evaluator import _run_subprocess_tasks
        seed = canonical_dual_shortfall_seed()
        task = plan_family_tasks(seed, ("case",), contexts=("3",))[0]
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache.json"
            metrics = {"key": []}
            worker = SimpleNamespace()
            def queue(rows, **kwargs):
                directory = next((root / "task_runs").glob("family_batch_*")) / "tasks"
                output = directory / "0.out.json"
                output.write_text(json.dumps({"context_id": "3", "case_id": "wrong-case",
                                              "status": "complete", "gap_pct": 1}))
                running = SimpleNamespace(destination=output, process=SimpleNamespace(returncode=0))
                self.assertFalse(kwargs["result_validator"](running))
                # A well-shaped result written before a process crashes is
                # still not a successful observation.
                output.write_text(json.dumps({"context_id": "3", "case_id": "case",
                                              "status": "complete", "gap_pct": 1}))
                running.process.returncode = 1
                self.assertFalse(kwargs["result_validator"](running))
                return QueueRunSummary(len(rows), 0, 0, 0, 0, ())
            with patch("sr.family.evaluator.run_task_queue", side_effect=queue):
                _run_subprocess_tasks([("key", seed, task, cache)], worker=worker,
                                      worker_module="unused", output_dir=root,
                                      metrics=metrics, workers=1)
            self.assertEqual(metrics["key"], [])
            self.assertFalse(cache.exists())



class Stage3FamilySchemaTests(unittest.TestCase):
    def test_stage3_artifact_hash_accepts_domain_builtin_motif(self) -> None:
        seed = canonical_dual_shortfall_seed()
        ast = json.loads(json.dumps(seed.ast))
        ast["left"]["args"][0]["left"]["args"][1]["arg"] = {
            "op": "motif",
            "name": "total_inventory",
        }
        search = MultiContextFamilySearch(
            case_ids=tuple(f"case_{i}" for i in range(8)),
            worker=lambda _family, _task: None,
            output_dir=Path("unused"),
            iterations=0,
            lead_times=(3,),
        )
        child = PolicyFamily(
            "domain_builtin_child",
            ast,
            seed.motifs,
            seed.parameters,
            parent_family_id=seed.family_id,
            mutation_action="replace_subtree",
            parent_family_hash=search.domain.semantic_hash(seed),
        )

        search.domain.validate_family(child)
        self.assertEqual(search._artifact_hash(child), search.domain.semantic_hash(child))

    def test_resume_rejects_manifest_identity_drift(self) -> None:
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            common = {
                "case_ids": tuple(f"case_{i}" for i in range(8)),
                "worker": lambda _family, _task: None,
                "output_dir": output,
                "iterations": 0,
                "workers": 1,
                "lead_times": (3,),
                "feature_init_pass": False,
            }
            MultiContextFamilySearch(**common).run()
            common["global_seed"] = 123
            with self.assertRaisesRegex(ValueError, "global_seed"):
                MultiContextFamilySearch(
                    resume=True, **common
                ).run()

    def test_resume_rejects_missing_or_changed_scoring_protocol(self) -> None:
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            search = MultiContextFamilySearch(
                case_ids=tuple(f"case_{i}" for i in range(8)),
                worker=lambda _family, _task: None,
                output_dir=output, iterations=0, workers=1,
                lead_times=(3,), feature_init_pass=False,
            )
            search.run()
            path = output / "RUN_MANIFEST.json"
            manifest = json.loads(path.read_text())
            for protocol in (None, "different-score"):
                payload = json.loads(json.dumps(manifest))
                if protocol is None:
                    payload["score"].pop("protocol")
                else:
                    payload["score"]["protocol"] = protocol
                path.write_text(json.dumps(payload))
                with self.assertRaisesRegex(ValueError, "score"):
                    MultiContextFamilySearch(
                        case_ids=search.case_ids, worker=search.worker,
                        output_dir=output, iterations=0, workers=1,
                        lead_times=(3,), feature_init_pass=False, resume=True,
                    ).run()

    def test_resume_rejects_experiment_signature_drift(self) -> None:
        with TemporaryDirectory() as temporary:
            output = Path(temporary)

            class Worker:
                def __init__(self, signature: str) -> None:
                    self.evaluation_protocol_signature = signature

                def __call__(self, _family, _task):
                    return None

            common = {
                "case_ids": tuple(f"case_{i}" for i in range(8)),
                "output_dir": output,
                "iterations": 0,
                "workers": 1,
                "lead_times": (3,),
                "feature_init_pass": False,
            }
            MultiContextFamilySearch(worker=Worker("suite"), **common).run()
            with self.assertRaisesRegex(ValueError, "experiment_signature"):
                MultiContextFamilySearch(
                    worker=Worker("different-suite"), resume=True, **common
                ).run()

    def test_resume_rejects_legacy_parent_selection_protocol(self) -> None:
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            search = MultiContextFamilySearch(
                case_ids=tuple(f"case_{i}" for i in range(8)),
                worker=lambda _family, _task: None,
                output_dir=output,
                iterations=0,
                workers=1,
                lead_times=(3,),
                resume=True,
                feature_init_pass=False,
            )
            manifest = search._run_manifest()
            score = manifest["score"]
            legacy = {
                "stage": "ML-MD",
                "lead_times": [3],
                "family_validation_lead_times": [3],
                "training_case_ids": manifest["training_case_ids"],
                "workers": 60,
                "iterations": 0,
                "feature_init_pass": manifest["feature_init_pass"],
                "proposals_per_iteration": manifest["proposals_per_iteration"],
                "parent_context_limit": manifest["parent_context_limit"],
                "gp_enabled": manifest["gp_enabled"],
                "optimizer_mode": manifest["optimizer_mode"],
                "fidelity": manifest["fidelity"],
                "screening": manifest["screening"],
                "score": {
                    "protocol": score["protocol"],
                    "cross_lead_mean_weight": score["cross_context_mean_weight"],
                    "cross_lead_p90_weight": 0.20,
                    "cross_lead_max_weight": 0.10,
                },
            }
            (output / "RUN_MANIFEST.json").write_text(
                json.dumps(legacy), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "parent_selection_protocol"):
                search._validate_or_write_manifest(manifest)
            self.assertEqual(json.loads((output / "RUN_MANIFEST.json").read_text()), legacy)

    def test_resume_allows_iteration_extension_but_not_checkpoint_rewind(self) -> None:
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            common = {
                "case_ids": tuple(f"case_{i}" for i in range(8)),
                "worker": lambda _family, _task: None,
                "output_dir": output,
                "workers": 1,
                "lead_times": (3,),
                "feature_init_pass": False,
            }
            with patch("sr.family.search.evaluate_families", return_value=()):
                MultiContextFamilySearch(iterations=0, **common).run()
                MultiContextFamilySearch(iterations=1, resume=True, **common).run()
            manifest = json.loads((output / "RUN_MANIFEST.json").read_text())
            self.assertEqual(manifest["iterations"], 1)
            (output / "CHECKPOINT.json").write_text(
                json.dumps({"iteration": 2, "archive": []}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "exceeds requested iterations"):
                MultiContextFamilySearch(iterations=1, resume=True, **common).run()
            unchanged = json.loads((output / "RUN_MANIFEST.json").read_text())
            self.assertEqual(unchanged["iterations"], 1)

    def test_parent_context_keeps_all_completed_families_below_limit(self) -> None:
        seed = canonical_dual_shortfall_seed()
        families = [seed]
        for index, name in enumerate(("global", "l3_best", "l5_best", "other"), start=1):
            families.append(PolicyFamily(
                name,
                {"op": "add", "args": [seed.ast, {"op": "const", "value": float(index)}]},
                seed.motifs,
                seed.parameters,
            ))
        values = {
            "canonical_capped_dual_shortfall": (9.0, 9.0, 9.0, 9.0),
            "global": (1.0, 1.0, 1.0, 1.0),
            "l3_best": (0.1, 4.0, 4.0, 4.0),
            "l5_best": (4.0, 0.1, 4.0, 4.0),
            "other": (3.0, 3.0, 3.0, 3.0),
        }
        archive = []
        for family in families:
            per_l = tuple(
                ScenarioScore(str(lead), value, value, value)
                for lead, value in zip((3, 5, 8, 10), values[family.family_id])
            )
            archive.append(FamilyEvaluation(
                family, (), (), per_l,
                1.0 if family.family_id == "global" else 2.0 + len(archive),
                "complete",
            ))
        with TemporaryDirectory() as temporary:
            search = MultiContextFamilySearch(
                case_ids=tuple(f"case_{i}" for i in range(8)),
                worker=lambda _family, _task: None,
                output_dir=Path(temporary),
                iterations=0,
            )
            selected = search._select_parents(families, archive, iteration=1)
        self.assertEqual(
            {family.family_id for family in selected},
            {"canonical_capped_dual_shortfall", "global", "l3_best", "l5_best", "other"},
        )

    def test_unevaluated_anchors_are_ineligible(self) -> None:
        seed = canonical_dual_shortfall_seed()
        families = [PolicyFamily(
            f"anchor-{i}",
            {"op": "add", "args": [seed.ast, {"op": "const", "value": float(i)}]},
            seed.motifs, seed.parameters,
        ) for i in range(8)]
        with TemporaryDirectory() as temporary:
            search = MultiContextFamilySearch(
                initial_families=families,
                case_ids=tuple(f"case_{i}" for i in range(8)),
                worker=lambda _family, _task: None,
                output_dir=Path(temporary), iterations=0,
            )
            for iteration in range(1, 9):
                self.assertEqual(search._select_parents(families, [], iteration), [])

    def test_family_deserialization_preserves_lineage(self) -> None:
        seed = canonical_dual_shortfall_seed()
        restored = family_from_json({
            "family_id": "restored_child",
            "ast": seed.ast,
            "motifs": seed.motifs,
            "parameters": [p.__dict__ for p in seed.parameters],
            "parent_family_id": seed.family_id,
            "mutation_action": "replace_subtree",
        })
        self.assertEqual(restored.parent_family_id, seed.family_id)
        self.assertEqual(restored.mutation_action, "replace_subtree")

    def test_search_state_round_trip_restores_all_registries(self) -> None:
        with TemporaryDirectory() as temporary:
            first = MultiContextFamilySearch(
                case_ids=tuple(f"case_{i}" for i in range(8)),
                worker=lambda _family, _task: None, output_dir=Path(temporary),
                iterations=0, lead_times=(2,),
            )
            first._parent_visits = {"parent": 3}
            first._parent_rewards = {"parent": 1.25}
            first._operator_visits = {"replace_subtree": 2}
            first._operator_rewards = {"replace_subtree": 0.75}
            motifs = {"reflected_x": {"ast": {"op": "const", "value": 1.0}, "description": "x"}}
            history = [{"iteration": 5, "active": {}, "pruned": []}]
            payload = first._search_state_payload(motifs, history)

            second = MultiContextFamilySearch(
                case_ids=tuple(f"case_{i}" for i in range(8)),
                worker=lambda _family, _task: None, output_dir=Path(temporary),
                iterations=0, lead_times=(2,),
            )
            restored_history, restored_motifs = second._restore_search_state(
                {"iteration": 5, "search_state": payload}
            )
            self.assertEqual(second._parent_visits, first._parent_visits)
            self.assertEqual(second._parent_rewards, first._parent_rewards)
            self.assertEqual(second._operator_visits, first._operator_visits)
            self.assertEqual(second._operator_rewards, first._operator_rewards)
            self.assertEqual(restored_motifs, motifs)
            self.assertEqual(restored_history, history)

    def test_checkpoint_requires_current_search_state(self) -> None:
        with TemporaryDirectory() as temporary:
            search = MultiContextFamilySearch(
                case_ids=tuple(f"case_{i}" for i in range(8)),
                worker=lambda *_: self.fail("must not evaluate"),
                output_dir=Path(temporary), iterations=0, lead_times=(2,))
            state = search._search_state_payload({}, [])
            for checkpoint in ({"iteration": 0}, {"search_state": {}},
                               {"search_state": {**state, "parent_identity": "family_id"}}):
                with self.subTest(checkpoint=checkpoint):
                    with self.assertRaisesRegex(ValueError, "checkpoint requires"):
                        search._restore_search_state(checkpoint)

    def test_resume_requires_every_current_manifest_identity_field(self) -> None:
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            search = MultiContextFamilySearch(
                case_ids=tuple(f"case_{i}" for i in range(8)), worker=lambda *_: None,
                output_dir=output, iterations=0, lead_times=(2,), resume=True)
            manifest = search._run_manifest()
            path = output / "RUN_MANIFEST.json"
            for key in ("initial_family_hashes", "feature_init_family_hashes", "global_seed",
                        "reference_mode", "experiment_signature", "evaluation_config",
                        "contexts", "family_validation_contexts", "fidelity"):
                old = {k: v for k, v in manifest.items() if k != key}
                path.write_text(json.dumps(old))
                with self.subTest(field=key), self.assertRaisesRegex(ValueError, key):
                    search._validate_or_write_manifest(manifest)
                self.assertEqual(json.loads(path.read_text()), old)

    def test_resume_at_iter0_skips_completed_feature_initialization(self) -> None:
        seed = canonical_dual_shortfall_seed()
        child = PolicyFamily("iter0_child",
            {"op": "add", "args": [seed.ast, {"op": "const", "value": 1.0}]},
            seed.motifs, seed.parameters, parent_family_id=seed.family_id,
            mutation_action="replace_subtree")
        with TemporaryDirectory() as temporary:
            calls = []
            def worker(family, task):
                calls.append(family.family_id)
                return CaseMetric(task.context_id, task.case_id, 1.0)
            kwargs = dict(case_ids=tuple(f"case_{i}" for i in range(8)), worker=worker,
                output_dir=Path(temporary), iterations=0, workers=1,
                lead_times=(2,), feature_init_pass=True)
            with patch("sr.family.search.feature_initialization_population", return_value=(child,)):
                MultiContextFamilySearch(**kwargs).run()
            self.assertTrue(calls)
            calls.clear()
            with patch("sr.family.search.feature_initialization_population") as initialize:
                restored = MultiContextFamilySearch(**kwargs, resume=True).run()
            initialize.assert_not_called()
            self.assertEqual(calls, [])
            self.assertEqual([row.family.family_id for row in restored], [child.family_id])

    def test_resume_rejects_checkpoint_family_missing_from_proposals(self) -> None:
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            search = MultiContextFamilySearch(
                case_ids=tuple(f"case_{i}" for i in range(8)),
                worker=lambda *_: self.fail("must not evaluate"), output_dir=output,
                iterations=1, lead_times=(3,), resume=True, feature_init_pass=False)
            (output / "RUN_MANIFEST.json").write_text(json.dumps(search._run_manifest()))
            (output / "CHECKPOINT.json").write_text(json.dumps({
                "iteration": 1, "archive": [{"family_hash": "missing-family", "status": "complete"}],
                "search_state": search._search_state_payload({}, [])}))
            with self.assertRaisesRegex(ValueError, "cannot be restored"):
                search.run()

    def test_resume_rejects_old_hash_even_with_matching_family_id(self) -> None:
        seed = canonical_dual_shortfall_seed()
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            search = MultiContextFamilySearch(
                case_ids=tuple(f"case_{i}" for i in range(8)),
                worker=lambda *_: self.fail("must not evaluate"), output_dir=output,
                iterations=0, lead_times=(3,), resume=True, feature_init_pass=False)
            (output / "RUN_MANIFEST.json").write_text(json.dumps(search._run_manifest()))
            (output / "CHECKPOINT.json").write_text(json.dumps({
                "iteration": 0, "archive": [{"family_id": seed.family_id,
                    "family_hash": "old-seed-hash", "status": "complete"}],
                "search_state": search._search_state_payload({}, [])}))
            with self.assertRaisesRegex(ValueError, "cannot be restored"):
                search.run()

    def test_checkpoint_without_manifest_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            search = MultiContextFamilySearch(
                case_ids=tuple(f"case_{i}" for i in range(8)), worker=lambda *_: None,
                output_dir=output, iterations=0, lead_times=(3,), resume=True)
            (output / "CHECKPOINT.json").write_text(json.dumps({"iteration": 0}))
            with self.assertRaisesRegex(ValueError, "RUN_MANIFEST"):
                search.run()
            self.assertFalse((output / "RUN_MANIFEST.json").exists())

    def test_feature_initialization_evaluates_only_children(self) -> None:
        seed = canonical_dual_shortfall_seed()
        child = PolicyFamily(
            "iter0_child",
            {"op": "add", "args": [seed.ast, {"op": "const", "value": 1.0}]},
            seed.motifs,
            seed.parameters,
            parent_family_id=seed.family_id,
            mutation_action="add_term",
        )
        evaluations = (
            FamilyEvaluation(child, (), (), (), 1.0, "complete"),
        )
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            with patch(
                "sr.family.search.feature_initialization_population",
                return_value=(child,),
            ), patch(
                "sr.family.search.evaluate_families",
                return_value=evaluations,
            ) as evaluate:
                result = MultiContextFamilySearch(
                    case_ids=tuple(f"case_{i}" for i in range(8)),
                    worker=lambda _family, _task: None,
                    output_dir=output,
                    iterations=0,
                    workers=1,
                    lead_times=(3,),
                    feature_init_pass=True,
                ).run()

            evaluated_families = evaluate.call_args.args[0]
            self.assertEqual(
                [family.family_id for family in evaluated_families],
                [child.family_id],
            )
            self.assertEqual([row.family.family_id for row in result], [child.family_id])
            audit = json.loads((output / "audit/iterations/ITERATION_AUDIT_0.json").read_text())
            self.assertEqual(audit["generated_count"], 1)
            self.assertEqual(audit["evaluated_count"], 1)

    def test_initial_population_contains_canonical_seed(self) -> None:
        self.assertEqual(
            [row.family_id for row in initial_population((canonical_dual_shortfall_seed(),))],
            ["canonical_capped_dual_shortfall"],
        )

    def test_parameter_contract_rejects_shared_dependence(self) -> None:
        with self.assertRaises(ValueError):
            FamilyParameter("K", "scale", 0.0, 3.0, context_dependence="shared")

    def test_prompt_states_cross_l_requirements(self) -> None:
        prompt = build_family_prompt([])
        self.assertIn("8 fixed training cases per context", prompt)
        self.assertIn("independently for each (context, case)", prompt)
        self.assertIn("same AST must validate in every context", prompt)
        self.assertIn("DOMAIN VOCABULARY", prompt)
        self.assertIn('"far_pipeline"', prompt)
        self.assertIn('"tail_sum_floor_l_div_2"', prompt)

    def test_literal_larger_state_is_rejected_for_small_lead_time(self) -> None:
        payload = {
            "family_id": "bad_p3",
            "ast": {"op": "state", "name": "p3"},
            "parameters": [],
        }
        with self.assertRaisesRegex(ValueError, "context case domain-L2"):
            family_from_json_impl(payload, domain=_domain_for((2, 3)))

    def test_proposal_cannot_shadow_builtin_motif(self) -> None:
        payload = {
            "family_id": "bad_shadow",
            "ast": {"op": "motif", "name": "total_pipeline"},
            "motifs": {
                "total_pipeline": {"op": "add", "args": [
                    {"op": "state", "name": "p1"},
                    {"op": "state", "name": "p10"},
                ]}
            },
            "parameters": [],
        }
        with self.assertRaisesRegex(ValueError, "shadow"):
            family_from_json(payload)

    def test_l_aware_tail_feature_materializes_without_invalid_states(self) -> None:
        seed = canonical_dual_shortfall_seed()
        family = PolicyFamily(seed.family_id + "_tail", {"op": "feature", "name": "tail_sum_2"}, {}, seed.parameters[:0], features=(FeatureSpec("weighted_tail", "tail_weighted", 3, (1.0, 0.5, 0.25)),))
        # Built-in tail_sum_2 works for all L, while the custom feature is
        # expanded only after L is known.
        family.validate()
        ast2, motifs2 = materialize_family(family, 2)
        self.assertEqual(ast2, {"op": "motif", "name": "tail_sum_2"})
        self.assertEqual(motifs2["weighted_tail"]["op"], "add")
        self.assertEqual(set(_state_names(motifs2["weighted_tail"])), {"p1", "p2"})
        self.assertAlmostEqual(
            evaluate_ast(motifs2["weighted_tail"], {"p1": 2, "p2": 6}, {}),
            7.0,
        )

    def test_structural_window_rules_resolve_per_lead_time(self) -> None:
        for rule, expected in (("L", 5), ("floor(L/2)", 2), ("ceil(L/2)", 3), ("fixed(4)", 4)):
            feature = FeatureSpec("f_" + rule.replace("/", "_"), "head_sum", rule)
            family = PolicyFamily("rule_" + rule, {"op": "feature", "name": feature.name}, {}, (), features=(feature,))
            family.validate()
            _, motifs = materialize_family(family, 5)
            self.assertCountEqual(
                _state_names(motifs[feature.name]),
                [f"p{i}" for i in range(1, expected + 1)],
            )
            self.assertEqual(
                evaluate_ast(motifs[feature.name], {f"p{i}": i for i in range(1, 6)}, {}),
                expected * (expected + 1) / 2,
            )

    def test_weighted_feature_dynamic_window_requires_enough_weights(self) -> None:
        with self.assertRaises(ValueError):
            PolicyFamily("bad_weighted_family", {"op": "feature", "name": "bad_weighted"}, {}, (), features=(FeatureSpec("bad_weighted", "tail_weighted", "L", (1.0,)),)).validate()
        feature = FeatureSpec("weighted_l", "tail_weighted", "L", tuple(float(i) for i in range(1, 11)))
        family = PolicyFamily("weighted_l_family", {"op": "feature", "name": "weighted_l"}, {}, (), features=(feature,))
        family.validate()

    def test_dynamic_linear_weight_rule_normalizes_after_resolving_window(self) -> None:
        tail = FeatureSpec(
            "tail_linear_half", "tail_weighted", "floor(L/2)",
            weight_rule="linear_desc_normalized",
        )
        head = FeatureSpec(
            "head_linear_third", "head_weighted", "floor(L/3)",
            weight_rule="linear_asc_normalized",
        )
        for feature, lead_time, expected in (
            (tail, 3, [(1.0, "p3")]),
            (tail, 5, [(2 / 3, "p5"), (1 / 3, "p4")]),
            (head, 10, [(1 / 6, "p1"), (2 / 6, "p2"), (3 / 6, "p3")]),
        ):
            family = PolicyFamily(feature.name, {"op": "feature", "name": feature.name}, {}, (), features=(feature,))
            family.validate()
            _, motifs = materialize_family(family, lead_time)
            feature_ast = motifs[feature.name]
            terms = feature_ast["args"] if feature_ast.get("op") == "add" else [feature_ast]
            actual = [(term["args"][0]["value"], term["args"][1]["name"]) for term in terms]
            self.assertEqual([state for _, state in actual], [state for _, state in expected])
            for (weight, _), (wanted, _) in zip(actual, expected):
                self.assertAlmostEqual(weight, wanted)
            self.assertAlmostEqual(sum(weight for weight, _ in actual), 1.0)

    def test_weight_rule_rejects_wrong_direction_or_explicit_weights(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported weight_rule"):
            FeatureSpec("bad_direction", "tail_weighted", "floor(L/2)", weight_rule="linear_asc_normalized")
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            FeatureSpec("bad_mixed", "tail_weighted", 3, (0.5, 0.3, 0.2), "linear_desc_normalized")

    def test_family_json_parses_dynamic_linear_weight_rule(self) -> None:
        family = family_from_json({
            "family_id": "dynamic_linear",
            "ast": {"op": "feature", "name": "tail_linear_half"},
            "features": [{
                "name": "tail_linear_half", "kind": "tail_weighted",
                "window": "floor(L/2)", "weights": [],
                "weight_rule": "linear_desc_normalized",
            }],
            "parameters": [],
        })
        self.assertEqual(family.features[0].weight_rule, "linear_desc_normalized")

    def test_iter0_uses_bounded_curated_feature_families(self) -> None:
        population = feature_initialization_population()
        features = {feature.name: feature for family in population for feature in family.features}
        self.assertLessEqual(len(population), 8)
        self.assertEqual(set(features), {
            "tail_linear_half_normalized",
            "head_linear_half_normalized",
            "tail_sum_ceil_l_div_2",
            "head_sum_ceil_l_div_2",
        })
        self.assertFalse(any(name.startswith("init_") for name in features))

    def test_iter0_caps_each_seed_independently(self) -> None:
        first = canonical_dual_shortfall_seed()
        second = PolicyFamily(
            "second_seed",
            {"op": "add", "args": [
                first.ast,
                {"op": "motif", "name": "demand_mean"},
            ]},
            first.motifs,
            first.parameters,
        )
        population = feature_initialization_population_impl(
            _domain_for(), (first, second), variants_per_family=3
        )
        counts = {}
        for family in population:
            counts[family.parent_family_id] = counts.get(family.parent_family_id, 0) + 1
        self.assertEqual(counts, {first.family_id: 3, second.family_id: 3})

    def test_prompt_registry_highlights_fixed_and_dynamic_linear_examples(self) -> None:
        registry = feature_registry([])["default_typed_features"]
        self.assertEqual(set(registry), set(_domain_for().common_features))
        prompt = build_family_prompt([])
        self.assertIn('"weight_rule": "linear_desc_normalized"', prompt)
        self.assertIn('"window": "floor(L/2)"', prompt)

    def test_batch_accepts_dynamic_linear_feature_mutation(self) -> None:
        seed = canonical_dual_shortfall_seed()
        class Client:
            def complete(self, _prompt):
                return json.dumps({
                    "features": [{
                        "name": "tail_linear_half", "kind": "tail_weighted",
                        "window": "floor(L/2)", "weights": [],
                        "weight_rule": "linear_desc_normalized",
                    }],
                    "mutations": [{
                        "family_id": "dynamic_linear_child",
                        "parent_family_hash": _domain_for().semantic_hash(seed),
                        "action": "replace_subtree",
                        "path": ["left", "args", 0, "right", "args", 1, "arg"],
                        "old_subtree": {"op": "motif", "name": "far_pipeline"},
                        "fragment": {"op": "feature", "name": "tail_linear_half"},
                    }],
                })
        child = llm_proposal_batch_fn(Client(), [seed], 1, None, 1)[0]
        self.assertEqual(child.features[0].weight_rule, "linear_desc_normalized")
        child.validate()

    def test_custom_sum_feature_may_share_builtin_prefix(self) -> None:
        feature = FeatureSpec("tail_sum_3_focus", "tail_sum", 3)
        family = PolicyFamily("custom_prefixed_sum", {"op": "feature", "name": feature.name}, {}, (), features=(feature,))
        family.validate()
        _, motifs = materialize_family(family, 5)
        self.assertIn(feature.name, motifs)

    def test_internal_subtree_replacement_changes_far_pipeline(self) -> None:
        seed = canonical_dual_shortfall_seed()
        feature = FeatureSpec("tail_weighted_m3", "tail_weighted", 3, (3.0, 2.0, 1.0))
        row = {
            "family_id": "replace_far_with_tail_weighted",
            "parent_family_id": seed.family_id,
            "action": "replace_subtree",
            "path": ["left", "args", 0, "right", "args", 1, "arg"],
            "old_subtree": {"op": "motif", "name": "far_pipeline"},
            "fragment": {"op": "feature", "name": "tail_weighted_m3"},
        }
        child = _apply_family_mutation(seed, row, (feature,))
        self.assertEqual(child.mutation_action, "replace_subtree")
        self.assertIn("tail_weighted_m3", str(child.ast))
        from sr.family.proposals import _get_path
        self.assertEqual(_get_path(child.ast, row["path"]), {"op": "feature", "name": "tail_weighted_m3"})
        child.validate()

    def test_batch_recovers_invalid_path_from_nested_old_subtree(self) -> None:
        seed = canonical_dual_shortfall_seed()

        class Client:
            def complete(self, _prompt):
                return json.dumps({"features": [], "mutations": [{
                    "family_id": "recover_nested_path",
                    "parent_family_id": seed.family_id,
                    "action": "replace_subtree",
                    # This incorrectly treats the inner minimum as an args node.
                    "path": ["left", "args", 0, "args", 0, "args", 1, "right", "args", 1, "arg"],
                    "fragment": {
                        "old_subtree": {"op": "motif", "name": "far_pipeline"},
                        "new_subtree": {"op": "motif", "name": "pipeline_lag_moment"},
                    },
                }]})

        rows = llm_proposal_batch_fn(Client(), [seed], 1, None, 1)
        self.assertEqual(len(rows), 1)
        self.assertIn("pipeline_lag_moment", str(rows[0].ast))

    def test_replacement_accepts_binary_left_right_add_from_real_llm_response(self) -> None:
        seed = canonical_dual_shortfall_seed()
        cases = (
            (
                "dual_shortfall_available_inventory",
                ["left", "args", 0, "left"],
                "S",
                "inventory_position",
                "available_inventory",
            ),
            (
                "dual_shortfall_total_pipeline",
                ["left", "args", 0, "right"],
                "F",
                "far_pipeline",
                "total_pipeline",
            ),
            (
                "dual_shortfall_near_pipeline",
                ["left", "args", 0, "right"],
                "F",
                "far_pipeline",
                "near_pipeline",
            ),
        )
        for family_id, path, parameter_name, old_name, new_name in cases:
            with self.subTest(family_id=family_id):
                binary_old = {
                    "op": "add",
                    "left": {"op": "parameter", "name": parameter_name},
                    "right": {"op": "neg", "arg": {"op": "motif", "name": old_name}},
                }
                binary_new = {
                    "op": "add",
                    "left": {"op": "parameter", "name": parameter_name},
                    "right": {"op": "neg", "arg": {"op": "feature", "name": new_name}},
                }
                child = _apply_family_mutation(seed, {
                    "family_id": family_id,
                    "parent_family_id": seed.family_id,
                    "action": "replace_subtree",
                    "path": path,
                    "old_subtree": binary_old,
                    "fragment": binary_new,
                }, ())
                self.assertEqual(child.family_id, family_id)
                self.assertEqual(child.mutation_action, "replace_subtree")
                self.assertIn(new_name, str(child.ast))
                child.validate()

    def test_ambiguous_old_subtree_requires_valid_path(self) -> None:
        seed = canonical_dual_shortfall_seed()
        parent = PolicyFamily(
            "duplicate_parent",
            {"op": "add", "args": [seed.ast, seed.ast]},
            seed.motifs,
            seed.parameters,
        )
        parent.validate()
        with self.assertRaisesRegex(ValueError, "ambiguous old_subtree"):
            _apply_family_mutation(parent, {
                "family_id": "ambiguous_replacement",
                "parent_family_id": parent.family_id,
                "action": "replace_subtree",
                "path": ["args", 99],
                "old_subtree": seed.ast,
                "fragment": {"op": "const", "value": 1.0},
            }, ())

    def test_delete_term_removes_one_add_operand(self) -> None:
        seed = canonical_dual_shortfall_seed()
        parent = PolicyFamily("add_parent", {"op": "add", "args": [seed.ast, {"op": "const", "value": 1.0}]}, {}, seed.parameters)
        parent.validate()
        child = _apply_family_mutation(parent, {
            "family_id": "delete_addend", "parent_family_id": parent.family_id,
            "action": "delete_term", "path": ["args", 1],
            "old_subtree": {"op": "const", "value": 1.0},
        }, ())
        self.assertEqual(child.mutation_action, "delete_term")
        self.assertEqual(child.ast, seed.ast)
        child.validate()

    def test_delete_wrapper_unwraps_safe_local_wrapper(self) -> None:
        seed = canonical_dual_shortfall_seed()
        parent = PolicyFamily("wrapped_parent", {"op": "neg", "arg": seed.ast}, {}, seed.parameters)
        parent.validate()
        child = _apply_family_mutation(parent, {
            "family_id": "unwrap_neg", "parent_family_id": parent.family_id,
            "action": "delete_wrapper", "path": [],
            "old_subtree": {"op": "neg", "arg": seed.ast},
        }, ())
        self.assertEqual(child.mutation_action, "delete_wrapper")
        self.assertEqual(child.ast, seed.ast)
        child.validate()

    def test_delete_term_cannot_create_empty_add(self) -> None:
        seed = canonical_dual_shortfall_seed()
        with self.assertRaisesRegex(ValueError, "requires a nonempty path"):
            _apply_family_mutation(seed, {
                "family_id": "bad_delete", "parent_family_id": seed.family_id,
                "action": "delete_term", "path": [],
            }, ())

    def test_family_hash_ignores_family_id_but_changes_definition(self) -> None:
        seed = canonical_dual_shortfall_seed()
        renamed = PolicyFamily("renamed", seed.ast, seed.motifs, seed.parameters, seed.description, seed.features)
        self.assertEqual(family_hash(seed), family_hash(renamed))
        changed = PolicyFamily("changed", {"op": "feature", "name": "tail_sum_2"}, seed.motifs, seed.parameters[:0], seed.description, seed.features)
        self.assertNotEqual(family_hash(seed), family_hash(changed))

    def test_semantic_hash_collapses_parameter_alpha_renaming(self) -> None:
        seed = canonical_dual_shortfall_seed()

        def rename(node):
            if isinstance(node, list):
                return [rename(item) for item in node]
            if not isinstance(node, dict):
                return node
            out = {key: rename(value) for key, value in node.items()}
            if out.get("op") == "parameter" and out.get("name") == "K":
                out["name"] = "K2"
            return out

        renamed = PolicyFamily(
            "renamed_scale", rename(seed.ast), seed.motifs,
            tuple(
                FamilyParameter("K2", p.kind, p.lower, p.upper, p.index_rule, p.context_dependence)
                if p.name == "K" else p
                for p in seed.parameters
            ),
        )
        renamed.validate()
        self.assertEqual(family_hash(seed), family_hash(renamed))
        self.assertEqual(semantic_family_hash(seed), semantic_family_hash(renamed))

    def test_semantic_hash_collapses_equivalent_typed_feature_aliases(self) -> None:
        first = PolicyFamily(
            "tail_alias_a", {"op": "feature", "name": "tail_a"}, {}, (),
            features=(FeatureSpec("tail_a", "tail_weighted", 4, (0.4, 0.3, 0.2, 0.1)),),
        )
        second = PolicyFamily(
            "tail_alias_b", {"op": "feature", "name": "tail_b"}, {}, (),
            features=(FeatureSpec("tail_b", "tail_weighted", 4, (0.4, 0.3, 0.2, 0.1)),),
        )
        first.validate(); second.validate()
        self.assertEqual(family_hash(first), family_hash(second))
        self.assertEqual(semantic_family_hash(first), semantic_family_hash(second))

    def test_semantic_hash_preserves_independent_same_domain_caps(self) -> None:
        seed = canonical_dual_shortfall_seed()
        duplicate_cap = PolicyFamily(
            "duplicate_cap",
            {"op": "minimum", "left": seed.ast, "right": {"op": "parameter", "name": "C2"}},
            seed.motifs,
            seed.parameters + (FamilyParameter("C2", "cap", 0.0, 200.0),),
        )
        duplicate_cap.validate()
        self.assertNotEqual(family_hash(seed), family_hash(duplicate_cap))
        self.assertNotEqual(semantic_family_hash(seed), semantic_family_hash(duplicate_cap))

    def test_semantic_hash_removes_safe_arithmetic_identities(self) -> None:
        seed = canonical_dual_shortfall_seed()
        identity = PolicyFamily(
            "seed_plus_zero", {"op": "add", "args": [seed.ast, {"op": "const", "value": 0.0}]},
            seed.motifs, seed.parameters,
        )
        identity.validate()
        self.assertEqual(semantic_family_hash(seed), semantic_family_hash(identity))

    def test_semantic_hash_removes_root_nonnegative_wrapper(self) -> None:
        seed = canonical_dual_shortfall_seed()
        for ast in (
            {"op": "maximum", "left": seed.ast, "right": {"op": "const", "value": 0}},
            {"op": "maximum", "left": {"op": "const", "value": 0.0}, "right": seed.ast},
        ):
            wrapped = PolicyFamily("wrapped", ast, seed.motifs, seed.parameters)
            wrapped.validate()
            self.assertEqual(semantic_family_hash(seed), semantic_family_hash(wrapped))

    def test_mutation_rejects_embedded_parent_or_semantic_no_op(self) -> None:
        seed = canonical_dual_shortfall_seed()
        with self.assertRaisesRegex(ValueError, "contain only the new term"):
            _apply_family_mutation(seed, {
                "family_id": "doubled_parent", "parent_family_id": seed.family_id,
                "action": "add_term",
                "fragment": {"op": "add", "args": [seed.ast, {"op": "const", "value": 1.0}]},
            }, ())
        with self.assertRaisesRegex(ValueError, "semantically identical"):
            _apply_family_mutation(seed, {
                "family_id": "zero_add", "parent_family_id": seed.family_id,
                "action": "add_term", "fragment": {"op": "const", "value": 0.0},
            }, ())

    def test_search_filters_semantic_alias_before_evaluation(self) -> None:
        seed = canonical_dual_shortfall_seed()
        alias = PolicyFamily(
            "seed_times_one", {"op": "mul", "args": [seed.ast, {"op": "const", "value": 1.0}]},
            seed.motifs, seed.parameters, parent_family_id=seed.family_id,
            mutation_action="add_term",
        )
        alias.validate()
        with TemporaryDirectory() as temporary:
            search = MultiContextFamilySearch(
                case_ids=tuple(f"case_{i}" for i in range(8)),
                worker=lambda _family, _task: None,
                output_dir=Path(temporary), iterations=0, lead_times=(3,),
            )
            self.assertEqual(search._novel_generation((alias,), (seed,)), ())

    def test_batch_protocol_accepts_multiple_policy_objects(self) -> None:
        seed = canonical_dual_shortfall_seed()
        payload = {"family_id": "batch", "ast": seed.ast, "motifs": seed.motifs, "parameters": [{"name": p.name, "kind": p.kind, "lower": p.lower, "upper": p.upper} for p in seed.parameters]}
        class Client:
            def __init__(self): self.calls = 0
            def complete(self, _prompt):
                self.calls += 1
                return json.dumps({"features": []}) if self.calls == 1 else json.dumps({"families": [{**payload, "family_id": f"batch_{i}"} for i in range(3)]})
        rows = llm_proposal_batch_fn(Client(), [seed], 1, None, 3)
        self.assertEqual(len(rows), 3)

    def test_batch_protocol_refills_underfilled_response(self) -> None:
        seed = canonical_dual_shortfall_seed()
        payload = {"family_id": "batch", "ast": seed.ast, "motifs": seed.motifs, "parameters": [{"name": p.name, "kind": p.kind, "lower": p.lower, "upper": p.upper} for p in seed.parameters]}
        class Client:
            def __init__(self): self.calls = 0
            def complete(self, _prompt):
                self.calls += 1
                if self.calls == 1:
                    return json.dumps({"features": []})
                if self.calls == 2:
                    return json.dumps({"families": [{**payload, "family_id": "only_one"}]})
                return json.dumps({"families": [{**payload, "family_id": f"refill_{self.calls}_{i}"} for i in range(2)]})
        client = Client()
        rows = llm_proposal_batch_fn(client, [seed], 1, None, 3)
        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual(client.calls, 2)

    def test_batch_protocol_repeatedly_repairs_underfilled_batch(self) -> None:
        seed = canonical_dual_shortfall_seed()
        class Client:
            def __init__(self): self.calls = 0
            def complete(self, _prompt):
                self.calls += 1
                if self.calls == 1:
                    return json.dumps({"features": [], "mutations": [{
                        "family_id": "one", "parent_family_id": seed.family_id,
                        "action": "add_term", "fragment": {"op": "const", "value": 1.0},
                    }]})
                return json.dumps({"mutations": [{
                    "family_id": f"repair_{self.calls}", "parent_family_id": seed.family_id,
                    "action": "add_term", "fragment": {"op": "const", "value": float(self.calls)},
                }]})
        client = Client()
        rows = llm_proposal_batch_fn(client, [seed], 1, None, 10)
        self.assertEqual(len(rows), 10)
        self.assertEqual(client.calls, 10)

    def test_batch_protocol_accepts_bare_array_from_repair(self) -> None:
        seed = canonical_dual_shortfall_seed()
        prompts = []

        class Client:
            def complete(self, prompt):
                prompts.append(prompt)
                if len(prompts) == 1:
                    return json.dumps({"features": [], "mutations": [{
                        "family_id": "initial_one", "parent_family_id": seed.family_id,
                        "action": "add_term", "fragment": {"op": "const", "value": 1.0},
                    }]})
                return json.dumps([{
                    "family_id": "repair_two", "parent_family_id": seed.family_id,
                    "action": "add_term", "fragment": {"op": "const", "value": 2.0},
                }])

        rows = llm_proposal_batch_fn(Client(), [seed], 1, None, 2)
        self.assertEqual([row.family_id for row in rows], ["initial_one", "repair_two"])
        self.assertIn('{"features":[],"mutations":[...]}', prompts[1])
        self.assertIn("do not return a bare array", prompts[1])

    def test_batch_protocol_accepts_bare_array_from_rescue(self) -> None:
        seed = canonical_dual_shortfall_seed()

        class Client:
            def __init__(self): self.calls = 0
            def complete(self, _prompt):
                self.calls += 1
                if self.calls == 1:
                    return "not json"
                return json.dumps([{
                    "family_id": "rescued", "parent_family_id": seed.family_id,
                    "action": "add_term", "fragment": {"op": "const", "value": 1.0},
                }])

        client = Client()
        rows = llm_proposal_batch_fn(client, [seed], 1, None, 1)
        self.assertEqual([row.family_id for row in rows], ["rescued"])
        self.assertEqual(client.calls, 2)

    def test_batch_protocol_caps_nonempty_repairs_at_ten_calls(self) -> None:
        seed = canonical_dual_shortfall_seed()
        row = {
            "family_id": "one", "parent_family_id": seed.family_id,
            "action": "add_term", "fragment": {"op": "const", "value": 1.0},
        }

        class Client:
            def __init__(self): self.calls = 0
            def complete(self, _prompt):
                self.calls += 1
                return json.dumps({"features": [], "mutations": [dict(row)]})

        client = Client()
        rows = llm_proposal_batch_fn(client, [seed], 1, None, 10)
        self.assertEqual(len(rows), 1)
        self.assertEqual(client.calls, 11)  # initial call + ten repairs

    def test_local_mutation_rejects_non_ast_fragment_before_generation(self) -> None:
        seed = canonical_dual_shortfall_seed()
        class Client:
            def __init__(self): self.calls = 0
            def complete(self, _prompt):
                self.calls += 1
                if self.calls == 1:
                    return json.dumps({"features": []})
                if self.calls == 2:
                    return json.dumps({"mutations": [{"family_id": "bad", "parent_family_id": seed.family_id, "action": "add_term", "fragment": "not-an-ast"}]})
                return json.dumps({"mutations": [{"family_id": "good", "parent_family_id": seed.family_id, "action": "add_term", "fragment": {"op": "const", "value": 1.0}}]})
        rows = llm_proposal_batch_fn(Client(), [seed], 1, None, 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].family_id, "good")

    def test_clip_numeric_bounds_are_normalized(self) -> None:
        seed = canonical_dual_shortfall_seed()
        class Client:
            def __init__(self): self.calls = 0
            def complete(self, _prompt):
                self.calls += 1
                if self.calls == 1:
                    return json.dumps({"features": []})
                return json.dumps({"mutations": [{
                    "family_id": "clip_bounds", "parent_family_id": seed.family_id,
                    "action": "add_term", "fragment": {"op": "mul", "args": [
                        {"op": "parameter", "name": "w"},
                        {"op": "clip", "arg": {"op": "feature", "name": "far_pipeline"}, "lower": 0, "upper": 10}
                    ]}, "new_parameters": [{"name": "w", "kind": "weight", "lower": -1, "upper": 1}]
                }]})
        rows = llm_proposal_batch_fn(Client(), [seed], 1, None, 1)
        self.assertEqual(len(rows), 1)

    def test_mutation_merges_parent_and_new_features(self) -> None:
        seed = canonical_dual_shortfall_seed()
        parent = PolicyFamily(seed.family_id + "_f", {"op": "add", "args": [seed.ast, {"op": "feature", "name": "f0"}]}, {}, seed.parameters, features=(FeatureSpec("f0", "tail_sum", 1),))
        class Client:
            def __init__(self): self.calls = 0
            def complete(self, _prompt):
                self.calls += 1
                if self.calls == 1: return json.dumps({"features": [{"name": "f1", "kind": "head_sum", "window": 1}]})
                return json.dumps({"mutations": [{"family_id": "child_f", "parent_family_id": parent.family_id, "action": "add_term", "fragment": {"op": "mul", "args": [{"op": "parameter", "name": "w1"}, {"op": "feature", "name": "f1"}]}, "new_parameters": [{"name": "w1", "kind": "weight", "lower": -1, "upper": 1}]}]})
        rows = llm_proposal_batch_fn(Client(), [parent], 1, None, 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual({f.name for f in rows[0].features}, {"f0", "f1"})

    def test_family_serialization_contract_includes_lineage_fields(self) -> None:
        seed = canonical_dual_shortfall_seed()
        self.assertIsNone(seed.parent_family_id)
        class Client:
            def __init__(self): self.calls = 0
            def complete(self, _prompt):
                self.calls += 1
                if self.calls == 1: return json.dumps({"features": []})
                return json.dumps({"mutations": [{"family_id": "lineage", "parent_family_id": seed.family_id, "action": "add_term", "fragment": {"op": "const", "value": 1}}]})
        child = llm_proposal_batch_fn(Client(), [seed], 1, None, 1)[0]
        self.assertEqual(child.parent_family_id, seed.family_id)
        self.assertEqual(child.parent_family_hash, _domain_for().semantic_hash(seed))
        self.assertEqual(child.mutation_action, "add_term")

    def test_prompt_includes_parent_performance_context(self) -> None:
        prompt = build_family_prompt([{
            "family_id": "parent",
            "description": "seed",
            "cross_context_score": 1.25,
            "status": "complete",
            "per_context": [{"context_id": "10", "robust_score": 0.9}],
        }])
        self.assertIn("PARENT PERFORMANCE CONTEXT", prompt)
        self.assertIn("1.25", prompt)
        self.assertIn("robust_score", prompt)

    def test_prompt_uses_soft_replacement_majority_without_forcing_replacement(self) -> None:
        prompt = build_family_prompt([])
        self.assertIn("more than half", prompt)
        self.assertIn("soft prior, not a quota", prompt)
        self.assertIn("Give meaningful opportunities to add_term", prompt)
        self.assertNotIn("make a concise one-step internal replacement", prompt)
        self.assertNotIn("Keep the replacement inside the existing policy skeleton", prompt)

    def test_prompt_forbids_root_nonnegative_wrapper(self) -> None:
        prompt = build_family_prompt([])
        self.assertIn("NEVER wrap the outermost policy AST", prompt)
        self.assertIn("maximum(policy, 0)", prompt)

    def test_batch_reflection_receives_complete_family_definition(self) -> None:
        seed = canonical_dual_shortfall_seed()
        payload = {"family_id": "reflect_me", "ast": seed.ast, "motifs": seed.motifs,
                   "parameters": [{"name": p.name, "kind": p.kind, "lower": p.lower, "upper": p.upper} for p in seed.parameters]}
        class Client:
            def __init__(self): self.calls = 0
            def complete(self, prompt):
                self.calls += 1
                if self.calls == 1: return json.dumps({"features": []})
                return json.dumps({"families": [payload]})
        class Reflection:
            def __init__(self): self.prompt = ""
            def complete(self, prompt):
                self.prompt = prompt
                return json.dumps({"accept": True})
        reflection = Reflection()
        rows = llm_proposal_batch_fn(Client(), [seed], 4, reflection, 1)
        self.assertEqual(len(rows), 1)
        self.assertIn('"ast"', reflection.prompt)
        self.assertIn('"parameters"', reflection.prompt)

    def test_periodic_reflection_extracts_only_cross_l_valid_motifs(self) -> None:
        seed = canonical_dual_shortfall_seed()
        class Client:
            def complete(self, _prompt):
                return json.dumps({"motifs": [
                    {"name": "safe_total", "ast": {"op": "motif", "name": "inventory_position"}},
                    {"name": "bad_raw", "ast": {"op": "state", "name": "p10"}},
                ]})
        out = llm_reflect_motifs(Client(), [seed], 5)
        self.assertEqual([row["name"] for row in out["accepted"]], ["safe_total"])
        self.assertTrue(out["rejected"])

    def test_periodic_reflection_resolves_family_feature_dependencies(self) -> None:
        feature = FeatureSpec("tail_half", "tail_sum", "ceil(L/2)")
        family = PolicyFamily("feature_owner", {"op": "feature", "name": feature.name}, {}, (), features=(feature,))
        class Client:
            def complete(self, _prompt):
                return json.dumps({"motifs": [{
                    "name": "tail_signal",
                    "ast": {"op": "feature", "name": feature.name},
                    "description": "reusable tail summary",
                }]})
        out = llm_reflect_motifs(Client(), [family], 5)
        self.assertEqual([row["name"] for row in out["accepted"]], ["tail_signal"])
        self.assertEqual(out["accepted"][0]["features"][0]["name"], feature.name)

    def test_scenario_score_uses_signed_gap_directly(self) -> None:
        for gap in (-2.0, 0.0, 0.5):
            score = scenario_score(
                [CaseMetric("3", f"case{i}", gap) for i in range(8)],
                "3",
            )
            self.assertAlmostEqual(score.robust_score, gap)
            self.assertAlmostEqual(cross_context_score((score,)), gap)

    def test_cross_lead_score_retains_moderate_worst_lead_penalty(self) -> None:
        rows = tuple(
            ScenarioScore(str(lead), 0.0, 0.0, value)
            for lead, value in zip((3, 5, 8, 10), (1.0, 2.0, 3.0, 4.0))
        )
        # The upper 20% of four equally weighted contexts is the worst context.
        self.assertAlmostEqual(cross_context_score(rows), 2.5 * 0.60 + 4.0 * 0.40)

    def test_task_planner_accepts_sparse_active_lead_times(self) -> None:
        rows = plan_family_tasks(
            canonical_dual_shortfall_seed(), [f"case{i}" for i in range(8)],
            contexts=("2", "5", "8", "10"),
        )
        self.assertEqual(len(rows), 32)
        self.assertEqual(sorted({row.context_id for row in rows}), ["10", "2", "5", "8"])
        for lead_time in (2, 5, 8, 10):
            per_l = [row for row in rows if row.context_id == str(lead_time)]
            self.assertEqual(len(per_l), 8)

    def test_strict_screen_does_not_reject_without_a_finite_incumbent(self) -> None:
        class Worker:
            def __call__(self, family, task):
                return CaseMetric(task.context_id, task.case_id, 1.0)
        screened, audit = screen_families(
            [canonical_dual_shortfall_seed()],
            [f"case{i}" for i in range(8)],
            Worker(), domain=_domain_for(),

            output_dir=__import__("pathlib").Path("/tmp/stage3_screen_test"), incumbent_score=float("inf"),
            workers=2,
        )
        self.assertEqual(screened, set())
        self.assertFalse(audit["enabled"])


if __name__ == "__main__":
    unittest.main()
