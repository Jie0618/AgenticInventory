from __future__ import annotations

import unittest
import time
from dataclasses import replace
from unittest.mock import patch

from optimization.models.types import EvaluationResult, ParameterSpec

from inventory.common.distributions import DiscreteDemand
from inventory.core import DeterministicLeadTime, LostSales, TransitionCosts
from inventory.core.factory import inventory_case
from optimization.policies.baselines.lead_time import baselines_for_lead_time
from evaluation.backends.lead_time.dp import solve_dp
from optimization import OptimizationConfig, SparseExactPolicyCostEvaluator, fit_inventory_constants
from optimization.solvers.pipeline import (
    StructuralAnalysis,
    _Archive,
    Candidate,
    _directional_overlap,
    _exact_local,
    _initial_exact_anchors,
    _mc_rerank_pre_shortlist,
    choose_route,
    optimize_policy_constants_pipeline,
)
from optimization.solvers.rules import (
    PIPELINE_RULES,
    certify_dimension_multiplier,
    resolve_certify_rules,
    resolve_pil_scalar_rules,
)


class PipelineTests(unittest.TestCase):
    def test_pil_scalar_route_and_certify_budget(self) -> None:
        analysis = StructuralAnalysis(
            "continuous",
            1,
            1,
            0,
            None,
            None,
            True,
            False,
            True,
            (),
            None,
            False,
            None,
            True,
            0,
            optimization_route_hint="pil_scalar",
        )
        self.assertEqual(
            choose_route(analysis, PIPELINE_RULES["certify"]),
            "R8_PIL_SCALAR",
        )
        rules = resolve_pil_scalar_rules(PIPELINE_RULES["certify"])
        self.assertEqual(rules.proposal_budget, 8_000)
        self.assertEqual(rules.soft_wall_time_seconds, 1_200.0)
        self.assertEqual(rules.exact_budget_total, 144)
        self.assertEqual(
            rules.initial_exact_budget
            + rules.rolling_exact_budget
            + rules.final_exact_budget,
            rules.exact_budget_total,
        )
        self.assertEqual(
            resolve_pil_scalar_rules(
                PIPELINE_RULES["certify"],
                explicit_proposal_override=True,
            ),
            PIPELINE_RULES["certify"],
        )

    def test_pil_scalar_route_finds_scalar_basin(self) -> None:
        class _Evaluator:
            optimization_route_hint = "pil_scalar"

            def evaluate(self, params, incumbent=None):
                del incumbent
                return EvaluationResult(
                    cost=(float(params["U"]) - 0.37) ** 2,
                    completed=True,
                    metadata={"fidelity": "exact"},
                )

        result = optimize_policy_constants_pipeline(
            (ParameterSpec("U", kind="continuous", lower=0.0, upper=1.0, default=0.5),),
            _Evaluator(),
            config=OptimizationConfig(mode="fast", total_evaluations=64, random_seed=7),
        )
        self.assertEqual(result.route, "R8_PIL_SCALAR")
        self.assertLess(abs(result.best_params["U"] - 0.37), 0.02)
        self.assertEqual(result.convergence_report["proposal_budget_total"], 64)

    def test_vbs_uses_only_safe_initial_anchors_and_reclaims_budget(self) -> None:
        class _ExactEvaluator:
            def evaluate(self, params, incumbent=None):
                return EvaluationResult(
                    cost=float(sum(params.values())),
                    completed=True,
                    metadata={"fidelity": "exact"},
                )

        vbs = next(
            policy
            for policy in baselines_for_lead_time(5, 20)
            if policy.name == "vector_base_stock"
        )
        rules = resolve_certify_rules(
            PIPELINE_RULES["certify"],
            parameter_count=len(vbs.parameter_specs),
            discrete_combination_count=None,
        )
        archive = _Archive(
            vbs.parameter_specs,
            _ExactEvaluator(),
            _ExactEvaluator(),
            rules,
            time.perf_counter(),
        )
        _initial_exact_anchors(archive, ())

        self.assertEqual(archive.phase_exact_used["initial"], 2)
        self.assertEqual(
            archive.shortlist_diagnostics["initial_anchor_strategy"],
            "vbs_safe_default_and_lower",
        )
        self.assertEqual(
            {row.source for row in archive.records.values()},
            {"default", "lower_bound"},
        )
        self.assertEqual(
            archive.final_exact_limit,
            rules.final_exact_budget + rules.initial_exact_budget - 2,
        )
        self.assertEqual(
            archive.final_exact_limit
            + archive.phase_exact_used["initial"]
            + rules.rolling_exact_budget,
            rules.exact_budget_total,
        )

    def test_certify_dimension_budget_schedule(self) -> None:
        base = PIPELINE_RULES["certify"]
        self.assertEqual(certify_dimension_multiplier(3), 0)
        self.assertEqual(certify_dimension_multiplier(8), 5)
        self.assertEqual(certify_dimension_multiplier(99), 7)

        unchanged = resolve_certify_rules(
            base,
            parameter_count=3,
            discrete_combination_count=None,
        )
        self.assertEqual(unchanged, base)

        scaled = resolve_certify_rules(
            base,
            parameter_count=8,
            discrete_combination_count=None,
        )
        self.assertEqual(scaled.proposal_budget, 60_000)
        self.assertEqual(scaled.exact_budget_total, 496)
        self.assertEqual(scaled.final_shortlist_budget, 186)
        self.assertEqual(
            scaled.initial_exact_budget
            + scaled.rolling_exact_budget
            + scaled.final_exact_budget,
            scaled.exact_budget_total,
        )

        max_scaled = resolve_certify_rules(
            base,
            parameter_count=99,
            discrete_combination_count=None,
        )
        self.assertEqual(max_scaled.proposal_budget, 76_000)
        self.assertEqual(max_scaled.exact_budget_total, 592)
        self.assertEqual(max_scaled.final_shortlist_budget, 222)

        # Complete finite enumeration and explicit experiment overrides retain
        # the route/budget contract.
        discrete_exact = resolve_certify_rules(
            base,
            parameter_count=8,
            discrete_combination_count=base.exact_enum_limit,
        )
        self.assertEqual(discrete_exact, base)
        explicit = resolve_certify_rules(
            base,
            parameter_count=8,
            discrete_combination_count=None,
            explicit_proposal_override=True,
        )
        self.assertEqual(explicit, base)

    def test_mc_rerank_normalizes_a_low_mixed_fidelity_share(self) -> None:
        class FakeFastEvaluator:
            calls = 0

            def rerank(self, candidates):
                self.calls += len(candidates)
                return len(candidates)

        class FakeArchive:
            fast_evaluator = FakeFastEvaluator()
            rules = PIPELINE_RULES["certify"]
            mc_rerank_count = 0
            shortlist_diagnostics = {}
            timings = {}

            def add_timing(self, name, seconds):
                self.timings[name] = self.timings.get(name, 0.0) + seconds

        rows = []
        for index in range(100):
            fidelity = "mc_fixed" if index < 2 else "stationary_1e-10"
            rows.append(
                Candidate(
                    index,
                    {"x": float(index)},
                    (float(index),),
                    "proposal",
                    fast=EvaluationResult(
                        cost=float(index),
                        completed=True,
                        metadata={"fidelity": fidelity},
                    ),
                )
            )
        archive = FakeArchive()
        archive.records = {row.key: row for row in rows}
        _mc_rerank_pre_shortlist(archive, count=40)
        self.assertGreater(archive.fast_evaluator.calls, 0)
        self.assertGreater(archive.shortlist_diagnostics["mc_rerank_evaluations"], 0)
        archive.rules = PIPELINE_RULES["standard"]
        archive.shortlist_diagnostics = {}
        _mc_rerank_pre_shortlist(archive, count=40)
        self.assertGreater(archive.fast_evaluator.calls, 0)

    def test_mc_rerank_keeps_full_calibration_for_a_mixed_pool(self) -> None:
        class FakeFastEvaluator:
            calls = 0

            def rerank(self, candidates):
                self.calls += len(candidates)
                return len(candidates)

        class FakeArchive:
            fast_evaluator = FakeFastEvaluator()
            rules = PIPELINE_RULES["certify"]
            mc_rerank_count = 0
            shortlist_diagnostics = {}
            timings = {}

            def add_timing(self, name, seconds):
                self.timings[name] = self.timings.get(name, 0.0) + seconds

        rows = []
        for index in range(100):
            fidelity = "mc_fixed" if index < 20 else "stationary_1e-10"
            rows.append(
                Candidate(
                    index,
                    {"x": float(index)},
                    (float(index),),
                    "proposal",
                    fast=EvaluationResult(
                        cost=float(index),
                        completed=True,
                        metadata={"fidelity": fidelity},
                    ),
                )
            )
        archive = FakeArchive()
        archive.records = {row.key: row for row in rows}
        _mc_rerank_pre_shortlist(archive, count=40)
        self.assertGreater(archive.fast_evaluator.calls, 0)

    def test_directional_overlap_budget_is_total_across_rounds(self) -> None:
        exact = EvaluationResult(cost=10.0, completed=True, metadata={"fidelity": "exact"})
        seed = Candidate(0, {"x": 0.0}, (0.0,), "seed", fast=exact, exact=exact)
        rows = [seed]
        for index in range(1, 6):
            rows.append(
                Candidate(
                    index,
                    {"x": float(index)},
                    (float(index),),
                    "proposal",
                    fast=EvaluationResult(
                        cost=9.0 - index,
                        completed=True,
                        metadata={"fidelity": "stationary_1e-10"},
                    ),
                )
            )

        class FakeArchive:
            records = {row.key: row for row in rows}
            exact_best_cost = 10.0
            final_exact_used = 0
            rules = replace(PIPELINE_RULES["fast"], overlap_budget=2)

            def evaluate_exact(self, params, **_kwargs):
                row = self.records[(float(params["x"]),)]
                row.exact = EvaluationResult(
                    cost=9.5, completed=True, metadata={"fidelity": "exact"}
                )
                self.final_exact_used += 1
                return row

        archive = FakeArchive()
        _directional_overlap(archive, rows)
        self.assertEqual(archive.final_exact_used, 2)

    def test_exact_local_moves_the_center_and_is_monotone(self) -> None:
        class QuadraticExact:
            def evaluate(self, params, incumbent=None):
                del incumbent
                return EvaluationResult(
                    cost=(float(params["x"]) - 0.2) ** 2
                    + (float(params["k"]) - 9.0) ** 2,
                    completed=True,
                    metadata={"fidelity": "exact"},
                )

        rules = replace(
            PIPELINE_RULES["fast"],
            exact_budget_total=16,
            final_exact_budget=16,
            exact_local_budget=8,
        )
        evaluator = QuadraticExact()
        archive = _Archive(
            (
                ParameterSpec("k", kind="integer", lower=0.0, upper=20.0),
                ParameterSpec("x", kind="continuous", lower=0.0, upper=1.0),
            ),
            evaluator,
            evaluator,
            rules,
            time.perf_counter(),
        )
        archive.evaluate_exact(
            {"k": 10.0, "x": 0.1},
            source="seed",
            phase="final",
            allow_early_stop=False,
        )
        before = archive.exact_best_cost
        _exact_local(archive, archive.specs)
        self.assertLessEqual(archive.exact_best_cost, before)
        self.assertEqual(archive.exact_best_params["k"], 9.0)
        self.assertAlmostEqual(archive.exact_best_params["x"], 0.2, places=12)
        self.assertEqual(archive.exact_count, rules.exact_budget_total)

    def test_route_boundaries_are_deterministic(self) -> None:
        rules = PIPELINE_RULES["fast"]
        self.assertEqual(
            choose_route(
                StructuralAnalysis("none", 0, 0, 0, None, None, True, False, True, (), None, False, None, True, 0),
                rules,
            ),
            "R0_PARAMETER_FREE",
        )
        self.assertEqual(
            choose_route(
                StructuralAnalysis("discrete", 1, 0, 1, rules.exact_enum_limit, None, True, False, True, (), 1, False, None, True, 0),
                rules,
            ),
            "R1_SMALL_DISCRETE_EXACT",
        )

    def test_guarded_stationary_is_strictly_close_to_exact(self) -> None:
        demand = DiscreteDemand.from_pmf({0: 0.35, 1: 0.4, 3: 0.25})
        reference = solve_dp(2, 1, 5, demand, 1.0, 4.0)
        spec = next(item for item in baselines_for_lead_time(2, 5) if item.name == "weighted_tilted_pic")
        params = {parameter.name: parameter.initial_value() for parameter in spec.parameter_specs}
        exact = SparseExactPolicyCostEvaluator(spec.ast, reference, 100, spec.motifs, max_cycle_period=0).evaluate(params)
        fast = SparseExactPolicyCostEvaluator(
            spec.ast,
            reference,
            100,
            spec.motifs,
            max_cycle_period=1,
            stationary_required_consecutive_hits=3,
            convergence_tolerance=1.0e-10,
        ).evaluate(params)
        self.assertLessEqual(abs(fast.cost - exact.cost), 1.0e-7 * max(1.0, abs(exact.cost)))
        self.assertTrue(fast.metadata["stationary_triggered"] or fast.metadata["exact_fallback"])

    def test_pipeline_default_returns_strict_exact_results(self) -> None:
        demand = DiscreteDemand.poisson(2, 4)
        reference = solve_dp(1, 1, 6, demand, 1.0, 4.0)
        spec = next(item for item in baselines_for_lead_time(1, 6) if item.name == "weighted_tilted_pic")
        evaluator = SparseExactPolicyCostEvaluator(spec.ast, reference, 30, spec.motifs, max_cycle_period=0)
        case = inventory_case(
            name="pipeline-case",
            demand=demand,
            n=6,
            costs=TransitionCosts(holding=1.0, shortage=4.0),
            lead_time=DeterministicLeadTime(1),
            backlog=LostSales(),
            capacity=6,
        )
        result = fit_inventory_constants(
            spec.ast,
            case,
            1,
            spec.parameter_specs,
            reference=reference,
            motifs=spec.motifs,
            config=OptimizationConfig(mode="fast", total_evaluations=32, random_seed=11),
            cost_evaluator=evaluator,
        )
        report = result.optimizer_metadata["convergence_report"]
        self.assertEqual(report["pipeline_version"], "current")
        self.assertEqual(report["screening_fidelity"], "disabled")
        self.assertGreater(report["exact_evaluations"], 0)
        self.assertTrue(result.success)

    def test_r1_certificate_reports_exact_budget_override(self) -> None:
        demand = DiscreteDemand.poisson(2, 3)
        reference = solve_dp(1, 1, 4, demand, 1.0, 4.0)
        spec = next(item for item in baselines_for_lead_time(1, 4) if item.name == "constant_order")
        evaluator = SparseExactPolicyCostEvaluator(spec.ast, reference, 20, spec.motifs, max_cycle_period=0)
        result = optimize_policy_constants_pipeline(
            spec.parameter_specs,
            evaluator,
            config=OptimizationConfig(mode="certify", total_evaluations=8),
        )
        report = result.convergence_report
        self.assertEqual(result.route, "R1_SMALL_DISCRETE_EXACT")
        self.assertTrue(report["certified_global"])
        self.assertTrue(report["exact_budget_override"])
        self.assertEqual(report["exact_evaluations"], report["proposal_count"])
        self.assertEqual(report["best_fidelity"], "exact")

    def test_r1_two_fidelity_exhausts_low_and_reserves_strict_evaluations(self) -> None:
        class _CountingEvaluator:
            def __init__(self, fidelity: str) -> None:
                self.fidelity = fidelity
                self.calls = 0

            def evaluate(self, params, incumbent=None):
                del incumbent
                self.calls += 1
                value = float(params["x"])
                return EvaluationResult(
                    cost=abs(value - 73.0),
                    completed=True,
                    metadata={"fidelity": self.fidelity},
                )

            def evaluate_many(self, params_batch, incumbent=None):
                return [self.evaluate(params, incumbent) for params in params_batch]

        class _TwoFidelityEvaluator(_CountingEvaluator):
            two_fidelity_exhaustive_route = True

            def __init__(self) -> None:
                super().__init__("exact")
                self.low = _CountingEvaluator("mc_fixed")

            def make_fast_evaluator(self, config, mode, route, parameter_count):
                del config, mode, route, parameter_count
                return self.low

        values = tuple(float(index) for index in range(200))
        spec = ParameterSpec(
            "x",
            kind="grid",
            lower=0.0,
            upper=199.0,
            values=values,
            default=50.0,
        )
        evaluator = _TwoFidelityEvaluator()
        result = optimize_policy_constants_pipeline(
            (spec,),
            evaluator,
            config=OptimizationConfig(
                mode="standard", total_evaluations=200, random_seed=9
            ),
        )
        report = result.convergence_report
        self.assertEqual(result.route, "R1_SMALL_DISCRETE_EXACT")
        self.assertFalse(report["certified_global"])
        self.assertFalse(report["exact_budget_override"])
        self.assertEqual(report["proposal_count"], 200)
        self.assertEqual(evaluator.low.calls, 200)
        self.assertGreater(evaluator.calls, 0)
        self.assertLessEqual(evaluator.calls, PIPELINE_RULES["standard"].exact_budget_total)
        self.assertEqual(result.best_params["x"], 73.0)

    def test_stationary_proposal_can_never_be_published_as_best(self) -> None:
        class _FakeEvaluator:
            def evaluate(self, params, incumbent=None):
                return EvaluationResult(cost=10.0, completed=True, metadata={"fidelity": "exact"})

            def stationary_copy(self, **kwargs):
                return _FakeFastEvaluator()

        class _FakeFastEvaluator:
            def evaluate(self, params, incumbent=None):
                return EvaluationResult(
                    cost=0.0,
                    completed=True,
                    metadata={"fidelity": "stationary_1e-10", "stationary_triggered": True},
                )

        spec = ParameterSpec("x", kind="continuous", lower=0.0, upper=1.0, default=0.5)
        result = optimize_policy_constants_pipeline(
            (spec,),
            _FakeEvaluator(),
            config=OptimizationConfig(mode="fast", total_evaluations=8),
        )
        self.assertEqual(result.best_cost, 10.0)

    def test_proposal_wall_is_checked_after_initial_exact_phase(self) -> None:
        clock = [0.0]

        class _SlowExactEvaluator:
            def evaluate(self, params, incumbent=None):
                clock[0] += 0.2
                return EvaluationResult(cost=10.0, completed=True, metadata={"fidelity": "exact"})

            def stationary_copy(self, **kwargs):
                return self

        spec = ParameterSpec("x", kind="continuous", lower=0.0, upper=1.0, default=0.5)
        fast_rules = replace(PIPELINE_RULES["fast"], soft_wall_time_seconds=0.15)
        with patch.dict(PIPELINE_RULES, {"fast": fast_rules}), patch(
            "optimization.solvers.pipeline.time.perf_counter",
            side_effect=lambda: clock[0],
        ):
            result = optimize_policy_constants_pipeline(
                (spec,),
                _SlowExactEvaluator(),
                config=OptimizationConfig(mode="fast", total_evaluations=8),
            )
        report = result.convergence_report
        self.assertTrue(report["wall_time_limited"])
        self.assertEqual(report["initial_exact_evaluations"], 4)
        self.assertEqual(report["proposal_count"], 0)
        self.assertGreaterEqual(report["final_exact_evaluations"], 1)

    def test_proposal_wall_does_not_block_final_exact(self) -> None:
        clock = [0.0]

        class _ExactEvaluator:
            def evaluate(self, params, incumbent=None):
                return EvaluationResult(cost=10.0, completed=True, metadata={"fidelity": "exact"})

            def stationary_copy(self, **kwargs):
                return _SlowFastEvaluator()

        class _SlowFastEvaluator:
            def evaluate(self, params, incumbent=None):
                clock[0] += 0.08
                return EvaluationResult(
                    cost=1.0,
                    completed=True,
                    metadata={"fidelity": "stationary_1e-10", "stationary_triggered": True},
                )

        spec = ParameterSpec("x", kind="continuous", lower=0.0, upper=1.0, default=0.5)
        fast_rules = replace(PIPELINE_RULES["fast"], soft_wall_time_seconds=0.05)
        with patch.dict(PIPELINE_RULES, {"fast": fast_rules}), patch(
            "optimization.solvers.pipeline.time.perf_counter",
            side_effect=lambda: clock[0],
        ):
            result = optimize_policy_constants_pipeline(
                (spec,),
                _ExactEvaluator(),
                config=OptimizationConfig(mode="fast", total_evaluations=8, random_seed=3),
            )
        report = result.convergence_report
        self.assertTrue(report["proposal_wall_time_limited"])
        self.assertEqual(report["search_stop_phase"], "proposal")
        self.assertFalse(report["exact_wall_time_limited"])
        self.assertGreaterEqual(report["final_exact_evaluations"], 1)
        self.assertEqual(result.best_cost, 10.0)

    def test_anchor_timeout_does_not_publish_partial_cost(self) -> None:
        demand = DiscreteDemand.poisson(2, 3)
        reference = solve_dp(1, 1, 4, demand, 1.0, 4.0)
        spec = next(item for item in baselines_for_lead_time(1, 4) if item.name == "weighted_tilted_pic")
        evaluator = SparseExactPolicyCostEvaluator(spec.ast, reference, 30, spec.motifs, max_cycle_period=0)
        evaluator.set_debug_context(
            {
                "evaluation_kind": "anchor",
                "evaluation_timeout_seconds": 0.0,
                "evaluation_deadline_at": time.perf_counter() - 1.0,
            }
        )
        result = evaluator.evaluate({parameter.name: parameter.initial_value() for parameter in spec.parameter_specs})
        self.assertFalse(result.completed)
        self.assertTrue(result.metadata["timed_out"])
        self.assertFalse(result.metadata["bound_valid"])
        self.assertEqual(result.cost, float("inf"))
        self.assertIn("partial_cost", result.metadata)
        self.assertNotIn("lower_bound", result.metadata)

    def test_initial_anchors_use_incumbent_early_stop_after_first_valid_anchor(self) -> None:
        class _AnchorAwareEvaluator:
            def __init__(self) -> None:
                self.context = {}
                self.anchor_calls = 0
                self.anchor_results = []

            def set_debug_context(self, context) -> None:
                self.context = dict(context or {})

            def stationary_copy(self, **kwargs):
                return self

            def evaluate(self, params, incumbent=None):
                if self.context.get("evaluation_kind") == "anchor":
                    self.anchor_calls += 1
                    if self.anchor_calls == 1:
                        result = EvaluationResult(
                            cost=float("inf"),
                            completed=False,
                            metadata={
                                "fidelity": "timeout_incomplete",
                                "timed_out": True,
                                "bound_valid": False,
                                "partial_cost": 7.0,
                            },
                        )
                    elif incumbent is not None:
                        result = EvaluationResult(
                            cost=float("inf"),
                            completed=False,
                            early_stopped=True,
                            metadata={
                                "fidelity": "exact",
                                "stop_reason": "incumbent_bound",
                                "timed_out": False,
                                "bound_valid": True,
                                "lower_bound": float(incumbent) + 1.0,
                            },
                        )
                    else:
                        result = EvaluationResult(
                            cost=100.0,
                            completed=True,
                            metadata={"fidelity": "exact"},
                        )
                    self.anchor_results.append(result)
                    return result
                return EvaluationResult(cost=100.0, completed=True, metadata={"fidelity": "exact"})

        spec = ParameterSpec("x", kind="continuous", lower=0.0, upper=1.0, default=0.5)
        evaluator = _AnchorAwareEvaluator()
        result = optimize_policy_constants_pipeline(
            (spec,),
            evaluator,
            config=OptimizationConfig(mode="fast", total_evaluations=8),
        )
        report = result.convergence_report
        self.assertEqual(result.best_cost, 100.0)
        self.assertEqual(report["anchor_timeout_count"], 1)
        self.assertGreaterEqual(report["anchor_early_stop_count"], 1)
        self.assertEqual(evaluator.anchor_results[0].cost, float("inf"))
        self.assertNotIn("lower_bound", evaluator.anchor_results[0].metadata)
        self.assertEqual(evaluator.anchor_results[1].cost, 100.0)

    def test_continuous_search_recovers_unused_phase_budget(self) -> None:
        class _ExactEvaluator:
            def evaluate(self, params, incumbent=None):
                return EvaluationResult(cost=10.0, completed=True, metadata={"fidelity": "exact"})

            def stationary_copy(self, **kwargs):
                return _FastEvaluator()

        class _FastEvaluator:
            def evaluate(self, params, incumbent=None):
                return EvaluationResult(
                    cost=1.0,
                    completed=True,
                    metadata={"fidelity": "stationary_1e-10", "stationary_triggered": True},
                )

        spec = ParameterSpec("x", kind="continuous", lower=0.0, upper=1.0, default=0.5)
        result = optimize_policy_constants_pipeline(
            (spec,),
            _ExactEvaluator(),
            config=OptimizationConfig(mode="standard", total_evaluations=256, random_seed=7),
        )
        report = result.convergence_report
        self.assertEqual(report["proposal_count"], 256)
        self.assertEqual(report["proposal_budget_unused"], 0)
        self.assertGreater(report["budget_recovery_proposals"], 0)
        self.assertEqual(report["search_stop_reason"], "proposal_budget_exhausted")
        self.assertLessEqual(report["fast_stationary_count"], report["fast_evaluator_calls"])

    def test_shortlist_reports_bounded_reduced_pool(self) -> None:
        class _ExactEvaluator:
            def evaluate(self, params, incumbent=None):
                return EvaluationResult(cost=10.0, completed=True, metadata={"fidelity": "exact"})

            def stationary_copy(self, **kwargs):
                return _FastEvaluator()

        class _FastEvaluator:
            def evaluate(self, params, incumbent=None):
                return EvaluationResult(
                    cost=1.0,
                    completed=True,
                    metadata={"fidelity": "stationary_1e-10", "stationary_triggered": True},
                )

        specs = (
            ParameterSpec("x", kind="continuous", lower=0.0, upper=1.0, default=0.5),
            ParameterSpec("y", kind="continuous", lower=0.0, upper=1.0, default=0.5),
        )
        result = optimize_policy_constants_pipeline(
            specs,
            _ExactEvaluator(),
            config=OptimizationConfig(mode="standard", total_evaluations=128, random_seed=9),
        )
        shortlist = result.convergence_report["shortlist"]
        self.assertLessEqual(shortlist["shortlist_input_records"], result.convergence_report["unique_parameter_points"])
        self.assertLessEqual(shortlist["shortlist_reduced_pool_count"], 128)
        self.assertEqual(shortlist["shortlist_selected_count"], 40)
        self.assertGreaterEqual(shortlist["shortlist_selected_best"], 1)

if __name__ == "__main__":
    unittest.main()
