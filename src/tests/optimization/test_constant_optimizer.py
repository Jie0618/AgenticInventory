from __future__ import annotations

import time
import unittest

import numpy as np

from inventory.common.distributions import DiscreteDemand
from optimization.policies.baselines.lead_time import baselines_for_lead_time
from evaluation.backends.lead_time.dp import solve_dp
from inventory.core.factory import inventory_case
from inventory.core import DeterministicLeadTime, LostSales, TransitionCosts
from optimization.backends.lead_time.rolling import rolling_reference, solve_optimal_cost_rolling
from inventory.policy.ast import add, maximum, neg, parameter, state
from optimization import (
    DenseExactPolicyCostEvaluator,
    OptimizationConfig,
    ParameterSpec,
    SparseExactPolicyCostEvaluator,
    canonical_parameter_key,
    canonicalize_params,
    fit_inventory_constants,
    optimize_policy_constants,
)


def Case(name, lead_time, h, b, demand, kind, n):
    return inventory_case(
        name=name, demand=demand, n=n, kind=kind,
        costs=TransitionCosts(holding=h, shortage=b),
        lead_time=DeterministicLeadTime(lead_time), backlog=LostSales(), capacity=n,
    )


class UnifiedConstantOptimizerTests(unittest.TestCase):
    def test_parameter_spec_rejects_invalid_public_values(self) -> None:
        invalid_factories = (
            lambda: ParameterSpec("", lower=0, upper=1),
            lambda: ParameterSpec("   ", lower=0, upper=1),
            lambda: ParameterSpec(None, lower=0, upper=1),
            lambda: ParameterSpec("x", lower=True, upper=1),
            lambda: ParameterSpec("x", lower="0", upper=1),
            lambda: ParameterSpec("x", lower=float("nan"), upper=1),
            lambda: ParameterSpec("x", lower=0, upper=float("inf")),
            lambda: ParameterSpec("x", lower=0, upper=1, default=float("nan")),
            lambda: ParameterSpec("x", lower=0, upper=1, default="0.5"),
            lambda: ParameterSpec(
                "x", lower=0, upper=1, monotone_group="g", monotone_index=1.5
            ),
            lambda: ParameterSpec(
                "x", lower=0, upper=1, monotone_group=" ", monotone_index=0
            ),
            lambda: ParameterSpec(
                "x", lower=0, upper=1, monotone_group=1, monotone_index=0
            ),
        )
        for factory in invalid_factories:
            with self.subTest(factory=factory), self.assertRaises(ValueError):
                factory()

    def test_optimization_config_validates_and_normalizes_public_values(self) -> None:
        invalid_values = (
            {"mode": "bogus"},
            {"rounding": "bogus"},
            {"mse_kind": "bogus"},
            {"strategy": ""},
            {"objective": None},
            {"require_policy_cost": "false"},
            {"maxiter": 1.5},
            {"random_seed": 2.5},
            {"action_cap": -1},
            {"mse_weight": float("nan")},
            {"mse_weight": True},
            {"mse_weight": "1.0"},
            {"policy_weight": float("inf")},
            {"initial_state": (0, 1.5)},
        )
        for values in invalid_values:
            with self.subTest(values=values), self.assertRaises(ValueError):
                OptimizationConfig(**values)
        config = OptimizationConfig(maxiter=8.0, random_seed=3.0, initial_state=(0.0, 1.0))
        self.assertEqual(config.maxiter, 8)
        self.assertEqual(config.random_seed, 3)
        self.assertEqual(config.initial_state, (0, 1))

    def test_parameter_kinds_and_canonical_key(self) -> None:
        specs = (
            ParameterSpec("i", kind="integer", lower=0, upper=5, default=2),
            ParameterSpec("g", kind="grid", lower=0, upper=1, values=(0.0, 0.25, 1.0)),
            ParameterSpec("c", kind="continuous", lower=0, upper=1),
        )
        params = canonicalize_params({"i": 2.6, "g": 0.3, "c": 0.12345678901234}, specs)
        self.assertEqual(params["i"], 3.0)
        self.assertEqual(params["g"], 0.25)
        self.assertEqual(canonical_parameter_key(params, specs)[2], round(params["c"], 12))

    def test_sparse_and_dense_exact_cost_agree(self) -> None:
        demand = DiscreteDemand.from_pmf({0: 0.5, 2: 0.5})
        dp = solve_dp(1, 8, 3, demand, 1.0, 3.0)
        spec = baselines_for_lead_time(1, 3)[1]
        params = {"S": 2.0}
        dense = DenseExactPolicyCostEvaluator(spec.ast, dp, spec.motifs).evaluate(params)
        sparse_evaluator = SparseExactPolicyCostEvaluator(spec.ast, dp, 8, spec.motifs)
        sparse = sparse_evaluator.evaluate(params)
        self.assertAlmostEqual(dense.cost, sparse.cost, places=10)
        forced_dense = SparseExactPolicyCostEvaluator(
            spec.ast, dp, 8, spec.motifs, dense_switch_fraction=0.0
        ).evaluate(params)
        forced_sparse = SparseExactPolicyCostEvaluator(
            spec.ast, dp, 8, spec.motifs, dense_switch_fraction=float("inf")
        ).evaluate(params)
        self.assertAlmostEqual(forced_dense.cost, forced_sparse.cost, places=10)
        self.assertGreater(forced_dense.metadata["dense_steps"], 0)
        self.assertEqual(forced_sparse.metadata["dense_steps"], 0)
        sparse_evaluator.configure_signature_bank(20)
        first_verified = sparse_evaluator.evaluate(params)
        cached_verified = sparse_evaluator.evaluate(params)
        self.assertAlmostEqual(first_verified.cost, cached_verified.cost, places=10)
        self.assertTrue(cached_verified.metadata["verified_action_map_cache_hit"])
        second_params = {"S": 1.0}
        dense_second = DenseExactPolicyCostEvaluator(spec.ast, dp, spec.motifs).evaluate(second_params)
        sparse_second = sparse_evaluator.evaluate(second_params)
        self.assertAlmostEqual(dense_second.cost, sparse_second.cost, places=10)
        stopped = SparseExactPolicyCostEvaluator(spec.ast, dp, 8, spec.motifs).evaluate(params, incumbent=0.0)
        self.assertTrue(stopped.early_stopped)
        self.assertFalse(stopped.completed)
        tolerance = 1.0e-12 * max(1.0, abs(sparse.cost))
        near_tie = SparseExactPolicyCostEvaluator(spec.ast, dp, 8, spec.motifs).evaluate(
            params,
            incumbent=sparse.cost - 0.5 * tolerance,
        )
        self.assertTrue(near_tie.completed)
        self.assertFalse(near_tie.early_stopped)
        pruned = SparseExactPolicyCostEvaluator(
            spec.ast, dp, 8, spec.motifs, probability_prune_threshold=1.0e-6
        ).evaluate(params)
        self.assertEqual(pruned.metadata["fidelity"], "probability_pruned")

    def test_lazy_and_eager_state_geometry_agree_for_l1_through_l8(self) -> None:
        demand = DiscreteDemand.from_pmf({0: 0.25, 1: 0.5, 2: 0.25})
        for lead_time in range(1, 9):
            dp = solve_dp(lead_time, 6, 2, demand, 1.0, 3.0)
            for spec in baselines_for_lead_time(lead_time, 2):
                params = {
                    parameter.name: parameter.initial_value()
                    for parameter in spec.parameter_specs
                }
                eager = SparseExactPolicyCostEvaluator(
                    spec.ast,
                    dp,
                    6,
                    spec.motifs,
                    max_cycle_period=0,
                )
                lazy = SparseExactPolicyCostEvaluator(
                    spec.ast,
                    dp,
                    6,
                    spec.motifs,
                    max_cycle_period=0,
                    full_state_geometry_limit=0,
                )
                eager_result = eager.evaluate(params)
                lazy_result = lazy.evaluate(params)
                self.assertFalse(eager.lazy_state_geometry)
                self.assertTrue(lazy.lazy_state_geometry)
                self.assertIsNotNone(eager._decoded_states)  # noqa: SLF001
                self.assertIsNone(lazy._decoded_states)  # noqa: SLF001
                self.assertAlmostEqual(
                    eager_result.cost,
                    lazy_result.cost,
                    places=10,
                    msg=f"L={lead_time} policy={spec.name}",
                )

    def test_state_geometry_mode_is_preserved_by_evaluator_copies(self) -> None:
        demand = DiscreteDemand.from_pmf({0: 0.5, 1: 0.5})
        dp = solve_dp(3, 4, 2, demand, 1.0, 2.0)
        spec = baselines_for_lead_time(3, 2)[0]
        evaluator = SparseExactPolicyCostEvaluator(
            spec.ast,
            dp,
            4,
            spec.motifs,
            full_state_geometry_limit=0,
        )
        self.assertTrue(evaluator.screening_copy().lazy_state_geometry)
        self.assertTrue(evaluator.stationary_copy().lazy_state_geometry)

    def test_stationary_dp_continuation_lower_bound_reuses_short_reference(self) -> None:
        demand = DiscreteDemand.from_pmf({0: 0.25, 2: 0.5, 5: 0.25})
        reference = solve_dp(2, 3, 5, demand, 1.0, 6.0)
        spec = next(
            item
            for item in baselines_for_lead_time(2, 5)
            if item.name == "vector_base_stock"
        )
        evaluator = SparseExactPolicyCostEvaluator(spec.ast, reference, 10, spec.motifs)
        initial_index = evaluator._encode_state((0, 0, 0), 6)
        bound = evaluator._dp_remaining_lower_bound(
            2,
            np.asarray([initial_index], dtype=np.int64),
            np.asarray([1.0]),
            6,
            3,
        )
        expected = reference.value(0, (0, 0, 0)) + 5 * evaluator._minimum_period_cost
        self.assertAlmostEqual(bound, expected, places=12)

    def test_rolling_zero_memory_and_compressed_continuation_bounds(self) -> None:
        demand = DiscreteDemand.from_pmf({0: 0.25, 2: 0.5, 5: 0.25})
        case = Case("rolling_bound", 2, 1.0, 6.0, demand, "unit", 5)
        reference = rolling_reference(case)
        solve_optimal_cost_rolling(
            reference,
            3,
            sample_size=1,
            continuation_bound_periods=3,
        )
        spec = next(
            item
            for item in baselines_for_lead_time(2, 5)
            if item.name == "vector_base_stock"
        )
        evaluator = SparseExactPolicyCostEvaluator(spec.ast, reference, 10, spec.motifs)
        state = (1, 2, 3)
        state_index = evaluator._encode_state(state, 6)
        bound = evaluator._dp_remaining_lower_bound(
            2,
            np.asarray([state_index], dtype=np.int64),
            np.asarray([1.0]),
            6,
            3,
        )
        dense = solve_dp(2, 3, 5, demand, 1.0, 6.0)
        expected = dense.value(0, state) + 5 * evaluator._minimum_period_cost
        self.assertAlmostEqual(bound, expected, places=11)

        reference.continuation_horizon = 0
        reference.continuation_values = None
        floor = evaluator._dp_remaining_lower_bound(
            2,
            np.asarray([state_index], dtype=np.int64),
            np.asarray([1.0]),
            6,
            3,
        )
        self.assertAlmostEqual(floor, 8 * evaluator._minimum_period_cost, places=12)

    def test_l1_inventory_position_reduction_matches_general_sparse_engine(self) -> None:
        demand = DiscreteDemand.from_pmf({0: 0.3, 1: 0.4, 3: 0.3})
        reference = solve_dp(1, 1, 8, demand, 1.0, 4.0)
        for name in ("base_stock", "capped_base_stock", "tilted_cbs", "tilted_pic"):
            spec = next(
                item for item in baselines_for_lead_time(1, 8) if item.name == name
            )
            params = {
                parameter.name: parameter.initial_value()
                for parameter in spec.parameter_specs
            }
            reduced = SparseExactPolicyCostEvaluator(
                spec.ast, reference, 40, spec.motifs
            ).evaluate(params)
            general = SparseExactPolicyCostEvaluator(
                spec.ast,
                reference,
                40,
                spec.motifs,
                use_inventory_position_reduction=False,
            ).evaluate(params)
            self.assertEqual(reduced.metadata["engine"], "l1_inventory_position")
            self.assertAlmostEqual(reduced.cost, general.cost, places=10, msg=name)

    def test_stable_transition_matrix_matches_pruned_general_aggregation(self) -> None:
        demand = DiscreteDemand.from_pmf({0: 0.55, 1: 0.25, 4: 0.2})
        reference = solve_dp(1, 1, 10, demand, 1.0, 4.0)
        spec = next(
            item
            for item in baselines_for_lead_time(1, 10)
            if item.name == "weighted_tilted_pic"
        )
        params = {"S": 7.0, "K": 0.8, "C": 3.0, "alpha": 0.15, "w1": 0.65}
        fast = SparseExactPolicyCostEvaluator(
            spec.ast,
            reference,
            80,
            spec.motifs,
            probability_prune_threshold=1.0e-8,
        ).evaluate(params)
        general = SparseExactPolicyCostEvaluator(
            spec.ast,
            reference,
            80,
            spec.motifs,
            probability_prune_threshold=1.0e-8,
            use_stable_transition_matrix=False,
        ).evaluate(params)
        self.assertAlmostEqual(fast.cost, general.cost, places=10)
        sparse_cached = SparseExactPolicyCostEvaluator(
            spec.ast,
            reference,
            80,
            spec.motifs,
            probability_prune_threshold=1.0e-8,
            stable_dense_cell_limit=0,
        ).evaluate(params)
        self.assertAlmostEqual(sparse_cached.cost, general.cost, places=10)
        self.assertGreater(sparse_cached.metadata["stable_sparse_steps"], 0)
        legacy_sparse_cached = SparseExactPolicyCostEvaluator(
            spec.ast,
            reference,
            80,
            spec.motifs,
            probability_prune_threshold=1.0e-8,
            stable_dense_cell_limit=0,
            use_csr_sparse_transition=False,
        ).evaluate(params)
        self.assertAlmostEqual(sparse_cached.cost, legacy_sparse_cached.cost, places=10)
        self.assertEqual(sparse_cached.metadata["stable_transition_mode"], "csr")
        self.assertEqual(legacy_sparse_cached.metadata["stable_transition_mode"], "sparse")

    def test_l2_large_partition_exact_behavior_key_requires_full_map_match(self) -> None:
        demand = DiscreteDemand.from_pmf({0: 0.5, 2: 0.5})
        reference = solve_dp(2, 1, 20, demand, 1.0, 4.0)
        spec = next(
            item
            for item in baselines_for_lead_time(2, 20)
            if item.name == "vector_base_stock"
        )
        evaluator = SparseExactPolicyCostEvaluator(spec.ast, reference, 20, spec.motifs)
        self.assertIsNotNone(evaluator._exact_behavior_states)
        self.assertGreater(len(evaluator._exact_behavior_states), 5000)
        zero_a = {"S0": 0.0, "S1": 0.0, "S2": 0.0}
        zero_b = {"S0": 0.0, "S1": 60.0, "S2": 60.0}
        nonzero = {"S0": 5.0, "S1": 5.0, "S2": 5.0}
        key_a = evaluator.exact_action_signature_key(zero_a)
        key_b = evaluator.exact_action_signature_key(zero_b)
        key_nonzero = evaluator.exact_action_signature_key(nonzero)
        self.assertEqual(key_a, key_b)
        self.assertNotEqual(key_a, key_nonzero)
        evaluator.disable_large_behavior_proof()
        self.assertIsNone(evaluator._exact_behavior_states)
        self.assertEqual(evaluator.exact_behavior_class_count, 0)

    def test_arbitrary_ast_action_projection_matches_uncompressed_actions(self) -> None:
        demand = DiscreteDemand.from_pmf({0: 0.4, 1: 0.35, 3: 0.25})
        reference = solve_dp(4, 1, 5, demand, 1.0, 4.0)
        ast = maximum(
            add(parameter("S"), neg(state("x")), neg(state("p1"))),
            {"op": "const", "value": 0.0},
        )
        params = {"S": 6.0}
        projected = SparseExactPolicyCostEvaluator(
            ast,
            reference,
            60,
            use_stable_transition_matrix=False,
        ).evaluate(params)
        general = SparseExactPolicyCostEvaluator(
            ast,
            reference,
            60,
            use_stable_transition_matrix=False,
            use_action_projection=False,
        ).evaluate(params)
        self.assertEqual(projected.metadata["action_projection_classes"], 36)
        self.assertAlmostEqual(projected.cost, general.cost, places=10)

    def test_arbitrary_ast_optimizer_matches_uncompressed_general_engine(self) -> None:
        demand = DiscreteDemand.from_pmf({0: 0.4, 1: 0.35, 3: 0.25})
        reference = solve_dp(3, 1, 4, demand, 1.0, 4.0)
        ast = maximum(
            add(parameter("S"), neg(state("x")), neg(state("p1"))),
            {"op": "const", "value": 0.0},
        )
        parameter_specs = (
            ParameterSpec("S", kind="integer", lower=0, upper=10, default=4),
        )
        accelerated = optimize_policy_constants(
            parameter_specs,
            SparseExactPolicyCostEvaluator(ast, reference, 40),
            config=OptimizationConfig(mode="fast"),
        )
        general = optimize_policy_constants(
            parameter_specs,
            SparseExactPolicyCostEvaluator(
                ast,
                reference,
                40,
                use_action_projection=False,
                use_stable_transition_matrix=False,
            ),
            config=OptimizationConfig(mode="fast"),
        )
        self.assertEqual(accelerated.status, "certified_discrete_global")
        self.assertEqual(accelerated.best_params, general.best_params)
        self.assertAlmostEqual(accelerated.best_cost, general.best_cost, places=10)

    def test_high_lead_time_fast_paths_match_general_engine_for_all_baselines(self) -> None:
        demand = DiscreteDemand.from_pmf({0: 0.35, 1: 0.4, 3: 0.25})
        for lead_time in (2, 3, 4):
            reference = solve_dp(lead_time, 1, 3, demand, 1.0, 4.0)
            for spec in baselines_for_lead_time(lead_time, 3):
                initial = {
                    parameter.name: parameter.initial_value()
                    for parameter in spec.parameter_specs
                }
                midpoint = {
                    parameter.name: (parameter.lower + parameter.upper) / 2.0
                    for parameter in spec.parameter_specs
                }
                for params in (initial, midpoint):
                    fast = SparseExactPolicyCostEvaluator(
                        spec.ast, reference, 35, spec.motifs, max_cycle_period=0
                    ).evaluate(params)
                    general = SparseExactPolicyCostEvaluator(
                        spec.ast,
                        reference,
                        35,
                        spec.motifs,
                        max_cycle_period=0,
                        use_action_projection=False,
                        use_stable_transition_matrix=False,
                    ).evaluate(params)
                    self.assertAlmostEqual(
                        fast.cost,
                        general.cost,
                        places=10,
                        msg=f"L={lead_time} {spec.name} {params}",
                    )

    def test_pruned_screening_steady_state_matches_full_horizon_propagation(self) -> None:
        demand = DiscreteDemand.from_pmf({0: 0.55, 1: 0.25, 4: 0.2})
        reference = solve_dp(1, 1, 10, demand, 1.0, 4.0)
        spec = next(
            item
            for item in baselines_for_lead_time(1, 10)
            if item.name == "weighted_tilted_pic"
        )
        params = {"S": 7.0, "K": 0.8, "C": 3.0, "alpha": 0.15, "w1": 0.65}
        accelerated = SparseExactPolicyCostEvaluator(
            spec.ast,
            reference,
            1000,
            spec.motifs,
            probability_prune_threshold=1.0e-8,
            max_cycle_period=1,
        ).evaluate(params)
        full = SparseExactPolicyCostEvaluator(
            spec.ast,
            reference,
            1000,
            spec.motifs,
            probability_prune_threshold=1.0e-8,
            max_cycle_period=0,
        ).evaluate(params)
        self.assertGreater(accelerated.metadata["cycle_detected_at"], 0)
        self.assertAlmostEqual(accelerated.cost, full.cost, places=8)

    def test_adaptive_aggregation_matches_sorted_sparse_for_all_baselines(self) -> None:
        demand = DiscreteDemand.from_pmf({0: 0.4, 1: 0.3, 2: 0.3})
        dp = solve_dp(2, 6, 2, demand, 1.0, 4.0)
        for spec in baselines_for_lead_time(2, 2):
            params = {
                parameter.name: parameter.initial_value()
                for parameter in spec.parameter_specs
            }
            legacy = SparseExactPolicyCostEvaluator(
                spec.ast,
                dp,
                6,
                spec.motifs,
                dense_switch_fraction=float("inf"),
                max_cycle_period=0,
            ).evaluate(params)
            adaptive = SparseExactPolicyCostEvaluator(
                spec.ast, dp, 6, spec.motifs, dense_switch_fraction=0.0
            ).evaluate(params)
            self.assertAlmostEqual(legacy.cost, adaptive.cost, places=10, msg=spec.name)

    def test_bounded_adaptive_csr_abort_falls_back_to_exact_sparse(self) -> None:
        demand = DiscreteDemand.from_pmf({0: 0.4, 1: 0.3, 2: 0.3})
        dp = solve_dp(2, 6, 2, demand, 1.0, 4.0)
        spec = next(
            item
            for item in baselines_for_lead_time(2, 2)
            if item.name == "base_stock"
        )
        params = {"S": 4.0}
        expected = SparseExactPolicyCostEvaluator(
            spec.ast,
            dp,
            12,
            spec.motifs,
            max_cycle_period=0,
            use_adaptive_reachable_csr=False,
        ).evaluate(params)
        accelerated = SparseExactPolicyCostEvaluator(
            spec.ast,
            dp,
            12,
            spec.motifs,
            max_cycle_period=0,
            adaptive_csr_warmup_periods=2,
        ).evaluate(params)
        self.assertAlmostEqual(accelerated.cost, expected.cost, places=11)
        self.assertTrue(accelerated.metadata["adaptive_csr_active"])
        self.assertEqual(accelerated.metadata["adaptive_csr_build_aborts"], 0)
        bounded = SparseExactPolicyCostEvaluator(
            spec.ast,
            dp,
            12,
            spec.motifs,
            max_cycle_period=0,
            adaptive_csr_warmup_periods=2,
            adaptive_csr_max_support_states=1,
            adaptive_csr_max_attempts=3,
        ).evaluate(params)
        self.assertTrue(bounded.completed)
        self.assertAlmostEqual(bounded.cost, expected.cost, places=11)
        self.assertEqual(bounded.metadata["adaptive_csr_build_attempts"], 1)
        self.assertEqual(bounded.metadata["adaptive_csr_build_aborts"], 1)
        self.assertEqual(
            bounded.metadata["adaptive_csr_last_abort_reason"], "support_limit"
        )
        self.assertFalse(bounded.metadata["adaptive_csr_active"])

    def test_incumbent_bound_runs_before_speculative_csr_build(self) -> None:
        demand = DiscreteDemand.from_pmf({0: 0.5, 2: 0.5})
        dp = solve_dp(2, 3, 2, demand, 1.0, 3.0)
        spec = next(
            item
            for item in baselines_for_lead_time(2, 2)
            if item.name == "base_stock"
        )
        evaluator = SparseExactPolicyCostEvaluator(
            spec.ast,
            dp,
            12,
            spec.motifs,
            max_cycle_period=0,
            adaptive_csr_warmup_periods=1,
        )
        stopped = evaluator.evaluate({"S": 4.0}, incumbent=0.0)
        self.assertTrue(stopped.early_stopped)
        self.assertEqual(stopped.metadata["stop_reason"], "incumbent_bound")
        self.assertEqual(stopped.metadata["adaptive_csr_build_attempts"], 0)

    def test_adaptive_csr_builder_honors_expired_deadline(self) -> None:
        demand = DiscreteDemand.from_pmf({0: 0.5, 1: 0.5})
        dp = solve_dp(2, 2, 2, demand, 1.0, 3.0)
        spec = next(
            item
            for item in baselines_for_lead_time(2, 2)
            if item.name == "base_stock"
        )
        evaluator = SparseExactPolicyCostEvaluator(spec.ast, dp, 8, spec.motifs)
        result, reason, peak_support, expansions = (
            evaluator._build_adaptive_reachable_csr(
                {"S": 3.0},
                np.asarray([evaluator._encode_state((0, 0, 0), 3)]),
                demand.values.astype(int),
                demand.probs.astype(float),
                3,
                None,
                demand.feature_dict(),
                {"h": 1.0, "b": 3.0},
                lambda *args, **kwargs: self.fail("expired build created a CSR"),
                max_support_states=100,
                deadline_at=time.perf_counter() - 1.0,
            )
        )
        self.assertIsNone(result)
        self.assertEqual(reason, "build_time_limit")
        self.assertEqual(peak_support, 1)
        self.assertEqual(expansions, 0)

    def test_certified_substochastic_screen_proves_rejection(self) -> None:
        demand = DiscreteDemand.from_pmf({0: 0.4, 1: 0.3, 2: 0.3})
        dp = solve_dp(2, 6, 2, demand, 1.0, 4.0)
        spec = next(
            item
            for item in baselines_for_lead_time(2, 2)
            if item.name == "base_stock"
        )
        params = {"S": 4.0}
        full = SparseExactPolicyCostEvaluator(
            spec.ast,
            dp,
            12,
            spec.motifs,
            max_cycle_period=0,
            use_adaptive_reachable_csr=False,
        ).evaluate(params)
        incumbent = 0.9 * full.cost
        screened = SparseExactPolicyCostEvaluator(
            spec.ast,
            dp,
            12,
            spec.motifs,
            max_cycle_period=0,
            adaptive_csr_warmup_periods=2,
            adaptive_csr_max_support_states=1,
            certified_screen_max_states=3,
            certified_screen_max_seconds=1.0,
        ).evaluate(params, incumbent=incumbent)
        self.assertTrue(screened.early_stopped)
        self.assertEqual(
            screened.metadata["stop_reason"], "certified_substochastic_bound"
        )
        self.assertTrue(screened.metadata["bound_valid"])
        self.assertGreater(screened.cost, incumbent)
        self.assertLessEqual(screened.cost, full.cost + 1.0e-12)

    def test_inconclusive_certified_screen_resumes_strict_exact(self) -> None:
        demand = DiscreteDemand.from_pmf({0: 0.4, 1: 0.3, 2: 0.3})
        dp = solve_dp(2, 6, 2, demand, 1.0, 4.0)
        spec = next(
            item
            for item in baselines_for_lead_time(2, 2)
            if item.name == "base_stock"
        )
        params = {"S": 4.0}
        full = SparseExactPolicyCostEvaluator(
            spec.ast,
            dp,
            12,
            spec.motifs,
            max_cycle_period=0,
            use_adaptive_reachable_csr=False,
        ).evaluate(params)
        screened = SparseExactPolicyCostEvaluator(
            spec.ast,
            dp,
            12,
            spec.motifs,
            max_cycle_period=0,
            adaptive_csr_warmup_periods=2,
            adaptive_csr_max_support_states=1,
            certified_screen_max_states=3,
            certified_screen_max_seconds=1.0,
        ).evaluate(params, incumbent=full.cost - 0.01)
        self.assertTrue(screened.early_stopped)
        self.assertEqual(screened.metadata["stop_reason"], "incumbent_bound")
        self.assertTrue(screened.metadata["certified_screen_attempted"])
        self.assertFalse(screened.metadata["certified_screen_proved"])
        self.assertLess(screened.metadata["certified_screen_retained_mass"], 1.0)
        self.assertGreater(
            screened.metadata["certified_screen_discarded_mass"], 0.0
        )
        self.assertGreaterEqual(
            screened.metadata["certified_screen_discarded_continuation_floor"],
            0.0,
        )
        self.assertLessEqual(
            screened.metadata["certified_screen_lower_bound"], full.cost
        )

    def test_continuation_floor_scales_with_substochastic_mass(self) -> None:
        demand = DiscreteDemand.from_pmf({0: 0.5, 2: 0.5})
        case = Case("substochastic_mass", 2, 1.0, 3.0, demand, "unit", 2)
        reference = rolling_reference(case)
        spec = next(
            item
            for item in baselines_for_lead_time(2, 2)
            if item.name == "base_stock"
        )
        evaluator = SparseExactPolicyCostEvaluator(spec.ast, reference, 10, spec.motifs)
        index = evaluator._encode_state((0, 0, 0), 3)
        bound = evaluator._dp_remaining_lower_bound(
            2,
            np.asarray([index], dtype=np.int64),
            np.asarray([0.25]),
            3,
            3,
        )
        self.assertAlmostEqual(
            bound, 0.25 * 8 * evaluator._minimum_period_cost, places=12
        )

    def test_fit_inventory_constants_rejects_mse_search_objective(self) -> None:
        demand = DiscreteDemand.from_pmf({0: 0.5, 1: 0.5})
        dp = solve_dp(1, 3, 2, demand, 1.0, 2.0)
        spec = baselines_for_lead_time(1, 2)[1]
        case = Case("fit-case", 1, 1.0, 2.0, demand, "test", 2)
        with self.assertRaisesRegex(ValueError, "only accepts exact policy cost"):
            fit_inventory_constants(
                spec.ast,
                case,
                3,
                spec.parameter_specs,
                reference=dp,
                motifs=spec.motifs,
                config=OptimizationConfig(mode="fast", total_evaluations=8, loss="raw_mse"),
            )

    def test_explicit_dp_cost_preserves_legacy_fit_normalization(self) -> None:
        demand = DiscreteDemand.from_pmf({0: 0.5, 1: 0.5})
        dp = solve_dp(1, 4, 2, demand, 1.0, 2.0)
        spec = baselines_for_lead_time(1, 2)[1]
        case = Case("fit-case", 1, 1.0, 2.0, demand, "test", 2)
        config = OptimizationConfig(mode="fast", total_evaluations=8, random_seed=17)
        legacy = fit_inventory_constants(
            spec.ast,
            case,
            4,
            spec.parameter_specs,
            reference=dp,
            motifs=spec.motifs,
            config=config,
        )
        explicit = fit_inventory_constants(
            spec.ast,
            case,
            4,
            spec.parameter_specs,
            reference=dp,
            motifs=spec.motifs,
            config=config,
            dp_optimal_cost=dp.value(0, (0, 0)),
        )
        self.assertEqual(legacy.params, explicit.params)
        self.assertAlmostEqual(legacy.policy_cost, explicit.policy_cost, places=12)
        self.assertAlmostEqual(legacy.normalized_policy_cost, explicit.normalized_policy_cost, places=12)
        self.assertAlmostEqual(legacy.objective_value, explicit.objective_value, places=12)

    def test_inventory_fit_accepts_complete_rolling_reference_bundle(self) -> None:
        demand = DiscreteDemand.from_pmf({0: 0.5, 1: 0.5})
        case = Case("rolling-fit-case", 1, 1.0, 2.0, demand, "test", 2)
        reference = rolling_reference(case)
        dp_cost, _seconds, samples = solve_optimal_cost_rolling(
            reference,
            4,
            sample_size=16,
            random_seed=17,
            backend="blocked_matrix",
        )
        spec = baselines_for_lead_time(1, 2)[1]
        evaluator = SparseExactPolicyCostEvaluator(
            spec.ast,
            reference,
            4,
            spec.motifs,
            max_cycle_period=0,
            use_stationary_continuation_bound=False,
        )
        result = fit_inventory_constants(
            spec.ast,
            case,
            4,
            spec.parameter_specs,
            samples=samples,
            metric_samples=samples,
            reference=reference,
            motifs=spec.motifs,
            config=OptimizationConfig(mode="fast", total_evaluations=8, random_seed=17),
            cost_evaluator=evaluator,
            dp_optimal_cost=dp_cost,
        )
        self.assertTrue(result.success)
        self.assertGreaterEqual(result.policy_cost, dp_cost)

    def test_inventory_fit_rejects_incomplete_rolling_reference_bundle(self) -> None:
        demand = DiscreteDemand.from_pmf({0: 0.5, 1: 0.5})
        case = Case("rolling-fit-case", 1, 1.0, 2.0, demand, "test", 2)
        reference = rolling_reference(case)
        spec = baselines_for_lead_time(1, 2)[1]
        with self.assertRaisesRegex(ValueError, "action samples"):
            fit_inventory_constants(
                spec.ast,
                case,
                4,
                spec.parameter_specs,
                reference=reference,
                motifs=spec.motifs,
                dp_optimal_cost=1.0,
            )

    def test_baseline_parameter_semantics_are_explicit(self) -> None:
        specs = {spec.name: spec for spec in baselines_for_lead_time(3, 10)}
        self.assertTrue(all(parameter.kind == "integer" for parameter in specs["capped_base_stock"].parameter_specs))
        self.assertEqual([parameter.kind for parameter in specs["tilted_pic"].parameter_specs], ["integer", "continuous", "integer", "continuous"])
        self.assertTrue(all(parameter.kind == "continuous" for parameter in specs["weighted_tilted_pic"].parameter_specs[4:]))


if __name__ == "__main__":
    unittest.main()
