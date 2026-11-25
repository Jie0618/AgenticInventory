from __future__ import annotations

import itertools
import unittest

import numpy as np

from inventory.common.distributions import DiscreteDemand
from optimization.policies.baselines.lead_time import baselines_for_lead_time
from evaluation.backends.lead_time.dp import solve_dp
from inventory.policy.ast import add, const, mul, parameter, state
from optimization.models.types import EvaluationResult, MonteCarloProposalConfig, OptimizationConfig
from optimization.backends.lead_time.exact_evaluator import SparseExactPolicyCostEvaluator
from optimization.backends.lead_time.mc_evaluator import (
    DemandTrajectoryBank,
    MonteCarloPolicyCostEvaluator,
)
from optimization.solvers.pipeline import (
    _make_fast_evaluator,
    _reported_mc_paths,
    _resolved_mc_config,
)
from optimization.solvers.proposal_evaluator import AdaptiveProposalEvaluator
from optimization.support.utils import _with_default_action_cap


class MonteCarloProposalTests(unittest.TestCase):
    def test_composed_report_uses_actual_full_proposal_paths(self) -> None:
        class ComposedLike:
            paths = 1024

        proposal, rerank = _reported_mc_paths(
            ComposedLike(), OptimizationConfig(mode="certify"), "certify"
        )
        self.assertEqual((proposal, rerank), (1024, 1024))

    def test_bank_is_reproducible(self) -> None:
        demand = DiscreteDemand.from_pmf({0: 0.25, 2: 0.75})
        first = DemandTrajectoryBank.create(demand, horizon=6, num_paths=32, seed=19)
        second = DemandTrajectoryBank.create(demand, horizon=6, num_paths=32, seed=19)
        self.assertTrue(np.array_equal(first.demand_indices, second.demand_indices))
        self.assertTrue(np.array_equal(first.demand_values, second.demand_values))

    def test_exhaustive_trajectory_mean_matches_exact(self) -> None:
        demand = DiscreteDemand.from_pmf({0: 0.5, 1: 0.5})
        reference = solve_dp(1, 3, 3, demand, 1.0, 4.0)
        spec = next(item for item in baselines_for_lead_time(1, 3) if item.name == "constant_order")
        params = {item.name: 1.0 for item in spec.parameter_specs}
        exact = SparseExactPolicyCostEvaluator(spec.ast, reference, 3, spec.motifs, max_cycle_period=0)
        all_paths = np.asarray(list(itertools.product(range(2), repeat=3)), dtype=np.int32)
        bank = DemandTrajectoryBank(
            all_paths,
            demand.values[all_paths],
            len(all_paths),
            3,
            0,
        )
        mc = MonteCarloPolicyCostEvaluator(
            spec.ast,
            reference,
            bank,
            spec.motifs,
            action_evaluator=exact,
            horizon=3,
        )
        result = mc.evaluate(params)
        self.assertAlmostEqual(result.cost, exact.evaluate(params).cost, places=12)

    def test_guard_falls_back_to_mc_without_partial_score(self) -> None:
        demand = DiscreteDemand.from_pmf({0: 0.3, 1: 0.4, 3: 0.3})
        reference = solve_dp(2, 3, 5, demand, 1.0, 4.0)
        spec = next(item for item in baselines_for_lead_time(2, 5) if item.name == "vector_base_stock")
        exact = SparseExactPolicyCostEvaluator(spec.ast, reference, 20, spec.motifs, max_cycle_period=0)
        stationary = exact.stationary_copy(
            proposal_complexity_guard={
                "enabled": True,
                "project_period_limit": True,
                "max_periods": 1,
                "max_active_states": 1,
                "max_generated_branches": 2,
                "max_seconds": 10.0,
                "growth_guard_enabled": False,
            }
        )
        bank = DemandTrajectoryBank.create(demand, 20, 32, 4)
        mc = MonteCarloPolicyCostEvaluator(
            spec.ast,
            reference,
            bank,
            spec.motifs,
            action_evaluator=stationary,
            horizon=20,
        )
        proposal = AdaptiveProposalEvaluator(stationary, mc)
        params = {item.name: item.initial_value() for item in spec.parameter_specs}
        result = proposal.evaluate(params)
        self.assertTrue(result.completed)
        self.assertEqual(result.metadata["fidelity"], "mc_fixed")
        self.assertTrue(result.metadata["mc_fallback"])
        self.assertFalse(result.metadata.get("partial_cost_is_bound", False))

    def test_period_guard_continues_a_cheap_chain_past_calibration_periods(self) -> None:
        demand = DiscreteDemand.from_pmf({0: 0.3, 1: 0.4, 3: 0.3})
        reference = solve_dp(1, 3, 5, demand, 1.0, 4.0)
        spec = next(
            item
            for item in baselines_for_lead_time(1, 5)
            if item.name == "constant_order"
        )
        exact = SparseExactPolicyCostEvaluator(
            spec.ast, reference, 20, spec.motifs, max_cycle_period=0
        )
        stationary = exact.stationary_copy(
            proposal_complexity_guard={
                "enabled": True,
                "project_period_limit": True,
                "max_periods": 1,
                "max_active_states": 100_000,
                "max_generated_branches": 100_000,
                "max_seconds": 10.0,
                "growth_guard_enabled": False,
            }
        )
        params = {item.name: item.initial_value() for item in spec.parameter_specs}
        result = stationary.evaluate(params)
        self.assertTrue(result.completed)
        self.assertNotEqual(
            result.metadata.get("proposal_guard_abort_reason"), "period_limit"
        )

    def test_period_guard_aborts_when_projected_work_exceeds_limit(self) -> None:
        demand = DiscreteDemand.from_pmf({0: 0.3, 1: 0.4, 3: 0.3})
        reference = solve_dp(1, 3, 5, demand, 1.0, 4.0)
        spec = next(
            item
            for item in baselines_for_lead_time(1, 5)
            if item.name == "constant_order"
        )
        exact = SparseExactPolicyCostEvaluator(
            spec.ast, reference, 20, spec.motifs, max_cycle_period=0
        )
        stationary = exact.stationary_copy(
            proposal_complexity_guard={
                "enabled": True,
                "project_period_limit": True,
                "max_periods": 1,
                "max_active_states": 100_000,
                "max_generated_branches": 100,
                "max_seconds": 10.0,
                "growth_guard_enabled": False,
            }
        )
        params = {item.name: item.initial_value() for item in spec.parameter_specs}
        result = stationary.evaluate(params)
        self.assertFalse(result.completed)
        self.assertEqual(
            result.metadata.get("proposal_guard_abort_reason"),
            "projected_transition_work_limit",
        )

    def test_stage1_defaults_are_fixed_by_mode(self) -> None:
        self.assertEqual(MonteCarloProposalConfig().paths, None)
        self.assertEqual(MonteCarloProposalConfig().warmup_candidates, 64)
        self.assertEqual(MonteCarloProposalConfig().abort_rate_mc_first, 0.50)
        self.assertEqual(MonteCarloProposalConfig().cost_switch_to_mc_ratio, 1.15)
        self.assertEqual(MonteCarloProposalConfig().cost_switch_to_probe_ratio, 0.85)
        self.assertEqual(MonteCarloProposalConfig().cost_switch_to_probe_confirmations, 3)
        self.assertEqual(MonteCarloProposalConfig().sentinel_candidates, 8)
        expected = {
            "fast": (256, 64, 5_000, 200_000, 0.2),
            "standard": (512, 96, 10_000, 500_000, 0.5),
            "certify": (1024, 128, 20_000, 1_000_000, 1.0),
        }
        for mode, values in expected.items():
            resolved = _resolved_mc_config(OptimizationConfig(mode=mode), mode)
            self.assertEqual(resolved.paths, values[0])
            self.assertEqual(resolved.stationary_probe_max_periods, values[1])
            self.assertEqual(resolved.stationary_probe_max_active_states, values[2])
            self.assertEqual(resolved.stationary_probe_max_generated_branches, values[3])
            self.assertEqual(resolved.stationary_probe_max_seconds, values[4])

    def test_mc_config_validates_and_normalizes_public_values(self) -> None:
        invalid_values = (
            {"paths": 1.5},
            {"seed": 2.7},
            {"warmup_candidates": True},
            {"enabled": "false"},
            {"routing_mode": None},
            {"abort_rate_no_mc": float("nan")},
            {"cost_switch_to_mc_ratio": float("inf")},
            {"stationary_probe_max_seconds": float("nan")},
            {"stationary_probe_max_seconds": True},
            {"stationary_probe_growth_ratio": "2.0"},
        )
        for values in invalid_values:
            with self.subTest(values=values), self.assertRaises(ValueError):
                MonteCarloProposalConfig(**values)
        config = MonteCarloProposalConfig(paths=8.0, seed=2.0)
        self.assertEqual(config.paths, 8)
        self.assertEqual(config.seed, 2)

    def test_mc_switch_survives_default_action_cap(self) -> None:
        config = OptimizationConfig(
            mode="fast",
            mc_proposal=MonteCarloProposalConfig(enabled=False),
        )
        resolved = _with_default_action_cap(config, 7)
        self.assertEqual(resolved.action_cap, 7)
        self.assertIsNotNone(resolved.mc_proposal)
        self.assertFalse(resolved.mc_proposal.enabled)

    def test_fast_low_dimensional_r7_uses_candidate_local_fallback(self) -> None:
        demand = DiscreteDemand.from_pmf({0: 0.5, 1: 0.5})
        reference = solve_dp(2, 5, 5, demand, 1.0, 4.0)
        spec = next(
            item for item in baselines_for_lead_time(2, 5) if item.name == "tilted_cbs"
        )
        exact = SparseExactPolicyCostEvaluator(spec.ast, reference, 5, spec.motifs, max_cycle_period=0)
        low_dim = _make_fast_evaluator(
            exact,
            OptimizationConfig(mode="fast"),
            "fast",
            "R7_MIXED_JOINT",
            3,
        )
        reference_l4 = solve_dp(4, 5, 5, demand, 1.0, 4.0)
        spec_l4 = next(
            item for item in baselines_for_lead_time(4, 5) if item.name == "tilted_cbs"
        )
        exact_l4 = SparseExactPolicyCostEvaluator(
            spec_l4.ast, reference_l4, 5, spec_l4.motifs, max_cycle_period=0
        )
        low_dim_l4 = _make_fast_evaluator(
            exact_l4,
            OptimizationConfig(mode="fast"),
            "fast",
            "R7_MIXED_JOINT",
            3,
        )
        high_dim = _make_fast_evaluator(
            exact,
            OptimizationConfig(mode="fast"),
            "fast",
            "R7_MIXED_JOINT",
            6,
        )
        standard = _make_fast_evaluator(
            exact,
            OptimizationConfig(mode="standard"),
            "standard",
            "R7_MIXED_JOINT",
            6,
        )
        certify = _make_fast_evaluator(
            exact,
            OptimizationConfig(mode="certify"),
            "certify",
            "R7_MIXED_JOINT",
            6,
        )
        self.assertEqual(low_dim.routing_mode, "probe_fallback")
        self.assertEqual(low_dim_l4.routing_mode, "auto")
        self.assertEqual(high_dim.routing_mode, "auto")
        self.assertEqual(standard._ensure_mc().bank.num_paths, 512)  # noqa: SLF001
        self.assertEqual(standard._ensure_mc().batch_size, 32)  # noqa: SLF001
        self.assertIsNone(standard._ensure_rerank_mc())  # noqa: SLF001
        self.assertEqual(certify._ensure_mc().bank.num_paths, 512)  # noqa: SLF001
        self.assertEqual(certify._ensure_mc().batch_size, 32)  # noqa: SLF001
        self.assertEqual(certify._ensure_rerank_mc().bank.num_paths, 1024)  # noqa: SLF001
        self.assertEqual(certify._ensure_rerank_mc().batch_size, 16)  # noqa: SLF001

    def test_candidate_batch_matches_scalar_mc(self) -> None:
        demand = DiscreteDemand.from_pmf({0: 0.5, 1: 0.5})
        reference = solve_dp(1, 4, 4, demand, 1.0, 4.0)
        spec = next(item for item in baselines_for_lead_time(1, 4) if item.name == "constant_order")
        exact = SparseExactPolicyCostEvaluator(spec.ast, reference, 4, spec.motifs, max_cycle_period=0)
        bank = DemandTrajectoryBank.create(demand, horizon=4, num_paths=24, seed=13)
        mc = MonteCarloPolicyCostEvaluator(
            spec.ast,
            reference,
            bank,
            spec.motifs,
            action_evaluator=exact,
            horizon=4,
            batch_size=8,
        )
        params = [{spec.parameter_specs[0].name: 0.0}, {spec.parameter_specs[0].name: 2.0}]
        scalar = np.vstack([mc.simulate_path_costs(row) for row in params])
        batched = mc.simulate_path_costs_many(params)
        np.testing.assert_array_equal(batched, scalar)
        mc.clear_cache()
        scalar_results = [mc.evaluate(row) for row in params]
        mc.clear_cache()
        batch_results = mc.evaluate_many(params)
        np.testing.assert_allclose(
            [result.cost for result in batch_results],
            [result.cost for result in scalar_results],
            rtol=0.0,
            atol=1.0e-12,
        )
        self.assertEqual(batch_results[0].metadata["engine"], "vectorized_mc_batch")

    def test_prepared_candidate_batch_matches_scalar_for_pipeline_formula(self) -> None:
        demand = DiscreteDemand.from_pmf({0: 0.4, 1: 0.4, 3: 0.2})
        reference = solve_dp(2, 7, 5, demand, 1.0, 4.0)
        spec = next(
            item
            for item in baselines_for_lead_time(2, 5)
            if item.name == "weighted_tilted_pic"
        )
        exact = SparseExactPolicyCostEvaluator(
            spec.ast,
            reference,
            7,
            spec.motifs,
            max_cycle_period=0,
        )
        bank = DemandTrajectoryBank.create(demand, horizon=7, num_paths=32, seed=23)
        mc = MonteCarloPolicyCostEvaluator(
            spec.ast,
            reference,
            bank,
            spec.motifs,
            action_evaluator=exact,
            horizon=7,
            batch_size=8,
        )
        params = [
            {"S": 4.0, "C": 1.0, "K": 0.8, "alpha": 0.2, "w1": 0.3, "w2": 0.9},
            {"S": 7.0, "C": 3.0, "K": 1.2, "alpha": 0.6, "w1": 1.0, "w2": 0.1},
            {"S": 9.0, "C": 2.0, "K": 1.0, "alpha": 0.0, "w1": 0.5, "w2": 0.5},
        ]
        scalar = np.vstack([mc.simulate_path_costs(row) for row in params])
        batched = mc.simulate_path_costs_many(params)
        np.testing.assert_array_equal(batched, scalar)

    def test_prepared_candidate_batch_does_not_cache_remaining_horizon(self) -> None:
        demand = DiscreteDemand.from_pmf({0: 0.5, 1: 0.5})
        reference = solve_dp(1, 6, 5, demand, 1.0, 4.0)
        ast = add(parameter("offset"), mul(const(0.5), state("remaining_horizon")))
        bank = DemandTrajectoryBank.create(demand, horizon=6, num_paths=16, seed=29)
        mc = MonteCarloPolicyCostEvaluator(
            ast,
            reference,
            bank,
            horizon=6,
            batch_size=8,
        )
        params = [{"offset": 0.0}, {"offset": 1.0}, {"offset": 2.0}]
        scalar = np.vstack([mc.simulate_path_costs(row) for row in params])
        batched = mc.simulate_path_costs_many(params)
        np.testing.assert_array_equal(batched, scalar)

    def test_prepared_candidate_batch_reuses_shared_state_subexpression_safely(self) -> None:
        demand = DiscreteDemand.from_pmf({0: 0.5, 2: 0.5})
        reference = solve_dp(2, 8, 6, demand, 1.0, 4.0)
        shared_gap = add(parameter("S"), mul(const(-1.0), state("x")))
        ast = add(shared_gap, mul(parameter("alpha"), shared_gap))
        bank = DemandTrajectoryBank.create(demand, horizon=8, num_paths=24, seed=31)
        mc = MonteCarloPolicyCostEvaluator(ast, reference, bank, horizon=8, batch_size=8)
        params = [
            {"S": 2.0, "alpha": 0.0},
            {"S": 5.0, "alpha": 0.5},
            {"S": 8.0, "alpha": 1.0},
        ]
        scalar = np.vstack([mc.simulate_path_costs(row) for row in params])
        batched = mc.simulate_path_costs_many(params)
        np.testing.assert_array_equal(batched, scalar)
        self.assertGreater(mc._last_batch_diagnostics["mc_batch_ast_cache_hits"], 0)  # noqa: SLF001

    def test_auto_route_lazily_creates_mc_only_on_guard_abort(self) -> None:
        class Probe:
            def evaluate(self, params, incumbent=None):
                del params, incumbent
                return EvaluationResult(cost=1.0, completed=True, metadata={})

        class FakeMC:
            def evaluate(self, params, incumbent=None):
                del params, incumbent
                return EvaluationResult(cost=2.0, completed=True, metadata={"fidelity": "mc_fixed"})

            def set_debug_context(self, context):
                del context

            def set_exact_anchor(self, params, cost):
                del params, cost

        created = []
        proposal = AdaptiveProposalEvaluator(
            Probe(),
            mc_factory=lambda: (created.append(True) or FakeMC()),
            routing_mode="auto",
            warmup_candidates=2,
        )
        first = proposal.evaluate({"a": 1.0})
        self.assertEqual(first.cost, 1.0)
        self.assertEqual(created, [])
        self.assertEqual(proposal.route_state, "warmup_probe")

        class GuardedProbe(Probe):
            def evaluate(self, params, incumbent=None):
                del params, incumbent
                return EvaluationResult(
                    cost=float("inf"),
                    completed=False,
                    metadata={
                        "proposal_guard_triggered": True,
                        "proposal_guard_abort_reason": "active_state_limit",
                    },
                )

        proposal.stationary_evaluator = GuardedProbe()
        second = proposal.evaluate({"a": 2.0})
        self.assertEqual(second.cost, 2.0)
        self.assertEqual(len(created), 1)
        self.assertEqual(proposal.fallback_count, 1)

        # One fallback is not enough to estimate MC runtime under the default
        # minimum observation count, so the route remains probe-first.
        proposal.stationary_evaluator = Probe()
        third = proposal.evaluate({"a": 3.0})
        self.assertEqual(third.cost, 1.0)
        self.assertEqual(len(created), 1)
        self.assertEqual(proposal.route_state, "probe_then_mc_fallback")
        self.assertEqual(
            proposal.route_switch_reason,
            "phase=default:warmup:hold:insufficient_mc_observations",
        )

    def test_auto_route_switches_to_mc_when_probe_first_cost_is_higher(self) -> None:
        class Probe:
            def __init__(self, guarded: bool = False) -> None:
                self.guarded = guarded
                self.calls = 0

            def evaluate(self, params, incumbent=None):
                del params, incumbent
                self.calls += 1
                if self.guarded:
                    return EvaluationResult(
                        cost=float("inf"),
                        completed=False,
                        metadata={
                            "proposal_guard_triggered": True,
                            "proposal_guard_abort_reason": "active_state_limit",
                            "probe_seconds": 2.0,
                        },
                    )
                return EvaluationResult(cost=1.0, completed=True, metadata={"probe_seconds": 0.25})

        class FakeMC:
            def evaluate(self, params, incumbent=None):
                del params, incumbent
                return EvaluationResult(
                    cost=2.0,
                    completed=True,
                    metadata={"fidelity": "mc_fixed", "mc_seconds": 1.0},
                )

            def evaluate_many(self, params_batch, incumbent=None):
                return [self.evaluate(params, incumbent=incumbent) for params in params_batch]

            def set_debug_context(self, context):
                del context

            def set_exact_anchor(self, params, cost):
                del params, cost

        created = []
        guarded_probe = Probe(guarded=True)
        proposal = AdaptiveProposalEvaluator(
            guarded_probe,
            mc_factory=lambda: (created.append(True) or FakeMC()),
            routing_mode="auto",
            warmup_candidates=4,
            sentinel_candidates=1,
            min_mc_observations=1,
        )
        for value in range(3):
            result = proposal.evaluate({"a": float(value)})
            self.assertEqual(result.cost, 2.0)
        guarded_probe.guarded = False
        fourth = proposal.evaluate({"a": 3.0})
        self.assertEqual(fourth.cost, 1.0)
        self.assertEqual(proposal.probe_count, 4)
        self.assertEqual(proposal.probe_abort_count, 3)
        self.assertEqual(proposal.route_state, "mc_first")
        self.assertIn(":switch_to_mc:", proposal.route_switch_reason)

        # MC-first keeps one sentinel probe per window. One favorable window is
        # not enough to switch back because sentinel samples are noisy.
        fifth = proposal.evaluate({"a": 4.0})
        self.assertEqual(fifth.cost, 1.0)
        self.assertEqual(proposal.probe_count, 5)
        self.assertEqual(len(created), 1)
        self.assertEqual(proposal.evaluate({"a": 5.0}).cost, 2.0)
        self.assertEqual(proposal.evaluate({"a": 6.0}).cost, 2.0)
        self.assertEqual(proposal.evaluate({"a": 7.0}).cost, 2.0)
        self.assertEqual(proposal.route_state, "mc_first")

        # A second favorable window is still insufficient.
        self.assertEqual(proposal.evaluate({"a": 8.0}).cost, 1.0)
        self.assertEqual(proposal.evaluate({"a": 9.0}).cost, 2.0)
        self.assertEqual(proposal.evaluate({"a": 10.0}).cost, 2.0)
        self.assertEqual(proposal.evaluate({"a": 11.0}).cost, 2.0)
        self.assertEqual(proposal.route_state, "mc_first")

        # The third consecutive favorable sentinel window confirms the return.
        self.assertEqual(proposal.evaluate({"a": 12.0}).cost, 1.0)
        self.assertEqual(proposal.evaluate({"a": 13.0}).cost, 2.0)
        self.assertEqual(proposal.evaluate({"a": 14.0}).cost, 2.0)
        self.assertEqual(proposal.evaluate({"a": 15.0}).cost, 2.0)
        self.assertEqual(proposal.route_state, "probe_then_mc_fallback")
        self.assertIn(":switch_to_probe:", proposal.route_switch_reason)

    def test_auto_route_rechecks_cost_in_each_probe_window(self) -> None:
        class Probe:
            def __init__(self) -> None:
                self.calls = 0

            def evaluate(self, params, incumbent=None):
                del params, incumbent
                self.calls += 1
                guarded = self.calls >= 3
                if guarded:
                    return EvaluationResult(
                        cost=float("inf"),
                        completed=False,
                        metadata={
                            "proposal_guard_triggered": True,
                            "proposal_guard_abort_reason": "active_state_limit",
                            "probe_seconds": 2.0,
                        },
                    )
                return EvaluationResult(cost=1.0, completed=True, metadata={"probe_seconds": 0.25})

        class FakeMC:
            def evaluate(self, params, incumbent=None):
                del params, incumbent
                return EvaluationResult(
                    cost=2.0,
                    completed=True,
                    metadata={"fidelity": "mc_fixed", "mc_seconds": 1.0},
                )

            def evaluate_many(self, params_batch, incumbent=None):
                return [self.evaluate(params, incumbent=incumbent) for params in params_batch]

            def set_debug_context(self, context):
                del context

            def set_exact_anchor(self, params, cost):
                del params, cost

        created = []
        proposal = AdaptiveProposalEvaluator(
            Probe(),
            mc_factory=lambda: (created.append(True) or FakeMC()),
            routing_mode="auto",
            warmup_candidates=2,
            sentinel_candidates=1,
            min_mc_observations=1,
        )

        # The first window has no aborts, so auto routing remains probe-first.
        self.assertEqual(proposal.evaluate({"a": 0.0}).cost, 1.0)
        self.assertEqual(proposal.evaluate({"a": 1.0}).cost, 1.0)
        self.assertEqual(proposal.route_state, "probe_then_mc_fallback")
        self.assertIn(":hold:insufficient_mc_observations", proposal.route_switch_reason)

        # The second window observes expensive probes and cheaper MC fallbacks,
        # then selects MC-first for the following window.
        self.assertEqual(proposal.evaluate({"a": 2.0}).cost, 2.0)
        self.assertEqual(proposal.evaluate({"a": 3.0}).cost, 2.0)
        self.assertEqual(proposal.route_state, "mc_first")
        self.assertEqual([row["decision"] for row in proposal.route_window_history], ["hold", "switch_to_mc"])
        self.assertEqual(proposal.evaluate({"a": 4.0}).cost, 2.0)
        self.assertEqual(proposal.probe_count, 5)
        self.assertEqual(len(created), 1)

    def test_certify_style_sparse_sentinels_reduce_monitoring_tax(self) -> None:
        class ExpensiveProbe:
            def evaluate(self, params, incumbent=None):
                del params, incumbent
                return EvaluationResult(
                    cost=float("inf"),
                    completed=False,
                    metadata={
                        "proposal_guard_triggered": True,
                        "proposal_guard_abort_reason": "transition_work_limit",
                        "probe_seconds": 2.0,
                    },
                )

        class CheapMC:
            def evaluate(self, params, incumbent=None):
                del params, incumbent
                return EvaluationResult(
                    cost=2.0,
                    completed=True,
                    metadata={"fidelity": "mc_fixed", "mc_seconds": 0.5},
                )

            def evaluate_many(self, params_batch, incumbent=None):
                return [self.evaluate(params, incumbent=incumbent) for params in params_batch]

            def set_debug_context(self, context):
                del context

            def set_exact_anchor(self, params, cost):
                del params, cost

        proposal = AdaptiveProposalEvaluator(
            ExpensiveProbe(),
            mc_factory=CheapMC,
            routing_mode="auto",
            warmup_candidates=4,
            sentinel_candidates=2,
            adaptive_sparse_sentinels=True,
            min_mc_observations=1,
        )
        proposal.evaluate_many([{"a": float(i)} for i in range(4)])
        first = proposal.route_window_history[-1]
        self.assertEqual(first["decision"], "switch_to_mc")
        self.assertEqual(first["sentinel_candidates_before"], 2)
        self.assertEqual(first["sentinel_candidates_after"], 1)

        proposal.evaluate_many([{"a": float(i)} for i in range(4, 8)])
        self.assertEqual(proposal.probe_count, 5)
        self.assertEqual(proposal.route_window_history[-1]["probe_count"], 1)


if __name__ == "__main__":
    unittest.main()
