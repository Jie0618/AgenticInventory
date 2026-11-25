from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from evaluation import select_evaluation_backend, solve_inventory_dp
from evaluation.backends.lead_time.dp import DPResult, solve_dp
from evaluation.dp import ComposedDPResult
from inventory.common.distributions import DiscreteDemand
from inventory.core import (
    DeterministicBacklog,
    DeterministicLeadTime,
    FixedShelfLife,
    GeometricBacklog,
    InventoryCase,
    InventoryModelConfig,
    KnownETALeadTime,
    LostSales,
    NoBacklog,
    TransitionCosts,
)
from inventory.policy.ast import parameter
from inventory.policy.ast import state
from optimization import OptimizationConfig, ParameterSpec, fit_inventory_policy
from optimization import InventorySearchContext


DEMAND = DiscreteDemand.from_pmf({0: 0.5, 1: 0.5}, name="bernoulli")


def _case(
    name: str,
    config: InventoryModelConfig,
    costs: TransitionCosts | None = None,
) -> InventoryCase:
    return InventoryCase(
        name,
        config,
        TransitionCosts(holding=1.0, shortage=4.0) if costs is None else costs,
        DEMAND,
        2,
    )


class EvaluationBackendRoutingTests(unittest.TestCase):
    def test_deterministic_lead_time_only_uses_established_array_dp(self) -> None:
        case = _case(
            "lt",
            InventoryModelConfig(
                lead_time=DeterministicLeadTime(2),
                backlog=LostSales(),
                capacity=2,
            ),
        )
        decision = select_evaluation_backend(case)
        self.assertTrue(decision.optimized)
        result = solve_inventory_dp(case, 3)
        direct = solve_dp(2, 3, 2, DEMAND, 1.0, 4.0)
        self.assertIsInstance(result, DPResult)
        self.assertEqual(result.backend, "lead_time_array")
        self.assertEqual(result.metadata["backend"], "lead_time_array")
        np.testing.assert_allclose(result.value_table, direct.value_table)
        np.testing.assert_array_equal(result.policy_table, direct.policy_table)

    def test_every_incompatible_semantic_uses_composed_backend(self) -> None:
        cases = (
            _case(
                "immediate-replenishment",
                InventoryModelConfig(
                    lead_time=DeterministicLeadTime(0),
                    backlog=LostSales(),
                    capacity=2,
                ),
            ),
            _case(
                "perishable",
                InventoryModelConfig(
                    perishability=FixedShelfLife(2),
                    backlog=LostSales(),
                    capacity=2,
                ),
            ),
            _case(
                "backlog",
                InventoryModelConfig(backlog=DeterministicBacklog(2), capacity=2),
            ),
            _case(
                "geometric",
                InventoryModelConfig(backlog=GeometricBacklog(0.8), capacity=2),
            ),
            _case(
                "stochastic-lead-time",
                InventoryModelConfig(
                    lead_time=KnownETALeadTime((1, 2), (0.5, 0.5)), capacity=2
                ),
            ),
            _case(
                "unprojected",
                InventoryModelConfig(lead_time=DeterministicLeadTime(1)),
            ),
            _case(
                "ordering-cost",
                InventoryModelConfig(
                    lead_time=DeterministicLeadTime(1), capacity=2
                ),
                TransitionCosts(holding=1.0, shortage=4.0, ordering=0.5),
            ),
        )
        for case in cases:
            with self.subTest(case=case.name):
                decision = select_evaluation_backend(case)
                self.assertFalse(decision.optimized)
                self.assertTrue(decision.reasons)
                if case.config.capacity is not None:
                    result = solve_inventory_dp(case, 2, max_states=1_000)
                    self.assertIsInstance(result, ComposedDPResult)
                    self.assertEqual(
                        result.metadata["backend"], "composed_finite_mdp"
                    )

    def test_disabled_and_lost_sales_backlog_share_the_hot_path(self) -> None:
        for backlog in (NoBacklog(), LostSales()):
            with self.subTest(backlog=backlog.mode.value):
                case = _case(
                    "lt-backlog-alias",
                    InventoryModelConfig(
                        lead_time=DeterministicLeadTime(1),
                        backlog=backlog,
                        capacity=2,
                    ),
                )
                self.assertTrue(select_evaluation_backend(case).optimized)

    def test_inactive_component_costs_do_not_disable_the_hot_path(self) -> None:
        case = _case(
            "lt-inactive-costs",
            InventoryModelConfig(
                lead_time=DeterministicLeadTime(1),
                backlog=LostSales(),
                capacity=2,
            ),
            TransitionCosts(
                holding=1.0,
                shortage=4.0,
                expiry=7.0,
                backlog=8.0,
                abandonment=9.0,
            ),
        )
        decision = select_evaluation_backend(case)
        self.assertTrue(decision.optimized)
        routed = solve_inventory_dp(case, 3)
        direct = solve_dp(1, 3, 2, DEMAND, 1.0, 4.0)
        np.testing.assert_allclose(routed.value_table, direct.value_table)
        np.testing.assert_array_equal(routed.policy_table, direct.policy_table)


class OptimizationBackendRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ast = parameter("q")
        self.specs = (
            ParameterSpec("q", kind="integer", lower=0, upper=2, default=1),
        )
        self.config = OptimizationConfig(mode="fast", total_evaluations=3)

    def test_lead_time_fit_keeps_established_optimizer(self) -> None:
        case = _case(
            "lt-fit",
            InventoryModelConfig(
                lead_time=DeterministicLeadTime(1),
                backlog=LostSales(),
                capacity=2,
            ),
        )
        result = fit_inventory_policy(
            self.ast, case, 2, self.specs, config=self.config
        )
        self.assertTrue(result.success)
        self.assertEqual(
            result.optimizer_metadata["evaluation_backend"], "lead_time_array"
        )

    def test_perishable_backlog_fit_uses_shared_composed_controller(self) -> None:
        case = _case(
            "composed-fit",
            InventoryModelConfig(
                perishability=FixedShelfLife(2),
                backlog=DeterministicBacklog(1),
                capacity=2,
            ),
            TransitionCosts(
                holding=1.0,
                shortage=4.0,
                expiry=0.5,
                backlog=0.25,
                abandonment=2.0,
            ),
        )
        result = fit_inventory_policy(
            self.ast, case, 2, self.specs, config=self.config
        )
        self.assertTrue(result.success)
        self.assertEqual(
            result.optimizer_metadata["evaluation_backend"],
            "composed_finite_mdp",
        )
        self.assertEqual(
            result.optimizer_metadata["exact_evaluator"],
            "composed_sparse_exact",
        )
        self.assertEqual(result.status, "certified_discrete_global")

    def test_sr_context_inherits_the_same_automatic_switch(self) -> None:
        lead_time_case = _case(
            "sr-lt",
            InventoryModelConfig(
                lead_time=DeterministicLeadTime(1),
                backlog=LostSales(),
                capacity=2,
            ),
        )
        composed_case = _case(
            "sr-composed",
            InventoryModelConfig(
                perishability=FixedShelfLife(2),
                backlog=DeterministicBacklog(1),
                capacity=2,
            ),
        )
        optimized = InventorySearchContext.build(lead_time_case, 2)
        general = InventorySearchContext.build(composed_case, 2)
        self.assertTrue(optimized.backend.optimized)
        self.assertIsInstance(optimized.reference, DPResult)
        self.assertFalse(general.backend.optimized)
        self.assertIsInstance(general.reference, ComposedDPResult)
        self.assertIn("oldest_backlog", general.motifs)
        result = general.fit(
            self.ast,
            parameter_specs=self.specs,
            config=self.config,
        )
        self.assertEqual(
            result.optimizer_metadata["evaluation_backend"],
            "composed_finite_mdp",
        )

    def test_sr_context_reuses_its_precomputed_reference(self) -> None:
        case = _case(
            "sr-reference-reuse",
            InventoryModelConfig(
                perishability=FixedShelfLife(2),
                backlog=LostSales(),
                capacity=2,
            ),
        )
        context = InventorySearchContext.build(case, 2)
        with patch(
            "optimization.solvers.api.solve_inventory_dp",
            side_effect=AssertionError("reference DP was recomputed"),
        ):
            result = context.fit(
                self.ast,
                parameter_specs=self.specs,
                config=self.config,
            )
        self.assertTrue(result.success)

    def test_sr_context_rejects_unsigned_composed_reference(self) -> None:
        case = _case(
            "sr-unsigned-reference",
            InventoryModelConfig(
                perishability=FixedShelfLife(2),
                backlog=LostSales(),
                capacity=2,
            ),
        )
        reference = solve_inventory_dp(case, 2)
        self.assertIsInstance(reference, ComposedDPResult)
        unsigned = replace(reference, metadata={})
        with self.assertRaisesRegex(ValueError, "unsigned composed reference"):
            InventorySearchContext.from_reference(case, 2, unsigned)

    def test_optimized_route_honors_nonzero_canonical_initial_state(self) -> None:
        case = InventoryCase(
            "nonzero-initial-state",
            InventoryModelConfig(
                lead_time=DeterministicLeadTime(1),
                backlog=LostSales(),
                capacity=2,
            ),
            TransitionCosts(holding=1.0, shortage=4.0),
            DEMAND,
            2,
            initial_state=(1, 0),
        )
        result = fit_inventory_policy(
            state("x"), case, 2, (), config=self.config
        )
        self.assertGreater(result.policy_cost, 0.0)

    def test_optimized_route_rejects_state_names_outside_case_schema(self) -> None:
        case = _case(
            "schema-bound-fit",
            InventoryModelConfig(
                lead_time=DeterministicLeadTime(1),
                backlog=LostSales(),
                capacity=2,
            ),
        )
        with self.assertRaises(ValueError):
            fit_inventory_policy(state("a1"), case, 2, (), config=self.config)


if __name__ == "__main__":
    unittest.main()
