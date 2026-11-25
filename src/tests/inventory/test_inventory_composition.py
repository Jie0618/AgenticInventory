from __future__ import annotations

import unittest

from evaluation.dp import solve_composed_dp
from evaluation.exact_mc import evaluate_composed_policy_exact
from inventory.core.factory import inventory_case
from inventory.common.distributions import DiscreteDemand
from inventory.core import (
    DeterministicLeadTime,
    FixedShelfLife,
    InventoryCase,
    LostSales,
    TransitionCosts,
)


class InventoryCompositionIntegrationTests(unittest.TestCase):
    def test_one_factory_composes_every_component_selection(self) -> None:
        demand = DiscreteDemand.from_pmf({0: 0.5, 1: 0.5}, name="bernoulli")
        costs = TransitionCosts(holding=1.0, shortage=4.0, expiry=2.0)
        cases = (
            inventory_case(name="lt", demand=demand, n=2, costs=costs,
                           lead_time=DeterministicLeadTime(1), backlog=LostSales(), capacity=2),
            inventory_case(name="p", demand=demand, n=2, costs=costs,
                           perishability=FixedShelfLife(2), backlog=LostSales(), capacity=2),
            inventory_case(name="lp", demand=demand, n=2, costs=costs,
                           lead_time=DeterministicLeadTime(1),
                           perishability=FixedShelfLife(2), backlog=LostSales(), capacity=2),
        )
        self.assertTrue(all(type(case) is InventoryCase for case in cases))
        self.assertEqual(
            [case.config.enabled_components for case in cases],
            [("lead_time",), ("perishability",), ("lead_time", "perishability")],
        )

    def test_each_factory_case_uses_shared_dp_and_exact_evaluation(self) -> None:
        demand = DiscreteDemand.from_pmf({0: 0.5, 1: 0.5}, name="bernoulli")
        costs = TransitionCosts(holding=1.0, shortage=4.0, expiry=2.0)
        cases = (
            inventory_case(name="lt", demand=demand, n=2, costs=costs,
                           lead_time=DeterministicLeadTime(1), backlog=LostSales(), capacity=2),
            inventory_case(name="p", demand=demand, n=2, costs=costs,
                           perishability=FixedShelfLife(2), backlog=LostSales(), capacity=2),
            inventory_case(name="lp", demand=demand, n=2, costs=costs,
                           lead_time=DeterministicLeadTime(1),
                           perishability=FixedShelfLife(2), backlog=LostSales(), capacity=2),
        )
        for case in cases:
            with self.subTest(case=case.name):
                result = solve_composed_dp(case, 2, max_states=100)
                policy = lambda state, remaining, result=result: result.action(2 - remaining, state)
                exact = evaluate_composed_policy_exact(case, policy, 2)
                self.assertAlmostEqual(exact.cost, result.value(0, case.initial_state))


if __name__ == "__main__":
    unittest.main()
