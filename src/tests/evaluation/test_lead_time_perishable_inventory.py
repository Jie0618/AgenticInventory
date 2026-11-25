from __future__ import annotations

import unittest

from evaluation.dp import solve_composed_dp
from evaluation.exact_mc import evaluate_composed_policy_exact, evaluate_composed_policy_mc
from inventory.core.factory import inventory_case
from inventory.common.distributions import DiscreteDemand
from inventory.core import (
    DeterministicLeadTime,
    FixedShelfLife,
    InventoryState,
    LostSales,
    NoLeadTime,
    TransitionCosts,
)


def _case(name: str, lead_time: int | None, shelf_life: int, h: float, b: float,
          w: float, demand: DiscreteDemand, n: int):
    return inventory_case(
        name=name, demand=demand, n=n,
        costs=TransitionCosts(holding=h, shortage=b, expiry=w),
        lead_time=None if lead_time is None else DeterministicLeadTime(lead_time),
        perishability=FixedShelfLife(shelf_life), backlog=LostSales(), capacity=n,
    )


class LeadTimePerishableInventoryTests(unittest.TestCase):
    def test_pipeline_arrival_fefo_aging_and_shift(self) -> None:
        case = _case("manual", 2, 2, 2.0, 7.0, 5.0,
                                         DiscreteDemand.from_pmf({0: 1.0}, name="zero"), 5)
        outcome = case.kernel.transition(InventoryState((1, 0), (2, 3), ()), 4, 2)
        self.assertEqual(outcome.next_state, InventoryState((1, 0), (3, 4), ()))
        self.assertEqual(outcome.holding_cost, 2.0)
        self.assertEqual(outcome.expiry_cost, 0.0)

    def test_explicitly_disabled_lead_time_is_the_same_as_omitted(self) -> None:
        demand = DiscreteDemand.from_pmf({0: 0.3, 2: 0.7}, name="two_point")
        pure = _case("pure", None, 3, 1.0, 4.0, 2.0, demand, 3)
        combined = inventory_case(
            name="combined", demand=demand, n=3,
            costs=TransitionCosts(holding=1.0, shortage=4.0, expiry=2.0),
            lead_time=NoLeadTime(), perishability=FixedShelfLife(3),
            backlog=LostSales(), capacity=3,
        )
        self.assertEqual(pure.config, combined.config)
        self.assertEqual(pure.schema, combined.schema)

    def test_shared_dp_exact_and_mc_agree(self) -> None:
        case = _case("evaluation", 1, 2, 1.0, 5.0, 2.0,
                                         DiscreteDemand.from_pmf({1: 1.0}, name="one"), 2)
        horizon = 4
        dp = solve_composed_dp(case, horizon)
        policy = lambda state, remaining: dp.action(horizon - remaining, state)
        exact = evaluate_composed_policy_exact(case, policy, horizon)
        mc = evaluate_composed_policy_mc(case, policy, horizon, paths=10, seed=3)
        self.assertAlmostEqual(exact.cost, dp.value(0, case.initial_state))
        self.assertEqual(mc.standard_error, 0.0)
        self.assertAlmostEqual(mc.mean_cost, exact.cost)


if __name__ == "__main__":
    unittest.main()
