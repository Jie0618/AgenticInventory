from __future__ import annotations

import unittest

from evaluation.dp import composed_state_count, enumerate_composed_states, solve_composed_dp
from evaluation.exact_mc import evaluate_composed_policy_exact, evaluate_composed_policy_mc
from inventory.core.factory import inventory_case
from inventory.common.distributions import DiscreteDemand
from inventory.core import FixedShelfLife, InventoryState, LostSales, TransitionCosts


def _case(name: str, shelf_life: int, h: float, b: float, w: float,
          demand: DiscreteDemand, n: int):
    return inventory_case(
        name=name, demand=demand, n=n,
        costs=TransitionCosts(holding=h, shortage=b, expiry=w),
        perishability=FixedShelfLife(shelf_life), backlog=LostSales(), capacity=n,
    )


class PerishableInventoryTests(unittest.TestCase):
    def test_age_state_grid_is_a_total_inventory_simplex(self) -> None:
        case = _case("grid", 2, 1.0, 4.0, 2.0,
                               DiscreteDemand.from_pmf({0: 1.0}, name="zero"), 3)
        states = enumerate_composed_states(case)
        self.assertEqual(composed_state_count(case), 10)
        self.assertTrue(all(sum(state.inventory) <= 3 for state in states))
        self.assertIn(InventoryState((1, 2), (), ()), states)
        self.assertNotIn(InventoryState((2, 2), (), ()), states)

    def test_fefo_transition_cost_timing_and_aging(self) -> None:
        case = _case("manual", 3, 2.0, 7.0, 5.0,
                               DiscreteDemand.from_pmf({0: 1.0}, name="zero"), 5)
        outcome = case.kernel.transition(InventoryState((2, 1, 0), (), ()), 2, 1)
        self.assertEqual(outcome.next_state, InventoryState((1, 2, 0), (), ()))
        self.assertEqual(outcome.holding_cost, 8.0)
        self.assertEqual(outcome.expiry_cost, 5.0)
        self.assertEqual(outcome.total_cost, 13.0)

    def test_shared_dp_exact_and_mc_agree(self) -> None:
        case = _case("evaluation", 2, 1.0, 5.0, 2.0,
                               DiscreteDemand.from_pmf({1: 1.0}, name="one"), 3)
        horizon = 4
        dp = solve_composed_dp(case, horizon)
        policy = lambda state, remaining: dp.action(horizon - remaining, state)
        exact = evaluate_composed_policy_exact(case, policy, horizon)
        mc = evaluate_composed_policy_mc(case, policy, horizon, paths=20, seed=17)
        self.assertAlmostEqual(exact.cost, dp.value(0, case.initial_state))
        self.assertEqual(mc.standard_error, 0.0)
        self.assertAlmostEqual(mc.mean_cost, exact.cost)


if __name__ == "__main__":
    unittest.main()
