from __future__ import annotations

import unittest
from itertools import product

import numpy as np

from inventory.common.distributions import DiscreteDemand
from evaluation.backends.lead_time.dp import solve_dp, solve_l0_dp, solve_lead_time_dp, verify_l0_base_stock


def _solve_lead_time_dp_reference(
    lead_time: int,
    horizon: int,
    n: int,
    demand: DiscreteDemand,
    h: float,
    b: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the pre-acceleration scalar-enumeration DP tables."""

    table_shape = (horizon + 1,) + (n + 1,) * (lead_time + 1)
    policy_shape = (horizon,) + (n + 1,) * (lead_time + 1)
    value_table = np.zeros(table_shape, dtype=float)
    policy_table = np.zeros(policy_shape, dtype=int)
    demand_values = demand.values.astype(int)
    demand_probs = demand.probs.astype(float)

    for t in range(horizon - 1, -1, -1):
        next_values = value_table[t + 1]
        for state in product(range(n + 1), repeat=lead_time + 1):
            x = state[0]
            pipeline = state[1:]
            available = x + pipeline[0]
            best_cost = float("inf")
            best_q = 0
            current_holding = h * np.maximum(available - demand_values, 0)
            current_shortage = b * np.maximum(demand_values - available, 0)

            for q in range(n + 1):
                expected_cost = 0.0
                for d, p, holding_cost, shortage_cost in zip(
                    demand_values, demand_probs, current_holding, current_shortage
                ):
                    next_x = int(max(available - int(d), 0))
                    next_x = min(next_x, n)
                    next_state = (next_x,) + pipeline[1:] + (q,)
                    expected_cost += float(p) * (
                        float(holding_cost) + float(shortage_cost) + float(next_values[next_state])
                    )
                if expected_cost < best_cost:
                    best_cost = expected_cost
                    best_q = q

            value_table[(t,) + state] = best_cost
            policy_table[(t,) + state] = best_q

    return value_table, policy_table


class BaselineDistributionAndDPTests(unittest.TestCase):
    def test_distribution_from_pmf_quantile_and_features(self) -> None:
        # PMF: D=0 has probability 0.25, and D=2 has probability 0.75.
        # Thus E[D]=1.5, and the median/q50 is 2.
        demand = DiscreteDemand.from_pmf({0: 0.25, 2: 0.75})

        self.assertAlmostEqual(demand.mean, 1.5)
        self.assertAlmostEqual(demand.std, np.sqrt(0.75))
        self.assertEqual(demand.quantile(0.50), 2)
        self.assertEqual(demand.quantile(0.90), 2)
        self.assertAlmostEqual(demand.expect(lambda d: np.maximum(d - 1, 0)), 0.75)

        features = demand.feature_dict()
        self.assertEqual(features["q50"], 2.0)
        self.assertEqual(features["support_max"], 2.0)

    def test_poisson_distribution_normalizes_truncated_tail(self) -> None:
        # With tail_to_max=True, probability mass beyond d_max is folded into the final point.
        demand = DiscreteDemand.poisson(lam=2.0, d_max=6)

        self.assertAlmostEqual(float(np.sum(demand.probs)), 1.0)
        self.assertEqual(demand.support_min, 0)
        self.assertEqual(demand.support_max, 6)
        self.assertGreaterEqual(demand.quantile(0.95), demand.quantile(0.50))

    def test_l0_dp_recovers_base_stock_structure(self) -> None:
        # L=0 is the most important sanity check: classical lost-sales finite-horizon DP
        # should recover the base-stock structure q(x)=max(S_t-x,0).
        demand = DiscreteDemand.poisson(lam=2.0, d_max=8)
        result = solve_l0_dp(horizon=5, n=6, demand=demand, h=1.0, b=5.0)
        check = verify_l0_base_stock(result)

        self.assertTrue(check["ok"], check)
        self.assertTrue(np.all(result.policy_table >= 0))
        self.assertTrue(np.all(np.isfinite(result.value_table)))
        self.assertGreater(result.value(0, 0), 0.0)

    def test_lead_time_dp_and_dispatch_match_scalar_bellman_recursion(self) -> None:
        cases = (
            (1, 4, 4, DiscreteDemand.poisson(lam=2.0, d_max=5), 1.0, 5.0),
            (2, 3, 3, DiscreteDemand.uniform(0, 3), 1.3, 4.7),
            (3, 2, 2, DiscreteDemand.from_pmf({0: 0.15, 2: 0.35, 5: 0.50}), 0.8, 3.2),
            (4, 2, 1, DiscreteDemand.from_pmf({0: 0.4, 3: 0.6}), 1.1, 2.9),
            (2, 1, 0, DiscreteDemand.uniform(0, 1), 1.0, 4.0),
        )

        for lead_time, horizon, n, demand, h, b in cases:
            with self.subTest(lead_time=lead_time, demand=demand.name):
                expected_values, expected_policy = _solve_lead_time_dp_reference(
                    lead_time=lead_time,
                    horizon=horizon,
                    n=n,
                    demand=demand,
                    h=h,
                    b=b,
                )
                for solver in (solve_lead_time_dp, solve_dp):
                    with self.subTest(solver=solver.__name__):
                        actual = solver(
                            lead_time=lead_time, horizon=horizon, n=n,
                            demand=demand, h=h, b=b,
                        )
                        np.testing.assert_array_equal(actual.value_table, expected_values)
                        np.testing.assert_array_equal(actual.policy_table, expected_policy)
if __name__ == "__main__":
    unittest.main()
