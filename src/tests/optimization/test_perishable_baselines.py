from __future__ import annotations

import unittest

from inventory.policy.ast import ast_check, compile_policy
from optimization.policies.baselines.perishable import baselines_for_perishable
from optimization.policies.seeds import policy_seed_asts_for_case
from inventory.common.distributions import DiscreteDemand
from inventory.core import FixedShelfLife, LostSales, TransitionCosts
from inventory.core.factory import inventory_case


class PerishableBaselineTests(unittest.TestCase):
    def test_core_catalog_is_complete_and_expectation_free(self) -> None:
        rows = baselines_for_perishable(4, 10)
        self.assertEqual(
            [row.name for row in rows],
            [
                "constant_order",
                "base_stock",
                "s_s",
                "capped_base_stock",
                "bsp_low",
                "weighted_base_stock",
            ],
        )
        for row in rows:
            ast_check(
                row.ast,
                motifs=row.motifs,
                allowed_state_names={"a1", "a2", "a3", "a4", "remaining_horizon"},
            )

    def test_scalar_inventory_policies_use_only_decision_time_stock(self) -> None:
        rows = {row.name: row for row in baselines_for_perishable(4, 10)}
        state = {"a1": 1, "a2": 2, "a3": 3, "a4": 99}

        base = compile_policy(
            rows["base_stock"].ast,
            {"S": 10},
            motifs=rows["base_stock"].motifs,
            action_cap=10,
            allowed_state_names={"a1", "a2", "a3", "a4", "remaining_horizon"},
        )
        capped = compile_policy(
            rows["capped_base_stock"].ast,
            {"S": 10, "Qmax": 2},
            motifs=rows["capped_base_stock"].motifs,
            action_cap=10,
            allowed_state_names={"a1", "a2", "a3", "a4", "remaining_horizon"},
        )
        reorder = compile_policy(
            rows["s_s"].ast,
            {"s": 6, "S": 10},
            motifs=rows["s_s"].motifs,
            action_cap=10,
            allowed_state_names={"a1", "a2", "a3", "a4", "remaining_horizon"},
        )
        self.assertEqual(base(state, {}, {}), 4)
        self.assertEqual(capped(state, {}, {}), 2)
        self.assertEqual(reorder(state, {}, {}), 4)

    def test_weighted_order_up_to_uses_age_profile(self) -> None:
        spec = next(
            row
            for row in baselines_for_perishable(4, 10)
            if row.name == "weighted_base_stock"
        )
        policy = compile_policy(
            spec.ast,
            {"S": 10, "omega1": 0.0, "omega2": 0.5, "omega3": 1.0},
            motifs=spec.motifs,
            action_cap=10,
            allowed_state_names={"a1", "a2", "a3", "a4", "remaining_horizon"},
        )
        self.assertEqual(policy({"a1": 4, "a2": 2, "a3": 1, "a4": 0}, {}, {}), 8)

    def test_perishable_cases_receive_baseline_search_seeds(self) -> None:
        case = inventory_case(
            name="perishable-seeds",
            demand=DiscreteDemand.from_pmf({0: 0.5, 1: 0.5}, name="two-point"),
            n=4,
            costs=TransitionCosts(holding=1, shortage=4, expiry=1),
            perishability=FixedShelfLife(3),
            backlog=LostSales(),
            capacity=4,
        )
        names = [source for source, _ in policy_seed_asts_for_case(case)]
        self.assertIn("baseline:base_stock", names)
        self.assertIn("baseline:weighted_base_stock", names)

if __name__ == "__main__":
    unittest.main()
