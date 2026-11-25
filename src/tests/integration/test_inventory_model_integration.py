from __future__ import annotations

import unittest

from evaluation.exact_mc import evaluate_composed_policy_exact, evaluate_composed_policy_mc
from inventory.common.distributions import DiscreteDemand
from inventory.core import (
    BacklogConfig,
    DeterministicLeadTime,
    FixedShelfLife,
    InventoryCase,
    InventoryModelConfig,
    TransitionCosts,
)


class InventoryModelIntegrationTests(unittest.TestCase):
    def test_three_components_compose_without_a_model_registry(self) -> None:
        case = InventoryCase(
            "all-components",
            InventoryModelConfig(
                lead_time=DeterministicLeadTime(1),
                perishability=FixedShelfLife(2),
                backlog=BacklogConfig.full(),
                capacity=3,
            ),
            TransitionCosts(holding=1.0, shortage=4.0, expiry=2.0, backlog=0.5),
            DiscreteDemand.from_pmf({0: 0.5, 2: 0.5}, name="bimodal"),
            3,
        )
        policy = lambda _state, _remaining: 1
        exact = evaluate_composed_policy_exact(case, policy, 3)
        mc = evaluate_composed_policy_mc(case, policy, 3, paths=2_000, seed=9)
        lower, upper = mc.confidence_interval(z=5.0)
        self.assertLessEqual(lower, exact.cost)
        self.assertLessEqual(exact.cost, upper)
        self.assertEqual(case.schema.names, ("a1", "a2", "p1", "B"))


if __name__ == "__main__":
    unittest.main()
