from __future__ import annotations

import unittest

import numpy as np

from inventory.common.distributions import DiscreteDemand
from inventory.core import (
    BacklogConfig,
    DeterministicLeadTime,
    FixedShelfLife,
    InventoryCase,
    InventoryModelConfig,
    InventoryState,
    TransitionCosts,
)
from evaluation.dp import composed_state_count, enumerate_composed_states, solve_composed_dp
from evaluation.exact_mc import evaluate_composed_policy_exact


class InventoryCaseContractTests(unittest.TestCase):
    def test_case_derives_schema_identity_and_zero_initial_state(self) -> None:
        config = InventoryModelConfig(
            lead_time=DeterministicLeadTime(2),
            perishability=FixedShelfLife(3),
            capacity=5,
        )
        case = InventoryCase(
            "demo",
            config,
            TransitionCosts(holding=1.0, shortage=4.0, expiry=2.0),
            DiscreteDemand.from_pmf({0: 0.5, 1: 0.5}, name="bernoulli"),
            5,
            kind="unit",
        )

        self.assertEqual(case.model_id, "lead_time+perishability")
        self.assertEqual(case.schema.names, ("a1", "a2", "a3", "p1", "p2"))
        self.assertEqual(case.initial_state, InventoryState.zeros(case.schema))
        self.assertEqual(case.initial_flat_state, (0, 0, 0, 0, 0))
        self.assertEqual(case.h, 1.0)
        self.assertEqual(case.b, 4.0)
        self.assertEqual(case.w, 2.0)
        self.assertTrue(case.case_signature)
        self.assertEqual(case.case_signature, case.signature)

    def test_case_normalizes_mapping_and_flat_initial_state(self) -> None:
        config = InventoryModelConfig(lead_time=DeterministicLeadTime(1))
        demand = DiscreteDemand.from_pmf({0: 1.0}, name="zero")
        case = InventoryCase(
            "mapped",
            config,
            {"h": 1.0, "b": 3.0},
            demand,
            4,
            initial_state=(2, 1),
        )
        self.assertEqual(case.costs, TransitionCosts(holding=1.0, shortage=3.0))
        self.assertEqual(case.initial_state, InventoryState((2,), (1,), ()))
        self.assertEqual(case.state_schema, case.schema)
        self.assertEqual(case.to_dict()["n"], 4)

    def test_capacity_and_state_shape_are_checked(self) -> None:
        config = InventoryModelConfig(lead_time=DeterministicLeadTime(1), capacity=2)
        demand = DiscreteDemand.from_pmf({0: 1.0}, name="zero")
        case = InventoryCase("separate-caps", config, TransitionCosts(), demand, 3)
        self.assertEqual(case.action_cap, 3)
        self.assertEqual(case.schema.capacity, 2)
        with self.assertRaises(ValueError):
            InventoryCase(
                "bad-state", config, TransitionCosts(), demand, 2, initial_state=(3, 0)
            )
        with self.assertRaisesRegex(ValueError, "n must be an integer"):
            InventoryCase(
                "boolean-action-cap",
                config,
                TransitionCosts(),
                demand,
                np.bool_(True),
            )

    def test_unbounded_case_does_not_infer_state_cap_from_action_cap(self) -> None:
        case = InventoryCase(
            "unprojected",
            InventoryModelConfig(
                lead_time=DeterministicLeadTime(1),
                backlog=BacklogConfig.full(),
            ),
            TransitionCosts(),
            DiscreteDemand.from_pmf({0: 1.0}, name="zero"),
            2,
        )
        outcome = case.kernel.transition(
            InventoryState((0,), (7,), (4,)), action=2, demand=20
        )
        self.assertIsNone(case.config.capacity)
        self.assertEqual(outcome.next_state.pipeline, (2,))
        self.assertEqual(outcome.next_state.backlog, (17,))
        with self.assertRaisesRegex(ValueError, "explicit config.capacity"):
            composed_state_count(case)

    def test_deterministic_backlog_grid_uses_a_total_capacity(self) -> None:
        case = InventoryCase(
            "finite-patience",
            InventoryModelConfig(backlog=BacklogConfig.deterministic(2), capacity=2),
            TransitionCosts(),
            DiscreteDemand.from_pmf({0: 1.0}, name="zero"),
            2,
        )
        states = enumerate_composed_states(case)
        self.assertEqual(composed_state_count(case), 18)
        self.assertEqual(len(states), 18)
        self.assertTrue(all(sum(state.backlog) <= case.n for state in states))

    def test_projected_dp_guards_horizon_times_state_storage(self) -> None:
        case = InventoryCase(
            "table-guard",
            InventoryModelConfig(capacity=2),
            TransitionCosts(),
            DiscreteDemand.from_pmf({0: 1.0}, name="zero"),
            1,
        )
        # Three states require (3 * horizon + 2) * 3 table cells: value,
        # policy, and cumulative-discard diagnostics all count toward storage.
        with self.assertRaisesRegex(MemoryError, "tables require 24 cells"):
            solve_composed_dp(case, 2, max_table_cells=23)

        result = solve_composed_dp(case, 2, max_table_cells=24)
        self.assertEqual(result.backend, "composed_compiled_finite_mdp")

    def test_projected_dp_skips_zero_probability_demand_support(self) -> None:
        case = InventoryCase(
            "zero-demand-dp",
            InventoryModelConfig(backlog=BacklogConfig.geometric(0.5), capacity=0),
            TransitionCosts(),
            DiscreteDemand.from_pmf({0: 1.0, 20_000: 0.0}, name="null-tail"),
            0,
        )
        result = solve_composed_dp(case, 1, max_states=1)
        self.assertEqual(result.value(0, case.initial_state), 0.0)

    def test_projected_dp_reports_discard_under_selected_optimal_policy(self) -> None:
        case = InventoryCase(
            "dp-discard-diagnostics",
            InventoryModelConfig(capacity=1),
            TransitionCosts(shortage=10.0),
            DiscreteDemand.from_pmf({0: 0.5, 3: 0.5}, name="bimodal"),
            2,
        )
        result = solve_composed_dp(case, 2, max_states=10)
        self.assertAlmostEqual(result.expected_discarded(0, case.initial_state), 1.25)
        self.assertAlmostEqual(result.metadata["expected_cumulative_discarded"], 1.25)

        optimal_policy = lambda state, remaining: result.action(2 - remaining, state)
        exact = evaluate_composed_policy_exact(case, optimal_policy, 2)
        self.assertAlmostEqual(exact.metadata["expected_cumulative_discarded"], 1.25)

    def test_projected_dp_rejects_noninteger_control_arguments(self) -> None:
        case = InventoryCase(
            "strict-dp-inputs",
            InventoryModelConfig(capacity=1),
            TransitionCosts(),
            DiscreteDemand.from_pmf({0: 1.0}, name="zero"),
            1,
        )
        calls = (
            ("horizon", lambda value: solve_composed_dp(case, value)),
            (
                "max_states",
                lambda value: solve_composed_dp(case, 1, max_states=value),
            ),
            (
                "max_table_cells",
                lambda value: solve_composed_dp(case, 1, max_table_cells=value),
            ),
        )
        for name, call in calls:
            for invalid in (
                1.5,
                np.bool_(True),
                np.array(1),
                np.array([1]),
                float("inf"),
            ):
                with self.subTest(name=name, invalid=invalid):
                    with self.assertRaisesRegex(
                        ValueError, f"{name} must be an integer"
                    ):
                        call(invalid)

        result = solve_composed_dp(case, 1)
        for accessor in (result.value, result.action, result.expected_discarded):
            with self.subTest(accessor=accessor):
                with self.assertRaisesRegex(ValueError, "period must be an integer"):
                    accessor(0.5, case.initial_state)


if __name__ == "__main__":
    unittest.main()
