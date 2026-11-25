from __future__ import annotations

import unittest

import numpy as np

from evaluation.diagnostics.state_abstraction import (
    StateAbstractionGroup,
    summarize_state_groups,
)
class StateInformationDiagnosticTests(unittest.TestCase):
    def test_bellman_regret_ignores_action_ties(self) -> None:
        group = StateAbstractionGroup(
            mass=1.0,
            action_mass=np.array([0.5, 0.5]),
            q_cost_mass=np.array([2.0, 2.0]),
            optimal_q_cost_mass=2.0,
        )
        summary = summarize_state_groups(
            {(0,): group}, reference_average_cost=4.0
        )
        self.assertAlmostEqual(float(summary["majority_action_error"]), 0.5)
        self.assertAlmostEqual(float(summary["bellman_regret_per_visited_period"]), 0.0)

    def test_bellman_regret_detects_incompatible_states(self) -> None:
        group = StateAbstractionGroup(
            mass=1.0,
            action_mass=np.array([0.5, 0.5]),
            q_cost_mass=np.array([1.5, 1.5]),
            optimal_q_cost_mass=1.0,
        )
        summary = summarize_state_groups(
            {(0,): group}, reference_average_cost=2.0
        )
        self.assertAlmostEqual(float(summary["bellman_regret_per_visited_period"]), 0.5)
        self.assertAlmostEqual(float(summary["bellman_regret_pct_of_dp_average_cost"]), 25.0)


if __name__ == "__main__":
    unittest.main()
