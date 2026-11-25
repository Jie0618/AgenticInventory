from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from inventory.common.distributions import DiscreteDemand
from inventory.core import (
    DeterministicLeadTime,
    InventoryCase,
    InventoryModelConfig,
    LostSales,
    TransitionCosts,
)
from evaluation.backends.lead_time.dp import solve_dp
from sr.inventory.single import SymbolicSearch, diagnostic_samples
from optimization import InventorySearchContext
from sr import SearchConfig
from sr.core.inventory import InventorySearchBindings
from sr.inventory.seeds import initial_seed_asts


class SymbolicSearchIntegrationTests(unittest.TestCase):
    def test_preloaded_samples_are_reused_without_reading_diagnostic_again(self) -> None:
        demand = DiscreteDemand.from_pmf({0: 1.0}, name="deterministic")
        dp = solve_dp(1, 6, 1, demand, 1.0, 2.0)
        case = InventoryCase(
            "tiny-preloaded",
            InventoryModelConfig(
                lead_time=DeterministicLeadTime(1),
                backlog=LostSales(),
                capacity=1,
            ),
            TransitionCosts(holding=1.0, shortage=2.0),
            demand,
            n=1,
        )
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            diagnostic = root / "diagnostic.json"
            diagnostic.write_text(json.dumps(_diagnostic(dp)), encoding="utf-8")
            samples = diagnostic_samples(diagnostic, 6)
            context = InventorySearchContext.from_reference(
                case,
                6,
                dp,
                samples=samples,
            )
            config = SearchConfig(iterations=1, optimizer_workers=1)
            bindings = InventorySearchBindings.build(context, config)

            with patch(
                "sr.inventory.single.diagnostic_samples",
                side_effect=AssertionError("diagnostic must not be read twice"),
            ):
                search = SymbolicSearch(
                    case_name=case.name,
                    context=context,
                    diagnostic_json=diagnostic,
                    output_dir=root / "output",
                    config=config,
                    bindings=bindings,
                    samples=samples,
                )

            self.assertEqual(search.samples, samples)

    def test_real_standard_search_is_reproducible(self) -> None:
        demand = DiscreteDemand.from_pmf({0: 0.5, 1: 0.5}, name="tiny")
        dp = solve_dp(1, 6, 1, demand, 1.0, 2.0)
        case = InventoryCase(
            "tiny",
            InventoryModelConfig(
                lead_time=DeterministicLeadTime(1),
                backlog=LostSales(),
                capacity=1,
            ),
            TransitionCosts(holding=1.0, shortage=2.0),
            demand,
            n=1,
        )
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            diagnostic = root / "diagnostic.json"
            diagnostic.write_text(json.dumps(_diagnostic(dp)), encoding="utf-8")
            config = SearchConfig(
                iterations=1,
                island_count=2,
                island_max_size=20,
                llm_proposals_per_iteration=1,
                gp_descendants_per_slot=1,
                standard_promotions_per_iteration=1,
                iteration_wait_seconds=1,
                iteration_idle_active_workers=0,
                screening_points_low_dim=16,
                screening_points_high_dim=8,
                optimizer_workers=2,
                evaluation_hard_timeout_seconds=40,
                memory_reserve_gb=0,
                random_seed=41,
            )
            context = InventorySearchContext.from_reference(case, 6, dp)
            bindings = InventorySearchBindings.build(context, config)
            smoke_seeds = initial_seed_asts(case)[:3]
            with patch(
                "sr.inventory.single.initial_seed_asts",
                return_value=smoke_seeds,
            ):
                first = SymbolicSearch(case_name="tiny", context=context, diagnostic_json=diagnostic, output_dir=root / "first", config=config, bindings=bindings).run()
                second = SymbolicSearch(case_name="tiny", context=context, diagnostic_json=diagnostic, output_dir=root / "second", config=config, bindings=bindings).run()
            self.assertTrue(first.archive)
            self.assertTrue((root / "first" / "REPORT.md").is_file())
            first_rows = [(row.ast_hash, row.params, row.policy_cost) for row in first.archive]
            second_rows = [(row.ast_hash, row.params, row.policy_cost) for row in second.archive]
            self.assertEqual(first_rows, second_rows)


def _diagnostic(dp) -> dict:
    period = 3
    rows = []
    states = [(x, p1) for x in range(dp.n + 1) for p1 in range(dp.n + 1)]
    for state in states:
        available = state[0] + state[1]
        q_costs = np.zeros(dp.n + 1, dtype=float)
        for demand, probability in zip(dp.demand.values, dp.demand.probs):
            demand = int(demand)
            next_x = min(max(available - demand, 0), dp.n)
            immediate = dp.h * max(available - demand, 0) + dp.b * max(demand - available, 0)
            q_costs += float(probability) * (immediate + dp.value_table[period + 1, next_x, :])
        rows.append(
            {
                "state": list(state),
                "state_features": {"x": state[0], "p1": state[1], "remaining_horizon": dp.horizon - period},
                "period": period,
                "probability": 1.0 / len(states),
                "dp_action": int(np.argmin(q_costs)),
                "q_cost_advantage": (q_costs - np.min(q_costs)).tolist(),
            }
        )
    return {
        "lead_time": dp.lead_time,
        "horizon": dp.horizon,
        "n": dp.n,
        "dp_average_cost": float(dp.value(0, (0, 0))) / dp.horizon,
        "representative_states": rows,
    }


if __name__ == "__main__":
    unittest.main()
