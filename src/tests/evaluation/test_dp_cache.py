from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from inventory.common.distributions import DiscreteDemand
from evaluation.backends.lead_time.dp import solve_dp
from inventory.core.factory import inventory_case
from inventory.core import DeterministicLeadTime, LostSales, TransitionCosts
from evaluation.backends.lead_time.cache import cache_case_dir, estimate_dp_bytes, load_or_create_dp, remove_case_cache
from optimization.backends.lead_time.rolling import (
    estimate_rolling_dp_bytes,
    rolling_reference,
    solve_optimal_cost_rolling,
)


def Case(name, lead_time, h, b, demand, kind, n):
    return inventory_case(
        name=name, demand=demand, n=n, kind=kind,
        costs=TransitionCosts(holding=h, shortage=b),
        lead_time=DeterministicLeadTime(lead_time), backlog=LostSales(), capacity=n,
    )


class DPSharedCacheTests(unittest.TestCase):
    def test_dp_reservation_includes_table_and_temporary_margin(self) -> None:
        gib = 1024**3
        # L4, n=14, H=1000: 11.32 GiB tables with the default 25% margin.
        self.assertAlmostEqual(estimate_dp_bytes(4, 14, 1000) / gib, 14.15, places=2)

    def test_rolling_reservation_is_independent_of_horizon_tables(self) -> None:
        mib = 1024**2
        self.assertLess(estimate_rolling_dp_bytes(4, 14) / mib, 40.0)
        self.assertGreater(estimate_rolling_dp_bytes(4, 14), 0)

        n = 20
        lead_time = 4
        base = n + 1
        legacy_cells = (
            2 * base ** (lead_time + 1)
            + (2 * n + 1) * base ** (lead_time - 1)
            + base**lead_time
        )
        legacy_bytes = int(legacy_cells * 8 * 1.25)
        self.assertEqual(
            estimate_rolling_dp_bytes(lead_time, n),
            estimate_rolling_dp_bytes(lead_time, n, backend="blocked_matrix"),
        )
        self.assertEqual(
            estimate_rolling_dp_bytes(lead_time, n, backend="expanded_dense"),
            legacy_bytes,
        )
        self.assertLess(
            estimate_rolling_dp_bytes(lead_time, n, backend="compressed"),
            legacy_bytes / 5,
        )

    def test_rolling_cost_and_samples_match_dense_dp(self) -> None:
        demand = DiscreteDemand.from_pmf({0: 0.25, 1: 0.5, 2: 0.25}, name="rolling_test")
        for lead_time in range(1, 5):
            case = Case("rolling_test", lead_time, 1.0, 3.0, demand, "unit", 2)
            horizon = 4
            dense = solve_dp(lead_time, horizon, case.n, demand, case.h, case.b)
            cost, _seconds, samples = solve_optimal_cost_rolling(
                rolling_reference(case),
                horizon,
                sample_size=8,
                random_seed=17,
            )
            self.assertAlmostEqual(cost, dense.value(0, (0,) * (lead_time + 1)), places=10)
            for sample in samples:
                state = (int(sample.state["x"]), *tuple(int(value) for value in sample.state["p"]))
                period = horizon - int(sample.state["remaining_horizon"])
                self.assertEqual(sample.target_action, dense.action(period, state))

    def test_compressed_and_legacy_rolling_match_exactly_for_l1_through_l8(self) -> None:
        demands = (
            DiscreteDemand.from_pmf({0: 0.25, 1: 0.5, 3: 0.25}, name="asymmetric"),
            DiscreteDemand.from_pmf({0: 0.5, 2: 0.5}, name="tie_prone"),
        )
        for lead_time in range(1, 9):
            for demand in demands:
                with self.subTest(lead_time=lead_time, demand=demand.name):
                    case = Case(
                        f"rolling_l{lead_time}_{demand.name}",
                        lead_time,
                        1.25,
                        3.75,
                        demand,
                        "unit",
                        2,
                    )
                    kwargs = {"sample_size": 19, "random_seed": 1700 + lead_time}
                    compressed = solve_optimal_cost_rolling(
                        rolling_reference(case), 5, backend="compressed", **kwargs
                    )
                    legacy = solve_optimal_cost_rolling(
                        rolling_reference(case), 5, backend="expanded_dense", **kwargs
                    )
                    self.assertEqual(compressed[0], legacy[0])
                    self.assertEqual(compressed[2], legacy[2])

    def test_blocked_matrix_matches_legacy_for_l1_through_l8(self) -> None:
        demands = (
            DiscreteDemand.from_pmf({0: 0.25, 1: 0.5, 3: 0.25}, name="asymmetric"),
            DiscreteDemand.from_pmf({0: 0.5, 2: 0.5}, name="tie_prone"),
        )
        for lead_time in range(1, 9):
            for demand in demands:
                with self.subTest(lead_time=lead_time, demand=demand.name):
                    case = Case(
                        f"blocked_l{lead_time}_{demand.name}",
                        lead_time,
                        1.25,
                        3.75,
                        demand,
                        "unit",
                        2,
                    )
                    kwargs = {"sample_size": 19, "random_seed": 4700 + lead_time}
                    blocked = solve_optimal_cost_rolling(
                        rolling_reference(case), 5, backend="blocked_matrix", **kwargs
                    )
                    legacy = solve_optimal_cost_rolling(
                        rolling_reference(case), 5, backend="expanded_dense", **kwargs
                    )
                    self.assertAlmostEqual(blocked[0], legacy[0], places=12)
                    self.assertEqual(blocked[2], legacy[2])

    def test_compressed_rolling_matches_dense_actions_for_l1_through_l8(self) -> None:
        demand = DiscreteDemand.from_pmf({0: 0.2, 1: 0.3, 4: 0.5}, name="l1_l8")
        for lead_time in range(1, 9):
            with self.subTest(lead_time=lead_time):
                case = Case(f"dense_l{lead_time}", lead_time, 0.75, 4.25, demand, "unit", 1)
                horizon = 4
                dense = solve_dp(lead_time, horizon, case.n, demand, case.h, case.b)
                cost, _seconds, samples = solve_optimal_cost_rolling(
                    rolling_reference(case),
                    horizon,
                    sample_size=13,
                    random_seed=2900 + lead_time,
                    backend="compressed",
                )
                self.assertEqual(cost, dense.value(0, (0,) * (lead_time + 1)))
                for sample in samples:
                    state = (
                        int(sample.state["x"]),
                        *tuple(int(value) for value in sample.state["p"]),
                    )
                    period = horizon - int(sample.state["remaining_horizon"])
                    self.assertEqual(sample.target_action, dense.action(period, state))

    def test_rolling_backend_validation(self) -> None:
        case = Case(
            "invalid_backend",
            1,
            1.0,
            3.0,
            DiscreteDemand.uniform(0, 1),
            "unit",
            2,
        )
        with self.assertRaisesRegex(ValueError, "backend"):
            solve_optimal_cost_rolling(rolling_reference(case), 2, backend="unknown")

    def test_rolling_retains_exact_compressed_continuation_layer(self) -> None:
        demand = DiscreteDemand.from_pmf({0: 0.25, 1: 0.5, 2: 0.25}, name="continuation")
        for lead_time in range(1, 5):
            for backend in ("blocked_matrix", "compressed", "expanded_dense"):
                with self.subTest(lead_time=lead_time, backend=backend):
                    case = Case("continuation", lead_time, 1.0, 3.0, demand, "unit", 2)
                    reference = rolling_reference(case)
                    solve_optimal_cost_rolling(
                        reference,
                        4,
                        sample_size=3,
                        random_seed=9,
                        backend=backend,
                        continuation_bound_periods=2,
                    )
                    dense = solve_dp(lead_time, 2, case.n, demand, case.h, case.b)
                    self.assertEqual(reference.continuation_horizon, 2)
                    self.assertEqual(
                        reference.continuation_values.shape,
                        (2 * case.n + 1,) + (case.n + 1,) * (lead_time - 1),
                    )
                    for state in np.ndindex(*((case.n + 1,) * (lead_time + 1))):
                        compressed_state = (state[0] + state[1], *state[2:])
                        self.assertAlmostEqual(
                            reference.continuation_values[compressed_state],
                            dense.value(0, state),
                            places=11,
                        )

    def test_rolling_continuation_horizon_is_validated(self) -> None:
        case = Case(
            "invalid_continuation",
            1,
            1.0,
            3.0,
            DiscreteDemand.uniform(0, 1),
            "unit",
            2,
        )
        with self.assertRaisesRegex(ValueError, "continuation_bound_periods"):
            solve_optimal_cost_rolling(
                rolling_reference(case), 2, continuation_bound_periods=3
            )

    def test_case_dp_is_created_once_and_reopened_read_only(self) -> None:
        case = Case(
            "cache_test",
            1,
            1.0,
            3.0,
            DiscreteDemand.from_pmf({0: 0.5, 1: 0.5}, name="cache_test_dist"),
            "unit",
            2,
        )
        with tempfile.TemporaryDirectory() as temporary:
            cache_root = Path(temporary)
            with patch("evaluation.backends.lead_time.cache.solve_dp", wraps=solve_dp) as solve:
                first = load_or_create_dp(case, 4, suite="unit", cache_dir=cache_root)
                second = load_or_create_dp(case, 4, suite="unit", cache_dir=cache_root)

            self.assertEqual(solve.call_count, 1)
            self.assertTrue(isinstance(second.value_table, np.memmap))
            self.assertTrue(isinstance(second.policy_table, np.memmap))
            self.assertFalse(second.value_table.flags.writeable)
            self.assertFalse(second.policy_table.flags.writeable)
            np.testing.assert_array_equal(first.value_table, second.value_table)
            np.testing.assert_array_equal(first.policy_table, second.policy_table)
            self.assertTrue((cache_case_dir(cache_root, "unit", case.name) / "metadata.json").exists())

            remove_case_cache(cache_root, "unit", case.name)
            self.assertFalse(cache_case_dir(cache_root, "unit", case.name).exists())


if __name__ == "__main__":
    unittest.main()
