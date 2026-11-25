from __future__ import annotations

import itertools
import unittest

import numpy as np

from inventory.common.distributions import DiscreteDemand
from optimization.policies.baselines.lead_time import baselines_for_lead_time
from evaluation.backends.lead_time.dp import solve_dp
from inventory.policy.ast import compile_policy
from optimization import SparseExactPolicyCostEvaluator, canonical_parameter_key, canonicalize_params
from optimization.policies.vbs import (
    detect_vbs_ast,
    effective_vbs_constraint_indexes,
    evaluate_vbs_actions,
)


class VectorBaseStockCanonicalizationTests(unittest.TestCase):
    @staticmethod
    def _spec(lead_time: int, n: int):
        return next(
            spec
            for spec in baselines_for_lead_time(lead_time, n)
            if spec.name == "vector_base_stock"
        )

    def test_only_vbs_parameters_are_marked(self) -> None:
        for policy in baselines_for_lead_time(4, 20):
            groups = [parameter.monotone_group for parameter in policy.parameter_specs]
            if policy.name == "vector_base_stock":
                self.assertTrue(
                    all(group == "vector_base_stock_targets" for group in groups)
                )
            else:
                self.assertTrue(all(group is None for group in groups), msg=policy.name)

    def test_only_registered_vbs_ast_is_detected(self) -> None:
        for lead_time in range(1, 6):
            for policy in baselines_for_lead_time(lead_time, 20):
                self.assertEqual(
                    detect_vbs_ast(policy.ast, lead_time),
                    policy.name == "vector_base_stock",
                    msg=f"L={lead_time} policy={policy.name}",
                )

    def test_dominated_constraint_examples(self) -> None:
        self.assertEqual(
            effective_vbs_constraint_indexes(
                {f"S{i}": 20.0 for i in range(6)}, 5, 20
            ),
            (5,),
        )
        self.assertEqual(
            effective_vbs_constraint_indexes(
                {f"S{i}": float(20 * i) for i in range(6)}, 5, 20
            ),
            (0,),
        )

    def test_dominated_constraint_actions_match_full_ast_exhaustively(self) -> None:
        for lead_time in range(1, 6):
            n = 2
            spec = self._spec(lead_time, n)
            high = (lead_time + 1) * n
            states = np.asarray(
                list(itertools.product(range(n + 1), repeat=lead_time + 1)),
                dtype=np.int32,
            )
            raw_vectors = (
                {f"S{i}": float(n) for i in range(lead_time + 1)},
                {f"S{i}": float(i * n) for i in range(lead_time + 1)},
                {
                    f"S{i}": float((7 * i + 5) % (high + 1))
                    for i in range(lead_time + 1)
                },
                {
                    f"S{i}": float((3 * i * i + 1) % (high + 1))
                    for i in range(lead_time + 1)
                },
            )
            for raw in raw_vectors:
                policy = compile_policy(
                    spec.ast, raw, motifs=spec.motifs, action_cap=n
                )
                expected = np.asarray(
                    [
                        policy(
                            {"x": int(values[0]), "p": tuple(int(x) for x in values[1:])},
                            {},
                            {},
                        )
                        for values in states
                    ],
                    dtype=np.int32,
                )
                actual, active = evaluate_vbs_actions(states, raw, lead_time, n)
                self.assertGreaterEqual(len(active), 1)
                np.testing.assert_array_equal(
                    actual,
                    expected,
                    err_msg=f"L={lead_time} raw={raw} active={active}",
                )

    def test_dominance_reduction_covers_all_small_target_vectors(self) -> None:
        n = 2
        for lead_time in range(1, 4):
            states = np.asarray(
                list(itertools.product(range(n + 1), repeat=lead_time + 1)),
                dtype=np.int32,
            )
            cumulative = np.cumsum(states, axis=1)
            high = (lead_time + 1) * n
            for targets in itertools.product(range(high + 1), repeat=lead_time + 1):
                params = {f"S{i}": float(value) for i, value in enumerate(targets)}
                full_raw = np.min(
                    np.asarray(targets, dtype=float)[None, :] - cumulative,
                    axis=1,
                )
                expected = np.clip(np.rint(full_raw), 0, n).astype(np.int32)
                actual, _active = evaluate_vbs_actions(
                    states, params, lead_time, n
                )
                np.testing.assert_array_equal(actual, expected)

    def test_suffix_min_is_nondecreasing_and_idempotent(self) -> None:
        spec = self._spec(4, 20)
        raw = {"S0": 90, "S1": 70, "S2": 80, "S3": 20, "S4": 60}
        canonical = canonicalize_params(raw, spec.parameter_specs)
        self.assertEqual(
            canonical,
            {"S0": 20.0, "S1": 20.0, "S2": 20.0, "S3": 20.0, "S4": 60.0},
        )
        self.assertEqual(canonicalize_params(canonical, spec.parameter_specs), canonical)

    def test_raw_and_canonical_vbs_actions_match_exhaustively(self) -> None:
        n = 2
        for lead_time in range(1, 6):
            spec = self._spec(lead_time, n)
            high = (lead_time + 1) * n
            raw_vectors = (
                {
                    f"S{i}": float((7 * i + 5) % (high + 1))
                    for i in range(lead_time + 1)
                },
                {f"S{i}": float(high - i) for i in range(lead_time + 1)},
                {
                    f"S{i}": float((3 * i * i + 1) % (high + 1))
                    for i in range(lead_time + 1)
                },
            )
            for raw in raw_vectors:
                canonical = canonicalize_params(raw, spec.parameter_specs)
                raw_policy = compile_policy(spec.ast, raw, motifs=spec.motifs, action_cap=n)
                canonical_policy = compile_policy(
                    spec.ast, canonical, motifs=spec.motifs, action_cap=n
                )
                for values in itertools.product(range(n + 1), repeat=lead_time + 1):
                    state = {"x": values[0], "p": values[1:]}
                    self.assertEqual(
                        raw_policy(state, {}, {}),
                        canonical_policy(state, {}, {}),
                        msg=f"L={lead_time} raw={raw} state={values}",
                    )

    def test_raw_and_canonical_exact_costs_match(self) -> None:
        demand = DiscreteDemand.from_pmf({0: 0.35, 1: 0.4, 3: 0.25})
        for lead_time in range(1, 6):
            spec = self._spec(lead_time, 3)
            high = (lead_time + 1) * 3
            reference = solve_dp(lead_time, 8, 3, demand, 1.0, 4.0)
            evaluator = SparseExactPolicyCostEvaluator(
                spec.ast, reference, 20, spec.motifs, max_cycle_period=0
            )
            raw = {
                f"S{i}": float((high - 2 * i) % (high + 1))
                for i in range(lead_time + 1)
            }
            canonical = canonicalize_params(raw, spec.parameter_specs)
            raw_result = evaluator.evaluate(raw)
            canonical_result = evaluator.evaluate(canonical)
            self.assertTrue(raw_result.completed)
            self.assertTrue(canonical_result.completed)
            self.assertAlmostEqual(raw_result.cost, canonical_result.cost, places=10)

    def test_specialized_and_generic_exact_costs_match(self) -> None:
        demand = DiscreteDemand.from_pmf({0: 0.35, 1: 0.4, 3: 0.25})
        for lead_time in range(1, 6):
            spec = self._spec(lead_time, 3)
            reference = solve_dp(lead_time, 8, 3, demand, 1.0, 4.0)
            specialized = SparseExactPolicyCostEvaluator(
                spec.ast, reference, 20, spec.motifs, max_cycle_period=0
            )
            generic = SparseExactPolicyCostEvaluator(
                spec.ast, reference, 20, spec.motifs, max_cycle_period=0
            )
            generic._is_vector_base_stock = False
            params = {f"S{i}": 3.0 for i in range(lead_time + 1)}
            specialized_result = specialized.evaluate(params)
            generic_result = generic.evaluate(params)
            self.assertTrue(specialized_result.completed)
            self.assertTrue(generic_result.completed)
            self.assertAlmostEqual(
                specialized_result.cost, generic_result.cost, places=12
            )

    def test_canonical_keys_remove_redundant_vbs_vectors(self) -> None:
        # On a five-value toy domain, 5^5 raw vectors reduce to the
        # C(5+5-1, 5)=126 nondecreasing representatives.
        spec = self._spec(4, 1)
        keys: set[tuple[object, ...]] = set()
        for values in itertools.product(range(5), repeat=5):
            raw = {f"S{i}": float(value) for i, value in enumerate(values)}
            canonical = canonicalize_params(raw, spec.parameter_specs)
            keys.add(canonical_parameter_key(canonical, spec.parameter_specs))
        self.assertEqual(len(keys), 126)
        self.assertGreater(3125 / len(keys), 24.0)


if __name__ == "__main__":
    unittest.main()
