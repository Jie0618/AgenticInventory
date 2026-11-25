from __future__ import annotations

import unittest

from optimization.policies.baselines.lead_time import baselines_for_lead_time
from inventory.policy.ast import ast_check, compile_policy


FORBIDDEN_OPS = {
    "expectation",
    "quantile",
    "lead_time_quantile",
    "projected_inventory",
    "projected_shortage",
    "projected_service_probability",
    "threshold_search",
    "projected_service_threshold",
    "myopic2_action",
    "indicator_leq",
    "indicator_geq",
    "positive_part",
}


def operators(node):
    if isinstance(node, dict):
        if "op" in node:
            yield node["op"]
        for value in node.values():
            yield from operators(value)
    elif isinstance(node, list):
        for value in node:
            yield from operators(value)


class SimpleBaselineTests(unittest.TestCase):
    def test_catalog_contains_only_simple_families(self) -> None:
        names = [spec.name for spec in baselines_for_lead_time(3, 10)]
        self.assertEqual(
            names,
            [
                "constant_order",
                "base_stock",
                "capped_base_stock",
                "vector_base_stock",
                "tilted_cbs",
                "tilted_pic",
                "weighted_tilted_pic",
                "aips_discounted_pipeline",
                "aips_hybrid",
            ],
        )

    def test_every_baseline_is_valid_and_expectation_free(self) -> None:
        for lead_time in range(1, 10):
            for spec in baselines_for_lead_time(lead_time, 10):
                ast_check(spec.ast, motifs=spec.motifs)
                self.assertFalse(FORBIDDEN_OPS.intersection(operators(spec.ast)))
                self.assertFalse(
                    any(FORBIDDEN_OPS.intersection(operators(ast)) for ast in spec.motifs.values())
                )

    def test_weighted_tilted_pic_action(self) -> None:
        spec = next(
            row
            for row in baselines_for_lead_time(2, 10)
            if row.name == "weighted_tilted_pic"
        )
        policy = compile_policy(
            spec.ast,
            params={"S": 10, "K": 0.8, "C": 2, "alpha": 0.2, "w1": 1, "w2": 0.5},
            motifs=spec.motifs,
            action_cap=10,
        )
        # Weighted inventory=1+2+0.5*4=5; gap=5; min(4,3)=3.
        self.assertEqual(policy({"x": 1, "p": [2, 4]}, {}, {}), 3)

    def test_aips_discounted_pipeline_dead_zone_is_strict(self) -> None:
        spec = next(
            row
            for row in baselines_for_lead_time(2, 10)
            if row.name == "aips_discounted_pipeline"
        )
        state = {"x": 1, "p": [2, 4]}
        common = {"S": 10, "omega": 0.5, "rho": 0.5}
        active = compile_policy(
            spec.ast,
            {**common, "tau": 5},
            motifs=spec.motifs,
            action_cap=10,
        )
        boundary = compile_policy(
            spec.ast,
            {**common, "tau": 6},
            motifs=spec.motifs,
            action_cap=10,
        )
        self.assertEqual(active(state, {}, {}), 3)
        self.assertEqual(boundary(state, {}, {}), 0)

    def test_aips_hybrid_behavior_equivalent_parameterization(self) -> None:
        spec = next(
            row
            for row in baselines_for_lead_time(2, 10)
            if row.name == "aips_hybrid"
        )
        policy = compile_policy(
            spec.ast,
            {
                "S": 10,
                "v1": 1.0,
                "v2": 0.5,
                "rho": 0.8,
                "delta": 2,
                "a_min": 3,
            },
            motifs=spec.motifs,
            action_cap=10,
        )
        self.assertEqual(policy({"x": 1, "p": [2, 4]}, {}, {}), 4)

    def test_vector_base_stock_contains_base_stock_for_arbitrary_l(self) -> None:
        for lead_time in (1, 4, 9):
            base_stock_spec = baselines_for_lead_time(lead_time, 10)[1]
            vector_spec = baselines_for_lead_time(lead_time, 10)[3]
            level = 7.0
            base_policy = compile_policy(
                base_stock_spec.ast,
                {"S": level},
                motifs=base_stock_spec.motifs,
                action_cap=10,
            )
            vector_policy = compile_policy(
                vector_spec.ast,
                {f"S{index}": level for index in range(lead_time + 1)},
                motifs=vector_spec.motifs,
                action_cap=10,
            )
            state = {"x": 1, "p": [index % 3 for index in range(1, lead_time + 1)]}
            self.assertEqual(base_policy(state, {}, {}), vector_policy(state, {}, {}))

    def test_all_documented_family_embeddings_match_actions(self) -> None:
        for lead_time in (1, 3, 5):
            specs = {spec.name: spec for spec in baselines_for_lead_time(lead_time, 10)}

            def policy(name, params):
                spec = specs[name]
                return compile_policy(spec.ast, params, motifs=spec.motifs, action_cap=10)

            comparisons = [
                (
                    policy("base_stock", {"S": 12}),
                    policy("capped_base_stock", {"S": 12, "C": 10}),
                ),
                (
                    policy("base_stock", {"S": 12}),
                    policy(
                        "vector_base_stock",
                        {f"S{index}": 12 for index in range(lead_time + 1)},
                    ),
                ),
                (
                    policy("capped_base_stock", {"S": 12, "C": 3}),
                    policy("tilted_cbs", {"S": 12, "C": 3, "alpha": 0}),
                ),
                (
                    policy("tilted_cbs", {"S": 12, "C": 3, "alpha": 0.25}),
                    policy(
                        "tilted_pic",
                        {"S": 12, "C": 3, "alpha": 0.25, "K": 1},
                    ),
                ),
                (
                    policy(
                        "tilted_pic",
                        {"S": 12, "C": 3, "alpha": 0.25, "K": 0.8},
                    ),
                    policy(
                        "weighted_tilted_pic",
                        {
                            "S": 12,
                            "C": 3,
                            "alpha": 0.25,
                            "K": 0.8,
                            **{f"w{index}": 1 for index in range(1, lead_time + 1)},
                        },
                    ),
                ),
            ]
            states = [
                {"x": x, "p": [(x + 2 * index) % 5 for index in range(1, lead_time + 1)]}
                for x in range(7)
            ]
            for parent, child in comparisons:
                for state in states:
                    self.assertEqual(parent(state, {}, {}), child(state, {}, {}))


if __name__ == "__main__":
    unittest.main()
