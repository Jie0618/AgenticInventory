from __future__ import annotations

import unittest

from inventory.core import (
    BacklogConfig,
    DeterministicBacklog,
    DeterministicLeadTime,
    FixedShelfLife,
    FullBacklog,
    GeometricBacklog,
    InventoryModelConfig,
    LostSales,
    NoBacklog,
    state_schema_for_config,
)
from inventory.policy.ast import ast_check, evaluate_ast
from inventory.policy.motifs import backlog_motifs, motifs_for_config


def _value(ast, state, *, mu=4.0):
    return evaluate_ast(ast, state, dist_features={"mu": mu, "cv": 0.5})


class BacklogMotifTests(unittest.TestCase):
    def test_disabled_and_lost_sales_have_no_backlog_motifs(self) -> None:
        self.assertEqual(backlog_motifs(NoBacklog()), {})
        self.assertEqual(backlog_motifs(LostSales()), {})

    def test_full_backlog_uses_scalar_state(self) -> None:
        motifs = backlog_motifs(FullBacklog())
        state = {"B": 6.0}
        self.assertEqual(_value(motifs["total_backlog"], state), 6.0)
        self.assertEqual(
            _value(motifs["expected_retained_backlog_if_unserved"], state), 6.0
        )
        self.assertEqual(
            _value(motifs["expected_abandoned_backlog_if_unserved"], state), 0.0
        )
        self.assertAlmostEqual(_value(motifs["backlog_to_mean_demand"], state), 1.2)

    def test_geometric_backlog_exposes_conditional_survival_pressure(self) -> None:
        motifs = backlog_motifs(GeometricBacklog(0.75))
        state = {"B": 8.0}
        self.assertEqual(
            _value(motifs["expected_retained_backlog_if_unserved"], state), 6.0
        )
        self.assertEqual(
            _value(motifs["expected_abandoned_backlog_if_unserved"], state), 2.0
        )

    def test_deterministic_backlog_uses_youngest_to_oldest_cohorts(self) -> None:
        motifs = backlog_motifs(DeterministicBacklog(3))
        state = {"b1": 2.0, "b2": 3.0, "b3": 5.0}
        expected = {
            "total_backlog": 10.0,
            "youngest_backlog": 2.0,
            "oldest_backlog": 5.0,
            "imminent_abandonment_if_unserved": 5.0,
            "retained_backlog_if_unserved": 5.0,
            "backlog_age_moment": 23.0,
            # safe_ratio is deliberately x / (1 + abs(y)).
            "mean_backlog_age": 23.0 / 11.0,
        }
        for name, expected_value in expected.items():
            with self.subTest(motif=name):
                self.assertAlmostEqual(_value(motifs[name], state), expected_value)

    def test_invalid_config_type_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            backlog_motifs("full")  # type: ignore[arg-type]


class ComposedMotifTests(unittest.TestCase):
    def test_every_composition_matches_its_state_schema(self) -> None:
        configs = (
            InventoryModelConfig(backlog=FullBacklog()),
            InventoryModelConfig(backlog=GeometricBacklog(0.8)),
            InventoryModelConfig(backlog=DeterministicBacklog(3)),
            InventoryModelConfig(
                lead_time=DeterministicLeadTime(2), backlog=FullBacklog()
            ),
            InventoryModelConfig(
                perishability=FixedShelfLife(3), backlog=DeterministicBacklog(2)
            ),
            InventoryModelConfig(
                lead_time=DeterministicLeadTime(2),
                perishability=FixedShelfLife(3),
                backlog=GeometricBacklog(0.8),
            ),
            InventoryModelConfig(
                lead_time=DeterministicLeadTime(0), backlog=LostSales()
            ),
        )
        for config in configs:
            with self.subTest(composition=config.composition_key):
                schema = state_schema_for_config(config)
                motifs = motifs_for_config(config)
                for name, ast in motifs.items():
                    with self.subTest(motif=name):
                        ast_check(
                            ast,
                            motifs=motifs,
                            allowed_state_names=schema.allowed_state_names,
                        )

    def test_lead_time_and_backlog_cross_component_values(self) -> None:
        config = InventoryModelConfig(
            lead_time=DeterministicLeadTime(2), backlog=FullBacklog()
        )
        motifs = motifs_for_config(config)
        state = {"x": 7.0, "p1": 2.0, "p2": 4.0, "B": 5.0}
        self.assertEqual(_value(motifs["inventory_position"], state), 13.0)
        self.assertEqual(_value(motifs["net_inventory_position"], state), 8.0)
        self.assertEqual(_value(motifs["available_after_backlog"], state), 4.0)
        self.assertEqual(_value(motifs["backlog_coverage"], state), 1.5)

    def test_perishable_and_backlog_cross_component_values(self) -> None:
        config = InventoryModelConfig(
            perishability=FixedShelfLife(3), backlog=DeterministicBacklog(2)
        )
        motifs = motifs_for_config(config)
        state = {"a1": 2.0, "a2": 3.0, "a3": 5.0, "b1": 1.0, "b2": 3.0}
        self.assertEqual(_value(motifs["inventory_position"], state), 10.0)
        self.assertEqual(_value(motifs["net_inventory_position"], state), 6.0)
        self.assertEqual(
            _value(motifs["effective_net_inventory_position"], state), 4.0
        )
        self.assertEqual(_value(motifs["available_after_backlog"], state), 6.0)

    def test_perishable_exposes_decision_inventory_second_moment(self) -> None:
        config = InventoryModelConfig(perishability=FixedShelfLife(3))
        motifs = motifs_for_config(config)
        state = {"a1": 2.0, "a2": 3.0, "a3": 5.0}
        # The temporary receipt bucket a3 is excluded from the pre-order
        # decision summary, matching decision_inventory and decision_age_moment.
        self.assertEqual(
            _value(motifs["decision_inventory_second_moment"], state), 13.0
        )

    def test_lead_time_perishable_composes_component_and_cross_motifs(self) -> None:
        config = InventoryModelConfig(
            lead_time=DeterministicLeadTime(2),
            perishability=FixedShelfLife(3),
        )
        motifs = motifs_for_config(config)
        state = {
            "a1": 2.0,
            "a2": 3.0,
            "a3": 5.0,
            "p1": 7.0,
            "p2": 11.0,
        }
        # Component-local summaries are merged.
        self.assertEqual(_value(motifs["total_pipeline"], state), 18.0)
        self.assertEqual(_value(motifs["expiring_inventory"], state), 2.0)
        # In a positive-lead-time composition a3 is real carried stock, not
        # the temporary immediate-replenishment receipt bucket.
        self.assertEqual(_value(motifs["decision_inventory"], state), 10.0)
        self.assertEqual(
            _value(motifs["decision_inventory_second_moment"], state), 38.0
        )
        # Composition-specific summaries combine on-hand ages and pipeline.
        self.assertEqual(_value(motifs["available_inventory"], state), 17.0)
        self.assertEqual(_value(motifs["inventory_position"], state), 28.0)
        self.assertEqual(
            _value(motifs["effective_inventory_position"], state), 26.0
        )
        self.assertEqual(_value(motifs["cumulative_inventory_1"], state), 17.0)
        self.assertEqual(_value(motifs["cumulative_inventory_2"], state), 28.0)

    def test_zero_period_lead_time_adds_no_pipeline_motifs(self) -> None:
        config = InventoryModelConfig(
            lead_time=DeterministicLeadTime(0), backlog=BacklogConfig.full()
        )
        motifs = motifs_for_config(config)
        self.assertNotIn("total_pipeline", motifs)
        self.assertNotIn("near_pipeline", motifs)


if __name__ == "__main__":
    unittest.main()
