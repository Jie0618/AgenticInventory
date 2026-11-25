"""Focused tests for the compositional inventory state/config contract."""

from __future__ import annotations

import unittest

import numpy as np

from inventory.common import DiscreteDemand
from inventory.core import (
    BacklogConfig,
    BacklogMode,
    CanonicalStateSchema,
    DeterministicLeadTime,
    FixedShelfLife,
    GeometricBacklog,
    InventoryModelConfig,
    InventoryState,
    KnownArrivalShelfLife,
    KnownEtaLeadTime,
    LeadTimeConfig,
    LeadTimeMode,
    PerishabilityMode,
    TransitionMode,
    TransitionSpec,
    build_state_schema,
    build_transition_kernel,
    composition_identity,
)


class InventoryCoreContractTests(unittest.TestCase):
    def test_root_api_uses_only_canonical_state_and_outcome(self) -> None:
        from inventory import (
            CanonicalStateSchema,
            CanonicalTransitionOutcome,
            StateSchema,
            TransitionOutcome,
        )
        from inventory.core import TransitionOutcome as CoreTransitionOutcome
        from inventory.core import StateSchema as CoreStateSchema

        self.assertIs(TransitionOutcome, CoreTransitionOutcome)
        self.assertIs(CanonicalTransitionOutcome, CoreTransitionOutcome)
        self.assertIs(StateSchema, CanonicalStateSchema)
        self.assertIs(CoreStateSchema, CanonicalStateSchema)

    def test_components_can_be_selected_independently(self) -> None:
        lead_only = InventoryModelConfig(lead_time=DeterministicLeadTime(3))
        lead_schema = build_state_schema(lead_only)
        self.assertEqual(lead_schema.names, ("x", "p1", "p2", "p3"))
        self.assertEqual(lead_schema.component_names["backlog"], ())

        perish_backlog = InventoryModelConfig(
            perishability=FixedShelfLife(2),
            backlog=GeometricBacklog(0.8),
        )
        schema = build_state_schema(perish_backlog)
        self.assertEqual(schema.names, ("a1", "a2", "B"))
        self.assertEqual(schema.inventory_dimension, 2)
        self.assertEqual(schema.pipeline_dimension, 0)
        self.assertEqual(schema.backlog_dimension, 1)

    def test_known_eta_support_uses_countdown_buckets_and_keeps_zero_immediate(
        self,
    ) -> None:
        config = InventoryModelConfig(
            lead_time=KnownEtaLeadTime((0, 2, 4), (0.2, 0.3, 0.5))
        )
        schema = build_state_schema(config)
        self.assertEqual(schema.names, ("x", "p1", "p2", "p3", "p4"))
        self.assertEqual(config.lead_time.mode, LeadTimeMode.KNOWN_ETA_STOCHASTIC)
        self.assertEqual(config.transition_mode, TransitionMode.KNOWN_EVENT_STOCHASTIC)

    def test_zero_probability_points_are_removed_from_stochastic_support(self) -> None:
        lead = KnownEtaLeadTime((0, 2, 9), (0.25, 0.75, 0.0))
        shelf_life = KnownArrivalShelfLife((1, 3, 8), (0.4, 0.6, 0.0))
        config = InventoryModelConfig(
            lead_time=lead,
            perishability=shelf_life,
        )

        self.assertEqual(lead.support, (0, 2))
        self.assertEqual(lead.probabilities, (0.25, 0.75))
        self.assertEqual(shelf_life.lifetime_support, (1, 3))
        self.assertEqual(shelf_life.lifetime_probabilities, (0.4, 0.6))
        self.assertEqual(build_state_schema(config).pipeline_dimension, 2)
        self.assertEqual(build_state_schema(config).inventory_dimension, 3)
        self.assertEqual(
            composition_identity(config),
            composition_identity(
                InventoryModelConfig(
                    lead_time=KnownEtaLeadTime((0, 2), (0.25, 0.75)),
                    perishability=KnownArrivalShelfLife((1, 3), (0.4, 0.6)),
                )
            ),
        )

    def test_state_round_trip_and_ast_mapping_share_coordinate_order(self) -> None:
        config = InventoryModelConfig(
            lead_time=DeterministicLeadTime(2),
            perishability=FixedShelfLife(3),
            backlog=BacklogConfig.deterministic(2),
            capacity=10,
        )
        schema = build_state_schema(config)
        state = InventoryState((1, 2, 3), (4, 5), (6, 0))
        flat = state.flatten(schema)
        self.assertEqual(flat, (1, 2, 3, 4, 5, 6, 0))
        self.assertEqual(InventoryState.from_flat(flat, schema), state)
        self.assertEqual(schema.mapping(flat, 7)["a1"], 1.0)
        self.assertEqual(schema.mapping(flat, 7)["p2"], 5.0)
        self.assertEqual(schema.mapping(flat, 7)["b1"], 6.0)
        self.assertEqual(schema.mapping(flat, 7)["remaining_horizon"], 7.0)

    def test_identity_changes_when_component_semantics_change(self) -> None:
        deterministic = InventoryModelConfig(
            lead_time=DeterministicLeadTime(2),
        )
        stochastic = InventoryModelConfig(
            lead_time=KnownEtaLeadTime((2,), (1.0,)),
        )
        self.assertNotEqual(
            composition_identity(deterministic), composition_identity(stochastic)
        )
        self.assertEqual(
            deterministic.composition_hash, composition_identity(deterministic)
        )
        self.assertIn("lead_time", deterministic.composition_id)

    def test_invalid_or_ambiguous_component_parameters_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DeterministicLeadTime(-1)
        with self.assertRaises(ValueError):
            KnownEtaLeadTime((1, 2), (1.0,))
        with self.assertRaises(ValueError):
            KnownEtaLeadTime((2, 1), (0.5, 0.5))
        with self.assertRaises(ValueError):
            FixedShelfLife(0)
        with self.assertRaises(ValueError):
            GeometricBacklog(1.1)
        with self.assertRaises(ValueError):
            InventoryState.from_flat((1.5,), build_state_schema(InventoryModelConfig()))

        boolean_quantities = (
            lambda: DeterministicLeadTime(np.bool_(True)),
            lambda: FixedShelfLife(np.bool_(True)),
            lambda: BacklogConfig.deterministic(np.bool_(True)),
            lambda: KnownEtaLeadTime((np.bool_(True),), (1.0,)),
            lambda: InventoryModelConfig(capacity=np.bool_(True)),
            lambda: InventoryState((np.bool_(True),), (), ()),
        )
        for construct in boolean_quantities:
            with self.subTest(construct=construct):
                with self.assertRaisesRegex(ValueError, "integer"):
                    construct()

    def test_integer_contract_rejects_arrays_and_nonfinite_values(self) -> None:
        for invalid in (np.array(1), np.array([1]), float("inf")):
            constructors = (
                lambda value=invalid: DeterministicLeadTime(value),
                lambda value=invalid: FixedShelfLife(value),
                lambda value=invalid: BacklogConfig.deterministic(value),
                lambda value=invalid: KnownEtaLeadTime((value,), (1.0,)),
                lambda value=invalid: InventoryModelConfig(capacity=value),
                lambda value=invalid: InventoryState((value,), (), ()),
            )
            for construct in constructors:
                with self.subTest(invalid=repr(invalid), construct=construct):
                    with self.assertRaisesRegex(ValueError, "integer"):
                        construct()

    def test_public_schema_does_not_truncate_support_metadata(self) -> None:
        arguments = {
            "model_id": "manual",
            "names": ("x",),
            "inventory_names": ("x",),
            "pipeline_names": (),
            "backlog_names": (),
        }
        for field, invalid in (
            ("lead_time_support", 1.5),
            ("lead_time_support", np.bool_(True)),
            ("lead_time_support", np.array([1])),
            ("lifetime_support", 1.5),
        ):
            with self.subTest(field=field, invalid=repr(invalid)):
                with self.assertRaisesRegex(ValueError, "integers"):
                    CanonicalStateSchema(**arguments, **{field: (invalid,)})

    def test_discrete_demand_rejects_noninteger_support_before_casting(self) -> None:
        for invalid in (
            1.5,
            True,
            np.bool_(True),
            np.array(1),
            np.array([1]),
            float("inf"),
        ):
            with self.subTest(invalid=repr(invalid)):
                with self.assertRaisesRegex(ValueError, "integer"):
                    DiscreteDemand([invalid], [1.0])

        for pmf in ({1.5: 1.0}, {True: 1.0}):
            with self.subTest(pmf=pmf):
                with self.assertRaisesRegex(ValueError, "integer"):
                    DiscreteDemand.from_pmf(pmf)

        self.assertEqual(DiscreteDemand([1.0], [1.0]).support_min, 1)

    def test_lost_sales_and_disabled_backlog_have_same_shape_but_distinct_identity(
        self,
    ) -> None:
        disabled = InventoryModelConfig()
        lost_sales = InventoryModelConfig(backlog=BacklogConfig.lost_sales())
        self.assertEqual(
            build_state_schema(disabled).names, build_state_schema(lost_sales).names
        )
        self.assertNotEqual(disabled.composition_hash, lost_sales.composition_hash)
        self.assertEqual(lost_sales.backlog.mode, BacklogMode.LOST_SALES)
        self.assertEqual(
            build_state_schema(lost_sales).backlog_mode, BacklogMode.LOST_SALES
        )
        self.assertEqual(
            build_state_schema(lost_sales).perishability_mode,
            PerishabilityMode.DISABLED,
        )

    def test_transition_rejects_an_unimplemented_event_order(self) -> None:
        with self.assertRaisesRegex(ValueError, "fixed event order"):
            TransitionSpec(
                TransitionMode.DETERMINISTIC,
                event_order=("choose_action", "observe_state"),
            )

    def test_transition_rejects_schema_from_another_configuration(self) -> None:
        config = InventoryModelConfig(lead_time=DeterministicLeadTime(1))
        wrong_schema = build_state_schema(
            InventoryModelConfig(lead_time=DeterministicLeadTime(2))
        )
        with self.assertRaisesRegex(ValueError, "schema does not match"):
            build_transition_kernel(config, schema=wrong_schema)

    def test_convenience_and_base_component_values_normalize_equally(self) -> None:
        convenience = InventoryModelConfig(lead_time=DeterministicLeadTime(2))
        base = InventoryModelConfig(lead_time=LeadTimeConfig.deterministic(2))
        self.assertEqual(convenience, base)
        self.assertEqual(convenience.composition_hash, base.composition_hash)


if __name__ == "__main__":
    unittest.main()
