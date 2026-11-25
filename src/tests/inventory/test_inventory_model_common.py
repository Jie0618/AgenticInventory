from __future__ import annotations

import unittest

from inventory.policy.ast import ASTValidationError, ast_check, state
from evaluation.backends.lead_time.models import FlatStateSchema


class InventoryModelCommonTests(unittest.TestCase):
    def test_state_schema_maps_coordinates_and_remaining_horizon(self) -> None:
        schema = FlatStateSchema("perishable", ("a1", "a2"))
        self.assertEqual(
            schema.mapping((2, 3), 7),
            {"a1": 2.0, "a2": 3.0, "remaining_horizon": 7.0},
        )
        with self.assertRaisesRegex(ValueError, "expects 2"):
            schema.mapping((1,), 7)

    def test_ast_state_schema_is_opt_in_and_default_remains_lead_time(self) -> None:
        with self.assertRaisesRegex(ASTValidationError, "unknown state feature"):
            ast_check(state("a1"))
        checked = ast_check(
            state("a1"), allowed_state_names={"a1", "a2", "remaining_horizon"}
        )
        self.assertEqual(checked.ast, state("a1"))
        ast_check(state("x"))
        ast_check(state("p12"))

if __name__ == "__main__":
    unittest.main()
