from __future__ import annotations

import unittest

from inventory.policy.ast import (
    ASTValidationError,
    add,
    ast_check,
    ast_hash,
    const,
    evaluate_ast,
    maximum,
    minimum,
    motif,
    mul,
    neg,
    parameter,
    state,
    sub,
)
from inventory.policy.motifs.lead_time import lead_time_motifs


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


class SimpleOperationalFormulaASTTests(unittest.TestCase):
    def test_base_stock_formula(self) -> None:
        motifs = lead_time_motifs(2)
        ast = maximum(add(parameter("S"), neg(motif("inventory_position"))), const(0.0))
        checked = ast_check(ast, motifs=motifs)
        self.assertEqual(
            evaluate_ast(checked.ast, (1, 2, 3), {"S": 10}, motifs=motifs), 4.0
        )

    def test_capped_formula(self) -> None:
        motifs = lead_time_motifs(1)
        gap = maximum(add(parameter("S"), neg(motif("inventory_position"))), const(0.0))
        ast = minimum(gap, parameter("C"))
        self.assertEqual(
            evaluate_ast(ast, (1, 1), {"S": 9, "C": 3}, motifs=motifs), 3.0
        )

    def test_canonicalization_removes_additive_identities_and_inverse_pairs(self) -> None:
        raw = add(
            state("x"),
            const(0.0),
            state("p1"),
            neg(state("p1")),
            sub(parameter("S"), parameter("S")),
        )
        checked = ast_check(raw)
        self.assertEqual(checked.ast, state("x"))
        self.assertEqual(checked.stats.parameters, ())

    def test_canonicalization_removes_multiplicative_identities(self) -> None:
        self.assertEqual(ast_check(mul(const(1.0), state("x"))).ast, state("x"))
        self.assertEqual(
            ast_check(mul(parameter("alpha"), const(0.0), state("x"))).ast,
            const(0.0),
        )

    def test_canonicalization_simplifies_local_unary_and_branch_identities(self) -> None:
        self.assertEqual(ast_check(neg(neg(state("x")))).ast, state("x"))
        self.assertEqual(ast_check(minimum(state("x"), state("x"))).ast, state("x"))
        self.assertEqual(ast_check(maximum(state("x"), state("x"))).ast, state("x"))

    def test_root_nonnegative_wrapper_has_one_policy_identity(self) -> None:
        policy = add(parameter("S"), neg(state("x")))
        plain = ast_check(policy)
        right_wrapped = ast_check(maximum(policy, const(0.0)))
        left_wrapped = ast_check(maximum(const(0.0), policy))
        self.assertEqual(right_wrapped.ast, plain.ast)
        self.assertEqual(left_wrapped.ast, plain.ast)
        self.assertEqual(right_wrapped.ast_hash, plain.ast_hash)
        self.assertEqual(left_wrapped.ast_hash, plain.ast_hash)
        self.assertEqual(ast_hash(maximum(policy, const(0.0))), ast_hash(policy))
        self.assertEqual(ast_hash(maximum(const(0.0), policy)), ast_hash(policy))

    def test_commutative_branch_operators_have_one_identity(self) -> None:
        left = add(parameter("S"), neg(state("x")))
        right = parameter("C")
        self.assertEqual(ast_hash(maximum(left, right)), ast_hash(maximum(right, left)))
        self.assertEqual(ast_hash(minimum(left, right)), ast_hash(minimum(right, left)))

    def test_nested_nonnegative_wrapper_remains_semantic(self) -> None:
        nested = add(maximum(state("x"), const(0.0)), parameter("S"))
        self.assertEqual(ast_check(nested).ast["args"][0]["op"], "maximum")

    def test_canonicalization_is_idempotent_and_preserves_evaluation(self) -> None:
        raw = maximum(
            add(parameter("S"), state("p1"), neg(state("p1")), const(2.0), const(-2.0)),
            const(0.0),
        )
        checked = ast_check(raw)
        self.assertEqual(ast_check(checked.ast).ast, checked.ast)
        for state_value in ((0, 0), (3, 7), (10, 2)):
            self.assertEqual(
                evaluate_ast(raw, state_value, {"S": 4.0}),
                evaluate_ast(checked.ast, state_value, {"S": 4.0}),
            )

    def test_expectation_is_outside_public_grammar(self) -> None:
        with self.assertRaises(ASTValidationError):
            ast_check({"op": "expectation", "vars": ["D"], "body": {"op": "const", "value": 0.0}})

    def test_demand_support_is_outside_public_grammar(self) -> None:
        for name in ("support_min", "support_max", "d_max"):
            terminal = {"op": "distribution", "name": name}
            with self.subTest(name=name):
                with self.assertRaisesRegex(ASTValidationError, "unknown distribution summary"):
                    ast_check(terminal)
                with self.assertRaisesRegex(ASTValidationError, "unknown distribution summary"):
                    ast_check(motif("cutoff"), motifs={"cutoff": terminal})
                for features in ({name: 1.0}, {"values": [0, 1], "probs": [0.5, 0.5]}):
                    with self.assertRaisesRegex(ASTValidationError, "unknown distribution summary"):
                        evaluate_ast(terminal, (0,), dist_features=features)
        ast_check({"op": "distribution", "name": "mu"})

    def test_redundant_piecewise_operators_are_outside_grammar(self) -> None:
        for node in (
            {"op": "positive_part", "arg": const(1.0)},
            {"op": "indicator_leq", "left": const(0.0), "right": const(1.0)},
            {"op": "indicator_geq", "left": const(1.0), "right": const(0.0)},
        ):
            with self.assertRaises(ASTValidationError):
                ast_check(node)

    def test_pipeline_imbalance_motif(self) -> None:
        motifs = lead_time_motifs(3)
        self.assertEqual(
            evaluate_ast(motif("pipeline_imbalance"), (0, 5, 2, 1), motifs=motifs),
            4.0,
        )

    def test_raw_pipeline_lag_moment_motif(self) -> None:
        motifs = lead_time_motifs(4)
        self.assertEqual(
            evaluate_ast(motif("pipeline_lag_moment"), (7, 5, 2, 1, 3), motifs=motifs),
            24.0,
        )

    def test_motifs_are_expectation_free(self) -> None:
        for lead_time in (1, 2, 5):
            for ast in lead_time_motifs(lead_time).values():
                self.assertFalse(FORBIDDEN_OPS.intersection(operators(ast)))
                ast_check(ast, motifs=lead_time_motifs(lead_time))


if __name__ == "__main__":
    unittest.main()
