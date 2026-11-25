"""Hardening tests for canonical exact and Monte Carlo evaluation."""

from __future__ import annotations

import unittest

import numpy as np

from inventory.common.distributions import DiscreteDemand
from inventory.core import (
    BacklogConfig,
    DeterministicLeadTime,
    InventoryCase,
    InventoryModelConfig,
    InventoryState,
    KnownEtaLeadTime,
    TransitionCosts,
    build_transition_kernel,
    state_schema_for_config,
)
from inventory.core.transition import KnownEventStochasticTransitionKernel
from evaluation.exact_mc import evaluate_composed_policy_exact, evaluate_composed_policy_mc


ZERO_DEMAND = DiscreteDemand.from_pmf({0: 1.0}, name="zero")


def _case() -> InventoryCase:
    return InventoryCase(
        "evaluation-hardening",
        InventoryModelConfig(
            lead_time=DeterministicLeadTime(1),
            capacity=2,
        ),
        TransitionCosts(holding=1.0, shortage=4.0),
        ZERO_DEMAND,
        2,
    )


class InventoryCoreEvaluationTests(unittest.TestCase):
    def test_exact_and_mc_report_cumulative_projection_discard(self) -> None:
        case = InventoryCase(
            "projection-diagnostics",
            InventoryModelConfig(capacity=1),
            TransitionCosts(),
            ZERO_DEMAND,
            2,
        )
        policy = lambda _state, _remaining: 2

        exact = evaluate_composed_policy_exact(case, policy, 2)
        self.assertEqual(exact.metadata["expected_cumulative_discarded"], 3.0)
        self.assertEqual(exact.metadata["projection_capacity"], 1)

        mc = evaluate_composed_policy_mc(case, policy, 2, paths=25, seed=11)
        self.assertEqual(mc.metadata["mean_cumulative_discarded"], 3.0)
        self.assertEqual(mc.metadata["discarded_standard_error"], 0.0)
        self.assertEqual(mc.metadata["projection_capacity"], 1)

    def test_all_transition_cost_components_are_exposed_and_accounted(self) -> None:
        case = InventoryCase(
            "all-cost-components",
            InventoryModelConfig(
                backlog=BacklogConfig.geometric(1.0),
                capacity=2,
            ),
            TransitionCosts(
                holding=2.0,
                shortage=3.0,
                expiry=5.0,
                backlog=7.0,
                abandonment=11.0,
                ordering=13.0,
            ),
            DiscreteDemand.from_pmf({1: 1.0}, name="unit-demand"),
            2,
        )
        self.assertEqual((case.h, case.b, case.w, case.g, case.a, case.k),
                         (2.0, 3.0, 5.0, 7.0, 11.0, 13.0))
        kernel = build_transition_kernel(case.config, case.costs, action_cap=case.action_cap)
        outcome = kernel.transition(case.initial_state, 1, 1, rng=np.random.default_rng(3))
        self.assertEqual(outcome.total_cost, sum((outcome.holding_cost,
                                                   outcome.shortage_cost,
                                                   outcome.expiry_cost,
                                                   outcome.backlog_cost,
                                                   outcome.abandonment_cost,
                                                   outcome.ordering_cost)))

    def test_cost_vector_aliases_and_case_identity_cover_all_components(self) -> None:
        base = InventoryCase(
            "cost-identity",
            InventoryModelConfig(capacity=2),
            TransitionCosts.from_mapping({"h": 1, "b": 2, "w": 3, "g": 4, "a": 5, "k": 6}),
            ZERO_DEMAND,
            2,
        )
        self.assertEqual(base.costs, TransitionCosts(1, 2, 3, 4, 5, 6))
        for field in ("holding", "shortage", "expiry", "backlog", "abandonment", "ordering"):
            values = {name: getattr(base.costs, name) for name in ("holding", "shortage", "expiry", "backlog", "abandonment", "ordering")}
            values[field] += 1.0
            changed = InventoryCase(
                "cost-identity",
                base.config,
                TransitionCosts(**values),
                ZERO_DEMAND,
                2,
            )
            with self.subTest(field=field):
                self.assertNotEqual(base.case_signature, changed.case_signature)

    def test_injected_kernel_must_match_the_complete_case_environment(self) -> None:
        case = _case()
        policy = lambda _state, _remaining: 0
        wrong_schema_kernel = build_transition_kernel(
            case.config,
            case.costs,
            schema=case.schema,
            action_cap=case.action_cap,
        )
        # Kernel construction protects this invariant too; mutate only to prove
        # that the evaluator does not trust an externally injected object.
        wrong_schema_kernel.schema = state_schema_for_config(
            case.config, model_id="wrong-model"
        )
        mismatches = {
            "configuration": build_transition_kernel(
                InventoryModelConfig(
                    lead_time=KnownEtaLeadTime((1,), (1.0,)),
                    capacity=2,
                ),
                case.costs,
                action_cap=case.action_cap,
            ),
            "costs": build_transition_kernel(
                case.config,
                TransitionCosts(holding=9.0, shortage=4.0),
                schema=case.schema,
                action_cap=case.action_cap,
            ),
            "schema": wrong_schema_kernel,
            "action cap": build_transition_kernel(
                case.config,
                case.costs,
                schema=case.schema,
                action_cap=case.action_cap + 1,
            ),
        }

        for label, kernel in mismatches.items():
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, label):
                    evaluate_composed_policy_exact(case, policy, 1, kernel=kernel)

    def test_exact_and_mc_reject_bool_and_fractional_policy_actions(self) -> None:
        case = _case()
        evaluators = (
            lambda policy: evaluate_composed_policy_exact(case, policy, 1),
            lambda policy: evaluate_composed_policy_mc(
                case, policy, 1, paths=1, seed=7
            ),
        )

        for evaluator in evaluators:
            for invalid in (
                True,
                1.5,
                np.array(1),
                np.array([1]),
                float("inf"),
            ):
                with self.subTest(evaluator=evaluator, invalid=invalid):
                    with self.assertRaisesRegex(ValueError, "must be an integer"):
                        evaluator(lambda _state, _remaining, value=invalid: value)

        # Integer-valued NumPy/Python scalar representations remain valid.
        result = evaluate_composed_policy_exact(case, lambda _state, _remaining: 1.0, 1)
        self.assertTrue(result.completed)

    def test_evaluators_reject_noninteger_control_arguments(self) -> None:
        case = _case()
        policy = lambda _state, _remaining: 0
        calls = (
            (
                "horizon",
                lambda value: evaluate_composed_policy_exact(case, policy, value),
            ),
            (
                "max_reachable_states",
                lambda value: evaluate_composed_policy_exact(
                    case, policy, 1, max_reachable_states=value
                ),
            ),
            (
                "horizon",
                lambda value: evaluate_composed_policy_mc(
                    case, policy, value, paths=1, seed=7
                ),
            ),
            (
                "paths",
                lambda value: evaluate_composed_policy_mc(
                    case, policy, 1, paths=value, seed=7
                ),
            ),
            (
                "seed",
                lambda value: evaluate_composed_policy_mc(
                    case, policy, 1, paths=1, seed=value
                ),
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

    def test_exact_skips_zero_probability_demand_and_current_states(self) -> None:
        demand = DiscreteDemand.from_pmf({0: 1.0, 1: 0.0}, name="zero-with-null")
        case = InventoryCase(
            "zero-demand-branch",
            InventoryModelConfig(backlog=BacklogConfig.full(), capacity=2),
            TransitionCosts(),
            demand,
            2,
        )

        def policy(state: InventoryState, _remaining: int) -> int:
            if state.total_backlog:
                raise AssertionError("policy called on a zero-probability state")
            return 0

        result = evaluate_composed_policy_exact(case, policy, 2, max_reachable_states=1)
        self.assertEqual(result.metadata["max_reachable_states"], 1)
        self.assertEqual(result.metadata["final_reachable_states"], 1)

    def test_exact_skips_zero_probability_transition_branches(self) -> None:
        case = InventoryCase(
            "zero-transition-branch",
            InventoryModelConfig(
                lead_time=KnownEtaLeadTime((0, 1), (1.0, 0.0)),
                capacity=2,
            ),
            TransitionCosts(),
            ZERO_DEMAND,
            2,
        )

        def policy(state: InventoryState, _remaining: int) -> int:
            if state.total_pipeline:
                raise AssertionError("policy called on a zero-probability state")
            return 1

        result = evaluate_composed_policy_exact(case, policy, 2, max_reachable_states=1)
        self.assertEqual(result.metadata["max_reachable_states"], 1)

    def test_reachable_limit_stops_branch_enumeration_incrementally(self) -> None:
        case = InventoryCase(
            "incremental-limit",
            InventoryModelConfig(
                lead_time=KnownEtaLeadTime((0, 1, 2), (0.25, 0.25, 0.5)),
                capacity=2,
            ),
            TransitionCosts(),
            ZERO_DEMAND,
            2,
        )

        class TrackingKernel(KnownEventStochasticTransitionKernel):
            yielded = 0

            def outcomes(self, *args, **kwargs):
                for branch in super().outcomes(*args, **kwargs):
                    self.yielded += 1
                    if self.yielded > 2:
                        raise AssertionError(
                            "enumerated beyond the reachable-state limit"
                        )
                    yield branch

        kernel = TrackingKernel(
            case.config,
            case.costs,
            schema=case.schema,
            action_cap=case.action_cap,
        )
        with self.assertRaisesRegex(MemoryError, "limit is 1"):
            evaluate_composed_policy_exact(
                case,
                lambda _state, _remaining: 1,
                1,
                kernel=kernel,
                max_reachable_states=1,
            )
        self.assertEqual(kernel.yielded, 2)


if __name__ == "__main__":
    unittest.main()
