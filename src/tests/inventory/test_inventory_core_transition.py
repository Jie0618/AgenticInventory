"""Numerical contracts for the compositional inventory transition kernel."""

from __future__ import annotations

import itertools
import unittest
from unittest import mock

import numpy as np

from inventory.common.distributions import DiscreteDemand
from inventory.core import (
    BacklogConfig,
    DeterministicLeadTime,
    ExogenousEvent,
    FixedShelfLife,
    GeometricBacklog,
    InventoryCase,
    InventoryModelConfig,
    InventoryState,
    KnownArrivalShelfLife,
    KnownEtaLeadTime,
    TransitionEnumerationError,
    TransitionCosts,
    build_transition_kernel,
)
from evaluation.dp import solve_composed_dp
from evaluation.exact_mc import evaluate_composed_policy_exact, evaluate_composed_policy_mc
from inventory.core.factory import inventory_case
from evaluation.backends.lead_time.transition import lead_time_transition
from evaluation.backends.lead_time.models import FlatTransitionOutcome


DETERMINISTIC_ZERO = DiscreteDemand.from_pmf({0: 1.0}, name="zero")


def LeadTimeCase(name, lead_time, h, b, demand, kind, n):
    return inventory_case(
        name=name, demand=demand, n=n, kind=kind,
        costs=TransitionCosts(holding=h, shortage=b),
        lead_time=DeterministicLeadTime(lead_time),
        backlog=BacklogConfig.lost_sales(), capacity=n,
    )


def PerishableCase(name, shelf_life, h, b, w, demand, n):
    return inventory_case(
        name=name, demand=demand, n=n,
        costs=TransitionCosts(holding=h, shortage=b, expiry=w),
        perishability=FixedShelfLife(shelf_life),
        backlog=BacklogConfig.lost_sales(), capacity=n,
    )


def LeadTimePerishableCase(name, lead_time, shelf_life, h, b, w, demand, n):
    return inventory_case(
        name=name, demand=demand, n=n,
        costs=TransitionCosts(holding=h, shortage=b, expiry=w),
        lead_time=None if lead_time == 0 else DeterministicLeadTime(lead_time),
        perishability=FixedShelfLife(shelf_life),
        backlog=BacklogConfig.lost_sales(), capacity=n,
    )


def _reference_lead_time_oracle(case, state, action, demand):
    """Independent copy of the scalar lead-time equations."""

    lead_time = int(case.lead_time)
    x = int(state[0])
    pipeline = tuple(int(value) for value in state[1:])
    available = x + (int(action) if lead_time == 0 else pipeline[0])
    ending = max(available - int(demand), 0)
    next_pipeline = () if lead_time == 0 else (*pipeline[1:], int(action))
    return FlatTransitionOutcome(
        next_state=(min(ending, int(case.n)), *next_pipeline),
        holding_cost=float(case.h) * float(ending),
        shortage_cost=float(case.b) * float(max(int(demand) - available, 0)),
    )


def _reference_age_oracle(case, ages, incoming, demand):
    """Independent copy of the FEFO/aging equations."""

    inventory = [int(value) for value in ages]
    inventory[-1] += int(incoming)
    unmet = int(demand)
    for index in range(len(inventory)):
        used = min(inventory[index], unmet)
        inventory[index] -= used
        unmet -= used
        if unmet == 0:
            break
    holding = sum(inventory)
    expired = inventory[0]
    aged = list((*inventory[1:], 0)) if len(inventory) > 1 else [0]
    excess = max(0, sum(aged) - int(case.n))
    for index in range(len(aged) - 1, -1, -1):
        removed = min(aged[index], excess)
        aged[index] -= removed
        excess -= removed
        if excess == 0:
            break
    return FlatTransitionOutcome(
        next_state=tuple(aged),
        holding_cost=float(case.h) * float(holding),
        shortage_cost=float(case.b) * float(unmet),
        expiry_cost=float(case.w) * float(expired),
    )


def _reference_perishable_oracle(case, state, action, demand):
    return _reference_age_oracle(case, state, action, demand)


def _reference_combined_oracle(case, state, action, demand):
    """Frozen copy of the old lead-time + perishable composition."""

    ages = tuple(int(value) for value in state[: case.shelf_life])
    pipeline = tuple(int(value) for value in state[case.shelf_life :])
    if case.lead_time == 0:
        incoming = int(action)
        next_pipeline = ()
    else:
        incoming = pipeline[0]
        next_pipeline = (*pipeline[1:], int(action))
    age_outcome = _reference_age_oracle(case, ages, incoming, demand)
    return FlatTransitionOutcome(
        next_state=(*age_outcome.next_state, *next_pipeline),
        holding_cost=age_outcome.holding_cost,
        shortage_cost=age_outcome.shortage_cost,
        expiry_cost=age_outcome.expiry_cost,
    )


class InventoryCoreTransitionTests(unittest.TestCase):
    def assertReferenceOutcomeEqual(self, legacy, canonical, schema) -> None:
        self.assertEqual(legacy.next_state, canonical.next_state.flatten(schema))
        self.assertAlmostEqual(legacy.holding_cost, canonical.holding_cost)
        self.assertAlmostEqual(legacy.shortage_cost, canonical.shortage_cost)
        self.assertAlmostEqual(legacy.expiry_cost, canonical.expiry_cost)
        self.assertAlmostEqual(legacy.total_cost, canonical.total_cost)

    def test_lead_time_kernel_matches_legacy_transition_exhaustively(self) -> None:
        for lead_time in range(3):
            case = LeadTimeCase(
                "lt", lead_time, 1.25, 4.5, DETERMINISTIC_ZERO, "test", 2
            )
            config = InventoryModelConfig(
                lead_time=DeterministicLeadTime(lead_time), capacity=case.n
            )
            kernel = build_transition_kernel(config, {"h": case.h, "b": case.b})
            for state in itertools.product(range(case.n + 1), repeat=lead_time + 1):
                for action in range(case.n + 1):
                    for demand in range(4):
                        legacy = _reference_lead_time_oracle(case, state, action, demand)
                        self.assertEqual(
                            lead_time_transition(case, state, action, demand), legacy
                        )
                        canonical = kernel.transition(state, action, demand)
                        self.assertReferenceOutcomeEqual(legacy, canonical, kernel.schema)

    def test_perishable_factory_uses_composed_kernel(self) -> None:
        for shelf_life in (1, 2, 3):
            case = PerishableCase(
                "p", shelf_life, 1.25, 4.5, 2.0, DETERMINISTIC_ZERO, 2
            )
            config = InventoryModelConfig(
                perishability=FixedShelfLife(shelf_life), capacity=case.n
            )
            kernel = build_transition_kernel(
                config, {"h": case.h, "b": case.b, "w": case.w}
            )
            for state in itertools.product(range(case.n + 1), repeat=shelf_life):
                if sum(state) > case.n:
                    continue
                for action in range(case.n + 1):
                    for demand in range(4):
                        legacy = _reference_perishable_oracle(case, state, action, demand)
                        canonical = kernel.transition(state, action, demand)
                        self.assertReferenceOutcomeEqual(legacy, canonical, kernel.schema)

    def test_combined_factory_uses_composed_kernel(self) -> None:
        for lead_time in (0, 1, 2):
            for shelf_life in (1, 2):
                case = LeadTimePerishableCase(
                    "lp",
                    lead_time,
                    shelf_life,
                    1.25,
                    4.5,
                    2.0,
                    DETERMINISTIC_ZERO,
                    2,
                )
                config = InventoryModelConfig(
                    lead_time=DeterministicLeadTime(lead_time),
                    perishability=FixedShelfLife(shelf_life),
                    capacity=case.n,
                )
                kernel = build_transition_kernel(
                    config, {"h": case.h, "b": case.b, "w": case.w}
                )
                for state in itertools.product(
                    range(case.n + 1), repeat=shelf_life + lead_time
                ):
                    if sum(state[:shelf_life]) > case.n:
                        continue
                    for action in range(case.n + 1):
                        for demand in range(4):
                            legacy = _reference_combined_oracle(
                                case, state, action, demand
                            )
                            canonical = kernel.transition(state, action, demand)
                            self.assertReferenceOutcomeEqual(
                                legacy, canonical, kernel.schema
                            )

    def test_known_eta_is_sampled_on_order_and_uses_countdown_pipeline(self) -> None:
        config = InventoryModelConfig(
            lead_time=KnownEtaLeadTime((0, 2), (0.25, 0.75)), capacity=10
        )
        kernel = build_transition_kernel(config)
        state = InventoryState((0,), (0, 0), ())
        branches = kernel.outcomes(state, action=4, demand=0)
        self.assertEqual(len(branches), 2)
        by_eta = {branch.outcome.event.lead_time: branch for branch in branches}
        self.assertAlmostEqual(by_eta[0].probability, 0.25)
        self.assertEqual(by_eta[0].next_state, InventoryState((4,), (0, 0), ()))
        self.assertAlmostEqual(by_eta[2].probability, 0.75)
        self.assertEqual(by_eta[2].next_state, InventoryState((0,), (0, 4), ()))

        countdown = kernel.outcomes(
            by_eta[2].next_state, action=0, demand=0, random_event={"lead_time": 0}
        )[0].next_state
        self.assertEqual(countdown, InventoryState((0,), (4, 0), ()))
        arrived = kernel.outcomes(
            countdown, action=0, demand=0, random_event={"lead_time": 0}
        )[0].next_state
        self.assertEqual(arrived, InventoryState((4,), (0, 0), ()))

    def test_known_eta_collisions_are_not_projected_without_explicit_cap(self) -> None:
        case = InventoryCase(
            "known-eta-collision",
            InventoryModelConfig(lead_time=KnownEtaLeadTime((1, 2), (0.5, 0.5))),
            TransitionCosts(),
            DETERMINISTIC_ZERO,
            5,
        )
        outcome = case.kernel.transition(
            InventoryState((0,), (0, 5), ()),
            action=5,
            demand=0,
            random_event={"lead_time": 1},
        )
        self.assertEqual(outcome.next_state.pipeline, (10, 0))
        self.assertEqual(outcome.discarded, 0)

    def test_known_eta_collision_projection_is_reported_as_discarded(self) -> None:
        case = InventoryCase(
            "known-eta-finite-grid-collision",
            InventoryModelConfig(
                lead_time=KnownEtaLeadTime((1, 2), (0.5, 0.5)),
                capacity=5,
            ),
            TransitionCosts(),
            DETERMINISTIC_ZERO,
            5,
        )
        outcome = case.kernel.transition(
            InventoryState((0,), (0, 5), ()),
            action=5,
            demand=0,
            random_event={"lead_time": 1},
        )

        self.assertEqual(outcome.next_state.pipeline, (5, 0))
        self.assertEqual(outcome.discarded, 5)

    def test_expired_is_not_an_alias_for_backlog_abandonment(self) -> None:
        with self.assertRaisesRegex(ValueError, "expired"):
            ExogenousEvent.from_value({"expired": 1})

    def test_due_receipt_is_one_batch_for_random_shelf_life(self) -> None:
        kernel = build_transition_kernel(
            InventoryModelConfig(
                lead_time=DeterministicLeadTime(1),
                perishability=KnownArrivalShelfLife((1, 3), (0.25, 0.75)),
                capacity=5,
            )
        )
        branches = kernel.outcomes(
            InventoryState((0, 0, 0), (4,), ()), action=0, demand=0
        )
        by_lifetime = {branch.outcome.event.lifetime: branch for branch in branches}

        # The aggregate p1 receipt is the batch unit in transition v1: its
        # four units share one lifetime draw rather than four independent
        # draws that would require a multinomial event representation.
        self.assertAlmostEqual(by_lifetime[1].probability, 0.25)
        self.assertEqual(
            by_lifetime[1].next_state,
            InventoryState((0, 0, 0), (0,), ()),
        )
        self.assertEqual(by_lifetime[1].outcome.expired, 4)
        self.assertAlmostEqual(by_lifetime[3].probability, 0.75)
        self.assertEqual(
            by_lifetime[3].next_state,
            InventoryState((0, 4, 0), (0,), ()),
        )

    def test_lead_time_backlog_and_perishable_backlog_compositions(self) -> None:
        lead_backlog = build_transition_kernel(
            InventoryModelConfig(
                lead_time=DeterministicLeadTime(1),
                backlog=BacklogConfig.full(),
                capacity=10,
            )
        )
        outcome = lead_backlog.transition(
            InventoryState((1,), (3,), (2,)), action=4, demand=5
        )
        self.assertEqual(outcome.served_backlog, 2)
        self.assertEqual(outcome.served_demand, 2)
        self.assertEqual(outcome.new_backlog, 3)
        self.assertEqual(outcome.next_state, InventoryState((0,), (4,), (3,)))

        perish_backlog = build_transition_kernel(
            InventoryModelConfig(
                perishability=FixedShelfLife(2),
                backlog=BacklogConfig.deterministic(2),
                capacity=10,
            )
        )
        outcome = perish_backlog.transition(
            InventoryState((2, 1), (), (1, 3)), action=2, demand=4
        )
        self.assertEqual(outcome.served_backlog, 4)
        self.assertEqual(outcome.new_backlog, 3)
        self.assertEqual(outcome.next_state, InventoryState((0, 0), (), (3, 0)))

    def test_geometric_backlog_exact_branches_and_event_override(self) -> None:
        kernel = build_transition_kernel(
            InventoryModelConfig(backlog=GeometricBacklog(0.5), capacity=10)
        )
        state = InventoryState((0,), (), (2,))
        branches = kernel.outcomes(state, action=0, demand=0)
        probabilities = {
            branch.outcome.abandoned: branch.probability for branch in branches
        }
        self.assertEqual(set(probabilities), {0, 1, 2})
        self.assertAlmostEqual(probabilities[0], 0.25)
        self.assertAlmostEqual(probabilities[1], 0.5)
        self.assertAlmostEqual(probabilities[2], 0.25)
        forced = kernel.outcomes(
            state, action=0, demand=0, random_event={"abandoned": 1}
        )
        self.assertEqual(len(forced), 1)
        self.assertEqual(forced[0].outcome.next_state.backlog, (1,))

    def test_geometric_backlog_large_exact_support_is_numerically_stable(self) -> None:
        kernel = build_transition_kernel(
            InventoryModelConfig(backlog=GeometricBacklog(0.5), capacity=2_000)
        )
        branches = kernel.outcomes(
            InventoryState((0,), (), (1_500,)), action=0, demand=0
        )

        self.assertGreater(len(branches), 1_000)
        self.assertAlmostEqual(sum(branch.probability for branch in branches), 1.0)
        mode = max(branches, key=lambda branch: branch.probability)
        self.assertIn(mode.outcome.abandoned, (750, 751))

    def test_geometric_backlog_checks_branch_budget_before_materializing(self) -> None:
        kernel = build_transition_kernel(
            InventoryModelConfig(backlog=GeometricBacklog(0.5)),
            max_outcomes=3,
        )

        with mock.patch("inventory.core.transition._binomial_branches") as materialize:
            with self.assertRaisesRegex(
                TransitionEnumerationError,
                "requires at least 6 branches; max_outcomes=3",
            ):
                kernel.outcomes(InventoryState((0,), (), (5,)), action=0, demand=0)
        materialize.assert_not_called()

    def test_backlog_carry_cost_uses_retained_post_survival_projected_quantity(
        self,
    ) -> None:
        kernel = build_transition_kernel(
            InventoryModelConfig(backlog=GeometricBacklog(0.5), capacity=3),
            TransitionCosts(backlog=7.0),
        )
        branch = kernel.outcomes(
            InventoryState((0,), (), (2,)),
            action=0,
            demand=4,
            random_event={"abandoned": 1},
        )[0]

        # Six units are eligible, one abandons, and the finite-grid projection
        # retains three.  Carry cost is charged only on those three units.
        self.assertEqual(branch.outcome.abandoned, 1)
        self.assertEqual(branch.outcome.discarded, 2)
        self.assertEqual(branch.next_state.backlog, (3,))
        self.assertEqual(branch.outcome.backlog_cost, 21.0)

    def test_core_transition_rejects_noninteger_action_and_demand(self) -> None:
        kernel = build_transition_kernel(InventoryModelConfig())
        state = InventoryState((0,), (), ())

        for invalid in (
            np.bool_(True),
            1.5,
            np.array(1),
            np.array([1]),
            float("inf"),
        ):
            for action, demand in ((invalid, 0), (0, invalid)):
                with self.subTest(action=repr(action), demand=repr(demand)):
                    with self.assertRaisesRegex(ValueError, "must be an integer"):
                        kernel.transition(state, action=action, demand=demand)

    def test_geometric_backlog_has_lost_sales_and_full_backlog_limits(self) -> None:
        state = InventoryState((0,), (), (2,))
        lost_sales_limit = build_transition_kernel(
            InventoryModelConfig(backlog=GeometricBacklog(0.0), capacity=10)
        ).outcomes(state, action=0, demand=3)
        self.assertEqual(len(lost_sales_limit), 1)
        self.assertEqual(lost_sales_limit[0].outcome.abandoned, 5)
        self.assertEqual(lost_sales_limit[0].next_state.backlog, (0,))

        full_backlog_limit = build_transition_kernel(
            InventoryModelConfig(backlog=GeometricBacklog(1.0), capacity=10)
        ).outcomes(state, action=0, demand=3)
        self.assertEqual(len(full_backlog_limit), 1)
        self.assertEqual(full_backlog_limit[0].outcome.abandoned, 0)
        self.assertEqual(full_backlog_limit[0].next_state.backlog, (5,))

    def test_deterministic_backlog_total_cap_preserves_older_cohorts(self) -> None:
        kernel = build_transition_kernel(
            InventoryModelConfig(backlog=BacklogConfig.deterministic(3), capacity=5)
        )
        outcome = kernel.transition(
            InventoryState((0,), (), (2, 2, 1)), action=0, demand=4
        )

        # b3 abandons, b1/b2 age, and four new units enter b1.  The raw next
        # vector (4, 2, 2) exceeds the total cap by three, which is removed
        # from the newest cohort rather than displacing older obligations.
        self.assertEqual(outcome.next_state.backlog, (1, 2, 2))
        self.assertEqual(outcome.abandoned, 1)
        self.assertEqual(outcome.discarded, 3)
        self.assertEqual(outcome.new_backlog, 4)

    def test_exact_mc_and_dp_use_the_same_stochastic_kernel(self) -> None:
        case = InventoryCase(
            "known-eta",
            InventoryModelConfig(
                lead_time=KnownEtaLeadTime((0, 1), (0.4, 0.6)),
                capacity=2,
            ),
            TransitionCosts(holding=1.0, shortage=5.0),
            DiscreteDemand.from_pmf({0: 0.5, 1: 0.5}, name="bernoulli"),
            2,
        )
        policy = lambda _state, _remaining: 1
        exact = evaluate_composed_policy_exact(case, policy, 4)
        mc = evaluate_composed_policy_mc(case, policy, 4, paths=2_000, seed=17)
        lower, upper = mc.confidence_interval(z=4.0)
        self.assertLessEqual(lower, exact.cost)
        self.assertLessEqual(exact.cost, upper)
        dp = solve_composed_dp(case, 3, max_states=100)
        optimal = lambda state, remaining: dp.action(3 - remaining, state)
        evaluated = evaluate_composed_policy_exact(case, optimal, 3)
        self.assertAlmostEqual(evaluated.cost, dp.value(0, case.initial_state))


if __name__ == "__main__":
    unittest.main()
