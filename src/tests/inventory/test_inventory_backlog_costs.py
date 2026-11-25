"""No shortage-entry fee in any backlog composition or evaluator."""
import numpy as np
import pytest

from inventory.common.distributions import DiscreteDemand
from inventory.core import (BacklogConfig, InventoryCase, InventoryModelConfig,
    InventoryState, TransitionCosts, DeterministicLeadTime, FixedShelfLife)
from inventory.policy.ast import parameter
from optimization.evaluators.composed import ComposedMonteCarloPolicyCostEvaluator


@pytest.mark.parametrize('backlog', [BacklogConfig.full(),
    BacklogConfig.deterministic(2), BacklogConfig.geometric(.5)])
def test_new_backlog_is_not_charged_shortage(backlog):
    case = InventoryCase('entry-fee', InventoryModelConfig(backlog=backlog, capacity=10),
        TransitionCosts(shortage=101, backlog=7, abandonment=11),
        DiscreteDemand.from_pmf({3: 1}), n=10)
    state = InventoryState((0,), (), (1,2) if backlog.patience else (3,))
    kwargs = {'random_event': {'abandoned': 2}} if backlog.mode.value == 'geometric' else {}
    tr = case.kernel.transition(state, 0, 3, **kwargs)
    assert tr.new_backlog == 3 and tr.shortage_cost == 0
    assert tr.backlog_cost == 7*sum(tr.next_state.backlog)
    assert tr.abandonment_cost == 11*tr.abandoned
    assert tr.total_cost == tr.backlog_cost+tr.abandonment_cost


@pytest.mark.parametrize('perishable', [False, True])
@pytest.mark.parametrize('backlog', [BacklogConfig.full(), BacklogConfig.deterministic(2)])
def test_batch_and_scalar_ignore_shortage_coefficient_with_backlog(backlog, perishable):
    # All other costs are held fixed: b cannot affect backlog cost or actions.
    results = []
    for b in (0, 101):
        config = InventoryModelConfig(lead_time=DeterministicLeadTime(1),
            backlog=backlog, capacity=2,
            **({'perishability': FixedShelfLife(2)} if perishable else {}))
        case = InventoryCase('batch-entry-fee', config,
            TransitionCosts(holding=2, shortage=b, backlog=3, abandonment=5,
                            expiry=7, ordering=11),
            DiscreteDemand.from_pmf({0: .5, 2: .5}), n=2)
        ev = ComposedMonteCarloPolicyCostEvaluator.from_case(parameter('Q'), case, 6, paths=8, seed=17)
        batch = [r.cost for r in ev.evaluate_many([{'Q': q} for q in (0,1,2)])]
        scalar = []
        for q in (0,1,2):
            costs = []
            for path in ev._demands:
                state, cost = case.initial_state, 0.
                for demand in path:
                    tr = case.kernel.transition(state, q, int(demand))
                    cost += tr.total_cost
                    state = tr.next_state
                costs.append(cost)
            scalar.append(np.mean(costs))
        np.testing.assert_allclose(batch, scalar, rtol=0, atol=1e-10)
        results.append(batch)
        assert case.identity_payload()['cost_semantics'] == 'backlog_no_entry_shortage'
    np.testing.assert_array_equal(*results)


def test_lost_sales_still_charge_shortage():
    case = InventoryCase('lost-sales', InventoryModelConfig(capacity=2),
        TransitionCosts(shortage=101), DiscreteDemand.from_pmf({2: 1}), n=2)
    tr = case.kernel.transition(case.initial_state, 0, 2)
    assert tr.shortage_cost == 202 and tr.new_backlog == 0
    assert 'cost_semantics' not in case.identity_payload()


@pytest.mark.parametrize('backlog', [BacklogConfig.full(),
    BacklogConfig.deterministic(1), BacklogConfig.geometric(.5)])
def test_exact_and_dp_costs_ignore_entry_shortage(backlog):
    from evaluation.exact_mc import evaluate_composed_policy_exact
    from evaluation.dp import solve_composed_dp
    values = []
    for b in (0, 99):
        case = InventoryCase('dp-no-entry-fee',
            InventoryModelConfig(backlog=backlog, capacity=1),
            TransitionCosts(holding=2, shortage=b, backlog=3, abandonment=7),
            DiscreteDemand.from_pmf({0: .5, 1: .5}), n=1)
        exact = evaluate_composed_policy_exact(case, lambda _s, _r: 0, 3)
        dp = solve_composed_dp(case, 3)
        values.append((exact.cost, dp.value(0, case.initial_state)))
    np.testing.assert_allclose(values[0], values[1], rtol=0, atol=1e-12)
