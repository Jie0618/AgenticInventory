from __future__ import annotations

import numpy as np

from evaluation.backends.composed.cache import load_or_create_composed_dp
from evaluation.routing import solve_inventory_dp
from inventory.common.distributions import DiscreteDemand
from inventory.core import (
    DeterministicLeadTime,
    DeterministicBacklog,
    FixedShelfLife,
    KnownArrivalShelfLife,
    LostSales,
    TransitionCosts,
)
from inventory.core.factory import inventory_case
from inventory.policy.ast import add, const, maximum, parameter, state, sub
from optimization.evaluators.composed import (
    ComposedExactPolicyCostEvaluator,
    ComposedMonteCarloPolicyCostEvaluator,
    TwoFidelityComposedMCEvaluator,
)
from optimization.models.types import MonteCarloProposalConfig, OptimizationConfig
from optimization.policies.baselines.perishable import baselines_for_perishable
from optimization.solvers.api import fit_inventory_policy_mc


def _case():
    return inventory_case(
        name="compiled-perishable-test",
        demand=DiscreteDemand.from_pmf({0: 0.2, 1: 0.5, 3: 0.3}),
        n=4,
        costs=TransitionCosts(holding=1.0, shortage=4.0, expiry=1.5),
        perishability=FixedShelfLife(3),
        backlog=LostSales(),
        capacity=4,
    )


def _evaluators(horizon: int = 12):
    case = _case()
    reference = solve_inventory_dp(case, horizon)
    spec = next(
        row for row in baselines_for_perishable(3, case.n) if row.name == "capped_base_stock"
    )
    scalar = ComposedExactPolicyCostEvaluator(
        spec.ast, case, horizon, spec.motifs
    )
    compiled = ComposedExactPolicyCostEvaluator(
        spec.ast,
        case,
        horizon,
        spec.motifs,
        compiled_transition=reference.compiled_transition,
    )
    return scalar, compiled


def _canonical_mc_cost(evaluator, params):
    """Independent path loop over the canonical kernel for regression checks."""

    exact = evaluator.exact_evaluator
    states = [exact.initial_state for _ in range(evaluator.paths)]
    costs = np.zeros(evaluator.paths, dtype=float)
    rng = np.random.default_rng(evaluator.seed ^ 0x5EED5EED)
    for period in range(exact.horizon):
        actions = exact._actions.actions(states, params, exact.horizon - period)
        next_states = []
        for path, (current, action) in enumerate(zip(states, actions)):
            outcome = exact._kernel.sample(
                current,
                int(action),
                int(evaluator._demands[path, period]),
                rng=rng,
            )
            costs[path] += outcome.total_cost
            next_states.append(outcome.next_state)
        states = next_states
    return float(np.mean(costs))


def test_compiled_exact_matches_canonical_scalar_kernel() -> None:
    scalar, compiled = _evaluators()
    for params in (
        {"S": 0, "Qmax": 0},
        {"S": 4, "Qmax": 2},
        {"S": 9, "Qmax": 4},
    ):
        expected = scalar.evaluate(params)
        actual = compiled.evaluate(params)
        assert actual.completed
        assert actual.metadata["backend"] == "compiled_finite_mdp_exact"
        assert np.isclose(actual.cost, expected.cost, rtol=0.0, atol=1.0e-10)


def test_compiled_stationary_exact_compiles_actions_once() -> None:
    _, compiled = _evaluators(horizon=12)
    original = compiled._actions.actions_array
    calls = 0

    def counted_actions(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    compiled._actions.actions_array = counted_actions
    compiled.evaluate({"S": 4, "Qmax": 2})
    assert calls == 1


def test_explicit_shared_bank_and_pathwise_costs_match_compiled_and_canonical() -> None:
    scalar, compiled = _evaluators(horizon=4)
    a = ComposedMonteCarloPolicyCostEvaluator(scalar, paths=5, seed=19)
    b = ComposedMonteCarloPolicyCostEvaluator(compiled, paths=5, seed=23)
    support = np.asarray(scalar.case.demand.values)
    bank = np.resize(support, (5, 4))
    a.bind_demand_paths(bank)
    b.bind_demand_paths(bank)
    params = {"S": 4, "Qmax": 2}
    assert np.array_equal(a.simulate_path_costs(params), b.simulate_path_costs(params))
    assert a.evaluate_many((params,))[0].cost == float(np.mean(a.simulate_path_costs(params)))
    # Rebinding invalidates a cached cost instead of returning the old bank's result.
    alternate = np.full((5, 4), support[0])
    a.bind_demand_paths(alternate)
    assert a.evaluate_many((params,))[0].cost == float(np.mean(a.simulate_path_costs(params)))
    import pytest
    with pytest.raises(ValueError, match="shape"):
        a.bind_demand_paths(bank[:2])
    with pytest.raises(ValueError, match="support"):
        a.bind_demand_paths(np.full((5, 4), float(np.max(support)) + 1))


def test_mc_only_optimizer_api_runs_perishable_and_composed_cases() -> None:
    demand = DiscreteDemand.from_pmf({0: 0.5, 1: 0.5})
    costs = TransitionCosts(holding=1.0, shortage=4.0, expiry=1.0)
    cases = (
        inventory_case(
            name="mc-only-p",
            demand=demand,
            n=2,
            costs=costs,
            perishability=FixedShelfLife(2),
            backlog=LostSales(),
            capacity=2,
        ),
        inventory_case(
            name="mc-only-lp",
            demand=demand,
            n=2,
            costs=costs,
            lead_time=DeterministicLeadTime(1),
            perishability=FixedShelfLife(2),
            backlog=LostSales(),
            capacity=2,
        ),
    )
    spec = next(row for row in baselines_for_perishable(2, 2) if row.name == "constant_order")
    config = OptimizationConfig(mode="fast", total_evaluations=4, random_seed=77)
    for case in cases:
        result = fit_inventory_policy_mc(
            spec.ast,
            case,
            horizon=5,
            parameter_specs=spec.parameter_specs,
            motifs=spec.motifs,
            config=config,
            proposal_paths=4,
            strict_paths=8,
            seed=101,
        )
        assert result.success
        assert result.policy_cost is not None
        assert result.optimizer_metadata["mc_only"] is True
        assert result.optimizer_metadata["evaluation_backend"] == "composed_mc_only"


def test_uncompiled_mc_batches_candidates_without_calling_scalar(monkeypatch) -> None:
    evaluator = ComposedMonteCarloPolicyCostEvaluator.from_case(
        parameter("q"), _case(), 9, paths=32, seed=919, batch_size=8
    )
    monkeypatch.setattr(
        evaluator,
        "evaluate",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("evaluate_many called scalar evaluate")
        ),
    )
    results = evaluator.evaluate_many([{"q": 1.0}, {"q": 2.0}, {"q": 3.0}])
    assert len(results) == 3
    assert all(row.metadata["mc_batch_size"] == 3 for row in results)


def test_two_fidelity_batches_strict_candidates_without_calling_scalar(monkeypatch) -> None:
    high = ComposedMonteCarloPolicyCostEvaluator.from_case(
        parameter("q"), _case(), 9, paths=16, seed=920, batch_size=8
    )
    low = ComposedMonteCarloPolicyCostEvaluator.from_case(
        parameter("q"), _case(), 9, paths=8, seed=919, batch_size=8
    )
    monkeypatch.setattr(
        high,
        "evaluate",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("strict evaluate_many called scalar evaluate")
        ),
    )
    results = TwoFidelityComposedMCEvaluator(high, low).evaluate_many(
        ({"q": 1.0}, {"q": 2.0})
    )
    assert len(results) == 2
    assert all(row.metadata["mc_empirical_strict"] is True for row in results)


def test_vectorized_canonical_mc_matches_kernel_for_stage3_compositions() -> None:
    demand = DiscreteDemand.from_pmf({0: 0.2, 1: 0.5, 3: 0.3})
    cases = (
        inventory_case(
            name="batch-p",
            demand=demand,
            n=6,
            costs=TransitionCosts(holding=1.0, shortage=4.0, expiry=1.5),
            perishability=FixedShelfLife(3),
            backlog=LostSales(),
            capacity=6,
        ),
        inventory_case(
            name="batch-lp",
            demand=demand,
            n=6,
            costs=TransitionCosts(holding=1.0, shortage=4.0, expiry=1.5),
            lead_time=DeterministicLeadTime(2),
            perishability=FixedShelfLife(3),
            backlog=LostSales(),
            capacity=6,
        ),
        inventory_case(
            name="batch-lb",
            demand=demand,
            n=6,
            costs=TransitionCosts(
                holding=1.0,
                shortage=4.0,
                backlog=0.2,
                abandonment=2.0,
            ),
            lead_time=DeterministicLeadTime(2),
            backlog=DeterministicBacklog(3),
            capacity=6,
        ),
    )
    for case in cases:
        inventory_position = add(
            *(state(name) for name in case.schema.inventory_names),
            *(state(name) for name in case.schema.pipeline_names),
        )
        policy = maximum(sub(parameter("S"), inventory_position), const(0.0))
        evaluator = ComposedMonteCarloPolicyCostEvaluator.from_case(
            policy, case, 17, paths=31, seed=919, batch_size=8
        )
        params_batch = ({"S": 2.0}, {"S": 5.0}, {"S": 8.0})
        expected = [_canonical_mc_cost(evaluator, params) for params in params_batch]
        actual = evaluator.evaluate_many(params_batch)
        assert all(
            row.metadata["backend"] == "composed_vectorized_transition_mc"
            for row in actual
        )
        assert np.allclose(
            [row.cost for row in actual], expected, rtol=0.0, atol=1.0e-12
        )


def test_compiled_stationary_mc_compiles_one_action_map_per_batch() -> None:
    _, compiled = _evaluators(horizon=12)
    original = compiled._actions.actions_array
    calls = 0

    def counted_actions(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    compiled._actions.actions_array = counted_actions
    mc = ComposedMonteCarloPolicyCostEvaluator(
        compiled, paths=64, seed=919, batch_size=8
    )
    mc.evaluate_many(
        [
            {"S": 2, "Qmax": 1},
            {"S": 4, "Qmax": 2},
            {"S": 7, "Qmax": 3},
        ]
    )
    assert calls == 1


def test_composed_certify_uses_two_fidelity_mc_path_budget() -> None:
    _, compiled = _evaluators(horizon=3)
    default_mc = compiled.make_fast_evaluator(OptimizationConfig(), "certify")
    configured_mc = compiled.make_fast_evaluator(
        OptimizationConfig(mc_proposal=MonteCarloProposalConfig(paths=1536)),
        "certify",
    )
    assert default_mc.mc_evaluator is None
    assert default_mc.rerank_mc_evaluator is None
    assert default_mc.routing_mode == "auto"
    assert configured_mc.routing_mode == "auto"
    # The proposal route is created lazily; certify retains a 1024-path
    # finalist route while screening at the 512-path fidelity.
    default_result = default_mc.evaluate_many([{"S": 2, "Qmax": 1}])[0]
    configured_result = configured_mc.evaluate_many([{"S": 2, "Qmax": 1}])[0]
    assert default_result.completed and configured_result.completed
    # A stationary proposal may finish the tiny horizon without creating MC;
    # when it falls back, the 512-path budget is retained.
    for evaluator in (default_mc, configured_mc):
        if evaluator.mc_evaluator is not None:
            assert evaluator.mc_evaluator.paths == 512
        assert evaluator.rerank_mc_evaluator is None


def test_composed_reference_cache_reuses_signed_compiled_result(tmp_path) -> None:
    case = _case()
    first = load_or_create_composed_dp(
        case, 9, suite="unit", cache_dir=tmp_path
    )
    second = load_or_create_composed_dp(
        case, 9, suite="unit", cache_dir=tmp_path
    )
    assert first.compiled_transition is not None
    assert second.compiled_transition is not None
    assert second.metadata["case_signature"] == case.case_signature
    assert np.array_equal(first.value_table, second.value_table)
    assert len(list(tmp_path.rglob("*.pickle"))) == 1


def test_compiled_evaluator_supports_nondefault_initial_state() -> None:
    case = _case()
    reference = solve_inventory_dp(case, 7)
    spec = next(
        row for row in baselines_for_perishable(3, case.n) if row.name == "base_stock"
    )
    initial_state = reference.states[-1]
    scalar = ComposedExactPolicyCostEvaluator(
        spec.ast, case, 7, spec.motifs, initial_state=initial_state
    )
    compiled = ComposedExactPolicyCostEvaluator(
        spec.ast,
        case,
        7,
        spec.motifs,
        initial_state=initial_state,
        compiled_transition=reference.compiled_transition,
    )
    params = {"S": 6}
    assert np.isclose(
        compiled.evaluate(params).cost,
        scalar.evaluate(params).cost,
        rtol=0.0,
        atol=1.0e-10,
    )


def test_compiled_exact_preserves_remaining_horizon_semantics() -> None:
    case = _case()
    reference = solve_inventory_dp(case, 7)
    ast = state("remaining_horizon")
    scalar = ComposedExactPolicyCostEvaluator(ast, case, 7)
    compiled = ComposedExactPolicyCostEvaluator(
        ast, case, 7, compiled_transition=reference.compiled_transition
    )
    assert np.isclose(
        compiled.evaluate({}).cost,
        scalar.evaluate({}).cost,
        rtol=0.0,
        atol=1.0e-10,
    )


def test_composed_stationary_proposal_is_generic_and_exact_remains_strict() -> None:
    scalar, compiled = _evaluators(horizon=12)
    stationary = compiled.make_fast_evaluator(
        OptimizationConfig(
            mc_proposal=MonteCarloProposalConfig(
                paths=16, lazy_init=False, routing_mode="auto"
            )
        ),
        "standard",
    )
    proposal = stationary.evaluate({"S": 4, "Qmax": 2})
    assert proposal.completed
    assert proposal.metadata["proposal_fidelity"] == "stationary_fast"
    assert proposal.metadata["backend"] == "compiled_finite_mdp_stationary"
    # Proposal routing must not change the strict final evaluator.
    assert np.isclose(
        compiled.evaluate({"S": 4, "Qmax": 2}).cost,
        scalar.evaluate({"S": 4, "Qmax": 2}).cost,
        rtol=0.0,
        atol=1.0e-10,
    )


def test_composed_horizon_dependent_policy_skips_stationary_probe() -> None:
    case = _case()
    reference = solve_inventory_dp(case, 7)
    ast = state("remaining_horizon")
    evaluator = ComposedExactPolicyCostEvaluator(
        ast, case, 7, compiled_transition=reference.compiled_transition
    )
    fast = evaluator.make_fast_evaluator(
        OptimizationConfig(
            mc_proposal=MonteCarloProposalConfig(
                paths=8, lazy_init=False, routing_mode="auto"
            )
        ),
        "standard",
    )
    assert getattr(fast.stationary_evaluator, "__class__", None).__name__ == (
        "ComposedExactPolicyCostEvaluator"
    )


def test_stochastic_composition_retains_scalar_fallback() -> None:
    case = inventory_case(
        name="stochastic-shelf-life-fallback",
        demand=DiscreteDemand.from_pmf({0: 0.5, 1: 0.5}),
        n=2,
        costs=TransitionCosts(holding=1.0, shortage=4.0, expiry=1.0),
        perishability=KnownArrivalShelfLife((2, 3), (0.5, 0.5)),
        backlog=LostSales(),
        capacity=2,
    )
    reference = solve_inventory_dp(case, 3)
    assert reference.compiled_transition is None
    assert reference.metadata["compiled_transition"] is False


def test_composed_cache_reuses_stochastic_reference(tmp_path) -> None:
    case = inventory_case(
        name="stochastic-cache-test",
        demand=DiscreteDemand.from_pmf({0: 0.5, 1: 0.5}),
        n=2,
        costs=TransitionCosts(holding=1.0, shortage=4.0, expiry=1.0),
        perishability=KnownArrivalShelfLife((2, 3), (0.5, 0.5)),
        backlog=LostSales(),
        capacity=2,
    )
    first = load_or_create_composed_dp(case, 3, suite="unit", cache_dir=tmp_path)
    second = load_or_create_composed_dp(case, 3, suite="unit", cache_dir=tmp_path)
    assert first.compiled_transition is None
    assert second.compiled_transition is None
    assert np.array_equal(first.value_table, second.value_table)
    assert len(list(tmp_path.rglob("*.pickle"))) == 1
