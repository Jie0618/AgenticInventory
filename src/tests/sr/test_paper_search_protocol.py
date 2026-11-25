"""Numerical contracts from EC.3, including evidence actually sent to the LLM."""
import json
import math
import random
from dataclasses import replace
from types import SimpleNamespace

import pytest

from inventory.common.distributions import DiscreteDemand
from inventory.core import DeterministicLeadTime, FixedShelfLife, TransitionCosts
from inventory.core.factory import inventory_case
from inventory.policy.ast import cap, const, parameter, state
from inventory.policy.family import FamilyParameter, PolicyFamily
from optimization.search_context import InventorySearchContext
from sr.core.diagnostics import decision_records
from sr.core.models import SearchCandidate, SearchConfig, SearchIndividual
from sr.core.search_control import adaptive_score, offspring_reward, select_parents
from sr.family.models import FamilyEvaluation
from sr.family.proposals import _apply_family_crossover, family_from_json, llm_reflect_motifs
from sr.family.search import MultiContextFamilySearch
from sr.family.validation import validate_search_family
from sr.inventory.preparation import prepare_case
from sr.inventory.screening import BellmanScreen
from sr.llm.search import SearchDomain, propose_asts
from tests.support.family import family_domain


def test_weighted_regret_and_spread_rank_probability_times_cost():
    bank = [
        {"state": [0], "period": 200, "probability": 1, "q_cost_advantage": [0, 20, 0]},
        {"state": [1], "period": 240, "probability": 9, "q_cost_advantage": [2, 0, 0]},
    ]
    out = decision_records(bank, {"a": [1, 0], "b": [2, 2]}, case="toy", limit=1)
    regret = out["regret"]["a"][0]
    assert regret["state"] == [0]  # 0.1*20 > 0.9*2
    assert regret["optimal_actions"] == [0, 2]
    assert regret["weighted_regret"] == pytest.approx(2)
    assert out["regret"]["b"][0]["weighted_regret"] == 0
    assert out["action_spread"][0]["state"] == [1]
    assert out["action_spread"][0]["weighted_action_spread"] == pytest.approx(1.8)
    assert decision_records([], {}, case="no-dp")["reason"] == "no_dp_reference"
    with pytest.raises(ValueError, match="action cap"):
        decision_records(bank, {"bad": [3, 0]}, case="toy")


def test_diagnostic_keeps_ten_per_parent_and_ten_spreads():
    bank = [{"state": [i], "period": 200+i, "probability": 1,
             "q_cost_advantage": [0, i]} for i in range(25)]
    out = decision_records(bank, {"a": [1]*25, "b": [0]*25}, case="toy")
    assert len(out["regret"]["a"]) == len(out["action_spread"]) == 10
    assert [row["state"][0] for row in out["regret"]["a"]] == list(range(24, 14, -1))


def test_completed_offspring_reward_and_ucb_follow_equations():
    assert offspring_reward(10, 8, parameter_frontier=True, complexity_frontier=False) == pytest.approx(.7)
    assert offspring_reward(10, 12, parameter_frontier=False, complexity_frontier=True) == .5
    assert offspring_reward(0, -1, parameter_frontier=True, complexity_frontier=True) == 5
    assert adaptive_score(2, 1.4, 10) == pytest.approx(.7+.8*math.sqrt(math.log(12)/3)+.5/3)


def test_parent_reserves_precede_reproducible_softmax():
    rows = [SimpleNamespace(ast_hash=str(i), score=float(i), params=i % 9,
                            complexity=50-i, behavior="common" if i < 38 else str(i),
                            recent=i) for i in range(40)]
    def select(seed):
        return select_parents(rows, key=lambda r: r.ast_hash, score=lambda r: r.score,
            parameters=lambda r: r.params, complexity=lambda r: r.complexity,
            behavior=lambda r: r.behavior, recent=lambda r: r.recent,
            visits={}, rewards={}, rng=random.Random(seed))
    first = select(11)
    assert len(first) == len({r.ast_hash for r in first}) == 24
    assert {"0", "1", "38", "39", "37", "36"} <= {r.ast_hash for r in first}
    assert [r.ast_hash for r in first] == [r.ast_hash for r in select(11)]
    assert {r.ast_hash for r in first} != {r.ast_hash for r in select(12)}


def test_dp_bank_has_16_periods_normalized_and_period_sensitive_fitted_actions(tmp_path):
    case = inventory_case(name="tiny", demand=DiscreteDemand.from_pmf({0: .5, 1: .5}),
        n=2, capacity=2, costs=TransitionCosts(holding=1, shortage=4),
        lead_time=DeterministicLeadTime(2))
    reference, path = prepare_case(case, 1000, tmp_path/"inputs", tmp_path/"cache")
    bank = json.loads(path.read_text())
    assert len(bank["sample_periods"]) == 16
    assert bank["sample_periods"][0] == 200 and bank["sample_periods"][-1] == 799
    assert sum(row["probability"] for row in bank["representative_states"]) == pytest.approx(1)
    context = InventorySearchContext.from_reference(case, 1000, reference)
    screen = BellmanScreen(path, context=context, motifs={}, random_seed=1)
    ast = cap(const(10), parameter("C"), "cyclic_integer")
    candidate = SearchCandidate(ast, "cap", "cap", 1, 1, ("C",), "test")
    parent = SearchIndividual(candidate, {"C": 1.5}, 1, 1, 0, "complete", "test", 0, 0, "")
    out = screen.parent_diagnostics([parent])
    assert out["occupancy_source"] == "dp_optimal"
    for row in out["regret"]["cap"]:
        t = row["period"]
        assert row["parent_action"] == round(1.5*(t+1))-round(1.5*t)
        assert row["weighted_regret"] == pytest.approx(
            row["probability"] * row["bellman_cost_difference"])


@pytest.mark.parametrize("perishable", [False, True], ids=["lead_time", "perishable"])
@pytest.mark.parametrize("stage", [1, 2])
def test_cyclic_production_clis_keep_both_dp_diagnostics(tmp_path, monkeypatch, perishable, stage):
    """Exercise real CLI wiring and DP evidence without a model call or costly fit."""
    import sys
    from sr.analysis.frontier import FrontierSeedBundle
    from sr.inventory import run_single, run_multi
    from sr.inventory.multi_evaluation import CaseEvaluationInput
    from sr.inventory.runtime import MultiCliDefaults, MultiRuntime
    from sr.inventory.single import SymbolicSearch, diagnostic_samples
    from sr.inventory.multi import MultiDistributionSearch
    from sr.multi.models import MultiObjective

    components = ({"perishability": FixedShelfLife(2)} if perishable
                  else {"lead_time": DeterministicLeadTime(2)})
    case = inventory_case(name="tiny", demand=DiscreteDemand.from_pmf({0: .5, 1: .5}),
        n=2, capacity=2, costs=TransitionCosts(holding=1, shortage=4, expiry=2), **components)
    reference, path = prepare_case(case, 8, tmp_path/"inputs", tmp_path/"cache")
    samples = diagnostic_samples(path, 8)
    context = InventorySearchContext.from_reference(case, 8, reference, samples=samples)
    config = SearchConfig(iterations=1, cyclic_cap=True, optimizer_workers=1, memory_reserve_gb=0)
    search_file = tmp_path/"search.json"
    search_file.write_text(json.dumps({"iterations": 1, "cyclic_cap": True,
                                      "optimizer_workers": 1, "memory_reserve_gb": 0}))
    evaluation_file = tmp_path/"evaluation.json"
    evaluation_file.write_text(json.dumps({
        "proposal_paths": 2, "strict_paths": 2, "evaluation_paths": 2,
        "screen_paths": 2, "screen_horizon": 8, "screen_parameter_points": 2,
    }))
    captured = []
    result = SimpleNamespace(case_name=case.name, case_names=(case.name,),
        archive=(SimpleNamespace(gap_pct=0, score=0),), iterations_completed=1,
        elapsed_seconds=0, output_dir=tmp_path/"output")

    def capture_search(**kwargs):
        search_type = SymbolicSearch if stage == 1 else MultiDistributionSearch
        captured.append(search_type(**kwargs))
        return SimpleNamespace(run=lambda: result)

    argv = ["search", "--output-dir", str(result.output_dir), "--no-write-run-logs",
            "--cyclic-evaluation-config", str(evaluation_file)]
    if stage == 1:
        argv += ["--case", case.name, "--horizon", "8",
                 "--diagnostic-root", str(path.parent), "--dp-cache-dir", str(tmp_path/"cache"),
                 "--search-config", str(search_file)]
        monkeypatch.setattr(sys, "argv", argv)
        monkeypatch.setattr(run_single, "SymbolicSearch", capture_search)
        run_single.main((case,), prepare_inputs=True, default_diagnostic_root=path.parent)
    else:
        bundle = FrontierSeedBundle((), {}, {}, {}, {})
        runtime = MultiRuntime((case.name,), MultiObjective(cvar_alpha=.8), config, 1, bundle,
            {case.name: CaseEvaluationInput(case.name, context, samples, 1)},
            {case.name: path}, "test-input-signature")
        monkeypatch.setattr(sys, "argv", argv)
        monkeypatch.setattr(run_multi, "load_runtime", lambda **kwargs: runtime)
        monkeypatch.setattr(run_multi, "MultiDistributionSearch", capture_search)
        defaults = MultiCliDefaults(search_file, search_file, tmp_path, path.parent,
                                   tmp_path/"cache", reference_suite="stage1-reference")
        run_multi.main((case,), defaults=defaults)

    search = captured[0]
    # Include a motif added by reflection after the MC/DP screens were constructed.
    search.motifs["new_summary"] = const(10)
    if stage == 1:
        search.screen.motifs["new_summary"] = const(10)
    else:
        search.screen.register_motif("new_summary", const(10))
    ast = cap({"op": "motif", "name": "new_summary"}, parameter("C"), "cyclic_integer")
    candidate = SearchCandidate(ast, "parent", "cap", 1, 1, ("C",), "test")
    parent = SearchIndividual(candidate, {"C": 1.5}, 1, 1, 0, "complete", "test", 0, 0, "")
    parents = [parent if stage == 1 else SimpleNamespace(case_results={case.name: parent})]
    evidence = search._parent_evidence(parents)
    diagnostic = evidence["diagnostics"][0]
    assert diagnostic["available"] and diagnostic["occupancy_source"] == "dp_optimal"
    assert diagnostic["regret"]["parent"] and diagnostic["action_spread"]
    assert evidence["case_evaluations"][0]["fitted_parameters"] == {"C": 1.5}
    for row in diagnostic["regret"]["parent"]:
        t = row["period"]
        assert row["parent_action"] == round(1.5*(t+1))-round(1.5*t)
    assert all(row["weighted_action_spread"] == 0 for row in diagnostic["action_spread"])


def test_mc_only_screen_without_parents_does_not_sample():
    from sr.inventory.mc import MonteCarloScreen
    screen = MonteCarloScreen(SimpleNamespace(case=SimpleNamespace(name="mc-only")), {})
    assert screen.parent_diagnostics(()) == {
        "case": "mc-only", "available": False, "reason": "no_parents",
    }


def test_crossover_credits_both_parents_and_ignores_pending(tmp_path):
    domain = family_domain((3,))
    a = PolicyFamily("a", parameter("S"), {}, (FamilyParameter("S", "target", 0, 10),))
    b = PolicyFamily("b", {"op": "sub", "left": parameter("T"), "right": {"op": "motif", "name": "inventory_position"}}, {}, (FamilyParameter("T", "target", 0, 10),))
    child = _apply_family_crossover(a, b, {"family_id": "child", "combination": "minimum"}, domain=domain)
    restored = family_from_json({**child.__dict__,
        "parameters": [p.__dict__ for p in child.parameters]}, domain=domain)
    assert restored.parent_family_hashes == child.parent_family_hashes
    search = MultiContextFamilySearch(domain=domain, initial_families=(a,b),
        case_ids=[str(i) for i in range(8)], worker=lambda *_: None,
        output_dir=tmp_path, iterations=0, feature_init_pass=False)
    search._validate_generation((child,), (a,b))
    archive = [FamilyEvaluation(a, (), (), (), 10, "complete"),
               FamilyEvaluation(b, (), (), (), 12, "complete")]
    search._update_bandits((child,), archive)
    assert not search._parent_visits
    archive.append(FamilyEvaluation(child, (), (), (), 8, "complete"))
    search._update_bandits((child,), archive)
    assert set(search._parent_visits.values()) == {1}
    assert sum(search._parent_rewards.values()) == pytest.approx(1.2)
    assert len(set(search._parent_rewards.values())) == 1
    assert domain.semantic_hash(replace(child, parent_family_hashes=())) == domain.semantic_hash(child)


def test_search_limits_expand_motifs_but_do_not_tighten_baseline_execution():
    domain = family_domain((3,))
    family = PolicyFamily("large", {"op": "motif", "name": "hidden"}, {
        "hidden": {"op": "add", "args": [const(1)]*70}}, ())
    domain.validate_family(family)
    with pytest.raises(ValueError, match="nodes"):
        validate_search_family(family, domain)


def test_reflection_receives_numeric_evidence_and_only_extracts_parent_subtrees():
    domain = family_domain((3,))
    family = PolicyFamily("stock", {"op": "motif", "name": "inventory_position"}, {}, ())
    evidence = {domain.semantic_hash(family): {"case_evaluations": [
        {"case_id": "case", "constants": {"S": 2.3}, "cost": 7.8}],
        "diagnostics": {"available": False, "reason": "no_dp_reference"}}}
    class Client:
        def complete(self, prompt):
            self.prompt = prompt
            return json.dumps({"motifs": [
                {"name": "stock_copy", "ast": family.ast},
                {"name": "invented_const", "ast": const(12345)},
                {"name": "copy2", "ast": family.ast},
                {"name": "copy3", "ast": family.ast}]})
    client = Client()
    out = llm_reflect_motifs(client, (family,), 5, domain=domain, parent_evaluations=evidence)
    assert "2.3" in client.prompt and "7.8" in client.prompt
    assert len(out["accepted"]) == 2
    assert any("subtree" in message for message in out["rejected"])


def test_single_prompts_use_fitted_params_and_never_mc_surrogate_as_bellman_regret():
    candidate = SearchCandidate(state("x"), "a", "x", 1, 1, (), "test",
                                detail={"bellman_regret_pct": 999})
    parent = SearchIndividual(candidate, {}, 7.8, 7.9, -1, "complete", "mc", 0, 0, "")
    class Client:
        def complete(self, prompt):
            self.prompt = prompt
            return '{"proposals":[{"action":"replace_subtree","parent_hash":"a","path":[],"ast":{"op":"const","value":1}}]}'
    client = Client()
    rows = propose_asts(client, (parent,), {}, SearchConfig(),
        SearchDomain("toy", "cost", ("x",)), {"diagnostics": [{"available": False, "reason": "no_dp_reference"}]})
    assert len(rows) == 1 and rows[0][2] == ("a",)
    assert "no_dp_reference" in client.prompt and "999" not in client.prompt


def test_stage3_adapters_request_twenty_edits_and_forward_same_evidence(tmp_path, monkeypatch):
    from sr.family.stage3_runner import _llm_functions
    from sr.llm import LLMConfig
    mutation_calls, reflection_calls = [], []
    monkeypatch.setattr(LLMConfig, "from_json", lambda _: object())
    monkeypatch.setattr("sr.llm.JSONChatClient", lambda *a, **k: object())
    monkeypatch.setattr("sr.family.proposals.llm_proposal_batch_fn",
                        lambda *a, **k: mutation_calls.append((a, k)) or ())
    monkeypatch.setattr("sr.family.proposals.llm_reflect_motifs",
                        lambda *a, **k: reflection_calls.append((a, k)) or {})
    args = SimpleNamespace(mutation_config=tmp_path/"config", reflection_config=None,
                           output_dir=tmp_path)
    proposal, reflection = _llm_functions(args, family_domain((3,)))
    parents, evidence = (object(),), {"evaluated_parent": {"cost": 123.4}}
    proposal(parents, 4, evidence)
    reflection(parents, 5, evidence)
    assert mutation_calls[0][1]["count"] == 20
    assert mutation_calls[0][1]["parent_evaluations"] is reflection_calls[0][1]["parent_evaluations"]
    assert reflection_calls[0][0][1] is parents
