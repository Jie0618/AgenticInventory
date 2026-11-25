"""No-DP spread, fixed simulated states, fitted constants, and actual prompt wiring."""
import json
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from inventory.common.distributions import DiscreteDemand
from inventory.core import DeterministicLeadTime, DeterministicBacklog, FixedShelfLife, TransitionCosts
from inventory.core.factory import inventory_case
from inventory.policy.ast import cap, const, evaluate_ast, parameter
from inventory.policy.family import FamilyParameter, PolicyFamily
from inventory.policy.family_domain import InventoryFamilyDomain
from optimization.evaluators.composed import cost_features_for_case
from sr.core.diagnostics import decision_records
from sr.core.models import SearchCandidate, SearchIndividual
from sr.family.diagnostics import FamilyParentDiagnostics, SHARED_DIAGNOSTICS
from sr.family.models import CaseMetric
from sr.family.proposals import _apply_family_mutation, llm_proposal_batch_fn, llm_reflect_motifs
from sr.family.search import MultiContextFamilySearch
from sr.inventory.diagnostics import MonteCarloDiagnostics
from sr.inventory.mc import MonteCarloScreen, MonteCarloSearchContext
from sr.single.controller import SingleSearchController
from sr.multi.screening import MultiCandidateScreen
from sr.multi.models import MultiObjective


def tiny_case(setting, name="tiny"):
    components = {}
    if setting != "perishable":
        components["lead_time"] = DeterministicLeadTime(2)
    if setting in ("perishable", "lead_time_perishable"):
        components["perishability"] = FixedShelfLife(2)
    if setting == "lead_time_backlog":
        components["backlog"] = DeterministicBacklog(2)
    return inventory_case(name=name, demand=DiscreteDemand.from_pmf({0: .25, 1: .75}),
        n=3, capacity=3, costs=TransitionCosts(holding=1, shortage=4, expiry=2,
                                            backlog=1, abandonment=4), **components)


def fitted_parent(key, value):
    ast = cap(const(10), parameter("C"), "cyclic_integer")
    candidate = SearchCandidate(ast, key, "cap", 1, 1, ("C",), "test")
    return SearchIndividual(candidate, {"C": value}, 1, 1, 0, "complete", "test", 0, 0, "")


def test_spread_without_bellman_values_is_ranked_and_regret_is_absent():
    bank = [{"state": [0], "period": 2, "probability": 1},
            {"state": [1], "period": 4, "probability": 9}]
    out = decision_records(bank, {"a": [0, 1], "b": [3, 2]}, case="no-dp")
    assert out["available"] and not out["regret_available"]
    assert out["regret_unavailable_reason"] == "no_dp_reference"
    assert "regret" not in out
    assert out["action_spread"][0]["state"] == [1]  # .9*1 > .1*3
    assert out["action_spread"][0]["weighted_action_spread"] == pytest.approx(.9)
    assert out["action_spread"][0]["parent_actions"] == {"a": 1, "b": 2}
    assert all("optimal_actions" not in row and "bellman_cost_difference" not in row
               for row in out["action_spread"])
    with pytest.raises(ValueError, match="mixes"):
        decision_records([bank[0], {**bank[1], "q_cost_advantage": [0, 1]}],
                         {"a": [0, 1]}, case="invalid")
    many = [{"state": [i], "period": 2, "probability": 1} for i in range(25)]
    assert len(decision_records(many, {"a": [0]*25, "b": list(range(25))},
                               case="top-ten")["action_spread"]) == 10


@pytest.mark.parametrize("setting", ["lead_time", "perishable", "lead_time_backlog", "lead_time_perishable"])
def test_mc_bank_is_reproducible_and_uses_canonical_fitted_actions(setting, monkeypatch):
    monkeypatch.setattr("evaluation.cache.load_or_create_inventory_dp",
                        lambda *a, **k: pytest.fail("diagnostics must not solve a DP"))
    case = tiny_case(setting)
    first = MonteCarloDiagnostics(case, 30)
    second = MonteCarloDiagnostics(case, 30)
    assert first.rows == second.rows
    assert first.signature == second.signature
    assert sum(row["probability"] for row in first.rows) == pytest.approx(1)
    assert all(6 <= row["period"] <= 23 for row in first.rows)
    assert len({row["period"] for row in first.rows}) == 16
    motifs = {"new_summary": {"op": "add", "args": [const(10), {"op": "state", "name": case.schema.names[0]}]}}
    ast = cap({"op": "motif", "name": "new_summary"}, parameter("C"), "cyclic_integer")
    actions = first.actions(ast, {"C": 1.5}, motifs)
    for row, action in zip(first.rows, actions):
        state = dict(zip(case.schema.names, row["state"]))
        state.update(time=row["period"], remaining_horizon=30-row["period"])
        raw = evaluate_ast(ast, state, {"C": 1.5}, case.demand.feature_dict(),
                           cost_features_for_case(case), motifs)
        assert action == int(np.clip(np.rint(raw), 0, case.n))
    records = first.records({"a": actions, "b": first.actions(const(0), {}, {})})
    assert not records["regret_available"] and "regret" not in records
    assert records["occupancy_source"] == "fixed_mc_reference"
    assert any(row["weighted_action_spread"] > 0 for row in records["action_spread"])
    for row in records["action_spread"]:
        assert row["weighted_action_spread"] == pytest.approx(
            row["probability"] * (max(row["parent_actions"].values()) - min(row["parent_actions"].values())))
    assert first.rows is first.rows  # The bank is reused rather than resampled per parent.


@pytest.mark.parametrize("multi", [False, True], ids=["stage1", "stage2"])
def test_mc_controllers_forward_spread_with_each_cases_fitted_constants(multi):
    names = ("first", "second") if multi else ("first",)
    screens = {}
    for name in names:
        case = tiny_case("lead_time_backlog", name)
        context = MonteCarloSearchContext(case, 12, 1, {})
        screens[name] = MonteCarloScreen(context, {})
    screen = MultiCandidateScreen(screens, MultiObjective()) if multi else screens["first"]
    controller = SimpleNamespace(screen=screen, case_name="first", motifs={})
    parents = [fitted_parent("a", .5), fitted_parent("b", 1.5)]
    if multi:
        parents = [SimpleNamespace(case_results={name: fitted_parent(key, value+i)
                    for i, name in enumerate(names)})
                   for key, value in (("a", .5), ("b", 1.5))]
    evidence = SingleSearchController._parent_evidence(controller, parents)
    assert len(evidence["diagnostics"]) == len(names)
    signatures = [row["bank_signature"] for row in evidence["diagnostics"]]
    assert all(row["available"] and row["action_spread"] and "regret" not in row
               for row in evidence["diagnostics"])
    expected = {("first", "a"): .5, ("first", "b"): 1.5,
                ("second", "a"): 1.5, ("second", "b"): 2.5}
    for record in evidence["case_evaluations"]:
        assert record["fitted_parameters"]["C"] == expected[(record["case"], record["parent_hash"])]
    again = SingleSearchController._parent_evidence(controller, parents)
    assert [row["bank_signature"] for row in again["diagnostics"]] == signatures


@pytest.mark.parametrize("setting,stage3,worker_name", [
    ("lead_time", "stage3_multi_leadtime_sr", "CanonicalFamilyWorker"),
    ("perishable", "stage3_multi_perishable_sr", "FamilyWorker"),
    ("lead_time_backlog", "stage3_multi_context_sr", "FamilyWorker"),
    ("lead_time_perishable", "stage3_multi_context_sr", "FamilyWorker"),
])
def test_all_production_family_workers_get_spread_without_refitting(tmp_path, setting, stage3, worker_name):
    module = import_module(f"experiments.{setting}.{stage3}.runtime.family_worker")
    root = Path(__file__).resolve().parents[3]
    values = json.loads((root/"experiments"/setting/stage3/"configs/stage3.json").read_text())
    values["horizon"] = 12
    config = tmp_path/"worker.json"
    config.write_text(json.dumps(values))
    worker = getattr(module, worker_name)(config_path=config, reference_cache_dir=tmp_path/"cache",
                                        reference_mode="best_baseline")
    case = next(iter(worker.case_map.values()))
    context_id = str(case.lead_time) if setting == "lead_time" else case.name.split("__", 1)[0]
    case_id = case.name.split("__", 1)[1]
    domain = InventoryFamilyDomain.from_cases({context_id: case}, training_case_count=1, cyclic_cap=True)
    a = PolicyFamily("a", cap(const(10), parameter("C"), "cyclic_integer"), {},
                     (FamilyParameter("C", "cap", 0, 3),))
    b = PolicyFamily("b", const(0), {}, ())
    search = MultiContextFamilySearch(domain=domain, initial_families=(a, b), case_ids=(case_id,),
        worker=worker, output_dir=tmp_path/"out", workers=1, iterations=0, feature_init_pass=False)
    assert isinstance(search._diagnostic_provider, FamilyParentDiagnostics)
    evaluations = {
        domain.semantic_hash(parent): {"case_evaluations": [
            {"context_id": context_id, "case_id": case_id, "constants": params}]}
        for parent, params in ((a, {"C": 1.5}), (b, {}))
    }
    supplied = search._diagnostic_provider((a, b), evaluations)
    shared = supplied[SHARED_DIAGNOSTICS]
    assert shared["available"] and not shared["regret_available"]
    out = shared["cases"][0]
    assert out["case"] == case.name and out["action_spread"] and "regret" not in out
    assert not (tmp_path/"cache").exists()  # No reference fitting or DP cache was needed.
    evaluations[domain.semantic_hash(a)]["case_evaluations"][0]["constants"] = {}
    with pytest.raises(ValueError, match="incomplete"):
        search._diagnostic_provider((a, b), evaluations)


def test_family_search_sends_identical_shared_spread_to_both_agents(tmp_path):
    case = tiny_case("lead_time_backlog")
    domain = InventoryFamilyDomain.from_cases({"ctx": case}, training_case_count=1, cyclic_cap=True)
    a = PolicyFamily("a", cap(const(10), parameter("C"), "cyclic_integer"), {},
                     (FamilyParameter("C", "cap", 0, 3),))
    b = PolicyFamily("b", const(0), {}, ())
    class Worker:
        case_map = {"ctx__case": case}
        config = {"horizon": 12}
        def __call__(self, family, task):
            return CaseMetric(task.context_id, task.case_id, 1.,
                              metadata={"constants": {p.name: 1.5 for p in family.parameters}})
    proposals, reflections = [], []
    def propose(parents, iteration, evidence):
        proposals.append(evidence)
        return (_apply_family_mutation(parents[0], {
            "family_id": f"child-{iteration}", "action": "add_term", "fragment": const(iteration+1),
        }, (), domain=domain),)
    def reflect(parents, iteration, evidence):
        reflections.append(evidence)
        return {"accepted": [], "rejected": []}
    from sr.family.template_init import PROTOCOL
    def initialize(_):
        return (a, b), {"status": "complete"}
    initialize.protocol = PROTOCOL
    seed = PolicyFamily("source-seed", const(99), {}, ())
    MultiContextFamilySearch(domain=domain, initial_families=(seed,), case_ids=("case",),
        worker=Worker(), output_dir=tmp_path, workers=1, iterations=5, proposal_fn=propose,
        reflection_fn=reflect, feature_init_pass=False, initialization_fn=initialize).run()
    assert reflections[0] is proposals[4]
    shared = reflections[0][SHARED_DIAGNOSTICS]
    saved = json.loads((tmp_path/"audit/parents/PARENT_EVIDENCE_ITER_5.json").read_text())
    assert saved["parents"][SHARED_DIAGNOSTICS] == shared
    assert shared["cases"][0]["action_spread"] and "regret" not in shared["cases"][0]
    class Client:
        def complete(self, prompt):
            self.prompt = prompt
            return '{"mutations": [], "motifs": []}'
    mutation, reflection = Client(), Client()
    evidence = {**proposals[0], SHARED_DIAGNOSTICS: shared}
    llm_proposal_batch_fn(mutation, (a, b), 1, count=0, domain=domain, parent_evaluations=evidence)
    llm_reflect_motifs(reflection, (a, b), 5, domain=domain, parent_evaluations=evidence)
    assert json.dumps(shared, sort_keys=True) in mutation.prompt
    assert json.dumps(shared, sort_keys=True) in reflection.prompt
