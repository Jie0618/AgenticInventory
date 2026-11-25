"""Regressions for the audited EC.3 branches and production numeric settings."""
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from inventory.policy.ast import ASTLimits, CAP_MODES, const, minimum, parameter
from inventory.policy.family import FamilyParameter, PolicyFamily
from inventory.policy.family_domain import InventoryFamilyDomain
from sr.core.models import SearchCandidate, SearchConfig, SearchIndividual
from sr.core.population import IslandPopulation
from sr.family.diagnostics import FamilyParentDiagnostics
from sr.family.models import CaseMetric, FamilyEvaluation
from sr.family.search import MultiContextFamilySearch
from sr.family.template_init import PROTOCOL as TEMPLATE_PROTOCOL
from sr.inventory.mc import MonteCarloScreen, MonteCarloSearchContext
from sr.llm.proposals import StructuralProposal, execute_structural_proposal
from sr.single.controller import SingleSearchController
from tests.sr.test_mc_parent_diagnostics import tiny_case, fitted_parent


@pytest.mark.parametrize("mode", sorted(CAP_MODES))
@pytest.mark.parametrize("reverse", [False, True])
def test_plain_minimum_converts_to_each_typed_cap_without_losing_limits(mode, reverse):
    args = (parameter("C"), parameter("S"))
    ast = minimum(*(args[::-1] if reverse else args))
    proposal = StructuralProposal("change_cap_type", "parent", None, None, "", mode=mode)
    converted = execute_structural_proposal(
        proposal, {"parent": ast}, {}, limits=ASTLimits(max_depth=10, max_nodes=64, max_parameters=8))
    assert converted["op"] == "cap" and converted["mode"] == mode
    assert converted["limit"] == parameter("C")
    assert converted["arg"] == parameter("S")
    with pytest.raises(ValueError):
        execute_structural_proposal(
            proposal, {"parent": ast}, {}, limits=ASTLimits(max_parameters=1))


def test_mc_novelty_compares_fitted_actions_instead_of_ast_identity():
    context = MonteCarloSearchContext(tiny_case("lead_time_backlog"), 12, 1, {})
    screen = MonteCarloScreen(context, {})
    parent = fitted_parent("one", 1.5)
    equivalent = replace(parent, candidate=replace(parent.candidate, ast_hash="other",
        ast={"op": "add", "args": [parent.candidate.ast, const(0)]}))
    assert screen.parent_behavior_signature(parent) == screen.parent_behavior_signature(equivalent)
    assert screen.parent_behavior_signature(parent) != screen.parent_behavior_signature(
        replace(parent, params={"C": 0}))
    controller = SimpleNamespace(screen=screen, case_name="tiny")
    stored = SingleSearchController._with_fitted_behavior(controller, parent)
    assert stored.candidate.detail["behavior_source"] == "fitted_actions_on_fixed_diagnostic_bank"
    assert screen._mc_diagnostics is not None


def test_stage3_novelty_uses_full_fitted_vectors_and_reuses_bank():
    case = tiny_case("lead_time_backlog")
    domain = InventoryFamilyDomain.from_cases({"ctx": case}, training_case_count=1)
    provider = FamilyParentDiagnostics(
        SimpleNamespace(config={"horizon": 12}, case_map={"ctx__case": case}), domain)
    parent = PolicyFamily("a", parameter("S"), {}, (FamilyParameter("S", "target", 0, 3),))
    other = replace(parent, family_id="b",
                    ast={"op": "add", "args": [parent.ast, const(0)]})
    def evaluation(family, value):
        return FamilyEvaluation(family, (), (
            CaseMetric("ctx", "case", 0, metadata={"constants": {"S": value}}),
        ), (), 0, "complete")
    assert provider.behavior_signature(evaluation(parent, 1)) == provider.behavior_signature(
        evaluation(other, 1))
    assert provider.behavior_signature(evaluation(parent, 1)) != provider.behavior_signature(
        evaluation(parent, 2))
    assert len(provider.banks) == 1 and len(provider._fitted_actions) == 2


def test_complete_frontier_retains_eight_parameter_candidate():
    candidate = SearchCandidate(const(1), "eight", "eight", 8, 12,
                                tuple(f"S{i}" for i in range(8)), "test")
    parent = SearchIndividual(candidate, {}, 1, 1, 0, "complete", "test", 0, 0, "")
    population = IslandPopulation(SearchConfig())
    population.add(parent)
    assert population.public_frontier() == (parent,)


def test_reflection_updates_library_without_injecting_policy():
    trial = SimpleNamespace(name="reflected", ast=const(1))
    population = IslandPopulation(SearchConfig())
    parent = fitted_parent("existing", 1.5)
    population.add(parent)
    candidates = []
    scheduled = []
    controller = SimpleNamespace(
        config=SearchConfig(), _ingested=set(), motifs={}, case_name="tiny", domain=None,
        population=population,
        reflection_client=object(), _reflections=[], _reflection_baselines={},
        screen=SimpleNamespace(motifs={}), scheduler=SimpleNamespace(),
        _remember=lambda candidate: candidates.append(candidate),
        _screen_candidates=lambda batch: scheduled.extend(batch),
        _reflection_representatives=lambda: (),
        _run_while_scheduling=lambda *args: (trial,),
    )
    SingleSearchController._maybe_reflect(controller, 5)
    assert controller.motifs == controller.screen.motifs == {"reflected": const(1)}
    assert controller._reflections == [trial]
    assert population.public_frontier() == (parent,)
    assert tuple(population.archive.values()) == (parent,)
    assert candidates == scheduled == []


def test_paper_initial_population_contains_only_llm_generalizations(tmp_path):
    case = tiny_case("lead_time_backlog")
    domain = InventoryFamilyDomain.from_cases({"ctx": case}, training_case_count=1)
    seed = PolicyFamily("seed", const(0), {}, ())
    children = tuple(PolicyFamily(f"child{i}", const(i), {}, ()) for i in range(1, 6))
    def initialize(_):
        return children, {"protocol": TEMPLATE_PROTOCOL, "status": "complete"}
    initialize.protocol = TEMPLATE_PROTOCOL
    calls = []
    def worker(family, task):
        calls.append(family.family_id)
        return CaseMetric(task.context_id, task.case_id, 0)
    kwargs = dict(domain=domain, initial_families=(seed,), case_ids=("case",),
                  worker=worker, output_dir=tmp_path, workers=1, iterations=0,
                  initialization_fn=initialize)
    result = MultiContextFamilySearch(**kwargs).run()
    assert {row.family.family_id for row in result} == {c.family_id for c in children}
    assert "seed" not in calls
    calls.clear()
    MultiContextFamilySearch(**kwargs, resume=True).run()
    assert calls == []


def test_paper_training_configs_keep_fitting_protocol():
    root = Path(__file__).resolve().parents[3]
    configs = paper_configs(root, training=True)
    for path in configs:
        settings = json.loads(path.read_text())
        from sr.inventory.runtime import HORIZON
        assert settings.get("horizon", HORIZON) == 1000, path
        assert settings["proposal_paths"] == 128, path
        assert settings.get("strict_paths", settings.get("tuning_paths")) == 1000, path
        assert settings["certify_proposal_paths"] == 512, path
        assert settings["certify_strict_paths"] == settings["evaluation_paths"] == 8000, path
        assert settings.get("standard_proposal_budget") is None, path
        assert settings.get("certify_proposal_budget") is None, path
        if path.name == "stage3.json":
            assert settings["stage2_seed_selection"] == "dual-frontier-union", path
            assert settings["llm_template_init_count_per_seed"] in (5, 6), path
            assert settings["llm_template_init_min_per_seed"] == 5, path
        context = MonteCarloSearchContext(tiny_case("lead_time_backlog"), 1000, 1,
            {**settings, "strict_paths": 1000}, fidelity="certify")
        assert context.paths("proposal") == 512 and context.paths("strict") == 8000


def paper_configs(root, *, training):
    """Select the shipped paper stages without constraining unrelated configs."""
    for setting, stage3 in (
        ("lead_time", "stage3_multi_leadtime_sr"),
        ("perishable", "stage3_multi_perishable_sr"),
        ("lead_time_backlog", "stage3_multi_context_sr"),
        ("lead_time_perishable", "stage3_multi_context_sr"),
    ):
        filename = "evaluation.json" if training else "search.json"
        for stage in ("stage1_single_distribution_sr", "stage2_multi_distribution_sr"):
            yield root / "experiments" / setting / stage / "configs" / filename
        if training:
            yield root / "experiments" / setting / stage3 / "configs/stage3.json"


def test_paper_search_configs_keep_structural_limits():
    root = Path(__file__).resolve().parents[3]
    for path in paper_configs(root, training=False):
        config = SearchConfig(**json.loads(path.read_text()))
        assert config.iterations == 100 and config.llm_proposals_per_iteration == 20
        assert config.reflection_iteration_interval == 5
        assert (config.max_parameters, config.max_depth, config.max_nodes,
                config.max_expanded_complexity) == (8, 10, 64, 150)
        assert config.gp_descendants_per_slot == 0
