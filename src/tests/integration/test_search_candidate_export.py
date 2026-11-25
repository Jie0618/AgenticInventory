"""Search outputs can enter independent MC campaigns without paper policy IDs."""
from dataclasses import asdict
from importlib import import_module
import json
from pathlib import Path
import sys

import pytest

from benchmarks.inventory_benchmark_family.mc_protocol import (
    load_campaign_config, load_campaign_selection, load_policy_set,
)
from benchmarks.inventory_benchmark_family.policy_assets import write_policy_set
from benchmarks.inventory_benchmark_family.tools.search_candidates import (
    ROOT, SETTINGS, export_search_candidates,
)
from inventory.policy.family import FamilyParameter, PolicyFamily


def checkpoint(tmp_path, *, portable=True):
    family = PolicyFamily(
        "new_llm_rule", {"op": "parameter", "name": "Q"}, {},
        (FamilyParameter("Q", "target", 0, 2),), "A new searched rule.")
    row = dict(family_id=family.family_id, family_hash="evaluated_hash",
               semantic_hash="evaluated_hash", status="complete", cross_context_score=1.0)
    if portable:
        row["family"] = asdict(family)
    else:
        proposals = tmp_path / "proposals"
        proposals.mkdir()
        (proposals / "PROPOSALS_ITER_1.json").write_text(json.dumps({
            "iteration": 1, "families": [{**asdict(family), "family_hash": "evaluated_hash"}]}))
        # An interrupted future proposal must not replace a committed definition.
        (proposals / "PROPOSALS_ITER_3.json").write_text(json.dumps({
            "iteration": 3, "families": [{
                **asdict(family), "family_hash": "evaluated_hash",
                "ast": {"op": "parameter", "name": "wrong"}}]}))
    archive = [
        {**row, "status": "timeout", "family_hash": "timeout", "semantic_hash": "timeout",
         "cross_context_score": -100},
        {**row, "family_hash": "nonfinite", "semantic_hash": "nonfinite",
         "cross_context_score": float("inf")},
        row,
    ]
    path = tmp_path / "CHECKPOINT.json"
    path.write_text(json.dumps(dict(iteration=1, archive=archive)))
    return path


@pytest.mark.parametrize("setting", SETTINGS)
def test_checkpoint_export_resolves_and_evaluates_new_family_in_every_setting(setting, tmp_path, monkeypatch):
    payload = export_search_candidates(checkpoint(tmp_path), setting)
    assert payload["selection_kind"] == "search_candidates"
    assert payload["selection_iteration"] == 1
    symbolic, = [p for p in payload["policies"] if p["type"] == "formula_family"]
    assert symbolic["id"] == "search_evaluated_hash"
    assert symbolic["family"]["parameters"] == [
        dict(name="Q", kind="target", lower=0, upper=2,
             index_rule="scalar", context_dependence="independent_per_case")]
    manifest = tmp_path / "policies/sets/new.json"
    write_policy_set(payload, manifest, externalize=True)
    model = import_module(f"experiments.{setting}.benchmark_campaigns.inventory_benchmark_family.mc_model")
    campaign = Path(model.__file__).parent
    settings = load_campaign_config(campaign / "configs/standard.json")
    if setting in ("lead_time", "perishable"):
        _, definitions = load_policy_set(manifest)
        definition, = [p for p in definitions if p.id == symbolic["id"]]
        case = model.MODEL.cases(settings, contexts=(3,))[0]
        # Campaign planning resolves without a case; workers supply the actual case.
        planned = model.resolve_policy(definition, 3, case.n)
        resolved = model.resolve_policy(definition, 3, case.n, case=case)
        assert planned.ast == resolved.ast
        from benchmarks.inventory_benchmark_family.run_one_mc import main as evaluate_one
        from experiments.benchmark_campaign import main as evaluate_campaign
        config = tmp_path / "evaluation.json"
        raw = json.loads((campaign / "configs/standard.json").read_text())
        raw.update(state_caps={"3": case.n}, horizon=2, proposal_budget=2,
                   proposal_paths=2, tuning_paths=3, evaluation_paths=4)
        config.write_text(json.dumps(raw))
        # Cover the actual campaign planner and numerical worker for a new name.
        evaluate_campaign([
            "--setting", setting, "--config", str(config), "--policy-set", str(manifest),
            "--run-dir", str(tmp_path / "results"), "--method-ids", symbolic["id"],
            "--limit-instances", "1", "--dry-run"])
        output = tmp_path / "measurement.json"
        monkeypatch.setattr(sys, "argv", [
            "run_one_mc", "--setting", setting, "--model-module", model.__name__,
            "--instance", case.name, "--policy-id", symbolic["id"],
            "--config", str(config), "--policy-set", str(manifest), "--output", str(output)])
        evaluate_one()
        row = json.loads(output.read_text())
        assert row["policies"][0]["policy_id"] == symbolic["id"]
        assert len(row["path_costs"][symbolic["id"]]) == 4
    else:
        frozen = load_campaign_selection(manifest, model.BASELINES)
        settings.update(contexts=[[3, 3]], horizon=2, fidelity="fast", proposal_budget=2,
                        proposal_paths=2, strict_paths=3, evaluation_paths=4)
        model.validate_campaign(settings, frozen, (symbolic["id"],), baseline_only=False)
        case = next(iter(model.cases(settings)))
        resolved, = [p for p in model.resolve(case, frozen) if p["id"] == symbolic["id"]]
        result = model.evaluate(case, resolved, settings)
        model.validate_cached(result, case, resolved, settings)
        assert result["policy"] == symbolic["id"]
        assert result["params"]["Q"] in (0, 1, 2)


def test_older_checkpoint_export_uses_committed_proposals_and_rejects_missing_selection(tmp_path):
    source = checkpoint(tmp_path, portable=False)
    payload = export_search_candidates(source, "lead_time", family_ids=["new_llm_rule"])
    symbolic, = [row for row in payload["policies"] if row["type"] == "formula_family"]
    assert symbolic["family"]["ast"] == {"op": "parameter", "name": "Q"}
    with pytest.raises(ValueError, match="completed candidate"):
        export_search_candidates(source, "lead_time", family_ids=["missing"])
    with pytest.raises(ValueError, match="Only 1"):
        export_search_candidates(source, "lead_time", top_k=2)


@pytest.mark.parametrize("portable", (True, False))
@pytest.mark.parametrize("organized", (True, False))
def test_export_ignores_truncated_uncommitted_proposals(tmp_path, portable, organized):
    source = checkpoint(tmp_path, portable=portable)
    directory = tmp_path / "proposals" if organized else tmp_path
    directory.mkdir(exist_ok=True)
    (directory / "PROPOSALS_ITER_2.json").write_text('{"iteration": 2, "families": [')
    if portable:
        # Embedded definitions must also make damaged historical proposals irrelevant.
        (directory / "PROPOSALS_ITER_0.json").write_text('{"iteration": 0, "families": [')
    payload = export_search_candidates(source, "lead_time")
    symbolic, = [row for row in payload["policies"] if row["type"] == "formula_family"]
    assert symbolic["family"]["ast"] == {"op": "parameter", "name": "Q"}


def test_export_only_recovers_selected_candidates_and_reports_corrupt_committed_proposals(tmp_path):
    source = checkpoint(tmp_path)
    saved = json.loads(source.read_text())
    row = saved["archive"][-1]
    saved["archive"].append({
        **{key: value for key, value in row.items() if key != "family"},
        "family_id": "older_candidate", "family_hash": "older_hash",
        "semantic_hash": "older_hash", "cross_context_score": 2.0})
    source.write_text(json.dumps(saved))
    proposals = tmp_path / "proposals"
    proposals.mkdir()
    (proposals / "PROPOSALS_ITER_1.json").write_text('{"iteration": 1, "families": [')
    # An unselected legacy candidate must not force reads of proposal files.
    payload = export_search_candidates(source, "lead_time")
    assert payload["policies"][-1]["display_name"] == "new_llm_rule"
    with pytest.raises(ValueError, match="invalid committed proposal: .*PROPOSALS_ITER_1.json"):
        export_search_candidates(source, "lead_time", top_k=2)


def test_freeze_cli_exports_checkpoint_and_preserves_input_mode(tmp_path, monkeypatch):
    from benchmarks.inventory_benchmark_family.tools.freeze_policies import main
    source = checkpoint(tmp_path)
    output = tmp_path / "policies/sets/search.json"
    monkeypatch.setattr(sys, "argv", [
        "freeze_policies", "--checkpoint", str(source), "--setting", "lead_time",
        "--family-ids", "new_llm_rule", "--output", str(output)])
    main()
    assert load_policy_set(output)[1][-1].id == "search_evaluated_hash"
    copy = output.with_name("copy.json")
    monkeypatch.setattr(sys, "argv", [
        "freeze_policies", "--input", str(output), "--output", str(copy)])
    main()
    assert load_policy_set(copy) == load_policy_set(output)


@pytest.mark.parametrize("setting", ("lead_time_perishable", "lead_time_backlog"))
def test_paper_selection_guards_remain_and_custom_ids_cannot_collide(setting):
    model = import_module(f"experiments.{setting}.benchmark_campaigns.inventory_benchmark_family.mc_model")
    campaign = Path(model.__file__).parent
    settings = load_campaign_config(campaign / "configs/standard.json")
    frozen = load_campaign_selection(campaign / "policies/sets/mc_final.json", model.BASELINES)
    ids = tuple(row["id"] for row in frozen["policies"] if row.get("family"))
    model.validate_campaign(settings, frozen, ids, baseline_only=False)
    with pytest.raises(ValueError, match="100-round"):
        model.validate_campaign(settings, {**frozen, "selection_iteration": 1}, ids, baseline_only=False)
    with pytest.raises(ValueError, match="100-round"):
        model.validate_campaign(settings, frozen, ("new_name",), baseline_only=False)
    custom = {**frozen, "selection_kind": "search_candidates", "selection_iteration": 1}
    with pytest.raises(ValueError, match="distinct from baseline"):
        model.validate_campaign(settings, custom, (model.BASELINES[0],), baseline_only=False)
    with pytest.raises(ValueError, match="unique pairs"):
        model.validate_campaign({**settings, "contexts": [[3, 3], [3, 3]]},
                                custom, ("new_name",), baseline_only=False)


@pytest.mark.parametrize("setting", ("lead_time_perishable", "lead_time_backlog"))
def test_custom_campaign_schedules_new_names_and_writes_complete_report(setting, tmp_path, monkeypatch, capsys):
    from experiments.benchmark_campaign import run_joint_campaign
    import experiments.benchmark_campaign as coordinator

    payload = export_search_candidates(checkpoint(tmp_path), setting)
    manifest = tmp_path / "policies/sets/new.json"
    write_policy_set(payload, manifest)
    campaign = ROOT / "experiments" / setting / "benchmark_campaigns/inventory_benchmark_family"
    settings = json.loads((campaign / "configs/standard.json").read_text())
    settings.update(contexts=[[3, 3]], horizon=2, proposal_budget=2,
                    proposal_paths=2, tuning_paths=3, evaluation_paths=4, fidelity="fast")
    config = tmp_path / "evaluation.json"
    config.write_text(json.dumps(settings))
    model = import_module(f"experiments.{setting}.benchmark_campaigns.inventory_benchmark_family.mc_model")
    case = next(iter(model.cases(settings)))
    output = tmp_path / "results"
    args = ["--config", str(config), "--policy-set", str(manifest), "--run-dir", str(output),
            "--instances", case.name, "--workers", "1"]
    run_joint_campaign(setting, [*args, "--method-ids", "search_evaluated_hash", "--dry-run"])
    plan = json.loads(capsys.readouterr().out)
    assert plan["tasks"] == 1
    assert plan["scheduled_method_ids"] == ["search_evaluated_hash"]
    assert not output.exists()

    def complete(pending, *, rows, **kwargs):
        # Exercise the real coordinator/report path without fitting every baseline.
        for i, (case, policy, path) in enumerate(pending):
            row = dict(case=case.name, policy=policy["id"], cost=10.0 + i,
                       L=3, **{"K" if setting == "lead_time_backlog" else "M": 3},
                       bank_seeds={"final": 123}, evaluation_paths=4)
            path.write_text(json.dumps(row))
            rows.append(row)

    monkeypatch.setattr(coordinator, "run_isolated_tasks", complete)
    run_joint_campaign(setting, args)
    assert (output / "SUMMARY.csv").is_file()
    assert (output / "OVERALL.csv").is_file()
    assert "search_evaluated_hash" in (output / "OVERALL.csv").read_text()
    protocol = json.loads((output / "MC_PROTOCOL.json").read_text())
    assert protocol["frozen_selection"]["selection_iteration"] == 1
    if setting == "lead_time_backlog":
        assert "search_evaluated_hash" in protocol["reference"]
        assert "BATS" not in protocol["reference"]
    with pytest.raises(ValueError, match="Unknown method"):
        run_joint_campaign(setting, [*args, "--method-ids", "unknown", "--dry-run"])
