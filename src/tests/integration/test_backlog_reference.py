"""Training reference rosters, numerical equivalence and shared caches."""
from runtime.artifacts import seed_from_text, stable_key
import json
from importlib import import_module

import pytest

from experiments.lead_time_backlog import reference
from experiments.lead_time_backlog.stage3_multi_context_sr.runtime.family_worker import FamilyWorker
from experiments.launch import stage_directories
from sr.family.composed_reference import catalog_signature
from sr.inventory.run_mc import _reference_one
from sr.family.composed_reference import select_reference_baselines


def test_stage1_and_stage3_reuse_legacy_reference_cache(tmp_path, monkeypatch):
    config_path = stage_directories("lead_time_backlog")[2] / "configs/stage3.json"
    family_worker = FamilyWorker(config_path=config_path, reference_cache_dir=tmp_path)
    case = next(iter(family_worker.case_map.values()))
    mode = "standard"
    # This is the pre-extraction cache identity, independent of worker module paths.
    protocol = "lead-time-backlog-best-formula-baseline-shared-bank"
    signature = catalog_signature(protocol, reference._catalog(
        case, family_worker.config.get("reference_baselines")
    ))
    key = (
        f"{protocol}|{signature}|best_baseline|{case.case_signature}|{mode}|"
        f"{json.dumps(family_worker.config, sort_keys=True)}"
    )
    path = tmp_path / (stable_key(key.encode()) + ".json")
    payload = {
        "cost": 12.5, "policy": "stored-policy", "params": {"S": 4.0},
        "reference_type": "best_formula_baseline_empirical_mc",
    }
    path.write_text(json.dumps(payload))

    def unexpected_refit(**kwargs):
        raise AssertionError("A signed existing reference must not be refitted")

    monkeypatch.setattr(reference, "fit_best_baseline_reference", unexpected_refit)
    assert family_worker.reference_details(case, mode) == (payload, True)
    assert _reference_one((
        case, reference.ReferenceWorker, config_path, tmp_path, mode,
    )) == (case.name, 12.5, "best_formula_baseline_empirical_mc", True)


def test_new_references_are_shared_and_keep_fitting_seeds(tmp_path, monkeypatch):
    config_path = stage_directories("lead_time_backlog")[2] / "configs/stage3.json"
    family_worker = FamilyWorker(config_path=config_path, reference_cache_dir=tmp_path)
    case = next(iter(family_worker.case_map.values()))
    calls = []

    def fit_reference(**kwargs):
        calls.append(kwargs)
        assert kwargs["seed_namespace"] == "ltb"
        assert kwargs["seed"]("ltb-final", case.name, "standard") == seed_from_text(f"ltb-final|{case.name}|standard".encode())
        return 9.0, "new-policy", {"S": 3.0}, [], 11, 22

    monkeypatch.setattr(reference, "fit_best_baseline_reference", fit_reference)
    early_worker = reference.ReferenceWorker(config_path=config_path, reference_cache_dir=tmp_path)
    payload, hit = early_worker.reference_details(case, "standard")
    assert not hit
    assert payload["params"] == {"S": 3.0}
    assert payload["tuning_seed"] == 11
    assert payload["evaluation_seed"] == 22
    assert family_worker.reference_details(case, "standard") == (payload, True)
    assert len(calls) == 1


@pytest.mark.parametrize("names", [None, [], "base_stock", ["base_stock", "base_stock"],
                                  ["unknown"], ["base_stock", "unknown"], [1]])
def test_reference_rosters_reject_invalid_membership(names):
    with pytest.raises(ValueError, match="reference_baselines"):
        select_reference_baselines({"base_stock": object()}, names)


@pytest.mark.parametrize("setting,reference_mode", [
    ("lead_time", "fp3"), ("lead_time", "best_baseline"),
    ("perishable", "best_baseline"), ("lead_time_perishable", "best_baseline"),
    ("lead_time_backlog", "best_baseline"),
])
def test_configured_references_fit_and_reuse_cache(setting, reference_mode, tmp_path):
    stage = stage_directories(setting)[2]
    module_name = f"experiments.{setting}.{stage.name}.runtime.family_worker"
    current = import_module(module_name)
    config = json.loads((stage / "configs/stage3.json").read_text())
    config.update(horizon=4, state_action_cap=3, proposal_paths=2, strict_paths=3,
                  tuning_paths=3, evaluation_paths=5, fast_proposal_budget=8,
                  certify_proposal_paths=2, certify_strict_paths=3, certify_proposal_budget=8)
    if setting == "lead_time":
        config["state_caps"] = {key: 3 for key in config["state_caps"]}
        worker_name = "CanonicalFamilyWorker"
    else:
        worker_name = "FamilyWorker"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    new_worker = getattr(current, worker_name)(
        config_path=config_path, reference_cache_dir=tmp_path / "new", reference_mode=reference_mode)
    case = next(iter(new_worker.case_map.values()))
    context = str(case.lead_time) if setting == "lead_time" else case.name.split("__")[0]
    case_id = case.name.split("__", 1)[1]
    new_worker.prepare_reference(context, case_id, "fast")
    new_path, = (tmp_path / "new").glob("*.json")
    payload = json.loads(new_path.read_text())
    if setting == "lead_time":
        assert [p.id for p in new_worker.reference_definitions] == config["reference_baselines"]
        assert payload.get("fp3_cost", payload.get("cost")) >= 0
    else:
        assert [p["policy"] for p in payload["candidates"]] == config["reference_baselines"]
    snapshot = new_path.stat().st_mtime_ns
    new_worker.prepare_reference(context, case_id, "fast")
    assert new_path.stat().st_mtime_ns == snapshot
    # Membership changes must invalidate the worker's evaluation identity.
    original_identity = new_worker.evaluation_protocol_signature
    config["reference_baselines"] = ["constant_order"]
    config_path.write_text(json.dumps(config))
    subset = getattr(current, worker_name)(
        config_path=config_path, reference_cache_dir=tmp_path / "new", reference_mode="best_baseline")
    assert subset.evaluation_protocol_signature != original_identity
    subset.prepare_reference(context, case_id, "fast")
    assert len(list((tmp_path / "new").glob("*.json"))) == 2
    assert new_path.stat().st_mtime_ns == snapshot
    config["reference_baselines"] += ["unknown"]
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Unknown reference_baselines"):
        getattr(current, worker_name)(config_path=config_path, reference_mode="best_baseline")
