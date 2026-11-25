"""Certification must refit stored frontiers through the real context interfaces."""
import json
import math
import sys

import pytest

from evaluation.routing import solve_inventory_dp
from inventory.common.distributions import DiscreteDemand
from inventory.core import DeterministicLeadTime, FixedShelfLife, LostSales, TransitionCosts
from inventory.core.factory import inventory_case
from inventory.policy.ast import ast_check, cap, const, motif, parameter
from optimization.samples.reference import reference_action_samples
from sr.inventory import certify_single as module


@pytest.mark.parametrize("setting", ("lead_time", "perishable"))
@pytest.mark.parametrize("cyclic", (False, True))
def test_certify_refits_reflected_frontier_with_real_context(setting, cyclic, tmp_path, monkeypatch):
    perishable = setting == "perishable"
    case = inventory_case(
        name=f"certify-{setting}", n=3, capacity=3,
        demand=DiscreteDemand.from_pmf({0: 0.5, 2: 0.5}),
        costs=TransitionCosts(holding=1.0, shortage=3.0, expiry=1.0),
        lead_time=None if perishable else DeterministicLeadTime(1),
        perishability=FixedShelfLife(2) if perishable else None,
        backlog=LostSales(),
    )
    horizon = 4
    reference = solve_inventory_dp(case, horizon)
    samples = reference_action_samples(reference)
    learned = const(1)
    ast = cap(motif("learned"), parameter("C"), mode="cyclic_integer") if cyclic else motif("learned")
    checked = ast_check(ast, motifs={"learned": learned})
    row = {
        "ast": ast, "ast_hash": checked.ast_hash, "gap_pct": 5.0,
        "num_parameters": 1 if cyclic else 0, "expanded_complexity": 4,
    }
    search = tmp_path / "search"
    search.mkdir()
    (search / "checkpoint.json").write_text(json.dumps({
        "config": {"cyclic_cap": cyclic},
        "reflection_motifs": [{"name": "learned", "ast": learned}],
        "parameter_profile": [row], "complexity_profile": [row],
    }))
    baselines = tmp_path / "baselines"
    baselines.mkdir()
    for i in range(7):
        (baselines / f"baseline-{i}.json").write_text(json.dumps({
            "name": case.name, "policy": f"baseline-{i}", "cost": 10.0 + i,
            "dp_cost": float(reference.value(0, case.initial_state)),
        }))
    settings = tmp_path / "evaluation.json"
    settings.write_text(json.dumps({
        "proposal_paths": 2, "strict_paths": 2, "evaluation_paths": 3,
        "screen_paths": 2, "screen_horizon": 4, "screen_parameter_points": 2,
        "certify_proposal_budget": 8,
    }))
    output = tmp_path / "certify.json"
    monkeypatch.setattr(module, "HORIZON", horizon)
    monkeypatch.setattr(
        module, "load_or_create_inventory_dp",
        lambda *args, **kwargs: reference,
    )
    monkeypatch.setattr(module, "diagnostic_samples", lambda *args: samples)
    monkeypatch.setattr(sys, "argv", [
        "certify", "--case", case.name, "--search-dir", str(search),
        "--output", str(output), "--dp-cache-dir", str(tmp_path / "dp"),
        "--diagnostic-root", str(tmp_path), "--baseline-results", str(baselines),
        "--cyclic-evaluation-config", str(settings),
    ])
    module.main(
        (case,), default_diagnostic_root=tmp_path,
        default_baseline_results=baselines,
    )
    result = json.loads(output.read_text())
    assert len(result["certify_results"]) == 1  # The two profiles share a finalist.
    assert len(result["seven_baselines"]) == 7
    fitted = result["certify_results"][0]
    assert fitted["ast_hash"] == checked.ast_hash
    assert math.isfinite(fitted["certify_cost"])
    assert set(fitted["params"]) == ({"C"} if cyclic else set())
    assert fitted["certify_gap_pct"] == pytest.approx(
        100.0 * (fitted["certify_cost"] / result["dp_cost"] - 1.0)
    )
    assert output.with_suffix(".md").is_file()
