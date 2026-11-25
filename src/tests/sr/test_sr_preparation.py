"""Numerical and sequencing checks for integrated per-case Stage1 inputs."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from inventory.common.distributions import DiscreteDemand
from inventory.core import DeterministicLeadTime, FixedShelfLife, TransitionCosts
from inventory.core.factory import inventory_case
from sr.inventory.preparation import prepare_case
from sr.inventory.single import diagnostic_samples


@pytest.mark.parametrize("perishable", [False, True])
def test_preparation_matches_bellman_and_reuses_inputs(tmp_path, monkeypatch, perishable):
    components = {"perishability": FixedShelfLife(2)} if perishable else {"lead_time": DeterministicLeadTime(2)}
    case = inventory_case(name="tiny", demand=DiscreteDemand.from_pmf({0: .5, 1: .5}),
                          n=2, capacity=2, costs=TransitionCosts(holding=1, shortage=4, expiry=2), **components)
    reference, path = prepare_case(case, 8, tmp_path / "inputs", tmp_path / "cache")
    payload = json.loads(path.read_text())
    assert diagnostic_samples(path, 8)
    for row in payload["representative_states"]:
        advantages = np.asarray(row["q_cost_advantage"])
        assert advantages.min() == pytest.approx(0)
        assert advantages[row["dp_action"]] == pytest.approx(0)
    before = path.read_bytes()
    from sr.inventory import preparation
    monkeypatch.setattr(preparation, "diagnostic", lambda *args: pytest.fail("recomputed diagnostic"))
    prepare_case(case, 8, tmp_path / "inputs", tmp_path / "cache")
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="identity differs"):
        prepare_case(case, 7, tmp_path / "inputs", tmp_path / "cache")




def test_stage2_reference_rejects_changed_config_and_nan(tmp_path):
    from sr.inventory.run_mc import prepared_reference
    case = SimpleNamespace(case_signature="a")
    path = tmp_path / "PREPARATION.json"
    payload = {"identity": {"case_signature": "a", "evaluation": {"horizon": 8},
                            "fidelity": "standard", "seed_namespace": "ltp"},
               "cost": 10., "kind": "best_formula_baseline_empirical_mc"}
    path.write_text(json.dumps(payload))
    assert prepared_reference(path, case, {"horizon": 8}, "standard", "ltp")["cost"] == 10
    with pytest.raises(ValueError, match="identity or reference"):
        prepared_reference(path, case, {"horizon": 9}, "standard", "ltp")
    payload["cost"] = float("nan")
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="identity or reference"):
        prepared_reference(path, case, {"horizon": 8}, "standard", "ltp")


def test_mc_stage2_exposes_resume_flag(tmp_path):
    from sr.inventory.run_mc import parser
    args = parser(2, tmp_path, prepare_per_case=True).parse_args([
        '--output-dir', str(tmp_path/'stage2'), '--checkpoint-root', str(tmp_path/'stage1'), '--resume'])
    assert args.resume




def test_dp_stage1_rejects_checkpoint_before_dp(tmp_path, monkeypatch):
    import sys
    from sr.inventory import run_single
    checkpoint = tmp_path/'out/checkpoint.json'
    checkpoint.parent.mkdir()
    checkpoint.write_text('{}')
    monkeypatch.setattr(sys, 'argv', ['single', '--case', 'tiny', '--output-dir', str(checkpoint.parent)])
    with pytest.raises(ValueError, match='Existing Stage1 checkpoint'):
        run_single.main((), prepare_inputs=True, default_diagnostic_root=tmp_path/'inputs')


def test_workflow_forwards_resume_only_to_search(tmp_path):
    from runtime.sr_workflow import WorkflowCliDefaults, parse_args, build_commands
    defaults = WorkflowCliDefaults(tmp_path, tmp_path/'init.py', tmp_path/'search.py',
        tmp_path/'experiment.json', tmp_path/'search.json', tmp_path/'stage1',
        tmp_path/'inputs', tmp_path/'cache', reference_suite='stage1-reference')
    args = parse_args(['--output-dir', str(tmp_path/'stage2'), '--resume'], defaults=defaults)
    commands = dict(build_commands(args, defaults=defaults))
    assert '--resume' in commands['search']
    assert '--resume' not in commands['initialization']


def test_case_suite_dry_run_does_not_create_output(tmp_path, monkeypatch):
    import sys
    from runtime.case_suite import CaseSuiteCliDefaults, main
    suite = tmp_path/'suite.json'
    suite.write_text(json.dumps({'name': 'tiny', 'cases': ['case'], 'workers': 1}))
    output = tmp_path/'absent'
    monkeypatch.setattr(sys, 'argv', ['suite', '--suite-config', str(suite),
        '--output-root', str(output), '--dry-run'])
    defaults = CaseSuiteCliDefaults(tmp_path, tmp_path/'search.py', tmp_path/'search.py',
        tmp_path/'inputs', tmp_path/'cache', tmp_path/'config.json', prepare_inputs_in_case=True)
    main(known_case_names={'case'}, defaults=defaults)
    assert not output.exists()






def test_stage2_workflow_resolves_paths_from_overridden_checkpoint_root(tmp_path):
    from runtime.sr_workflow import WorkflowCliDefaults, parse_args, build_commands
    defaults = WorkflowCliDefaults(tmp_path, tmp_path / "init.py", tmp_path / "search.py",
        tmp_path / "experiment.json", tmp_path / "search.json", tmp_path / "original",
        tmp_path / "legacy-inputs", tmp_path / "cache", reference_suite="stage1-reference")
    selected = tmp_path / "selected-stage1"
    args = parse_args(["--output-dir", str(tmp_path / "out"), "--checkpoint-root", str(selected)],
                      defaults=defaults)
    for _, command in build_commands(args, defaults=defaults):
        assert command[command.index("--diagnostic-root") + 1] == str(selected / ".inputs")
        assert command[command.index("--dp-cache-dir") + 1] == str(selected / ".dp_cache")
    explicit = parse_args(["--output-dir", str(tmp_path / "out"), "--checkpoint-root", str(selected),
        "--diagnostic-root", str(tmp_path / "custom-inputs"),
        "--dp-cache-dir", str(tmp_path / "custom-cache")], defaults=defaults)
    assert explicit.diagnostic_root == tmp_path / "custom-inputs"
    assert explicit.dp_cache_dir == tmp_path / "custom-cache"
