"""Lightweight entry-point, policy-roster and report contracts."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.launch import ROOT, SETTINGS, stage_command, stage_directories


def option(command, name):
    matches = [command[i + 1] for i, value in enumerate(command[:-1]) if value == name]
    return matches[-1]


@pytest.mark.parametrize("setting", ("lead_time_backlog", "lead_time_perishable"))
def test_joint_benchmark_policy_sets_explicitly_declare_the_registered_roster(setting):
    from importlib import import_module
    from benchmarks.inventory_benchmark_family.mc_protocol import (
        BASELINE_TYPES, load_policy_set, load_campaign_selection, load_campaign_config)

    campaign = ROOT / "experiments" / setting / "benchmark_campaigns/inventory_benchmark_family"
    model = import_module(f"experiments.{setting}.benchmark_campaigns.inventory_benchmark_family.mc_model")
    for name in ("mc_final", "common_baselines"):
        path = campaign / f"policies/sets/{name}.json"
        _, definitions = load_policy_set(path)
        assert {p.id for p in definitions if p.type in BASELINE_TYPES} == set(model.BASELINES)
        frozen = load_campaign_selection(path, model.BASELINES)
        assert all(p.get("family") or p.get("name") == "codex" for p in frozen["policies"])
    config = load_campaign_config(campaign / "configs/standard.json")
    assert config["strict_paths"] == 1000
    assert config["fidelity"] == "certify"
    assert config["proposal_budget"] is None


@pytest.mark.parametrize("setting", ("lead_time_backlog", "lead_time_perishable"))
def test_joint_benchmark_report_validates_membership(setting, tmp_path):
    from copy import deepcopy
    from types import SimpleNamespace
    from benchmarks.inventory_benchmark_family.tools.build_mc_report import write_joint_report

    axis = "K" if setting == "lead_time_backlog" else "M"
    cases = [SimpleNamespace(name=f"L3_{axis}3")]
    model = SimpleNamespace(SETTING=setting)
    methods = ("base_stock", "capped_base_stock")
    rows = [dict(case=cases[0].name, policy=pid, cost=10. + j,
                 L=3, **{axis: 3}, bank_seeds={"final": 123}, evaluation_paths=5)
            for j, pid in enumerate(methods)]
    write_joint_report(tmp_path, rows, cases, model=model, method_ids=methods)
    assert (tmp_path / "SUMMARY.csv").is_file()
    assert (tmp_path / "OVERALL.csv").is_file()
    with pytest.raises(ValueError, match="incomplete"):
        write_joint_report(tmp_path, rows[:-1], cases, model=model, method_ids=methods)
    mismatched = deepcopy(rows)
    mismatched[0]["bank_seeds"]["final"] = 456
    with pytest.raises(ValueError, match="banks differ"):
        write_joint_report(tmp_path, mismatched, cases, model=model, method_ids=methods)


@pytest.mark.parametrize("context_fields", (("lead_time",), ("shelf_life",), ("L", "M"), ("L", "K")))
def test_compact_score_tables_do_not_require_measurements(context_fields, tmp_path):
    from reporting.scores import load_scores
    from reporting.tables import write_csv

    identifier = "policy_id" if len(context_fields) == 1 else "policy"
    row = {identifier: "base_stock", "count": 4, "mean_gap_pct": 1.,
           "cvar80_gap_pct": 2., "fitness": 1.4}
    context = dict(row, **dict.fromkeys(context_fields, 3))
    write_csv(tmp_path / "SUMMARY.csv", [context])
    write_csv(tmp_path / "OVERALL.csv", [dict(row, count=1)])
    contexts, overall = load_scores(tmp_path, context_fields)
    assert contexts == [context]
    assert overall[0]["fitness"] == 1.4
    assert not (tmp_path / "RESULTS.csv").exists()
    write_csv(tmp_path / "SUMMARY.csv", [context, context])
    with pytest.raises(ValueError, match="Duplicate score"):
        load_scores(tmp_path, context_fields)
    write_csv(tmp_path / "SUMMARY.csv", [dict(context, fitness=float("nan"))])
    with pytest.raises(ValueError, match="Nonfinite score"):
        load_scores(tmp_path, context_fields)
    write_csv(tmp_path / "SUMMARY.csv", [context])
    write_csv(tmp_path / "OVERALL.csv", [dict(row, **{identifier: "other_policy"})])
    with pytest.raises(ValueError, match="rosters differ"):
        load_scores(tmp_path, context_fields)


@pytest.mark.parametrize("setting", SETTINGS)
def test_pipeline_paths_are_connected_and_stage_overrides_are_isolated(setting, tmp_path):
    roots = {f"STAGE{i}": str(tmp_path / f"stage {i}") for i in (1, 2, 3)}
    env = {
        **roots, "PYTHON_BIN": sys.executable, "EXPERIMENT_PIPELINE": "1",
        "OUTPUT_DIR": str(tmp_path / "unrelated"), "CHECKPOINT_ROOT": "unrelated",
        "STAGE2_CHECKPOINT": "unrelated", "EVALUATION_CONFIG": "unrelated",
    }
    first, second, third = [stage_command(setting, i, environ=env) for i in (1, 2, 3)]
    output_key = "--output-root" if setting in ("lead_time", "perishable") else "--output-dir"
    assert option(first, output_key) == roots["STAGE1"]
    assert option(second, "--checkpoint-root") == roots["STAGE1"]
    assert option(third, "--stage2-checkpoint") == str(Path(roots["STAGE2"]) / "checkpoint.json")
    assert "unrelated" not in first + second + third
    assert option(third, "--output-dir") == roots["STAGE3"]


def test_single_case_and_explicit_cli_overrides_preserve_spaces(tmp_path):
    command = stage_command(
        "lead_time", 1,
        ["--case", "case name", "--output-dir", str(tmp_path / "explicit result"), "--workers", "2"],
        environ={"OUTPUT_ROOT": str(tmp_path / "suite root"), "WORKERS": "60"},
    )
    assert command[2:5] == [str(ROOT / "experiments/dispatch.py"), "lead_time", "single"]
    assert "--suite-config" not in command
    assert option(command, "--output-dir") == str(tmp_path / "explicit result")
    assert option(command, "--workers") == "2"


def test_explicit_python_and_stage2_checkpoint_are_retained(tmp_path):
    command = stage_command("lead_time", 3, environ={
        "PYTHON_BIN": "/custom environment/bin/python",
        "STAGE2_CHECKPOINT": str(tmp_path / "seed checkpoint.json"),
        "ITERATIONS": "100", "WORKERS": "80",
    })
    assert command[:2] == ["/custom environment/bin/python", "-u"]
    assert "--seed-set" not in command
    assert option(command, "--workers") == "80"
    assert option(command, "--stage2-checkpoint") == str(tmp_path / "seed checkpoint.json")


@pytest.mark.parametrize("setting", SETTINGS)
@pytest.mark.parametrize("stage", (1, 2, 3))
def test_launch_uses_current_python_environment(setting, stage):
    assert stage_command(setting, stage, environ={})[0] == sys.executable


@pytest.mark.parametrize("setting,stage", (
    ("lead_time", 1), ("perishable", 2),
    ("lead_time_backlog", 3), ("lead_time_perishable", 3),
))
def test_stage_help_does_not_create_results_or_controller_log(setting, stage, tmp_path):
    output = tmp_path / "unused results"
    log = tmp_path / "unused logs" / "controller.log"
    env = dict(
        os.environ, PYTHON_BIN=sys.executable, OUTPUT_DIR=str(output),
        OUTPUT_ROOT=str(output), LOG_FILE=str(log), USE_CONDA_RUN="0", DRY_RUN="0",
    )
    env.pop("EXPERIMENT_PIPELINE", None)
    result = subprocess.run(
        ["bash", str(stage_directories(setting)[stage - 1] / "run.sh"), "--help"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert not output.exists()
    assert not log.parent.exists()
