"""Public example selection and reporting, without fitting or simulation."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def example(monkeypatch):
    directory = Path(__file__).resolve().parents[3] / "examples"
    monkeypatch.syspath_prepend(str(directory))
    spec = spec_from_file_location("compare_policies_example", directory / "compare_policies.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def result(policy, cost):
    return dict(
        policy=policy, parameter_count=1, params={"S": 2},
        tuning_cost_per_period=cost, evaluation_cost_per_period=cost,
        standard_error_per_period=0.01, status="complete",
    )


@pytest.mark.parametrize(("arguments", "expected"), (
    (["--setting", "lead_time"], (
        ("lead_time", "co"), ("lead_time", "base_stock"), ("lead_time", "cbs"),
        ("lead_time", "weighted_tilted_pic"), ("lead_time", "SCDS"), ("lead_time", "IA-SCDS"),
    )),
    (["--setting", "perishable", "--policies", "RLCS", "EABS"],
     (("perishable", "EABS"), ("perishable", "RLCS"))),
    (["--policy", "base_stock"],
     (("lead_time", "base_stock"), ("perishable", "base_stock"))),
))
def test_cli_selects_catalog_order_and_shares_banks(example, monkeypatch, capsys, arguments, expected):
    banks = []
    calls = []

    def create(demand, horizon, paths, seed):
        bank = SimpleNamespace(seed=seed)
        banks.append(bank)
        return bank

    def fit(case, policy, args, tuning_bank, evaluation_bank):
        setting = "perishable" if case.shelf_life else "lead_time"
        calls.append((setting, policy["id"], tuning_bank, evaluation_bank))
        assert (args.tuning_paths, args.proposal_budget) == (4096, 4096)
        return result(policy["id"], 1.0)

    monkeypatch.setattr(example, "MCDemandBank", SimpleNamespace(create=create))
    monkeypatch.setattr(example, "fit_policy", fit)
    monkeypatch.setattr(sys, "argv", ["compare_policies.py", *arguments])
    example.main()
    assert tuple((setting, policy) for setting, policy, *_ in calls) == expected
    for group_index, setting in enumerate(dict.fromkeys(s for s, _ in expected)):
        tuning, evaluation = banks[2 * group_index:2 * group_index + 2]
        assert tuning.seed == 123 and evaluation.seed == 20261125
        assert tuning is not evaluation
        assert all(t is tuning and e is evaluation for s, _, t, e in calls if s == setting)
    output = capsys.readouterr().out
    assert output.count("ranked by MC/period") == len({s for s, _ in expected})


@pytest.mark.parametrize(("arguments", "message"), (
    (["--setting", "lead_time", "--policy", "RLCS"], "do not belong"),
    (["--evaluation-seed", "123"], "must differ"),
    (["--proposal-paths", "4097"], "must not exceed"),
))
def test_cli_rejects_incompatible_comparisons_before_work(example, monkeypatch, capsys, arguments, message):
    monkeypatch.setattr(example, "comparison_case", lambda _: pytest.fail("started an invalid comparison"))
    monkeypatch.setattr(sys, "argv", ["compare_policies.py", *arguments])
    with pytest.raises(SystemExit) as error:
        example.main()
    assert error.value.code == 2
    assert message in capsys.readouterr().err


def test_report_separates_settings_and_ranks_displayed_ties(example):
    rows = [
        dict(setting="lead_time", **result("third", 2.0)),
        dict(setting="perishable", **result("other", 0.1)),
        dict(setting="lead_time", **result("tie", 1.0000002)),
        dict(setting="lead_time", **result("first", 1.0000001)),
    ]
    report = example.format_report(rows)
    lead, perishable = report.split("perishable —", 1)
    table = [line for line in lead.splitlines() if line.startswith("| ")]
    ranked = [[cell.strip() for cell in line.split("|")[1:-1]] for line in table[2:5]]
    assert [(row[0], row[1]) for row in ranked] == [("1", "first"), ("1", "tie"), ("3", "third")]
    assert "other" not in lead
    assert "other" in perishable and "third" not in perishable
    assert "Fitted parameters" in lead and "S=2" in lead
