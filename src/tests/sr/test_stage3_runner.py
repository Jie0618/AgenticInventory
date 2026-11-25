from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sr.family.stage3_runner import (
    Stage3RunnerSpec,
    Stage3Setup,
    run_stage3,
    validate_stage3_config_declarations,
)


@pytest.fixture(autouse=True)
def configured_llm(monkeypatch):
    def proposal(*args):
        return ()
    proposal.initialize = lambda parents: ((), {"status": "complete"})
    monkeypatch.setattr("sr.family.stage3_runner._llm_functions", lambda *args: (proposal, None))


def test_shared_runner_preserves_setting_owned_search_inputs(tmp_path, capsys) -> None:
    (tmp_path / "config.json").write_text("{}")
    domain = SimpleNamespace(context_ids=("C1",))
    family = SimpleNamespace(family_id="seed")
    worker = object()
    captured = {}

    def prepare(args, _reference_mode):
        return Stage3Setup(
            domain=domain,
            initial_families=(family,),
            case_ids=("case",),
            worker=worker,
            feature_init_families=(family,),
        )

    spec = Stage3RunnerSpec(
        setting="lead_time",
        description="test",
        default_output_dir=tmp_path / "default",
        default_config=tmp_path / "config.json",
        default_reference_mode="fp3",
        reference_modes=("fp3", "best_baseline"),
        prepare=prepare,
    )

    class FakeSearch:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self):
            return (SimpleNamespace(cross_context_score=1.25),)

    with patch("sr.family.stage3_runner.MultiContextFamilySearch", FakeSearch):
        run_stage3(spec, [
            "--output-dir", str(tmp_path / "run"),
            "--iterations", "7",
            "--workers", "3",
        ])

    assert captured["domain"] is domain
    assert captured["initial_families"] == (family,)
    assert captured["case_ids"] == ("case",)
    assert captured["worker"] is worker
    assert captured["feature_init_families"] == (family,)
    assert captured["reference_mode"] == "fp3"
    assert captured["iterations"] == 7
    assert captured["workers"] == 3
    assert '"best_score": 1.25' in capsys.readouterr().out


def test_shared_runner_allows_zero_iteration_smoke_runs(tmp_path, capsys) -> None:
    (tmp_path / "config.json").write_text("{}")
    domain = SimpleNamespace(context_ids=("C1",))
    family = SimpleNamespace(family_id="seed")

    def prepare(args, _reference_mode):
        return Stage3Setup(
            domain=domain,
            initial_families=(family,),
            case_ids=("case",),
            worker=object(),
        )

    spec = Stage3RunnerSpec(
        setting="lead_time",
        description="test",
        default_output_dir=tmp_path / "default",
        default_config=tmp_path / "config.json",
        default_reference_mode="fp3",
        reference_modes=("fp3",),
        prepare=prepare,
    )

    class EmptySearch:
        def __init__(self, **_kwargs):
            pass

        def run(self):
            return ()

    with patch("sr.family.stage3_runner.MultiContextFamilySearch", EmptySearch):
        run_stage3(spec, ["--output-dir", str(tmp_path / "run"), "--iterations", "0"])

    output = capsys.readouterr().out
    assert '"families": 0' in output
    assert '"best_score": null' in output


def test_shared_runner_wires_llm_template_initializer(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text('{"llm_template_init": true}')
    family = SimpleNamespace(family_id="rule")
    calls = []
    def initialize(parents):
        calls.append(parents)
        return (), {"status": "complete"}
    def proposal(*args):
        return ()
    proposal.initialize = initialize
    monkeypatch.setattr("sr.family.stage3_runner._llm_functions", lambda *args: (proposal, None))
    class Search:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
        def run(self):
            assert self.kwargs["initialization_fn"] is initialize
            self.kwargs["initialization_fn"](self.kwargs["initial_families"])
            return ()
    monkeypatch.setattr("sr.family.stage3_runner.MultiContextFamilySearch", Search)
    spec = Stage3RunnerSpec(
        setting="lead_time", description="init wiring",
        default_output_dir=tmp_path / "out", default_config=tmp_path / "config.json",
        default_reference_mode="fp3", reference_modes=("fp3",),
        prepare=lambda *_: Stage3Setup(domain=object(), initial_families=(family,),
                                       case_ids=("case",), worker=object()))
    run_stage3(spec, ["--iterations", "0"])
    assert calls == [(family,)]


def test_config_declarations_fail_on_context_drift() -> None:
    config = {
        "schema_version": 2,
        "setting": "lead_time_perishable",
        "training_contexts": [[2, 2]],
        "reference": "best-formula",
        "optimizer": "two-fidelity",
    }
    with pytest.raises(ValueError, match="training_contexts"):
        validate_stage3_config_declarations(
            config,
            schema_version=2,
            setting="lead_time_perishable",
            training_contexts=((2, 2), (3, 3)),
            reference="best-formula",
            optimizer="two-fidelity",
        )


def test_invalid_resume_does_not_start_setting_preparation(tmp_path, monkeypatch):
    config = tmp_path/'config.json'
    config.write_text('{}')
    events = []
    class Search:
        def __init__(self, **kwargs):
            pass
        def validate_run_identity(self):
            events.append('validate')
            raise ValueError('immutable field changed')
        def run(self):
            pytest.fail('must not run')
    monkeypatch.setattr('sr.family.stage3_runner.MultiContextFamilySearch', Search)
    spec = Stage3RunnerSpec(
        setting='lead_time', description='preflight', default_output_dir=tmp_path/'out',
        default_config=config, default_reference_mode='fp3', reference_modes=('fp3',),
        prepare=lambda *_: Stage3Setup(domain=object(), initial_families=(object(),),
            case_ids=('case',), worker=object(), before_search=lambda: events.append('prepare')))
    with pytest.raises(ValueError, match='immutable'):
        run_stage3(spec, ['--resume'])
    assert events == ['validate']


@pytest.mark.parametrize('argv', [['--workers', '0'], ['--iterations', '-1']])
def test_invalid_counts_are_rejected_before_adapter_preparation(tmp_path, argv):
    spec = Stage3RunnerSpec(
        setting='lead_time', description='counts', default_output_dir=tmp_path/'out',
        default_config=tmp_path/'missing.json', default_reference_mode='fp3',
        reference_modes=('fp3',), prepare=lambda *_: pytest.fail('adapter called'))
    with pytest.raises(SystemExit):
        run_stage3(spec, argv)


def test_stage3_cannot_disable_template_initialization(tmp_path):
    config = tmp_path / "config.json"
    config.write_text('{"llm_template_init": false}')
    spec = Stage3RunnerSpec(
        setting="lead_time", description="required initialization",
        default_output_dir=tmp_path / "out", default_config=config,
        default_reference_mode="fp3", reference_modes=("fp3",),
        prepare=lambda *_: pytest.fail("adapter must not prepare disabled initialization"))
    with pytest.raises(SystemExit):
        run_stage3(spec, ["--dry-run"])


def test_stage3_requires_llm_configuration(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text('{}')
    monkeypatch.setattr("sr.family.stage3_runner._llm_functions", lambda *args: (None, None))
    spec = Stage3RunnerSpec(
        setting="lead_time", description="required LLM",
        default_output_dir=tmp_path / "out", default_config=config,
        default_reference_mode="fp3", reference_modes=("fp3",),
        prepare=lambda *_: Stage3Setup(domain=object(), initial_families=(object(),),
                                      case_ids=("case",), worker=object()))
    with pytest.raises(SystemExit):
        run_stage3(spec, [])
    assert not (tmp_path / "out").exists()
