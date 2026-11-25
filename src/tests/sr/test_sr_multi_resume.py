import json
import time
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sr.core.engine import AgenticSearchEngine
from sr.core.models import SearchConfig
from sr.inventory.multi import MultiDistributionSearch
from sr.multi.models import MultiObjective


def restored_search(tmp_path):
    search = object.__new__(MultiDistributionSearch)
    search.config = SearchConfig()
    search.objective = MultiObjective()
    search.output_dir = tmp_path
    search.initialization_signature = "same-inputs"
    search.case_names = ("a", "b")
    search._started_at = time.perf_counter()
    search.motifs = {}
    search._rejections = {}
    search._ingested = set()
    search._known_hashes = set()
    search._known_structures = set()
    search.initialization_cache_dir = tmp_path / "init"
    search.scheduler = Mock(result_dir=tmp_path / "standard_evaluations",
                            completed={}, group_candidates={}, group_states={})
    search.screen = SimpleNamespace(screens={})
    search.population = Mock()
    search.bindings = Mock()
    search.case_inputs = {name: SimpleNamespace(random_seed=1) for name in search.case_names}
    payload = {"config": asdict(search.config), "objective": asdict(search.objective),
               "initialization_signature": "same-inputs", "case_names": ["a", "b"],
               "iteration": 26, "motifs": {}, "archive_size": 0}
    (tmp_path / "checkpoint.json").write_text(json.dumps(payload))
    return search, payload


def test_resume_reuses_complete_observations_and_queues_partial(tmp_path, monkeypatch):
    search, payload = restored_search(tmp_path)
    candidate = SimpleNamespace(ast={"op": "const", "value": 0}, ast_hash="hash")
    search.bindings.candidate.return_value = candidate
    row = {"ast": candidate.ast, "source": "original", "evaluation_signature": "historical"}
    directory = search.scheduler.result_dir / "hash"
    directory.mkdir(parents=True)
    for name in search.case_names:
        (directory / (name + ".json")).write_text(json.dumps(row))
    sentinel = object()
    loader = Mock(return_value=sentinel)
    monkeypatch.setattr("sr.evaluation.standard.load_standard_result", loader)
    monkeypatch.setattr("sr.multi.models.aggregate_case_results", lambda *a: sentinel)
    search._restore_completed_run()
    assert search._resume_iteration == 26
    search.population.add.assert_called_once_with(sentinel)
    search.scheduler.submit_group.assert_not_called()
    assert loader.call_args.args[-1] == "historical"
    assert json.loads((tmp_path / "audit/resume.json").read_text())["restored_groups"] == 1
    assert not (tmp_path / "RESUME_AUDIT.json").exists()
    (directory / "b.json").unlink()
    search.scheduler.reset_mock()
    search._restore_completed_run()
    search.scheduler.submit_group.assert_called_once_with(candidate, search.case_inputs)


def test_resume_rejects_changed_inputs(tmp_path):
    search, payload = restored_search(tmp_path)
    search.initialization_signature = "different"
    with pytest.raises(ValueError, match="do not match"):
        search._restore_completed_run()


def test_resume_allows_only_increasing_timeout(tmp_path):
    from dataclasses import replace

    search, payload = restored_search(tmp_path)
    search.config = replace(search.config, evaluation_hard_timeout_seconds=1500)
    search._restore_completed_run()
    audit = json.loads((tmp_path / "audit/resume.json").read_text())
    assert audit["previous_timeout_seconds"] == 900
    assert audit["timeout_seconds"] == 1500

    search.config = replace(search.config, evaluation_hard_timeout_seconds=600)
    with pytest.raises(ValueError, match="do not match"):
        search._restore_completed_run()

    search.config = replace(search.config, evaluation_hard_timeout_seconds=1500, max_parameters=9)
    with pytest.raises(ValueError, match="do not match"):
        search._restore_completed_run()


def test_engine_continues_with_next_iteration():
    engine = AgenticSearchEngine()
    engine.config = SimpleNamespace(iterations=27, island_count=1, standard_promotions_per_iteration=1,
                                    max_outstanding_tasks=1)
    engine._resume_iteration = 26
    engine.scheduler = Mock()
    engine.population = Mock()
    engine._last_ingest_timing = engine._last_proposal_timing = {}
    for name in ("_write_progress", "_evaluate_initial_seeds", "_ingest_completed", "_iteration_rng",
                 "_pending_by_parent", "_propose", "_screen_candidates", "_select_promotions",
                 "_write_checkpoint", "_result", "_write_final_report", "_maybe_reflect"):
        setattr(engine, name, Mock())
    engine._submit_promotions = Mock(return_value=[])
    engine._propose.return_value = []
    engine._screen_candidates.return_value = []
    engine._run()
    assert engine._propose.call_args.args[0] == 27
    engine._write_checkpoint.assert_called_with(27)
    engine._maybe_reflect.assert_called_once_with(27)
