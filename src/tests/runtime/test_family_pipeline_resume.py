"""Exercise interruption and resume through the real single-machine CPU backend."""
from dataclasses import replace
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.tests.support.family import canonical_dual_shortfall_seed
from sr.family.pipeline import FamilyPipeline
from sr.family.search import MultiContextFamilySearch
from src.tests.support.family import family_domain


def test_local_subprocess_pipeline_recovers_pending_families(tmp_path):
    domain = replace(family_domain((3,)), training_case_count=2)
    seed = canonical_dual_shortfall_seed()
    child = replace(
        seed, family_id="child",
        ast={"op": "add", "args": [seed.ast, {"op": "const", "value": 1}]},
        parent_family_id=seed.family_id, parent_family_hash=domain.semantic_hash(seed),
        mutation_action="add_term",
    )
    config = tmp_path / "config.json"
    config.write_text('{"sleep": 0}')
    worker = SimpleNamespace(
        config_path=config, config={"evaluation_hard_timeout_seconds": 10},
        reference_cache_dir=tmp_path, reference_mode="fixture",
        subprocess_module="src.tests.support.family_subprocess_worker",
    )
    kwargs = dict(
        domain=domain, initial_families=(seed,),
        case_ids=("c0", "c1"), worker=worker,
        output_dir=tmp_path / "search", workers=2, iterations=1,
        feature_init_pass=False,
        scheduling={"async_evaluation": True, "iteration_wait_seconds": 0},
    )
    original = FamilyPipeline.collect

    def interrupted(pipeline, seconds=None):
        if seconds is None:
            raise RuntimeError("interrupted after checkpoint")
        return original(pipeline, seconds)

    with patch.object(FamilyPipeline, "collect", interrupted):
        with pytest.raises(RuntimeError, match="interrupted after checkpoint"):
            MultiContextFamilySearch(**kwargs, proposal_fn=lambda *_: (child,)).run()
    checkpoint = kwargs["output_dir"] / "CHECKPOINT.json"
    assert json.loads(checkpoint.read_text())["pending_family_hashes"] == [domain.semantic_hash(child)]

    def unexpected_proposal(*_):
        pytest.fail("resume must reuse persisted proposals")

    rows = MultiContextFamilySearch(**kwargs, resume=True, proposal_fn=unexpected_proposal).run()
    assert len(rows) == 2
    assert all(len(row.metrics) == 2 for row in rows)
    saved = json.loads(checkpoint.read_text())
    assert saved["pending_family_hashes"] == []
    assert {row["family"]["family_id"] for row in saved["archive"]} == {seed.family_id, child.family_id}
    assert all(row["family"]["ast"] for row in saved["archive"])
