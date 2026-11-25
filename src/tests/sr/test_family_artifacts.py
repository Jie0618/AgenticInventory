from sr.family.artifacts import artifact_files, read_path, write_path


def test_history_lookup_preserves_flat_runs_and_prefers_organized_copy(tmp_path):
    old = tmp_path / "PROPOSALS_ITER_0.json"
    old.write_text("old")
    assert read_path(tmp_path, old.name) == old
    new = write_path(tmp_path, old.name)
    new.write_text("new")
    second = tmp_path / "PROPOSALS_ITER_1.json"
    second.write_text("legacy continuation")
    assert read_path(tmp_path, old.name) == new
    assert artifact_files(tmp_path, "PROPOSALS_ITER_*.json") == (new, second)
    assert old.read_text() == "old"


def test_missing_artifact_reads_do_not_create_directories(tmp_path):
    assert not read_path(tmp_path, "LLM_TEMPLATE_INIT.json").exists()
    assert artifact_files(tmp_path, "MOTIF_REFLECTION_ITER_*.json") == ()
    assert list(tmp_path.iterdir()) == []
