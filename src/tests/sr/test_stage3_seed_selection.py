"""Boundary selection, archive completeness, and portable transfer regressions."""
import json

import pytest

from sr.family.seed_selection import load_archive, select_frontier_seeds


def row(level, rank, params=3):
    return dict(ast_hash=f"{level}-{rank}", expanded_complexity=level,
                score=100-level+rank/100, num_parameters=params)


def test_dual_frontier_union_keeps_eight_parameters_and_is_deterministic():
    frontier = [row(i, 0, 8 if i == 4 else 3) for i in range(9)]
    archive = frontier + [row(i, 1) for i in range(9)]
    selected, audit = select_frontier_seeds(frontier[::-1], archive[::-1])
    assert {r['ast_hash'] for r in selected} == {r['ast_hash'] for r in frontier}
    assert any(r['num_parameters'] == 8 for r in selected)
    assert select_frontier_seeds(frontier, archive) == (selected, audit)


def test_selected_pool_is_exactly_the_two_rule_boundaries_union():
    parameter = [row(1, 0), row(4, 1)]
    complexity = [row(1, 0), row(7, 2)]
    archive = parameter + complexity + [row(8, 3)]
    selected, _ = select_frontier_seeds(parameter + complexity, archive)
    assert {r['ast_hash'] for r in selected} == {'1-0', '4-1', '7-2'}


def test_selection_rejects_incomplete_inventory_instead_of_silently_dropping_frontier():
    with pytest.raises(ValueError, match="frontier candidates missing"):
        select_frontier_seeds([row(1, 0), row(2, 0)], [row(1, 0)])


def test_backlog_transfer_audits_first_unportable_candidate_and_keeps_both_frontiers(tmp_path):
    from experiments.lead_time_backlog.stage3_multi_context_sr.cases import cases
    from experiments.lead_time_backlog.stage3_multi_context_sr.family import families_from_stage2_checkpoint
    from inventory.policy.ast import const
    from inventory.policy.family_domain import InventoryFamilyDomain
    case = next(case for case in cases()
                if case.lead_time == 3 and case.config.backlog.patience == 3)
    domain = InventoryFamilyDomain.from_cases({"L3K3": case}, training_case_count=1)
    bad = {**row(1, 0), "ast": {"op": "state", "name": "unknown_coordinate"}}
    a, b = ({**row(2, 0), "ast": const(1)}, {**row(3, 0), "ast": const(2)})
    checkpoint = tmp_path / "checkpoint.json"
    payload = dict(iteration=1, config={"iterations": 1},
                   parameter_frontier=[bad, a], complexity_frontier=[b, a])
    checkpoint.write_text(json.dumps(payload))
    audit = []
    result = families_from_stage2_checkpoint(checkpoint, domain, audit=audit)
    assert {family.ast["value"] for family in result} == {1, 2}
    assert any(r.get("source_hash") == bad["ast_hash"] and r["status"] == "rejected"
               for r in audit)
    assert {r["source_hash"] for r in audit if r["status"] == "selected"} == {
        a["ast_hash"], b["ast_hash"]}
    payload.update(parameter_frontier=[bad], complexity_frontier=[bad])
    checkpoint.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="no portable family seeds"):
        families_from_stage2_checkpoint(checkpoint, domain)


def test_archive_requires_scores_for_complete_candidates(tmp_path):
    payload = dict(case_names=['a', 'b'], complexity_frontier=[])
    record = dict(ast_hash='candidate', expanded_complexity=35,
                  parameter_names=['S'], complete_cases=2)
    (tmp_path/'ARCHIVE_INVENTORY.jsonl').write_text(json.dumps(record)+'\n')
    with pytest.raises(ValueError, match='missing archive scores'):
        load_archive(tmp_path/'checkpoint.json', payload)
    (tmp_path/'STAGE3_SCORE_INDEX.json').write_text(json.dumps({'candidate': .17}))
    rows = load_archive(tmp_path/'checkpoint.json', payload)
    assert rows[0]['num_parameters'] == 1
    assert rows[0]['score'] == .17


def test_archive_can_reconstruct_missing_event_score_from_raw_cases(tmp_path):
    payload = dict(case_names=['a', 'b'], complexity_frontier=[])
    record = dict(ast_hash='candidate', expanded_complexity=35,
                  parameter_names=[], complete_cases=2)
    (tmp_path/'ARCHIVE_INVENTORY.jsonl').write_text(json.dumps(record)+'\n')
    root = tmp_path/'standard_evaluations/candidate'
    root.mkdir(parents=True)
    for name, gap in [('a', 1.), ('b', 3.)]:
        (root/f'{name}.json').write_text(json.dumps({'gap_pct': gap}))
    assert load_archive(tmp_path/'checkpoint.json', payload)[0]['score'] == pytest.approx(2.4)


def test_archive_rejects_export_missing_initialization_candidates(tmp_path):
    payload = dict(case_names=['a', 'b'], archive_size=2, complexity_frontier=[])
    standard = dict(ast_hash='standard', expanded_complexity=35,
                    parameter_names=['S'], complete_cases=2)
    initial = dict(ast_hash='initial', expanded_complexity=35,
                   parameter_names=['S'], complete_cases=2)
    inventory = tmp_path/'ARCHIVE_INVENTORY.jsonl'
    inventory.write_text(json.dumps(standard)+'\n')
    (tmp_path/'STAGE3_SCORE_INDEX.json').write_text(json.dumps({'standard': .17, 'initial': .16}))
    with pytest.raises(ValueError, match='incomplete archive inventory: 1/2'):
        load_archive(tmp_path/'checkpoint.json', payload)
    inventory.write_text(json.dumps(standard)+'\n'+json.dumps(initial)+'\n')
    assert {r['ast_hash'] for r in load_archive(tmp_path/'checkpoint.json', payload)} == {'standard', 'initial'}


def test_resume_cannot_silently_change_many_seed_parent_selection(tmp_path):
    from sr.family.search import MultiContextFamilySearch
    old = dict(contexts=['3'], training_case_ids=['a'], parent_context_limit=4)
    (tmp_path/'RUN_MANIFEST.json').write_text(json.dumps(old))
    search = object.__new__(MultiContextFamilySearch)
    search.output_dir = tmp_path
    search.resume = True
    current = {**old, 'parent_selection_protocol': 'elite-rotating-anchor-specialist'}
    with pytest.raises(ValueError, match='parent_selection_protocol'):
        search._validate_or_write_manifest(current, persist=False)
