"""Model-neutral loading and reconstruction of durable SR artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from inventory.policy.ast import AST, ast_check, ast_to_formula
from sr.core.grammar import expand_motifs
from sr.core.models import SearchCandidate, SearchIndividual
from sr.core.pareto import (
    complexity_frontier,
    complexity_profile,
    parameter_frontier,
    parameter_profile,
)


def load_standard_result(
    path: Path, motifs: Mapping[str, AST]
) -> SearchIndividual:
    row = json.loads(Path(path).read_text(encoding="utf-8"))
    checked = ast_check(row["ast"], motifs=motifs)
    candidate = SearchCandidate(
        ast=checked.ast,
        ast_hash=checked.ast_hash,
        formula=str(row.get("formula") or ast_to_formula(checked.ast)),
        macro_complexity=float(row["macro_complexity"]),
        expanded_complexity=float(row["expanded_complexity"]),
        parameter_names=tuple(str(name) for name in row["parameter_names"]),
        source=str(row["source"]),
        parents=tuple(str(value) for value in row.get("parents", [])),
        iteration=int(row.get("iteration", 0)),
        island=int(row.get("island", 0)),
        detail=dict(row.get("candidate_detail", {})),
    )
    return SearchIndividual(
        candidate=candidate,
        params=dict(row["params"]),
        policy_cost=float(row["policy_cost"]),
        dp_cost=float(row["dp_cost"]),
        gap_pct=float(row["gap_pct"]),
        status=str(row["status"]),
        route=str(row["route"]),
        optimizer_seconds=float(row["optimizer_seconds"]),
        completed_at=float(row["completed_at"]),
        result_path=str(path),
        optimizer_metadata=dict(row.get("optimizer_metadata", {})),
    )


def individual_summary(
    row: SearchIndividual, motifs: Mapping[str, AST]
) -> dict[str, object]:
    expanded = expand_motifs(row.candidate.ast, motifs)
    return {
        "ast_hash": row.ast_hash,
        "formula": row.candidate.formula,
        "ast": row.candidate.ast,
        "expanded_ast": expanded,
        "expanded_formula": ast_to_formula(expanded),
        "source": row.candidate.source,
        "parents": list(row.candidate.parents),
        "params": dict(row.params),
        "num_parameters": row.num_parameters,
        "expanded_complexity": row.expanded_complexity,
        "policy_cost": row.policy_cost,
        "dp_cost": row.dp_cost,
        "gap_pct": row.gap_pct,
        "status": row.status,
        "route": row.route,
        "optimizer_seconds": row.optimizer_seconds,
        "result_path": row.result_path,
    }


def rebuild_checkpoint_profiles(
    checkpoint: Mapping[str, object],
    archive: Sequence[SearchIndividual],
    motifs: Mapping[str, AST],
) -> dict[str, object]:
    """Rebuild all profile/frontier views from durable completed results."""

    rebuilt = dict(checkpoint)
    expected = int(rebuilt["archive_size"])
    if len(archive) < expected:
        raise RuntimeError(
            f"archive is missing durable results: checkpoint expected at least "
            f"{expected}, found {len(archive)}"
        )
    rebuilt["archive_size"] = len(archive)
    scheduler_counts = dict(rebuilt.get("scheduler_counts", {}))
    scheduler_counts["completed"] = len(archive)
    rebuilt["scheduler_counts"] = scheduler_counts
    eligible = tuple(archive)
    rebuilt["schema_version"] = 2
    rebuilt["parameter_profile"] = [
        individual_summary(row, motifs) for row in parameter_profile(eligible)
    ]
    rebuilt["complexity_profile"] = [
        individual_summary(row, motifs) for row in complexity_profile(eligible)
    ]
    rebuilt["parameter_frontier"] = [
        individual_summary(row, motifs) for row in parameter_frontier(eligible)
    ]
    rebuilt["complexity_frontier"] = [
        individual_summary(row, motifs) for row in complexity_frontier(eligible)
    ]
    return rebuilt


__all__ = ["individual_summary", "load_standard_result", "rebuild_checkpoint_profiles"]
