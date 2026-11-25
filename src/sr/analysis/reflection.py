"""Scenario-neutral evidence rules for reflected motifs."""

from __future__ import annotations

from typing import Sequence

from sr.core.models import SearchIndividual
from sr.core.tree import iter_subtrees


def motif_promotion_evidence(
    motif_name: str,
    baseline_gap_pct: float,
    individuals: Sequence[SearchIndividual],
) -> bool:
    """Require completed improvement in two independent lineages."""

    improved = []
    for row in individuals:
        used = any(
            ref.subtree.get("op") == "motif"
            and ref.subtree.get("name") == motif_name
            for ref in iter_subtrees(row.candidate.ast)
        )
        if used and row.gap_pct < float(baseline_gap_pct) - 1.0e-9:
            improved.append(row)
    lineages = {
        tuple(row.candidate.parents) if row.candidate.parents else (row.ast_hash,)
        for row in improved
    }
    return len(improved) >= 2 and len(lineages) >= 2


__all__ = ["motif_promotion_evidence"]
