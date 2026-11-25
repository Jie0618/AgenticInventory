"""Model-neutral helpers for cross-case policy-structure transfer studies."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def transfer_matrix(
    results: Sequence[Mapping[str, Any]],
    source_names: Sequence[str],
    target_names: Sequence[str],
    field: str,
) -> dict[str, dict[str, float]]:
    lookup = {
        (str(row["source_case"]), str(row["target_case"])): float(row[field])
        for row in results
    }
    return {
        source: {target: lookup[(source, target)] for target in target_names}
        for source in source_names
    }


__all__ = ["transfer_matrix"]
