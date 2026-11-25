"""Small selection primitives shared by search controllers."""

from __future__ import annotations

from typing import Protocol, Sequence, TypeVar


T = TypeVar("T")


class ScreenedRow(Protocol):
    candidate: object
    behavior_signature: str
    bellman_regret_pct: float


def spread_rows(rows: Sequence[T], count: int) -> tuple[T, ...]:
    if count <= 0 or not rows:
        return ()
    if len(rows) <= count:
        return tuple(rows)
    if count == 1:
        return (rows[0],)
    return tuple(
        rows[round(index * (len(rows) - 1) / (count - 1))]
        for index in range(count)
    )


def select_promotions(
    rows: Sequence[ScreenedRow], limit: int
) -> tuple[ScreenedRow, ...]:
    """Select a quality-ranked, parameter- and behavior-diverse batch."""

    if limit <= 0:
        return ()
    best_by_behavior: dict[str, ScreenedRow] = {}
    for row in rows:
        current = best_by_behavior.get(row.behavior_signature)
        candidate = row.candidate
        key = (
            row.bellman_regret_pct,
            candidate.expanded_complexity,
            candidate.ast_hash,
        )
        if current is None or key < (
            current.bellman_regret_pct,
            current.candidate.expanded_complexity,
            current.candidate.ast_hash,
        ):
            best_by_behavior[row.behavior_signature] = row
    buckets: dict[int, list[ScreenedRow]] = {}
    for row in best_by_behavior.values():
        buckets.setdefault(row.candidate.num_parameters, []).append(row)
    selected: dict[str, ScreenedRow] = {}
    quota = max(1, int(limit) // max(1, len(buckets)))
    for parameter_count in sorted(buckets):
        ranked = sorted(
            buckets[parameter_count],
            key=lambda row: (
                row.bellman_regret_pct,
                row.candidate.expanded_complexity,
                row.candidate.ast_hash,
            ),
        )
        for row in ranked[:quota]:
            selected[row.candidate.ast_hash] = row
    for row in sorted(
        rows,
        key=lambda row: (
            row.bellman_regret_pct,
            row.candidate.num_parameters,
            row.candidate.expanded_complexity,
            row.candidate.ast_hash,
        ),
    ):
        if len(selected) >= int(limit):
            break
        selected.setdefault(row.candidate.ast_hash, row)
    return tuple(
        sorted(
            selected.values(),
            key=lambda row: (row.bellman_regret_pct, row.candidate.ast_hash),
        )[: int(limit)]
    )


__all__ = ["ScreenedRow", "select_promotions", "spread_rows"]
