"""Deterministic dual Pareto frontiers used by SR search and reporting."""

from __future__ import annotations

from typing import Callable, Sequence, TypeVar

from .models import SearchIndividual


T = TypeVar("T")


def nondominated(items: Sequence[T], objective: Callable[[T], tuple[float, float]]) -> tuple[T, ...]:
    # Equal objective coordinates are one Pareto point. Keep a deterministic
    # representative rather than printing many action-equivalent ASTs.
    unique: dict[tuple[float, float], T] = {}
    for item in items:
        point = objective(item)
        if point not in unique or _stable_key(item) < _stable_key(unique[point]):
            unique[point] = item
    # In two dimensions, sorting by the first objective reduces the exact
    # nondominance scan from quadratic to O(n log n).  After sorting by
    # (first, second), a point is nondominated iff its second objective is
    # strictly below every earlier point's second objective.  Equal objective
    # coordinates were already reduced to one deterministic representative.
    rows = sorted(unique.values(), key=lambda item: (*objective(item), _stable_key(item)))
    front = []
    best_second = float("inf")
    for item in rows:
        second = objective(item)[1]
        if second < best_second:
            front.append(item)
            best_second = second
    return tuple(sorted(front, key=lambda item: (*objective(item), _stable_key(item))))


def parameter_frontier(items: Sequence[SearchIndividual]) -> tuple[SearchIndividual, ...]:
    return nondominated(items, lambda row: (float(row.gap_pct), float(row.num_parameters)))


def complexity_frontier(items: Sequence[SearchIndividual]) -> tuple[SearchIndividual, ...]:
    return nondominated(items, lambda row: (float(row.gap_pct), float(row.expanded_complexity)))


def dual_frontier(items: Sequence[SearchIndividual]) -> tuple[SearchIndividual, ...]:
    combined = {row.ast_hash: row for row in (*parameter_frontier(items), *complexity_frontier(items))}
    return tuple(sorted(combined.values(), key=lambda row: (row.gap_pct, row.num_parameters, row.expanded_complexity, row.ast_hash)))


def candidate_on_dual_frontier(
    existing: Sequence[SearchIndividual],
    candidate: SearchIndividual,
) -> bool:
    """Return whether ``candidate.ast_hash`` appears on either augmented frontier.

    This is exactly the membership question previously answered by constructing
    ``dual_frontier([*existing, candidate])``.  Checking only the candidate's
    objective points makes it O(n), which matters because offspring reward is
    computed once for every completed optimizer task.  An older archive entry
    with the same hash is included deliberately, preserving replacement and
    equal-coordinate tie behaviour.
    """

    rows = (*existing, candidate)
    target_key = _stable_key(candidate)
    return _identifier_on_frontier(
        rows,
        target_key,
        lambda row: (float(row.gap_pct), float(row.num_parameters)),
    ) or _identifier_on_frontier(
        rows,
        target_key,
        lambda row: (float(row.gap_pct), float(row.expanded_complexity)),
    )


def parameter_profile(items: Sequence[SearchIndividual]) -> tuple[SearchIndividual, ...]:
    """Return the best formula at every represented parameter count."""
    best: dict[int, SearchIndividual] = {}
    for row in items:
        key = int(row.num_parameters)
        if key not in best or (row.gap_pct, row.expanded_complexity, row.ast_hash) < (
            best[key].gap_pct,
            best[key].expanded_complexity,
            best[key].ast_hash,
        ):
            best[key] = row
    return tuple(best[key] for key in sorted(best))


def complexity_profile(items: Sequence[SearchIndividual]) -> tuple[SearchIndividual, ...]:
    """Return the best formula at every represented expanded-complexity value."""
    best: dict[float, SearchIndividual] = {}
    for row in items:
        key = float(row.expanded_complexity)
        if key not in best or (row.gap_pct, row.num_parameters, row.ast_hash) < (
            best[key].gap_pct,
            best[key].num_parameters,
            best[key].ast_hash,
        ):
            best[key] = row
    return tuple(best[key] for key in sorted(best))


def _dominates(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return left[0] <= right[0] and left[1] <= right[1] and (left[0] < right[0] or left[1] < right[1])


def _stable_key(item: object) -> str:
    # Do not use getattr(item, "ast_hash", repr(item)): Python evaluates the
    # default argument eagerly, so it built a full dataclass/AST repr millions
    # of times even though every SearchIndividual has an ast_hash.
    key = getattr(item, "ast_hash", None)
    return str(key) if key is not None else repr(item)


def _identifier_on_frontier(
    items: Sequence[T],
    target_key: str,
    objective: Callable[[T], tuple[float, float]],
) -> bool:
    entries = [(objective(item), _stable_key(item)) for item in items]
    for point, key in entries:
        if key != target_key:
            continue
        # Equal coordinates are represented by the lexicographically smallest
        # stable key, matching nondominated().  If the same key occurs twice,
        # either occurrence still represents the same identifier.
        if any(other_point == point and other_key < key for other_point, other_key in entries):
            continue
        if not any(_dominates(other_point, point) for other_point, _ in entries):
            return True
    return False
