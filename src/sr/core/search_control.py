"""Paper EC.24--27: completed-lineage credit and reserved softmax parents."""
from __future__ import annotations

import math
import random
from collections import Counter
from typing import Callable, Mapping, Sequence, TypeVar

from .pareto import nondominated

T = TypeVar("T")
SEARCH_CONTROL_PROTOCOL = "paper-ec3"
REWARD_EPSILON = 1e-12
PARENT_LIMIT = 24
PROPOSAL_COUNT = 20


def offspring_reward(parent_score: float, child_score: float, *,
                     parameter_frontier: bool, complexity_frontier: bool) -> float:
    improvement = max(0.0, parent_score - child_score) / (abs(parent_score) + REWARD_EPSILON)
    return min(4.0, improvement) + .5 * (parameter_frontier + complexity_frontier)


def adaptive_score(visits: int, reward_sum: float, total: int) -> float:
    return ((reward_sum / visits if visits else 0.0)
            + .8 * math.sqrt(math.log(total + 2) / (1 + visits)) + .5 / (1 + visits))


def select_parents(rows: Sequence[T], *, key: Callable[[T], str],
                   score: Callable[[T], float], parameters: Callable[[T], int],
                   complexity: Callable[[T], float], behavior: Callable[[T], str],
                   recent: Callable[[T], object], visits: Mapping[str, int],
                   rewards: Mapping[str, float], rng: random.Random,
                   count: int = PARENT_LIMIT) -> tuple[T, ...]:
    """Reserve 2 quality/2 novelty/2 recent/6 frontier slots, then softmax."""
    count = min(PARENT_LIMIT, max(0, count))
    rows = list({key(row): row for row in rows if math.isfinite(score(row))}.values())
    selected: dict[str, T] = {}
    quality = lambda row: (score(row), parameters(row), complexity(row), key(row))

    def add(group, limit):
        for row in group:
            if len(selected) >= count or limit <= 0:
                break
            if key(row) not in selected:
                selected[key(row)] = row
                limit -= 1

    add(sorted(rows, key=quality), 2)
    frequencies = Counter(behavior(row) for row in rows)
    add(sorted(rows, key=lambda row: (frequencies[behavior(row)], quality(row))), 2)
    add(sorted(rows, key=lambda row: (recent(row), key(row)), reverse=True), 2)
    fronts = (
        nondominated(rows, lambda row: (score(row), float(parameters(row)))),
        nondominated(rows, lambda row: (score(row), complexity(row))),
    )
    for front in fronts:
        available = [row for row in front if key(row) not in selected]
        indices = ([round(i * (len(available) - 1) / 2) for i in range(3)]
                   if len(available) > 3 else range(len(available)))
        add([available[i] for i in indices], 3)
    total = sum(visits.values())
    pool = sorted((row for row in rows if key(row) not in selected), key=key)
    while pool and len(selected) < count:
        scores = [adaptive_score(visits.get(key(row), 0), rewards.get(key(row), 0.0), total)
                  for row in pool]
        top = max(scores)
        chosen = rng.choices(range(len(pool)), weights=[math.exp((value - top) / .5)
                                                       for value in scores], k=1)[0]
        row = pool.pop(chosen)
        selected[key(row)] = row
    return tuple(selected.values())
