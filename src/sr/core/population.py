"""Island populations with Pareto safety and bandit-guided breeding."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

from .models import SearchCandidate, SearchConfig, SearchIndividual
from .pareto import complexity_frontier, dual_frontier, parameter_frontier


@dataclass
class BreedingStat:
    completed: int = 0
    reward_sum: float = 0.0
    successes: int = 0

    @property
    def mean_reward(self) -> float:
        return self.reward_sum / self.completed if self.completed else 0.0


class IslandPopulation:
    def __init__(self, config: SearchConfig) -> None:
        self.config = config
        self.islands: list[dict[str, SearchIndividual]] = [dict() for _ in range(config.island_count)]
        self.archive: dict[str, SearchIndividual] = {}
        self.parent_stats: dict[str, BreedingStat] = defaultdict(BreedingStat)
        self.operator_stats: dict[str, BreedingStat] = defaultdict(BreedingStat)

    def add(self, individual: SearchIndividual) -> bool:
        previous = self.archive.get(individual.ast_hash)
        if previous is not None and previous.gap_pct <= individual.gap_pct:
            return False
        reward = self._offspring_reward(individual)
        self._credit_lineage(individual.candidate, reward)
        self.archive[individual.ast_hash] = individual
        island = int(individual.candidate.island) % self.config.island_count
        self.islands[island][individual.ast_hash] = individual
        self._prune(island)
        return True

    def public_frontier(self) -> tuple[SearchIndividual, ...]:
        return dual_frontier(tuple(self.archive.values()))

    def parents(
        self,
        island: int,
        count: int,
        rng: random.Random,
        *,
        pending_by_parent: Mapping[str, int] | None = None,
    ) -> tuple[SearchIndividual, ...]:
        # The growing evaluated population is the parent pool (EC.25). Islands
        # remain scheduling metadata and do not hide completed candidates.
        from .search_control import select_parents
        return select_parents(
            list(self.archive.values()), key=lambda row: row.ast_hash,
            score=lambda row: row.gap_pct, parameters=lambda row: row.num_parameters,
            complexity=lambda row: row.expanded_complexity, behavior=_behavior,
            recent=lambda row: (row.candidate.iteration, row.completed_at),
            visits={key: stat.completed for key, stat in self.parent_stats.items()},
            rewards={key: stat.reward_sum for key, stat in self.parent_stats.items()},
            rng=rng, count=count,
        )

    def operator_weights(self, operators: Sequence[str]) -> dict[str, float]:
        priors = {name: value for name, value in zip(
            ("minimum", "maximum", "subtree", "cap", "scale", "elastic", "feature", "unwrap", "immigrant"),
            (18.0, 6.0, 12.0, 12.0, 8.0, 10.0, 14.0, 8.0, 12.0),
        )}
        total = sum(self.operator_stats[name].completed for name in operators)
        priors["cap_mode"] = 12.0
        scores = {
            name: self._stat_score(self.operator_stats[name], total, pending=0)
            for name in operators
        }
        if not scores:
            return {}
        top = max(scores.values())
        temperature = max(1.0e-6, self.config.operator_bandit_temperature)
        prior_total = sum(priors.get(name, 1.0) for name in scores) or 1.0
        raw = {
            name: (priors.get(name, 1.0) / prior_total)
            * math.exp(max(-50.0, min(0.0, (score - top) / temperature)))
            for name, score in scores.items()
        }
        floor = max(0.0, min(1.0, self.config.operator_min_probability))
        normalizer = sum(raw.values()) or 1.0
        return {name: floor + (1.0 - floor * len(raw)) * raw[name] / normalizer for name in raw}

    def parent_weights(
        self,
        parents: Sequence[SearchIndividual],
        pending_by_parent: Mapping[str, int] | None = None,
    ) -> dict[str, float]:
        pending = pending_by_parent or {}
        scores = {row.ast_hash: self._bandit_score(row.ast_hash, pending.get(row.ast_hash, 0)) for row in parents}
        if not scores:
            return {}
        top = max(scores.values())
        temperature = max(1.0e-6, self.config.parent_bandit_temperature)
        raw = {key: math.exp(max(-50.0, min(0.0, (value - top) / temperature))) for key, value in scores.items()}
        total = sum(raw.values()) or 1.0
        return {key: value / total for key, value in raw.items()}

    def migrate(self, iteration: int) -> None:
        if iteration <= 0 or iteration % self.config.migration_interval != 0:
            return
        outgoing = [list(dual_frontier(list(island.values())))[: self.config.migration_count] for island in self.islands]
        for index, rows in enumerate(outgoing):
            destination = (index + 1) % self.config.island_count
            for row in rows:
                self.islands[destination].setdefault(row.ast_hash, row)
            self._prune(destination)

    def breeding_summary(self) -> dict[str, object]:
        def summary(stats: Mapping[str, BreedingStat]) -> dict[str, object]:
            return {
                key: {"completed": row.completed, "mean_reward": row.mean_reward, "successes": row.successes}
                for key, row in sorted(stats.items())
                if row.completed
            }
        return {"parents": summary(self.parent_stats), "operators": summary(self.operator_stats)}

    def _offspring_reward(self, child: SearchIndividual) -> float:
        parents = [self.archive[key] for key in child.candidate.parents if key in self.archive]
        if not parents:
            return 0.0
        from .search_control import offspring_reward
        augmented = [*self.archive.values(), child]
        return offspring_reward(
            min(row.gap_pct for row in parents), child.gap_pct,
            parameter_frontier=any(row.ast_hash == child.ast_hash for row in parameter_frontier(augmented)),
            complexity_frontier=any(row.ast_hash == child.ast_hash for row in complexity_frontier(augmented)),
        )

    def _credit_lineage(self, candidate: SearchCandidate, reward: float) -> None:
        parents = tuple(dict.fromkeys(candidate.parents))
        if parents:
            share = reward / len(parents)
            for parent in parents:
                stat = self.parent_stats[parent]
                stat.completed += 1
                stat.reward_sum += share
                stat.successes += int(reward > 0)
        if candidate.source.startswith("gp:"):
            name = candidate.source.split(":", 1)[1]
            stat = self.operator_stats[name]
            stat.completed += 1
            stat.reward_sum += reward
            stat.successes += int(reward > 0)

    def _bandit_score(self, ast_hash: str, pending: int) -> float:
        total = sum(row.completed for row in self.parent_stats.values())
        return self._stat_score(self.parent_stats[ast_hash], total, pending)

    def _stat_score(self, stat: BreedingStat, total: int, pending: int) -> float:
        visits = stat.completed + max(0, int(pending))
        exploration = self.config.parent_bandit_exploration * math.sqrt(math.log1p(total + 1) / (1 + visits))
        prior = self.config.parent_bandit_newcomer_bonus / (1 + visits)
        return stat.mean_reward + exploration + prior

    @staticmethod
    def _quality_key(row: SearchIndividual) -> tuple[float, int, float, str]:
        return (row.gap_pct, row.num_parameters, row.expanded_complexity, row.ast_hash)

    @staticmethod
    def _parameter_specialists(rows: Sequence[SearchIndividual]) -> list[SearchIndividual]:
        buckets: dict[int, list[SearchIndividual]] = defaultdict(list)
        for row in rows:
            buckets[row.num_parameters].append(row)
        return [min(buckets[key], key=IslandPopulation._quality_key) for key in sorted(buckets)]

    def _prune(self, island: int) -> None:
        rows = list(self.islands[island].values())
        limit = self.config.island_max_size
        if len(rows) <= limit:
            return
        retained = {row.ast_hash: row for row in dual_frontier(rows)}
        for row in self._parameter_specialists(rows):
            retained[row.ast_hash] = row
        behavior_counts: dict[str, int] = defaultdict(int)
        for row in rows:
            behavior_counts[_behavior(row)] += 1
        novelty = sorted(rows, key=lambda row: (behavior_counts[_behavior(row)], -row.candidate.iteration, self._quality_key(row)))
        for row in novelty[: max(2, limit // 6)]:
            retained[row.ast_hash] = row
        newcomers = sorted(rows, key=lambda row: (self.parent_stats[row.ast_hash].completed, -row.candidate.iteration, self._quality_key(row)))
        for row in newcomers[: max(2, limit // 10)]:
            retained[row.ast_hash] = row
        bandit = [row for row in rows if row.ast_hash not in retained]
        from runtime.artifacts import seed_from_text
        rng = random.Random(seed_from_text(
            f"{self.config.random_seed}:{island}:" + ",".join(sorted(row.ast_hash for row in rows))))
        scores = {row.ast_hash: self._bandit_score(row.ast_hash, 0) for row in bandit}
        for row in _softmax_sample_without_replacement(
            bandit,
            scores,
            max(0, limit - len(retained)),
            self.config.parent_bandit_temperature,
            rng,
        ):
            retained[row.ast_hash] = row
        if len(retained) > limit:
            protected = {row.ast_hash for row in dual_frontier(rows)} | {row.ast_hash for row in self._parameter_specialists(rows)}
            ordered = sorted(retained.values(), key=lambda row: (row.ast_hash not in protected, self._quality_key(row)))
            retained = {row.ast_hash: row for row in ordered[:limit]}
        self.islands[island] = retained


def _behavior(row: SearchIndividual) -> str:
    return str(row.candidate.detail.get("behavior_signature", "")) or row.ast_hash


def _spread(rows: Sequence[SearchIndividual], count: int) -> tuple[SearchIndividual, ...]:
    if count <= 0 or not rows:
        return ()
    if len(rows) <= count:
        return tuple(rows)
    if count == 1:
        return (rows[0],)
    return tuple(rows[round(index * (len(rows) - 1) / (count - 1))] for index in range(count))


def _softmax_sample_without_replacement(
    rows: Sequence[SearchIndividual],
    scores: Mapping[str, float],
    count: int,
    temperature: float,
    rng: random.Random,
) -> tuple[SearchIndividual, ...]:
    pool = list(rows)
    output: list[SearchIndividual] = []
    temperature = max(1.0e-6, float(temperature))
    while pool and len(output) < count:
        values = [scores[row.ast_hash] for row in pool]
        top = max(values)
        weights = [math.exp(max(-50.0, min(0.0, (value - top) / temperature))) for value in values]
        chosen = rng.choices(pool, weights=weights, k=1)[0]
        output.append(chosen)
        pool.remove(chosen)
    return tuple(output)
