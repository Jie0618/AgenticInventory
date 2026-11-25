"""Island population adapter for aggregate multi-case individuals."""

from __future__ import annotations

import random
from typing import Mapping, Sequence

from sr.core.models import SearchConfig
from sr.core.population import IslandPopulation
from sr.multi.models import MultiSearchIndividual


class MultiIslandPopulation(IslandPopulation):
    def add(
        self,
        individual: MultiSearchIndividual,
        *,
        initial_islands: Sequence[int] = (),
    ) -> bool:
        inserted = super().add(individual)  # type: ignore[arg-type]
        if not inserted:
            return False
        for island in initial_islands:
            index = int(island) % self.config.island_count
            self.islands[index][individual.ast_hash] = individual  # type: ignore[assignment]
            self._prune(index)
        return True

    def parents(
        self,
        island: int,
        count: int,
        rng: random.Random,
        *,
        pending_by_parent: Mapping[str, int] | None = None,
    ) -> tuple[MultiSearchIndividual, ...]:
        return super().parents(island, count, rng, pending_by_parent=pending_by_parent)


__all__ = ["MultiIslandPopulation"]
