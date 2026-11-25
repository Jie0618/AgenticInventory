"""Canonical inventory composition root for generic single-case SR search."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from optimization.models.types import SupervisedSample
from optimization.search_context import InventorySearchContext
from sr.single.controller import SingleSearchController
from sr.core.inventory import InventorySearchBindings
from sr.core.models import ReflectionMotif, ScreenedCandidate, SearchCandidate, SearchConfig
from sr.core.population import IslandPopulation
from sr.core.randomness import derived_seed

from sr.inventory.seeds import initial_seed_asts
from sr.inventory.screening import BellmanScreen
from sr.inventory.evaluation import AsyncStandardScheduler


class SymbolicSearch(SingleSearchController):
    def __init__(
        self,
        *,
        case_name: str,
        context: InventorySearchContext,
        diagnostic_json: Path | None,
        output_dir: Path,
        config: SearchConfig = SearchConfig(),
        bindings: InventorySearchBindings,
        samples: Sequence[SupervisedSample] | None = None,
        mutation_client: object | None = None,
        reflection_client: object | None = None,
        write_run_logs: bool = True,
        screen: object | None = None,
    ) -> None:
        self.case_name = str(case_name)
        self.context = context
        self.diagnostic_json = None if diagnostic_json is None else Path(diagnostic_json)
        self.output_dir = Path(output_dir)
        self.config = config
        self.mutation_client = mutation_client
        self.reflection_client = reflection_client
        if bindings.config != config:
            raise ValueError("search bindings and controller must share one SearchConfig")
        if screen is None and bindings.context != context:
            raise ValueError("search bindings must own the supplied search context")
        if bindings.case != context.case:
            raise ValueError("search bindings and context must share one inventory case")
        self.bindings = bindings
        self.domain = bindings.domain
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.motifs = dict(bindings.motifs)
        self.samples = (
            diagnostic_samples(self.diagnostic_json, context.horizon)
            if samples is None
            else tuple(samples)
        )
        if not self.samples and screen is None:
            raise ValueError("symbolic search requires at least one diagnostic sample")
        self.screen = screen if screen is not None else BellmanScreen(
            self.diagnostic_json,
            context=context,
            motifs=self.motifs,
            random_seed=config.random_seed,
        )
        self.scheduler = AsyncStandardScheduler(
            config,
            result_dir=self.output_dir / "standard_results",
            event_log=(self.output_dir / "events.jsonl") if write_run_logs else None,
        )
        self.population = IslandPopulation(config)
        self._ingested: set[str] = set()
        self._known_hashes: set[str] = set()
        self._known_structures: set[str] = set()
        self._reflections: list[ReflectionMotif] = []
        self._reflection_baselines: dict[str, float] = {}
        self._rejections: dict[str, int] = {}
        self._last_ingest_timing: dict[str, float | int] = {}
        self._last_proposal_timing: dict[str, float | int] = {}
        self._started_at = time.perf_counter()

    def _evaluate_initial_seeds(self) -> None:
        candidates = []
        for index, (source, ast) in enumerate(initial_seed_asts(self.context.case)):
            try:
                candidate = self.bindings.candidate(ast, motifs=self.motifs, source=source, island=index % self.config.island_count)
                if self._remember(candidate): candidates.append(candidate)
            except (ValueError, KeyError, TypeError, OverflowError):
                self._reject("invalid_initial_seed")
        hashes = self._submit_promotions(self._screen_candidates(candidates), 0, limit=None)
        if hashes: self.scheduler.wait_for_batch(hashes)
        self._ingest_completed()
        if not self.population.archive:
            raise RuntimeError(
                "no initial Standard evaluation completed; "
                f"rejections={dict(sorted(self._rejections.items()))}; "
                f"scheduler={self.scheduler.counts()}; "
                f"failures={dict(sorted(self.scheduler.failures.items()))}"
            )

    def _submit_promotions(self, rows: Sequence[ScreenedCandidate], iteration: int, *, limit: int | None = 0) -> list[str]:
        selected = tuple(rows) if limit is None else self._select_promotions(rows, limit or self.config.standard_promotions_per_iteration)
        hashes = []
        for row in selected:
            candidate = replace(row.candidate, detail={**dict(row.candidate.detail), "bellman_regret_pct": row.bellman_regret_pct, "behavior_signature": row.behavior_signature, "discarded_screen_params": dict(row.screen_params)})
            seed = derived_seed(self.config.random_seed, self.case_name, candidate.ast_hash, iteration)
            self.scheduler.submit(candidate, case_name=self.case_name, context=self.context, samples=self.samples, motifs=self.motifs, random_seed=seed)
            hashes.append(candidate.ast_hash)
        return hashes

def diagnostic_samples(path: Path, horizon: int) -> tuple[SupervisedSample, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    samples = []
    for row in payload["representative_states"]:
        state_features = row.get("state_features")
        if isinstance(state_features, dict):
            state = {
                str(name): float(value) for name, value in state_features.items()
            }
        else:
            # Alternate-format reader for the packed lead-time diagnostic
            # schema. Diagnostics with named state features use the branch above.
            flat = tuple(int(value) for value in row["state"])
            state = {"x": flat[0], "p": tuple(flat[1:])}
        state.setdefault("remaining_horizon", max(1, int(horizon) // 2))
        if "period" in row:
            state["time"] = int(row["period"])
        samples.append(SupervisedSample(state=state, target_action=float(row["dp_action"]), weight=float(row["probability"])))
    if not samples: raise ValueError(f"diagnostic has no representative states: {path}")
    return tuple(samples)


__all__ = ["SymbolicSearch", "diagnostic_samples"]
