"""Scenario-neutral controller differences for multi-case SR search."""

from __future__ import annotations

import time
from dataclasses import asdict, replace
from typing import Mapping, Sequence

from inventory.policy.ast import AST
from runtime.artifacts import atomic_json
from sr.analysis.multi_report import MultiSearchRunResult, multi_summary, quality_key, write_multi_search_report
from sr.single.controller import SingleSearchController
from sr.core.models import SearchCandidate
from sr.core.pareto import complexity_frontier, complexity_profile, parameter_frontier, parameter_profile
from sr.multi.models import MultiSearchIndividual
from sr.multi.screening import MultiScreenedCandidate, select_multi_promotions


class MultiSearchController(SingleSearchController):
    @staticmethod
    def _select_promotions(rows: Sequence[MultiScreenedCandidate], limit: int) -> tuple[MultiScreenedCandidate, ...]:
        return select_multi_promotions(rows, limit)

    @staticmethod
    def _write_final_report(result: MultiSearchRunResult) -> None:
        write_multi_search_report(result)

    def _screen_candidates(self, candidates: Sequence[SearchCandidate]) -> list[MultiScreenedCandidate]:
        rows = []
        for candidate in candidates:
            try:
                row = self.screen.score(candidate, self.config)
                if candidate.num_parameters == 0 and all(value.startswith("constant-action:") for value in row.case_behavior_signatures.values()):
                    self._reject("trivial_parameter_free_behavior")
                    continue
                rows.append(row)
            except (ValueError, KeyError, TypeError, IndexError, FloatingPointError, OverflowError):
                self._reject("multi_screen_failure")
        return rows

    def _submit_promotions(self, rows: Sequence[MultiScreenedCandidate], iteration: int, *, limit: int | None = 0) -> list[str]:
        selected = tuple(rows) if limit is None else select_multi_promotions(rows, limit or self.config.standard_promotions_per_iteration)
        hashes = []
        for row in selected:
            candidate = replace(row.candidate, detail={
                **dict(row.candidate.detail),
                "bellman_regret_pct": row.score,
                "multi_screen_mean_regret_pct": row.mean_regret_pct,
                "multi_screen_mean_top_k_regret_pct": row.mean_top_k_regret_pct,
                "multi_screen_case_regrets_pct": dict(row.case_regrets_pct),
                "behavior_signature": row.behavior_signature,
                "case_behavior_signatures": dict(row.case_behavior_signatures),
                "discarded_case_screen_params": {name: dict(values) for name, values in row.case_screen_params.items()},
            })
            self.scheduler.submit_group(candidate, self.case_inputs)
            hashes.append(candidate.ast_hash)
        return hashes

    def _ingest_completed(self) -> int:
        started = time.perf_counter()
        inserted = 0
        for ast_hash, individual in sorted(self.scheduler.completed.items()):
            if ast_hash in self._ingested:
                continue
            self._ingested.add(ast_hash)
            if self.population.add(self._with_fitted_behavior(individual)):
                inserted += 1
        if inserted:
            self._update_motif_promotions()
        self._last_ingest_timing = {"inserted": inserted, "total_seconds": round(time.perf_counter() - started, 6)}
        return inserted

    def _pending_by_parent(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for candidate in self.scheduler.pending_candidates():
            for parent in set(candidate.parents):
                counts[parent] = counts.get(parent, 0) + 1
        return counts

    def _write_checkpoint(self, iteration: int) -> None:
        archive = tuple(sorted(self.population.archive.values(), key=quality_key))
        eligible = archive
        payload = {
            "schema_version": 1,
            "mode": "multi_distribution_standard_search",
            "case": self.case_name,
            "case_names": list(self.case_names),
            "iteration": int(iteration),
            "config": asdict(self.config),
            "objective": self.objective.to_payload(),
            "elapsed_seconds": time.perf_counter() - self._started_at,
            "scheduler_counts": self.scheduler.counts(),
            "rejections": dict(sorted(self._rejections.items())),
            "archive_size": len(archive),
            "parameter_profile": [multi_summary(row, self.motifs) for row in parameter_profile(eligible)],
            "complexity_profile": [multi_summary(row, self.motifs) for row in complexity_profile(eligible)],
            "parameter_frontier": [multi_summary(row, self.motifs) for row in parameter_frontier(eligible)],
            "complexity_frontier": [multi_summary(row, self.motifs) for row in complexity_frontier(eligible)],
            "reflection_motifs": [{"name": row.name, "ast": row.ast, "rationale": row.rationale, "version": row.version, "status": row.status} for row in self._reflections],
            "motifs": dict(sorted(self.motifs.items())),
            "breeding": self.population.breeding_summary(),
            "initialization_signature": self.initialization_signature,
        }
        atomic_json(payload, self.output_dir / "checkpoint.json")

    def _result(self, iterations_completed: int) -> MultiSearchRunResult:
        archive = tuple(sorted(self.population.archive.values(), key=quality_key))
        eligible = archive
        return MultiSearchRunResult(
            case_name=self.case_name,
            case_names=self.case_names,
            output_dir=self.output_dir,
            archive=archive,
            parameter_frontier=parameter_frontier(eligible),
            complexity_frontier=complexity_frontier(eligible),
            parameter_profile=parameter_profile(eligible),
            complexity_profile=complexity_profile(eligible),
            scheduler_counts=self.scheduler.counts(),
            iterations_completed=iterations_completed,
            elapsed_seconds=time.perf_counter() - self._started_at,
            reflection_motifs=tuple(self._reflections),
            motifs=dict(self.motifs),
        )


__all__ = ["MultiSearchController"]
