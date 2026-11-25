"""Scenario-neutral single-population SR controller hooks."""

from __future__ import annotations

import json
import multiprocessing as mp
import queue as queue_module
import random
import time
from dataclasses import asdict, replace
from typing import Mapping, Sequence

from inventory.policy.ast import AST
from runtime.artifacts import atomic_json
from sr.analysis.artifacts import individual_summary
from sr.analysis.reflection import motif_promotion_evidence
from sr.analysis.single_report import SearchRunResult, reflection_summary, write_search_report
from sr.core.engine import AgenticSearchEngine
from sr.core.gp import structural_signature
from sr.core.models import ReflectionMotif, ScreenedCandidate, SearchCandidate, SearchIndividual
from sr.core.pareto import complexity_frontier, complexity_profile, parameter_frontier, parameter_profile
from sr.core.randomness import derived_seed
from sr.core.selection import select_promotions
from sr.llm import SearchDomain, propose_asts, reflect_motifs


def _auxiliary_worker(output_queue: object, function: object, args: tuple[object, ...]) -> None:
    try:
        output_queue.put(("completed", function(*args)))  # type: ignore[operator,attr-defined]
    except BaseException as exc:
        output_queue.put(("failed", type(exc).__name__, str(exc)))  # type: ignore[attr-defined]


def _reflect_auxiliary(client: object, individuals: Sequence[SearchIndividual], motifs: Mapping[str, AST], version: int, context: Mapping[str, object], domain: SearchDomain) -> tuple[ReflectionMotif, ...]:
    return reflect_motifs(client, individuals, motifs, version=version, context=context, domain=domain)  # type: ignore[arg-type]


class SingleSearchController(AgenticSearchEngine[SearchRunResult]):
    """Common proposal, screening, reflection, and persistence behavior."""

    def _iteration_rng(self, iteration: int, island: int) -> random.Random:
        return random.Random(derived_seed(self.config.random_seed, self.case_name, iteration, island))

    @staticmethod
    def _select_promotions(rows: Sequence[ScreenedCandidate], limit: int) -> tuple[ScreenedCandidate, ...]:
        return select_promotions(rows, limit)  # type: ignore[return-value]

    @staticmethod
    def _write_final_report(result: SearchRunResult) -> None:
        write_search_report(result)

    def _write_progress(self, event: str, *, iteration: int, **detail: object) -> None:
        frontier = self.population.public_frontier()
        row = {
            "progress": "search", "time": time.time(), "event": str(event),
            "case": self.case_name, "iteration": int(iteration),
            "iterations_total": int(self.config.iterations),
            "elapsed_seconds": round(time.perf_counter() - self._started_at, 3),
            "archive_size": len(self.population.archive),
            "best_gap_pct": None if not frontier else min(item.gap_pct for item in frontier),
            "scheduler": dict(sorted(self.scheduler.counts().items())), **detail,
        }
        print(json.dumps(row, sort_keys=True), flush=True)

    def _propose(self, iteration: int, island: int, parents: Sequence[SearchIndividual], rng: random.Random, pending_by_parent: Mapping[str, int]) -> list[SearchCandidate]:
        started = time.perf_counter()
        self._selected_parents = tuple(parents)
        self._selected_evidence = self._parent_evidence(parents)
        atomic_json({"iteration": iteration, **self._selected_evidence},
                    self.output_dir / f"parent_evidence_{iteration}.json")
        llm_seconds = 0.0
        raw = []
        if self.mutation_client is not None and parents:
            llm_started = time.perf_counter()
            try:
                llm_rows = self._run_while_scheduling(propose_asts, self.mutation_client, parents, self.motifs, self.config, self.domain, self._selected_evidence)
            except Exception as exc:
                self._reject(f"llm_failure:{type(exc).__name__}")
                llm_rows = ()
            llm_seconds = time.perf_counter() - llm_started
            for ast, rationale, parent_hashes in llm_rows:
                try:
                    raw.append(self.bindings.candidate(ast, motifs=self.motifs, source="llm:structural", parents=parent_hashes, iteration=iteration, island=island, detail={"rationale": rationale}))
                except (ValueError, KeyError, TypeError, OverflowError):
                    self._reject("invalid_llm_candidate")
        gp_started = time.perf_counter()
        raw.extend(self.bindings.descendants(
            parents,
            count=self.config.llm_proposals_per_iteration * self.config.gp_descendants_per_slot,
            motifs=self.motifs, rng=rng, iteration=iteration, island=island,
            operator_weights=self.population.operator_weights(("minimum", "maximum", "subtree", "cap", "scale", "elastic", "feature", "unwrap", "immigrant") + (("cap_mode",) if self.config.cyclic_cap else ())),
            parent_weights=self.population.parent_weights(parents, pending_by_parent),
        ))
        gp_seconds = time.perf_counter() - gp_started
        dedup_started = time.perf_counter()
        accepted: list[SearchCandidate] = []
        batch_hashes: set[str] = set()
        batch_structures: set[str] = set()
        for candidate in raw:
            signature = structural_signature(candidate.ast)
            if candidate.ast_hash in self._known_hashes or candidate.ast_hash in batch_hashes:
                self._reject("ast_duplicate"); continue
            if signature in self._known_structures or signature in batch_structures:
                self._reject("structural_duplicate"); continue
            batch_hashes.add(candidate.ast_hash); batch_structures.add(signature)
            self._known_hashes.add(candidate.ast_hash); self._known_structures.add(signature)
            accepted.append(candidate)
        self._last_proposal_timing = {
            "llm_seconds": round(llm_seconds, 6), "gp_seconds": round(gp_seconds, 6),
            "dedup_seconds": round(time.perf_counter() - dedup_started, 6),
            "total_seconds": round(time.perf_counter() - started, 6),
            "raw_candidates": len(raw), "accepted_candidates": len(accepted),
        }
        return accepted

    def _screen_candidates(self, candidates: Sequence[SearchCandidate]) -> list[ScreenedCandidate]:
        rows = []
        for candidate in candidates:
            try:
                row = self.screen.score(candidate, self.config)
                if candidate.num_parameters == 0 and row.behavior_signature.startswith("constant-action:"):
                    self._reject("trivial_parameter_free_behavior"); continue
                rows.append(row)
            except (ValueError, KeyError, TypeError, IndexError, FloatingPointError, OverflowError):
                self._reject("screen_failure")
        return rows

    def _ingest_completed(self) -> int:
        started = time.perf_counter(); population_seconds = 0.0; inserted = 0; newly_seen = 0
        for ast_hash, individual in sorted(self.scheduler.completed.items()):
            if ast_hash in self._ingested: continue
            newly_seen += 1; self._ingested.add(ast_hash); add_started = time.perf_counter()
            if self.population.add(self._with_fitted_behavior(individual)): inserted += 1
            population_seconds += time.perf_counter() - add_started
        motif_seconds = 0.0
        if inserted:
            motif_started = time.perf_counter(); self._update_motif_promotions(); motif_seconds = time.perf_counter() - motif_started
        self._last_ingest_timing = {"newly_seen": newly_seen, "inserted": inserted, "population_add_seconds": round(population_seconds, 6), "motif_promotion_seconds": round(motif_seconds, 6), "total_seconds": round(time.perf_counter() - started, 6)}
        return inserted

    def _pending_by_parent(self) -> dict[str, int]:
        candidates = [task.candidate for task in self.scheduler.pending]
        candidates.extend(active.task.candidate for active in self.scheduler.active.values())
        counts: dict[str, int] = {}
        for candidate in candidates:
            for parent in set(candidate.parents): counts[parent] = counts.get(parent, 0) + 1
        return counts

    def _with_fitted_behavior(self, individual):
        """Store behavior from completed fits, replacing the screening proxy."""
        from sr.core.diagnostics import action_signature
        screens = getattr(self.screen, "screens", None)
        if screens is None:
            screens = {self.case_name: self.screen}
        signatures = {}
        for name, screen in sorted(screens.items()):
            provider = getattr(screen, "parent_behavior_signature", None)
            if provider is None:
                # Custom, non-inventory screens may define their own signatures.
                continue
            fitted = (individual.case_results[name] if hasattr(individual, "case_results")
                      else individual)
            signatures[name] = provider(fitted)
        if not signatures:
            return individual
        candidate = replace(individual.candidate, detail={
            **individual.candidate.detail,
            "behavior_signature": action_signature(signatures),
            "case_behavior_signatures": signatures,
            "behavior_source": "fitted_actions_on_fixed_diagnostic_bank",
        })
        return replace(individual, candidate=candidate)

    def _parent_evidence(self, parents) -> dict[str, object]:
        """Keep fitting outcomes separate from cheap screening surrogates."""
        screens = getattr(self.screen, "screens", None)
        if screens is None:
            screens = {self.case_name: self.screen}
        diagnostics = []
        records = []
        for name, screen in screens.items():
            fitted = [row.case_results[name] if hasattr(row, "case_results") else row
                      for row in parents]
            provider = getattr(screen, "parent_diagnostics", None)
            diagnostics.append(provider(fitted) if provider else
                               {"case": name, "available": False, "reason": "no_dp_reference"})
            for row in fitted:
                context = getattr(screen, "context", None)
                specs = (context.parameter_specs_for_ast(row.candidate.ast, self.motifs)
                         if context is not None and hasattr(context, "parameter_specs_for_ast")
                         else context.parameter_specs(row.candidate.parameter_names)
                         if context is not None else ())
                if specs:
                    from optimization.policies.parameters import apply_cap_modes
                    specs = apply_cap_modes(specs, row.candidate.ast, self.motifs)
                records.append({"case": name, "parent_hash": row.ast_hash,
                                "ast": row.candidate.ast, "formula": row.candidate.formula,
                                "parameter_domains": [asdict(spec) for spec in specs],
                                "fitted_parameters": dict(row.params),
                                "policy_cost": row.policy_cost, "reference_cost": row.dp_cost,
                                "gap_pct": row.gap_pct, "status": row.status})
        return {"case_evaluations": records, "diagnostics": diagnostics}

    def _maybe_reflect(self, iteration: int) -> None:
        if iteration <= 0 or iteration % self.config.reflection_iteration_interval:
            return
        if self.reflection_client is None: return
        completed = len(self._ingested)
        best = min((row.gap_pct for row in self.population.public_frontier()), default=float("inf"))
        try:
            trials = self._run_while_scheduling(_reflect_auxiliary, self.reflection_client, self._reflection_representatives(), self.motifs, len(self._reflections) + 1, {"case": self.case_name, "iteration": iteration, "completed_standard": completed, "best_gap_pct": best, **getattr(self, "_selected_evidence", {})}, self.domain)
        except Exception as exc:
            self._reject(f"reflection_failure:{type(exc).__name__}"); return
        for trial in trials:
            if trial.name in self.motifs: continue
            self.motifs[trial.name] = trial.ast
            if hasattr(self.screen, "register_motif"): self.screen.register_motif(trial.name, trial.ast)
            else: self.screen.motifs[trial.name] = trial.ast
            if hasattr(self.scheduler, "register_motif"): self.scheduler.register_motif(trial.name, trial.ast)
            self._reflections.append(trial); self._reflection_baselines[trial.name] = best

    def _reflection_representatives(self) -> tuple[SearchIndividual, ...]:
        # Reflection sees exactly the parents and evidence used by mutation.
        return tuple(getattr(self, "_selected_parents", ()))

    def _update_motif_promotions(self) -> None:
        for index, motif_row in enumerate(self._reflections):
            if motif_row.status != "promoted" and motif_promotion_evidence(motif_row.name, self._reflection_baselines.get(motif_row.name, float("-inf")), tuple(self.population.archive.values())):
                self._reflections[index] = replace(motif_row, status="promoted")

    def _remember(self, candidate: SearchCandidate) -> bool:
        signature = structural_signature(candidate.ast)
        if candidate.ast_hash in self._known_hashes or signature in self._known_structures: return False
        self._known_hashes.add(candidate.ast_hash); self._known_structures.add(signature); return True

    def _run_while_scheduling(self, function: object, *args: object) -> object:
        context = self.scheduler._context(); queue = context.Queue(maxsize=1)
        process = context.Process(target=_auxiliary_worker, args=(queue, function, tuple(args))); process.daemon = False; process.start()
        message = None
        try:
            while message is None and process.is_alive():
                self.scheduler.poll()
                try: message = queue.get_nowait()
                except queue_module.Empty: pass
                process.join(timeout=0.05)
            process.join(timeout=0.2)
            if message is None:
                try: message = queue.get(timeout=0.5)
                except queue_module.Empty as exc: raise RuntimeError(f"auxiliary operation exited without result; exitcode={process.exitcode}") from exc
            if message[0] != "completed": raise RuntimeError(f"auxiliary operation failed: {message[1]}: {message[2]}")
            return message[1]
        finally:
            if process.is_alive(): process.terminate(); process.join(timeout=1.0)
            try: queue.close()
            except Exception: pass

    def _reject(self, reason: str) -> None:
        self._rejections[reason] = self._rejections.get(reason, 0) + 1

    def _write_checkpoint(self, iteration: int) -> None:
        archive = sorted(self.population.archive.values(), key=lambda row: (row.gap_pct, row.num_parameters, row.expanded_complexity, row.ast_hash))
        eligible = archive
        payload = {"schema_version": 2, "case": self.case_name, "iteration": int(iteration), "config": asdict(self.config), "elapsed_seconds": time.perf_counter() - self._started_at, "scheduler_counts": self.scheduler.counts(), "rejections": dict(sorted(self._rejections.items())), "archive_size": len(archive), "parameter_profile": [individual_summary(row, self.motifs) for row in parameter_profile(eligible)], "complexity_profile": [individual_summary(row, self.motifs) for row in complexity_profile(eligible)], "parameter_frontier": [individual_summary(row, self.motifs) for row in parameter_frontier(eligible)], "complexity_frontier": [individual_summary(row, self.motifs) for row in complexity_frontier(eligible)], "reflection_motifs": [reflection_summary(row) for row in self._reflections], "breeding": self.population.breeding_summary()}
        atomic_json(payload, self.output_dir / "checkpoint.json")

    def _result(self, iterations_completed: int) -> SearchRunResult:
        archive = tuple(sorted(self.population.archive.values(), key=lambda row: (row.gap_pct, row.num_parameters, row.expanded_complexity, row.ast_hash)))
        eligible = archive
        return SearchRunResult(case_name=self.case_name, output_dir=self.output_dir, archive=archive, parameter_frontier=parameter_frontier(eligible), complexity_frontier=complexity_frontier(eligible), parameter_profile=parameter_profile(eligible), complexity_profile=complexity_profile(eligible), scheduler_counts=self.scheduler.counts(), iterations_completed=iterations_completed, elapsed_seconds=time.perf_counter() - self._started_at, reflection_motifs=tuple(self._reflections), motifs=dict(self.motifs))


__all__ = ["SingleSearchController"]
