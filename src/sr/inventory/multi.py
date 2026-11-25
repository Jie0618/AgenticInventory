"""Canonical inventory composition root for generic multi-case SR search."""

from __future__ import annotations

import time
import json
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

from inventory.policy.ast import AST
from sr.analysis.frontier import FrontierSeedBundle
from sr.analysis.multi_report import search_evaluation_dir
from sr.core.gp import structural_signature
from sr.core.inventory import InventorySearchBindings
from sr.core.models import ReflectionMotif, SearchCandidate, SearchConfig
from sr.multi.controller import MultiSearchController
from sr.multi.models import MultiObjective
from sr.multi.population import MultiIslandPopulation
from sr.multi.screening import MultiCandidateScreen

from sr.inventory.multi_evaluation import CaseEvaluationInput, MultiStandardScheduler
from sr.inventory.initialization import run_frontier_initialization
from sr.inventory.screening import BellmanScreen


class MultiDistributionSearch(MultiSearchController):
    def __init__(
        self,
        *,
        case_inputs: Mapping[str, CaseEvaluationInput],
        diagnostic_paths: Mapping[str, Path],
        seed_bundle: FrontierSeedBundle,
        initialization_cache_dir: Path,
        initialization_signature: str,
        output_dir: Path,
        config: SearchConfig,
        objective: MultiObjective,
        max_pair_retries: int = 5,
        mutation_client: object | None = None,
        reflection_client: object | None = None,
        write_run_logs: bool = True,
        screens: Mapping[str, object] | None = None,
        resume: bool = False,
    ) -> None:
        self.case_inputs = {str(name): value for name, value in case_inputs.items()}
        self.case_names = tuple(self.case_inputs)
        if set(diagnostic_paths) != set(self.case_names):
            raise ValueError("diagnostic paths do not match case inputs")
        if any(value.context is None for value in self.case_inputs.values()):
            raise ValueError("multi-case search requires numerical search contexts")
        contexts = tuple(value.context for value in self.case_inputs.values())
        assert all(context is not None for context in contexts)
        compositions = {context.case.config.composition_hash for context in contexts}
        if len(compositions) != 1:
            raise ValueError("all cases must share one inventory composition")
        self.case_name = "multi:" + "+".join(self.case_names)
        self.context = max(contexts, key=lambda row: row.case.n)
        self.diagnostic_json = None if screens is not None else Path(diagnostic_paths[self.case_names[0]])
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.objective = objective
        self.mutation_client = mutation_client
        self.reflection_client = reflection_client
        self.seed_bundle = seed_bundle
        self.initialization_cache_dir = Path(initialization_cache_dir)
        self.initialization_signature = str(initialization_signature)
        self.max_pair_retries = int(max_pair_retries)
        self.bindings = (
            InventorySearchBindings.build(self.context, config) if screens is None
            else InventorySearchBindings.build_context_free(self.context.case, config)
        )
        self.domain = self.bindings.domain
        self.motifs: dict[str, AST] = dict(seed_bundle.motifs)
        screens = screens if screens is not None else {
            name: BellmanScreen(
                Path(diagnostic_paths[name]),
                context=self.case_inputs[name].context,
                motifs=self.motifs,
                random_seed=config.random_seed,
            )
            for name in self.case_names
        }
        self.screen = MultiCandidateScreen(screens, objective)
        self.samples = self.case_inputs[self.case_names[0]].samples
        self.scheduler = MultiStandardScheduler(
            config,
            case_names=self.case_names,
            objective=objective,
            motifs=self.motifs,
            result_dir=search_evaluation_dir(self.output_dir),
            event_log=(self.output_dir / "events.jsonl") if write_run_logs else None,
            max_pair_retries=self.max_pair_retries,
        )
        self.population = MultiIslandPopulation(config)
        self._ingested: set[str] = set()
        self._known_hashes: set[str] = set()
        self._known_structures: set[str] = set()
        self._reflections: list[ReflectionMotif] = []
        self._reflection_baselines: dict[str, float] = {}
        self._rejections: dict[str, int] = {}
        self._last_ingest_timing: dict[str, float | int] = {}
        self._last_proposal_timing: dict[str, float | int] = {}
        self._started_at = time.perf_counter()
        self.resume = resume
        self._resume_iteration = 0
        if (self.output_dir / "checkpoint.json").exists() and not resume:
            raise ValueError("Existing Stage2 checkpoint requires --resume or a new output directory")

    def _evaluate_initial_seeds(self) -> None:
        if self.resume:
            self._restore_completed_run()
            return
        initialized = run_frontier_initialization(
            self.seed_bundle,
            self.case_inputs,
            config=self.config,
            objective=self.objective,
            cache_dir=self.initialization_cache_dir,
            initialization_signature=self.initialization_signature,
            max_pair_retries=self.max_pair_retries,
        )
        for individual in initialized.individuals:
            self._known_hashes.add(individual.ast_hash)
            self._known_structures.add(structural_signature(individual.candidate.ast))
            self._ingested.add(individual.ast_hash)
            self.population.add(self._with_fitted_behavior(individual),
                                initial_islands=self.seed_bundle.initial_islands.get(individual.ast_hash, ()))
        if not self.population.archive:
            raise RuntimeError("multi-distribution initialization produced no population")

    def _restore_completed_run(self) -> None:
        """Recover durable case results; bandit counters are not stored."""
        from runtime.artifacts import atomic_json
        from sr.core.models import ReflectionMotif
        from sr.evaluation.standard import load_standard_result
        from sr.multi.models import aggregate_case_results
        from sr.core.randomness import derived_seed

        payload = json.loads((self.output_dir / "checkpoint.json").read_text())
        previous_config = dict(payload["config"])
        current_config = asdict(self.config)
        previous_timeout = previous_config.pop("evaluation_hard_timeout_seconds")
        current_timeout = current_config.pop("evaluation_hard_timeout_seconds")
        if (payload["initialization_signature"] != self.initialization_signature
                or tuple(payload["case_names"]) != self.case_names
                or MultiObjective(**payload["objective"]).to_payload() != self.objective.to_payload()
                or previous_config != current_config
                or current_timeout < previous_timeout):
            raise ValueError("Stage2 resume inputs/config do not match the checkpoint")
        self._resume_iteration = int(payload["iteration"])
        if not 0 <= self._resume_iteration <= self.config.iterations:
            raise ValueError("invalid resume iteration")
        self._started_at -= float(payload.get("elapsed_seconds", 0))
        for name, ast in payload["motifs"].items():
            if name in self.motifs and self.motifs[name] != ast:
                raise ValueError(f"resume motif changed: {name}")
            self.motifs[name] = ast
            self.scheduler.register_motif(name, ast)
            self.screen.register_motif(name, ast)
        self._reflections = [ReflectionMotif(**row) for row in payload.get("reflection_motifs", ())]
        self._rejections.update(payload.get("rejections", {}))
        restored = partial = invalid = 0
        directories = {}
        for result_root in (self.initialization_cache_dir / "evaluations", self.scheduler.result_dir):
            if result_root.exists():
                for directory in sorted(result_root.iterdir()):
                    if directory.is_dir():
                        directories[directory.name] = directory
        for directory in directories.values():
            if not directory.is_dir():
                continue
            paths = [directory / f"{name}.json" for name in self.case_names]
            existing = [p for p in paths if p.exists()]
            if not existing:
                continue
            first = json.loads(existing[0].read_text())
            candidate = self.bindings.candidate(
                first["ast"], motifs=self.motifs, source=first["source"],
                parents=first.get("parents", ()), iteration=int(first.get("iteration", 0)),
                island=int(first.get("island", 0)), detail=first.get("candidate_detail", {}),
            )
            if candidate.ast_hash != directory.name:
                raise ValueError(f"resume AST hash mismatch: {directory}")
            case_results = {}
            for name, path in zip(self.case_names, paths):
                if not path.exists():
                    continue
                row = json.loads(path.read_text())
                seed = derived_seed(self.case_inputs[name].random_seed, candidate.ast_hash, name)
                # Preserve saved numerical observations with their original
                # implementation signature; never relabel them as new evaluations.
                result = load_standard_result(path, candidate, name, seed, row["evaluation_signature"])
                if result is not None:
                    case_results[name] = result
                else:
                    invalid += 1
            if len(case_results) == len(self.case_names):
                individual = aggregate_case_results(candidate, case_results, self.case_names, self.objective)
                self.scheduler.completed[candidate.ast_hash] = individual
                self.scheduler.group_candidates[candidate.ast_hash] = candidate
                self.scheduler.group_states[candidate.ast_hash] = "completed"
                self.population.add(self._with_fitted_behavior(individual))
                self._ingested.add(candidate.ast_hash)
                restored += 1
            else:
                # Partial groups use current strict cache validation; any
                # mismatching observations are recomputed, not forged.
                self.scheduler.submit_group(candidate, self.case_inputs)
                partial += 1
            self._known_hashes.add(candidate.ast_hash)
            self._known_structures.add(structural_signature(candidate.ast))
        if restored < int(payload["archive_size"]):
            raise ValueError(f"resume recovered only {restored} groups, checkpoint has {payload['archive_size']}")
        atomic_json({"iteration": self._resume_iteration, "restored_groups": restored,
                     "previous_timeout_seconds": previous_timeout,
                     "timeout_seconds": current_timeout,
                     "partial_groups_requeued": partial, "invalid_case_rows": invalid,
                     "bandit_state": "reconstructed population; historical counters not persisted",
                     "historical_results": "original signatures retained; new tasks use current signatures"},
                    self.output_dir / "audit" / "resume.json")

__all__ = ["MultiDistributionSearch"]
