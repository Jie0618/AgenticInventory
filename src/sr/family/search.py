from __future__ import annotations

import json
import math
import time
import inspect
import random
from dataclasses import asdict

from sr.core.search_control import (SEARCH_CONTROL_PROTOCOL, PARENT_LIMIT, PROPOSAL_COUNT,
                                    select_parents, offspring_reward)
from sr.core.diagnostics import DIAGNOSTIC_PROTOCOL
from .diagnostics import FamilyParentDiagnostics, SHARED_DIAGNOSTICS
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .proposals import FEATURE_INIT_PROTOCOL, feature_initialization_population
from .evaluator import Worker, evaluate_families
from inventory.policy.family import PolicyFamily
from inventory.policy.family_domain import InventoryFamilyDomain
from .models import FamilyEvaluation
from .scoring import validate_score_protocol
from inventory.common.fitness import SCORE_PROTOCOL
from .planner import plan_family_tasks
from sr.core.tree import contains_subtree
from runtime.artifacts import atomic_json
from .artifacts import artifact_files, read_path, write_path


ProposalFn = Callable[..., Sequence[PolicyFamily]]
ReflectionFn = Callable[..., Mapping[str, object]]

PARENT_CONTEXT_LIMIT = PARENT_LIMIT


class MultiContextFamilySearch:
    def __init__(self, *, domain: InventoryFamilyDomain, initial_families: Sequence[PolicyFamily], case_ids: Sequence[str], worker: Worker, output_dir: Path, workers: int = 60, iterations: int = 100, global_seed: int = 20260811, proposal_fn: ProposalFn | None = None, reflection_fn: ReflectionFn | None = None, resume: bool = False, fidelity: str = "standard", feature_init_pass: bool = False, feature_init_families: Sequence[PolicyFamily] | None = None, reference_mode: str = "best_baseline", score_protocol: str = SCORE_PROTOCOL, initialization_fn: Callable | None = None, scheduling: Mapping[str, object] | None = None) -> None:
        if workers <= 0 or iterations < 0:
            raise ValueError("workers must be positive and iterations must be nonnegative")
        if len(case_ids) != domain.training_case_count:
            raise ValueError("case_ids must match the domain training-case count")
        self.case_ids = tuple(str(x) for x in case_ids)
        self.domain = domain
        self.initial_families = tuple(initial_families)
        if not self.initial_families:
            raise ValueError("family search requires at least one experiment-provided seed")
        for family in self.initial_families:
            self.domain.validate_family(family)
        self._anchor_hashes = {
            self.domain.semantic_hash(family) for family in self.initial_families
        }
        self.score_protocol = validate_score_protocol(score_protocol)
        from .template_init import PROTOCOL
        if initialization_fn is not None and getattr(initialization_fn, "protocol", None) != PROTOCOL:
            raise ValueError("unsupported LLM initialization protocol")
        self.initialization_fn = initialization_fn
        self.reference_mode = str(reference_mode)
        self.worker = worker
        self._diagnostic_provider = getattr(worker, "parent_diagnostics", None)
        if self._diagnostic_provider is None and hasattr(worker, "case_map"):
            self._diagnostic_provider = FamilyParentDiagnostics(worker, domain)
        self.scheduling = dict(scheduling or {})
        wait_seconds = float(self.scheduling.get("iteration_wait_seconds", 300))
        if not math.isfinite(wait_seconds) or wait_seconds < 0:
            raise ValueError("iteration_wait_seconds must be finite and nonnegative")
        if int(self.scheduling.get("max_pending_families", 20)) <= 0:
            raise ValueError("max_pending_families must be positive")
        self.output_dir = Path(output_dir)
        self.workers = workers
        self.iterations = iterations
        self.global_seed = global_seed
        self.contexts = domain.context_ids
        self.resume = bool(resume)
        self.fidelity = str(fidelity)
        self.feature_init_pass = bool(feature_init_pass)
        self.feature_init_families = (
            self.initial_families
            if feature_init_families is None
            else tuple(feature_init_families)
        )
        initial_semantics = {
            self.domain.semantic_hash(family) for family in self.initial_families
        }
        for family in self.feature_init_families:
            self.domain.validate_family(family)
            if self.domain.semantic_hash(family) not in initial_semantics:
                raise ValueError(
                    "feature_init_families must be selected from initial_families"
                )
        if self.fidelity not in {"fast", "standard", "certify"}:
            raise ValueError("fidelity must be fast, standard, or certify")
        self.proposal_fn = proposal_fn
        self.reflection_fn = reflection_fn
        self._parent_visits: dict[str, int] = {}
        self._parent_rewards: dict[str, float] = {}
        self._operator_visits: dict[str, int] = {}
        self._operator_rewards: dict[str, float] = {}

    def _llm_initial_candidates(self) -> tuple[PolicyFamily, ...]:
        """Persist model proposals before evaluation so interruption reuses them."""
        from .proposals import family_from_json
        path = read_path(self.output_dir, "LLM_TEMPLATE_INIT.json")
        if path.exists():
            payload = json.loads(path.read_text())
            expected = getattr(self.initialization_fn, "protocol", None)
            if expected is not None and payload.get("protocol") != expected:
                raise ValueError("LLM initialization protocol mismatch; use a new output directory")
            families = tuple(family_from_json(row, domain=self.domain)
                             for row in payload["families"])
        else:
            families, audit = self.initialization_fn(self.initial_families)
            if audit.get("status") == "failed":
                raise RuntimeError("LLM template initialization failed; retry before starting search")
            payload = {**audit, "protocol": self.initialization_fn.protocol, "families": []}
            from dataclasses import asdict
            for family in families:
                from .validation import validate_search_family
                validate_search_family(family, self.domain)
                payload["families"].append(asdict(family))
            path = write_path(self.output_dir, "LLM_TEMPLATE_INIT.json")
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, indent=2) + "\n")
            temporary.replace(path)
        if payload.get("status") == "failed":
            raise RuntimeError("LLM template initialization failed; retry before starting search")
        from .validation import validate_search_family
        for family in families:
            validate_search_family(family, self.domain)
        return self._novel_generation(families, self.initial_families)

    def _semantic_hash(self, family: PolicyFamily) -> str:
        return self.domain.semantic_hash(family)

    def _artifact_hash(self, family: PolicyFamily) -> str:
        """Return the domain-aware identity written to current artifacts."""

        return self._semantic_hash(family)

    def _unique_by_semantic(self, families: Sequence[PolicyFamily]) -> list[PolicyFamily]:
        """Keep one representative per executable family."""
        unique: dict[str, PolicyFamily] = {}
        for family in families:
            unique.setdefault(self._semantic_hash(family), family)
        return list(unique.values())

    def _novel_generation(self, generated: Sequence[PolicyFamily], population: Sequence[PolicyFamily]) -> tuple[PolicyFamily, ...]:
        """Drop semantic aliases before they can reach a numerical queue."""

        seen = {self._semantic_hash(family) for family in population}
        novel: list[PolicyFamily] = []
        for family in generated:
            key = self._semantic_hash(family)
            if key in seen:
                continue
            seen.add(key)
            novel.append(family)
        return tuple(novel)

    def _run_manifest(self) -> dict[str, object]:
        config_path = getattr(self.worker, "config_path", None)
        evaluation_config = json.loads(Path(config_path).read_text()) if config_path is not None else None
        return {
            "stage": "multi_context_family_search",
            "contexts": list(self.contexts),
            "family_validation_contexts": list(self.contexts),
            "training_case_ids": list(self.case_ids),
            "initial_family_hashes": sorted(self._anchor_hashes),
            "global_seed": int(self.global_seed),
            "workers": self.workers,
            "scheduling": self.scheduling,
            "iterations": self.iterations,
            "feature_init_pass": self.feature_init_pass,
            "llm_template_init": self.initialization_fn is not None,
            "llm_template_init_protocol": getattr(self.initialization_fn, "protocol", None),
            "feature_init_protocol": FEATURE_INIT_PROTOCOL,
            "feature_init_family_hashes": sorted(
                self.domain.semantic_hash(family)
                for family in self.feature_init_families
            ),
            "proposals_per_iteration": PROPOSAL_COUNT,
            "parent_context_limit": PARENT_CONTEXT_LIMIT,
            "parent_selection_protocol": SEARCH_CONTROL_PROTOCOL,
            "diagnostic_protocol": DIAGNOSTIC_PROTOCOL,
            "gp_enabled": False,
            "optimizer_mode": self.fidelity,
            "fidelity": self.fidelity,
            "optimizer_budget_contract": "canonical_fast_standard_certify_with_two_fidelity_mc",
            "screening": {
                "mode": "disabled",
                "reason": "family-level early screening is not reliable with heterogeneous tasks",
            },
            "cache": "content_hash_task_cache",
            "deduplication": "semantic_cross_context_hash",
            "evaluator": "domain_selected_mc_backend",
            "reference_mode": self.reference_mode,
            "experiment_signature": str(getattr(
                self.worker, "evaluation_protocol_signature", "legacy"
            )),
            "evaluation_config": evaluation_config,
            "score": {
                "protocol": self.score_protocol,
                "within_context": "0.6*mean+0.4*CVaR80",
                "across_contexts": "0.6*mean+0.4*CVaR80",
                "cross_context_mean_weight": 0.6,
                "cvar_alpha": 0.8, "tail_weight": 0.4,
                "reference_gaps": "signed_fixed_reference",
            },
        }

    def _validate_or_write_manifest(
        self, manifest: Mapping[str, object], *, persist: bool = True
    ) -> None:
        """Prevent a resume from silently changing the experiment identity."""

        def identity(payload: Mapping[str, object]) -> dict[str, object]:
            score = dict(payload.get("score", {}))
            score_identity = {
                "protocol": score.get("protocol"),
                "mean_weight": score.get("cross_context_mean_weight"),
                "tail_weight": score.get("tail_weight"),
                "cvar_alpha": score.get("cvar_alpha"),
            }
            screening = dict(payload.get("screening", {}))
            required = {
                "contexts", "family_validation_contexts", "training_case_ids",
                "feature_init_pass", "llm_template_init", "llm_template_init_protocol",
                "proposals_per_iteration", "parent_context_limit", "parent_selection_protocol",
                "diagnostic_protocol", "gp_enabled", "scheduling", "fidelity", "screening", "score",
                "initial_family_hashes", "feature_init_family_hashes", "feature_init_protocol",
                "global_seed", "reference_mode", "experiment_signature", "evaluation_config",
            }
            missing = required - payload.keys()
            if missing:
                raise ValueError("run manifest missing required fields: " + ", ".join(sorted(missing)))
            contexts = payload["contexts"]
            validation_contexts = payload["family_validation_contexts"]
            training_case_ids = payload["training_case_ids"]
            result: dict[str, object] = {
                "contexts": [str(value) for value in contexts],
                "family_validation_contexts": [
                    str(value) for value in validation_contexts
                ],
                "training_case_ids": [str(value) for value in training_case_ids],
                "feature_init_pass": payload.get("feature_init_pass"),
                "llm_template_init": payload["llm_template_init"],
                "llm_template_init_protocol": payload.get("llm_template_init_protocol"),
                "proposals_per_iteration": payload.get("proposals_per_iteration"),
                "parent_context_limit": payload.get("parent_context_limit"),
                "parent_selection_protocol": payload["parent_selection_protocol"],
                "diagnostic_protocol": payload.get("diagnostic_protocol"),
                "gp_enabled": payload.get("gp_enabled"),
                "scheduling": payload["scheduling"],
                "fidelity": payload["fidelity"],
                "screening_mode": screening.get("mode"),
                "score": score_identity,
            }
            for key in (
                "initial_family_hashes",
                "feature_init_family_hashes",
                "feature_init_protocol",
                "global_seed",
                "reference_mode",
                "experiment_signature",
                "evaluation_config",
            ):
                result[key] = payload[key]
            return result

        path = self.output_dir / "RUN_MANIFEST.json"
        if self.resume and (self.output_dir / "CHECKPOINT.json").exists() and not path.exists():
            raise ValueError("checkpoint requires its current RUN_MANIFEST.json")
        if path.exists():
            if not self.resume:
                raise FileExistsError(
                    f"run manifest already exists; pass resume=True or use a new output directory: {path}"
                )
            existing = json.loads(path.read_text(encoding="utf-8"))
            existing_identity = identity(existing)
            current_identity = identity(manifest)
            old_config = dict(existing.get("evaluation_config") or {})
            new_config = dict(manifest.get("evaluation_config") or {})
            execution_changes = list(existing.get("execution_changes", ()))
            old_timeout = old_config.pop("evaluation_hard_timeout_seconds", None)
            new_timeout = new_config.pop("evaluation_hard_timeout_seconds", None)
            if (old_config == new_config and old_timeout is not None and new_timeout is not None
                    and math.isfinite(float(new_timeout)) and 0 < float(old_timeout) <= float(new_timeout)):
                existing_identity["evaluation_config"] = manifest["evaluation_config"]
                if old_timeout != new_timeout:
                    execution_changes.append({"field": "evaluation_hard_timeout_seconds",
                                              "previous": old_timeout, "current": new_timeout})
            mismatches = [
                key
                for key in sorted(set(existing_identity) | set(current_identity))
                if existing_identity.get(key) != current_identity.get(key)
            ]
            if mismatches:
                raise ValueError(
                    "run manifest mismatch for immutable fields: "
                    + ", ".join(sorted(mismatches))
                )
            updated = dict(existing)
            updated.update(manifest)
            if execution_changes:
                updated["execution_changes"] = execution_changes
            manifest = updated
        if persist:
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            temporary.replace(path)

    def validate_run_identity(self) -> None:
        """Read-only preflight before setting-owned reference preparation."""
        self._validate_or_write_manifest(self._run_manifest(), persist=False)
        checkpoint = self.output_dir / "CHECKPOINT.json"
        if self.resume and checkpoint.exists():
            iteration = int(json.loads(checkpoint.read_text()).get("iteration", 0))
            if not 0 <= iteration <= self.iterations:
                raise ValueError(f"checkpoint iteration {iteration} exceeds requested iterations {self.iterations}")

    def run(self) -> tuple[FamilyEvaluation, ...]:
        self._pipeline = None
        try:
            return self._run()
        finally:
            if self._pipeline is not None:
                self._pipeline.close()

    def _run(self) -> tuple[FamilyEvaluation, ...]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(self.worker, "reference_cache_dir"):
            self.worker.run_reference_cache_dir = self.output_dir / "reference_cache"
        archive: list[FamilyEvaluation] = []
        population = list(self.initial_families)
        feature_history: list[dict[str, object]] = []
        reflected_motifs: dict[str, Mapping[str, object]] = {}
        checkpoint = self.output_dir / "CHECKPOINT.json"
        start_iteration = 0
        cp: Mapping[str, object] | None = None
        if self.resume and checkpoint.exists():
            cp = json.loads(checkpoint.read_text(encoding="utf-8"))
            start_iteration = int(cp.get("iteration", 0))
            if start_iteration > self.iterations:
                raise ValueError(
                    f"checkpoint iteration {start_iteration} exceeds requested iterations {self.iterations}"
                )
        manifest = self._run_manifest()
        self._validate_or_write_manifest(
            manifest, persist=not (self.resume and cp is not None)
        )
        if cp is not None:
            feature_history, reflected_motifs = self._restore_search_state(cp)
            from .proposals import family_from_json
            for path in sorted(artifact_files(self.output_dir, "PROPOSALS_ITER_*.json"),
                               key=lambda path: int(path.stem.rsplit("_", 1)[-1])):
                payload = json.loads(path.read_text(encoding="utf-8"))
                proposal_iteration = int(payload.get("iteration", -1))
                # A proposal file is written before its numerical checkpoint.
                # Never let an interrupted/future proposal contaminate a
                # resumed population or parent prompt.
                if proposal_iteration < 0 or proposal_iteration > start_iteration:
                    continue
                for row in payload.get("families", []):
                    if isinstance(row, dict) and isinstance(row.get("ast"), dict):
                        try:
                            family = family_from_json(row, domain=self.domain)
                            population.append(family)
                        except Exception: pass
            # Reconstruct only the families recorded in the last completed
            # checkpoint. An experiment seed may be present in ``population``
            # without having been evaluated by iter0, so evaluating the whole
            # population here would change the experiment during resume.
            known_semantics = {self._semantic_hash(family) for family in population}
            completed_semantic_keys = {
                str(row["family_hash"])
                for row in cp["archive"]
                if row["status"] == "complete"
            }
            missing_checkpoint_families = completed_semantic_keys - known_semantics
            if missing_checkpoint_families:
                preview = ", ".join(sorted(missing_checkpoint_families)[:5])
                raise ValueError(
                    "checkpoint references families that cannot be restored from "
                    f"the seed/proposal artifacts: {preview}"
                )
            population = self._unique_by_semantic(population)
            completed_families = [
                family for family in population
                if self._semantic_hash(family) in completed_semantic_keys
            ]
            archive.extend(evaluate_families(
                completed_families, self.case_ids, self.worker,
                domain=self.domain,
                output_dir=self.output_dir, workers=self.workers,
                global_seed=self.global_seed, fidelity=self.fidelity,
                score_protocol=self.score_protocol,
            ))
            self._validate_or_write_manifest(manifest)
        # An audit precedes its checkpoint and may survive an interrupted
        # initialization. Only a durable checkpoint commits the population.
        feature_init_complete = cp is not None
        if (self.feature_init_pass or self.initialization_fn is not None) and start_iteration == 0 and not feature_init_complete:
            init_families = feature_initialization_population(
                self.domain, self.feature_init_families
            ) if self.feature_init_pass else ()
            if self.initialization_fn is not None:
                init_families = (*init_families, *self._llm_initial_candidates())
            existing_semantic = {self._semantic_hash(family) for family in population}
            init_families = [f for f in init_families if self._semantic_hash(f) not in existing_semantic]
            init_families = self._unique_by_semantic(init_families)
            write_path(self.output_dir, "PROPOSALS_ITER_0.json").write_text(json.dumps({"iteration": 0, "count": len(init_families), "families": [{"family_id": f.family_id, "family_hash": self._artifact_hash(f), "semantic_hash": self._semantic_hash(f), "parent_family_id": f.parent_family_id, "parent_family_hash": f.parent_family_hash, "parent_family_hashes": list(f.parent_family_hashes), "mutation_action": f.mutation_action, "description": f.description, "ast": f.ast, "motifs": f.motifs, "parameters": [p.__dict__ for p in f.parameters], "features": [x.__dict__ for x in f.features]} for f in init_families]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            init_batch = self._unique_by_semantic(init_families)
            init_eval = evaluate_families(
                init_batch, self.case_ids, self.worker,
                domain=self.domain,
                output_dir=self.output_dir, workers=self.workers,
                global_seed=self.global_seed, fidelity=self.fidelity,
                score_protocol=self.score_protocol,
            )
            for row in init_eval:
                if row.status != "complete" or not math.isfinite(row.cross_context_score):
                    raise RuntimeError(
                        f"initial family {row.family.family_id} failed evaluation: "
                        f"status={row.status}, score={row.cross_context_score}"
                    )
            archive.extend(init_eval)
            if not init_eval:
                raise RuntimeError("no complete initial families remain after evaluation; inspect rejected_families")
            successful = {self._semantic_hash(row.family) for row in init_eval}
            population = [family for family in init_batch
                          if self._semantic_hash(family) in successful]
            write_path(self.output_dir, "ITERATION_AUDIT_0.json").write_text(json.dumps({"iteration": 0, "generated_count": len(init_families), "evaluated_count": len(init_eval), "generated": [{"family_id": f.family_id, "parent_family_id": f.parent_family_id, "mutation_action": f.mutation_action, "family_hash": self._artifact_hash(f), "semantic_hash": self._semantic_hash(f)} for f in init_families], "evaluated": [{"family_id": r.family.family_id, "cross_context_score": r.cross_context_score, "status": r.status, "metric_count": len(r.metrics), "semantic_hash": self._semantic_hash(r.family)} for r in init_eval], "invariants": {"all_evaluated_complete": all(r.status == "complete" for r in init_eval), "active_contexts": list(self.contexts), "fidelity": self.fidelity}}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            atomic_json({"iteration": 0, "generated_families": len(init_families), "evaluated_families": len(init_eval), "screened_out": 0, "archive": [self._summary(r) for r in archive], "search_state": self._search_state_payload(reflected_motifs, feature_history)}, self.output_dir / "CHECKPOINT.json")
            # Iteration zero is the template initialization pass; the
            # normal LLM loop retains its existing 1..N audit numbering.
            start_iteration = 0
        # Checkpoints are progress/audit records.  Full metric tensors are not
        # serialized yet, so an interrupted family is safely recomputed rather
        # than incorrectly treated as a completed archive entry.
        if any(row.status != "complete" or not math.isfinite(row.cross_context_score)
               for row in archive):
            raise RuntimeError("restored or initialized family evaluation failed")
        scheduling = getattr(self, "scheduling", {})
        if scheduling.get("async_evaluation", False):
            if not archive and start_iteration == 0:
                initial_rows = evaluate_families(
                    population, self.case_ids, self.worker, domain=self.domain,
                    output_dir=self.output_dir, workers=self.workers,
                    global_seed=self.global_seed, fidelity=self.fidelity,
                    score_protocol=self.score_protocol)
                if any(row.status != "complete" or not math.isfinite(row.cross_context_score)
                       for row in initial_rows):
                    raise RuntimeError("initial family evaluation failed")
                archive.extend(initial_rows)
            if not archive:
                raise RuntimeError("no complete initial families remain after evaluation; inspect rejected_families")
            from .pipeline import FamilyPipeline
            self._pipeline = FamilyPipeline(
                worker=self.worker, domain=self.domain, case_ids=self.case_ids,
                output_dir=self.output_dir, workers=self.workers,
                global_seed=self.global_seed, fidelity=self.fidelity,
                score_protocol=self.score_protocol,
                max_pending_families=int(scheduling.get("max_pending_families", 20)),
            )
        settled = {self._semantic_hash(row.family) for row in archive}

        def ingest(rows):
            fresh = [row for row in rows if self._semantic_hash(row.family) not in settled]
            archive.extend(fresh)
            self._update_bandits([row.family for row in fresh], archive)
            settled.update(self._semantic_hash(row.family) for row in fresh)
            archive.sort(key=lambda row: row.cross_context_score)
            for row in fresh:
                atomic_json(self._summary(row), self.output_dir / "completed_families" /
                            f"{self._semantic_hash(row.family)}.json")
            return fresh

        if self._pipeline is not None and cp is not None:
            pending_keys = set(cp.get("pending_family_hashes", ()))
            restored = {self._semantic_hash(family): family for family in population}
            if pending_keys - restored.keys():
                raise ValueError("checkpoint pending families missing from proposal artifacts")
            ingest(self._pipeline.submit([restored[key] for key in sorted(pending_keys)]))
        for iteration in range(start_iteration, self.iterations):
            if self._pipeline is not None:
                ingest(self._pipeline.capacity())
            # Generate a fresh batch on every iteration, including iteration
            # zero. Experiment-provided seeds remain eligible as anchors.
            generated = ()
            if self.proposal_fn is not None:
                # Feed the proposal model the best evaluated parents and their
                # per-context diagnostics. Keep experiment seeds as anchors and
                # cap context size so prompts do not grow with the archive.
                eligible = population if self._pipeline is None else [
                    family for family in population if self._semantic_hash(family) in settled]
                prompt_population = self._select_parents(eligible, archive, iteration)
                eval_context = {
                    self._semantic_hash(family): {
                        "cross_context_score": next((row.cross_context_score for row in reversed(archive) if self._semantic_hash(row.family) == self._semantic_hash(family)), None),
                        "per_context": [
                            {"context_id": score.context_id, "robust_score": score.robust_score}
                            for row in reversed(archive) if self._semantic_hash(row.family) == self._semantic_hash(family) for score in row.per_context
                        ],
                        "status": next((row.status for row in reversed(archive) if self._semantic_hash(row.family) == self._semantic_hash(family)), "unevaluated"),
                        "ast": family.ast, "motifs": family.motifs,
                        "parameters": [p.__dict__ for p in family.parameters],
                        "features": [f.__dict__ for f in family.features],
                        "case_evaluations": [
                            {"context_id": metric.context_id, "case_id": metric.case_id,
                             "gap_pct": metric.gap_pct, **dict(metric.metadata)}
                            for row in archive
                            if self._semantic_hash(row.family) == self._semantic_hash(family)
                            for metric in row.metrics
                        ],
                        "diagnostics": {"available": False, "reason": "no_dp_reference"},
                    }
                    for family in prompt_population
                }
                provider = self._diagnostic_provider
                if provider is not None:
                    supplied = provider(tuple(prompt_population), eval_context)
                    for key, diagnostics in supplied.items():
                        if key == SHARED_DIAGNOSTICS:
                            eval_context[key] = diagnostics
                        elif key in eval_context:
                            eval_context[key]["diagnostics"] = diagnostics
                atomic_json({"iteration": iteration + 1, "parents": eval_context},
                            write_path(self.output_dir, f"PARENT_EVIDENCE_ITER_{iteration + 1}.json"))
                # Preserve compatibility with small test/custom proposal
                # functions written against the original two-argument API,
                # without swallowing a genuine TypeError raised by a proposal
                # implementation.
                accepts_context = False
                accepts_motifs = False
                accepts_known_definitions = False
                try:
                    n_parameters = len(inspect.signature(self.proposal_fn).parameters)
                    accepts_context = n_parameters >= 3
                    accepts_motifs = n_parameters >= 4
                    accepts_known_definitions = n_parameters >= 5
                except (TypeError, ValueError):
                    accepts_context = True
                known_definition_keys = tuple(sorted({self._semantic_hash(family) for family in population}))
                if accepts_known_definitions:
                    generated = tuple(self.proposal_fn(
                        tuple(prompt_population), iteration, eval_context,
                        reflected_motifs, known_definition_keys,
                    ))
                elif accepts_motifs:
                    generated = tuple(self.proposal_fn(tuple(prompt_population), iteration, eval_context, reflected_motifs))
                elif accepts_context:
                    generated = tuple(self.proposal_fn(tuple(prompt_population), iteration, eval_context))
                else:
                    generated = tuple(self.proposal_fn(tuple(prompt_population), iteration))
                # A proposal function can be called outside the built-in LLM
                # adapter.  Defend the numerical queue here as well, so a
                # semantic alias can never consume CPU merely because a custom
                # proposer skipped the adapter's local checks.
                generated = self._novel_generation(generated, population)
                if not generated:
                    # Never silently spend a full cross-context evaluation on the
                    # seed when the configured LLM proposer failed (quota,
                    # auth, or repeated schema failure).  A zero-proposal
                    # iteration is an invalid search iteration and must be
                    # surfaced immediately for operator action.
                    failure = {
                        "iteration": iteration + 1,
                        "reason": "proposal_fn_returned_zero_valid_families",
                        "required_target": PROPOSAL_COUNT,
                    }
                    write_path(self.output_dir, f"PROPOSAL_FAILURE_ITER_{iteration + 1}.json").write_text(
                        json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                    )
                    raise RuntimeError(
                        f"iteration {iteration + 1} produced zero valid family proposals; "
                        "aborting before optimization"
                    )
                # A non-empty, fully validated batch is sufficient to make
                # progress.  LLM responses may legitimately contain fewer
                # than twenty usable mutations after local structural checks;
                # do not discard an otherwise valid iteration solely for
                # under-filling.  Empty responses remain fail-closed.
                if len(generated) == 0:
                    raise RuntimeError(
                        f"iteration {iteration + 1} produced zero valid family proposals; "
                        "aborting before optimization"
                    )
                # Under-filled batches are allowed as long as they are
                # non-empty.  A transient empty/invalid LLM response is still
                # fail-closed above, but discarding a valid batch of 1..5
                # families causes avoidable experiment aborts and defeats the
                # intended resume semantics for fast runs.
                self._validate_generation(generated, population)
                population.extend(generated)
            write_path(self.output_dir, f"PROPOSALS_ITER_{iteration + 1}.json").write_text(
                json.dumps({"iteration": iteration + 1, "count": len(generated), "families": [
                    {"family_id": family.family_id, "family_hash": self._artifact_hash(family), "semantic_hash": self._semantic_hash(family), "parent_family_id": family.parent_family_id, "parent_family_hash": family.parent_family_hash, "parent_family_hashes": list(family.parent_family_hashes), "mutation_action": family.mutation_action, "description": family.description, "ast": family.ast, "motifs": family.motifs, "parameters": [p.__dict__ for p in family.parameters], "features": [f.__dict__ for f in family.features]}
                    for family in generated
                ]}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            population = self._unique_by_semantic(population)
            evaluated_semantic_hashes = {
                self._semantic_hash(row.family) for row in archive
                if row.status == "complete"
            }
            batch = [
                family for family in population
                if self._semantic_hash(family) not in evaluated_semantic_hashes
            ]
            # Structural checks are always gates.  We deliberately do not use
            # performance screening: a family is judged only after all
            # heterogeneous (L, distribution) tasks have run.
            # Domain validation owns the executable limits for both seeds and
            # mutations; do not silently drop valid families using a second cap.
            screened_hashes: set[str] = set()
            survivors = [family for family in batch if self._semantic_hash(family) not in screened_hashes]
            if self._pipeline is not None:
                evaluated_rows = ingest(self._pipeline.submit(survivors))
                evaluated_rows.extend(ingest(self._pipeline.collect(
                    float(scheduling.get("iteration_wait_seconds", 300)))))
            else:
                evaluated_rows = list(evaluate_families(
                    survivors, self.case_ids, self.worker,
                    domain=self.domain,
                    output_dir=self.output_dir, workers=self.workers,
                    global_seed=self.global_seed, fidelity=self.fidelity,
                    score_protocol=self.score_protocol,
                ))
                archive.extend(evaluated_rows)
            # A family is eligible for evolutionary selection only after the
            # complete active-L × case task tensor has been produced.  Fail closed here so
            # a worker crash, malformed output, or accidental exact route
            # cannot be mistaken for a valid score.
            for row in evaluated_rows:
                expected = len(self.contexts) * len(self.case_ids)
                if row.status != "complete" or len(row.metrics) != expected or not math.isfinite(row.cross_context_score):
                    raise RuntimeError(
                        f"family {row.family.family_id} failed completeness invariant: "
                        f"status={row.status}, metrics={len(row.metrics)}/{expected}, "
                        f"score={row.cross_context_score}"
                    )
                if any(task.evaluator != "mc" or task.fidelity != self.fidelity for task in row.tasks):
                    raise RuntimeError(f"family {row.family.family_id} contains an unexpected fidelity task")
            self._write_iteration_audit(iteration + 1, generated, evaluated_rows, archive)
            for family in batch:
                if self._semantic_hash(family) in screened_hashes:
                    archive.append(FamilyEvaluation(
                        family,
                        plan_family_tasks(
                            family, self.case_ids, contexts=self.contexts,
                            global_seed=self.global_seed, fidelity=self.fidelity,
                            evaluator="mc",
                        ),
                        (), (), float("inf"), status="screened_out",
                    ))
            archive.sort(key=lambda row: row.cross_context_score)
            if self._pipeline is None:
                self._update_bandits(generated, archive)
            if self.reflection_fn is not None and (iteration + 1) % 5 == 0:
                reflection_parents = tuple(prompt_population) if self.proposal_fn else ()
                reflection_evidence = eval_context if self.proposal_fn else {}
                if len(inspect.signature(self.reflection_fn).parameters) >= 3:
                    reflection = dict(self.reflection_fn(reflection_parents, iteration + 1, reflection_evidence))
                else:
                    reflection = dict(self.reflection_fn(reflection_parents, iteration + 1))
                accepted_reflections = []
                rejected_reflections = list(reflection.get("rejected", []))
                for row in reflection.get("accepted", []):
                    definition = {
                        "ast": row["ast"], "description": row.get("description", ""),
                        "features": list(row.get("features", ())),
                    }
                    name = str(row["name"])
                    if name in reflected_motifs and reflected_motifs[name] != definition:
                        rejected_reflections.append(f"reflected motif name collision: {name}")
                        continue
                    reflected_motifs[name] = definition
                    accepted_reflections.append(row)
                reflection["accepted"] = accepted_reflections
                reflection["rejected"] = rejected_reflections
                write_path(self.output_dir, f"MOTIF_REFLECTION_ITER_{iteration + 1}.json").write_text(json.dumps(dict(reflection), indent=2, sort_keys=True) + "\n", encoding="utf-8")
            active_features: dict[str, object] = {}
            for family in population:
                for feature in family.features:
                    row = active_features.setdefault(feature.name, {"kind": feature.kind, "window": feature.window, "weights": list(feature.weights), "weight_rule": feature.weight_rule, "families": []})
                    row["families"] = sorted(set(row["families"]) | {family.family_id})
            feature_history.append({"iteration": iteration + 1, "active": active_features, "pruned": []})
            (self.output_dir / "FEATURE_REGISTRY.json").write_text(json.dumps({"iterations": feature_history}, indent=2) + "\n", encoding="utf-8")
            atomic_json({"iteration": iteration + 1, "generated_families": len(generated), "evaluated_families": len(evaluated_rows), "pending_family_hashes": sorted(self._pipeline.pending) if self._pipeline else [], "screened_out": len(screened_hashes), "archive": [self._summary(row) for row in archive], "search_state": self._search_state_payload(reflected_motifs, feature_history)}, self.output_dir / "CHECKPOINT.json")
        if self._pipeline is not None:
            final_rows = ingest(self._pipeline.collect())
            previous_checkpoint = json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
            atomic_json({**previous_checkpoint, "iteration": self.iterations,
                         "pending_family_hashes": [], "drain_evaluated_families": len(final_rows),
                         "archive": [self._summary(row) for row in archive],
                         "search_state": self._search_state_payload(reflected_motifs, feature_history)},
                        self.output_dir / "CHECKPOINT.json")
        return tuple(archive)

    def _search_state_payload(self, reflected_motifs: Mapping[str, Mapping[str, object]], feature_history: Sequence[Mapping[str, object]]) -> dict[str, object]:
        return {
            "parent_identity": "semantic_hash",
            "parent_visits": dict(self._parent_visits),
            "parent_rewards": dict(self._parent_rewards),
            "operator_visits": dict(self._operator_visits),
            "operator_rewards": dict(self._operator_rewards),
            "reflected_motifs": dict(reflected_motifs),
            "feature_history": list(feature_history),
        }

    def _restore_search_state(self, checkpoint: Mapping[str, object]) -> tuple[list[dict[str, object]], dict[str, Mapping[str, object]]]:
        """Restore the complete state written by the current search."""
        state = checkpoint.get("search_state")
        required = {"parent_identity", "parent_visits", "parent_rewards",
                    "operator_visits", "operator_rewards", "reflected_motifs", "feature_history"}
        if not isinstance(state, Mapping) or required - state.keys():
            raise ValueError("checkpoint requires complete search_state")
        if state["parent_identity"] != "semantic_hash":
            raise ValueError("checkpoint requires semantic_hash parent identity")
        self._parent_visits = {str(k): int(v) for k, v in dict(state["parent_visits"]).items()}
        self._parent_rewards = {str(k): float(v) for k, v in dict(state["parent_rewards"]).items()}
        self._operator_visits = {str(k): int(v) for k, v in dict(state["operator_visits"]).items()}
        self._operator_rewards = {str(k): float(v) for k, v in dict(state["operator_rewards"]).items()}
        motifs = {str(k): dict(v) for k, v in dict(state["reflected_motifs"]).items()}
        history = [dict(row) for row in state["feature_history"]]
        return history, motifs

    def _validate_generation(self, generated: Sequence[PolicyFamily], population: Sequence[PolicyFamily]) -> None:
        """Enforce one-step local lineage before any numerical task starts."""
        parent_by_hash = {self._semantic_hash(family): family for family in population}
        parents_by_id: dict[str, list[PolicyFamily]] = {}
        for parent in population:
            parents_by_id.setdefault(parent.family_id, []).append(parent)
        allowed = {"add_term", "minimum_term", "maximum_term", "add_cap", "replace_subtree", "replace_feature", "delete_term", "delete_wrapper", "crossover"}
        if self.domain.cyclic_cap:
            allowed.add("set_cap_mode")
        # Stage 3 is intended to search internal policy structure.  Permit a
        # small additive minority for exploration, but reject batches that
        # regress to the old ``B + correction`` pattern.
        seen: set[str] = set()
        parent_semantic_hashes = {self._semantic_hash(parent) for parent in population}
        seen_semantic_hashes: set[str] = set()
        for family in generated:
            if family.family_id in seen:
                raise RuntimeError(f"duplicate family id in mutation batch: {family.family_id}")
            seen.add(family.family_id)
            semantic_hash = self._semantic_hash(family)
            if semantic_hash in seen_semantic_hashes or semantic_hash in parent_semantic_hashes:
                raise RuntimeError(
                    f"family {family.family_id} is semantically duplicate with a generated or existing family"
                )
            seen_semantic_hashes.add(semantic_hash)
            parent = parent_by_hash.get(str(family.parent_family_hash or ""))
            if parent is None and family.parent_family_hash is None:
                legacy_matches = parents_by_id.get(str(family.parent_family_id or ""), [])
                if len(legacy_matches) == 1:
                    parent = legacy_matches[0]
            if parent is None:
                raise RuntimeError(
                    f"family {family.family_id} has unknown or ambiguous parent "
                    f"hash={family.parent_family_hash!r}, id={family.parent_family_id!r}"
                )
            if family.mutation_action not in allowed:
                raise RuntimeError(f"family {family.family_id} has invalid local action {family.mutation_action!r}")
            if family.mutation_action == "crossover":
                if (len(set(family.parent_family_hashes)) != 2
                        or family.parent_family_hash not in family.parent_family_hashes
                        or any(key not in parent_by_hash for key in family.parent_family_hashes)):
                    raise RuntimeError("crossover must record two known distinct parents")
                first, second = [parent_by_hash[key] for key in family.parent_family_hashes]
                if family.ast not in [
                    {"op": op, "left": first.ast, "right": second.ast}
                    for op in ("minimum", "maximum")
                ]:
                    raise RuntimeError("crossover must combine both complete parent expressions")
            if family.mutation_action in {"add_term", "minimum_term", "maximum_term", "add_cap"}:
                parent_ast = parent.ast
                child_ast = family.ast
                if not _contains_exact_subtree(child_ast, parent_ast):
                    raise RuntimeError(f"family {family.family_id} action {family.mutation_action} does not preserve parent AST")
            if family.mutation_action == "set_cap_mode":
                from sr.core.tree import iter_subtrees, replace_subtree
                matches = []
                for ref in iter_subtrees(parent.ast):
                    node = ref.subtree
                    if node["op"] == "cap":
                        for mode in ("integer", "continuous", "cyclic_integer"):
                            matches.append(replace_subtree(parent.ast, ref.path, {**node, "mode": mode}))
                    elif node["op"] == "minimum":
                        for a, b in (("left", "right"), ("right", "left")):
                            limit = node[a]
                            if limit.get("op") == "parameter" and any(p.name == limit["name"] and p.kind == "cap" for p in parent.parameters):
                                for mode in ("integer", "continuous", "cyclic_integer"):
                                    matches.append(replace_subtree(parent.ast, ref.path, {"op": "cap", "arg": node[b], "limit": limit, "mode": mode}))
                if family.ast not in matches:
                    raise RuntimeError("set_cap_mode must change exactly one cap slot")
            from .validation import validate_search_family
            validate_search_family(family, self.domain)

    def _write_iteration_audit(self, iteration: int, generated: Sequence[PolicyFamily], evaluated: Sequence[FamilyEvaluation], archive: Sequence[FamilyEvaluation]) -> None:
        payload = {
            "iteration": int(iteration),
            "generated_count": len(generated),
            "generated": [{"family_id": f.family_id, "parent_family_id": f.parent_family_id, "parent_family_hash": f.parent_family_hash, "parent_family_hashes": list(f.parent_family_hashes), "mutation_action": f.mutation_action, "family_hash": self._artifact_hash(f), "semantic_hash": self._semantic_hash(f), "complexity": self.domain.complexity_stats(f)} for f in generated],
            "evaluated_count": len(evaluated),
            "pending_family_hashes": sorted(self._pipeline.pending) if getattr(self, "_pipeline", None) else [],
            "evaluated": [{"family_id": row.family.family_id, "family_hash": self._artifact_hash(row.family), "semantic_hash": self._semantic_hash(row.family), "status": row.status, "task_count": len(row.tasks), "metric_count": len(row.metrics), "cross_context_score": row.cross_context_score, "per_context": [{"context_id": score.context_id, "mean_gap": score.mean_gap, "cvar80_gap": score.cvar80_gap, "robust_score": score.robust_score} for score in row.per_context]} for row in evaluated],
            "archive_complete_families": sum(1 for row in archive if row.status == "complete"),
            "invariants": {"exactly_twenty_mutations": len(generated) == PROPOSAL_COUNT, "distinct_mutation_hashes": len({self._artifact_hash(f) for f in generated}) == len(generated), "distinct_semantic_mutations": len({self._semantic_hash(f) for f in generated}) == len(generated), "all_evaluated_complete": all(row.status == "complete" and len(row.metrics) == len(self.contexts) * len(self.case_ids) for row in evaluated), "all_tasks_mc": all(all(task.evaluator == "mc" and task.fidelity == self.fidelity for task in row.tasks) for row in evaluated), "fidelity": self.fidelity, "active_contexts": list(self.contexts)},
        }
        write_path(self.output_dir, f"ITERATION_AUDIT_{iteration}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _select_parents(self, population: Sequence[PolicyFamily], archive: Sequence[FamilyEvaluation], iteration: int) -> list[PolicyFamily]:
        positions = {self._semantic_hash(family): index for index, family in enumerate(population)}
        available = set(positions)
        rows = [row for row in archive if row.status == "complete"
                and self._semantic_hash(row.family) in available
                and math.isfinite(row.cross_context_score)]
        stats = {self._semantic_hash(row.family): self.domain.complexity_stats(row.family)
                 for row in rows}
        def complexity(row):
            values = stats[self._semantic_hash(row.family)]
            return float(values["expanded_complexity_max"])
        def behavior(row):
            provider = getattr(self._diagnostic_provider, "behavior_signature", None)
            if provider is not None:
                return provider(row)
            signatures = [(metric.context_id, metric.case_id, metric.metadata.get("behavior_signature"))
                          for metric in row.metrics]
            return (str(sorted(signatures)) if any(value is not None for _, _, value in signatures)
                    else self._semantic_hash(row.family))
        selected = select_parents(
            rows, key=lambda row: self._semantic_hash(row.family),
            score=lambda row: row.cross_context_score,
            parameters=lambda row: len(row.family.parameters), complexity=complexity,
            behavior=behavior,
            recent=lambda row: positions[self._semantic_hash(row.family)],
            visits=self._parent_visits, rewards=self._parent_rewards,
            rng=random.Random(f"{self.global_seed}:{iteration}"), count=PARENT_CONTEXT_LIMIT,
        )
        return [row.family for row in selected]

    def _update_bandits(self, generated: Sequence[PolicyFamily], archive: Sequence[FamilyEvaluation]) -> None:
        from sr.core.pareto import nondominated
        rows = [row for row in archive if row.status == "complete" and math.isfinite(row.cross_context_score)]
        scores = {self._semantic_hash(row.family): row.cross_context_score for row in rows}
        par_front = {self._semantic_hash(row.family) for row in nondominated(
            rows, lambda row: (row.cross_context_score, float(len(row.family.parameters))))}
        cmp_front = {self._semantic_hash(row.family) for row in nondominated(
            rows, lambda row: (row.cross_context_score,
                              float(self.domain.complexity_stats(row.family)["expanded_complexity_max"])))}
        for child in generated:
            child_key = self._semantic_hash(child)
            if child_key not in scores:
                continue  # Pending or failed offspring never receive credit.
            parents = list(child.parent_family_hashes)
            if not parents and child.parent_family_hash:
                parents = [child.parent_family_hash]
            if not parents and child.parent_family_id:
                matches = {self._semantic_hash(row.family) for row in rows
                           if row.family.family_id == child.parent_family_id}
                if len(matches) == 1:
                    parents = list(matches)
            parents = list(dict.fromkeys(parents))
            if not parents or any(parent not in scores for parent in parents):
                continue
            reward = offspring_reward(min(scores[parent] for parent in parents), scores[child_key],
                                      parameter_frontier=child_key in par_front,
                                      complexity_frontier=child_key in cmp_front)
            for parent in parents:
                self._parent_visits[parent] = self._parent_visits.get(parent, 0) + 1
                self._parent_rewards[parent] = self._parent_rewards.get(parent, 0.0) + reward / len(parents)
            if child.mutation_action:
                op = child.mutation_action
                self._operator_visits[op] = self._operator_visits.get(op, 0) + 1
                self._operator_rewards[op] = self._operator_rewards.get(op, 0.0) + reward

    def _summary(self, result: FamilyEvaluation) -> dict[str, object]:
        return {"family_id": result.family.family_id, "family": asdict(result.family), "family_hash": self._artifact_hash(result.family), "semantic_hash": self._semantic_hash(result.family), "status": result.status, "cross_context_score": result.cross_context_score, "complexity": self.domain.complexity_stats(result.family), "per_context": [{"context_id": row.context_id, "robust_score": row.robust_score} for row in result.per_context]}


def _contains_exact_subtree(node: object, target: object) -> bool:
    """Return whether an AST contains an exact structural parent subtree."""
    return contains_subtree(node, target)
