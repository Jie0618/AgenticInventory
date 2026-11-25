#!/usr/bin/env python3
"""Lead-time + backlog Stage-III family-search adapter."""

from __future__ import annotations

from sr.family.artifacts import write_path
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from runtime.artifacts import atomic_json

from inventory.policy.family_domain import InventoryFamilyDomain
from sr.family.proposals import (
    FEATURE_INIT_PROTOCOL,
    FEATURE_INIT_VARIANTS_PER_FAMILY,
)
from sr.family.stage3_runner import (
    Stage3RunnerSpec,
    Stage3Setup,
    run_stage3,
    validate_stage3_config_declarations,
)

from experiments.lead_time_backlog.stage3_multi_context_sr.cases import CASE_SUFFIXES, CONTEXTS, cases
from experiments.lead_time_backlog.stage3_multi_context_sr.family import (
    families_from_stage2_checkpoint,
)
from experiments.lead_time_backlog.stage3_multi_context_sr.runtime.family_worker import FamilyWorker


STAGE_DIR = Path(__file__).resolve().parents[1]


def _prepare_reference_task(task):
    config, cache, context, case_id, fidelity = task
    worker = FamilyWorker(config_path=config, reference_cache_dir=cache)
    case = worker.case_map[f"{context}__{case_id}"]
    payload, hit = worker.reference_details(case, fidelity)
    return {"case": case.name, "cost": payload["cost"], "policy": payload["policy"],
            "cache_hit": hit}


def _configure_parser(parser):
    parser.set_defaults(iterations=100)
    parser.add_argument("--stage2-checkpoint", "--seed-artifacts", dest="seed_artifacts", type=Path, required=True,
                        help="Stage-2 multi-distribution checkpoint.json")


def _prepare(args: argparse.Namespace, reference_mode: str) -> Stage3Setup:
    if args.workers <= 0 or args.iterations < 0:
        raise ValueError("workers must be positive and iterations must be nonnegative")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_stage3_config_declarations(
        config,
        schema_version=2,
        setting="lead_time_backlog",
        training_contexts=CONTEXTS,
        reference="best_formula_baseline_empirical_mc_shared_final_bank",
        optimizer="two_fidelity_composed_mc",
    )
    training_cases = cases(int(config.get("state_action_cap", 20)))
    representatives = {
        f"L{lead}_K{patience}": next(
            row for row in training_cases
            if row.name.startswith(f"L{lead}_K{patience}__")
        )
        for lead, patience in CONTEXTS
    }
    domain = InventoryFamilyDomain.from_cases(
        representatives, training_case_count=len(CASE_SUFFIXES),
        cyclic_cap=bool(config.get("cyclic_cap_enabled", False))
    )
    seed_audit = []
    seeds = tuple(families_from_stage2_checkpoint(
        args.seed_artifacts, domain, audit=seed_audit,
        selection=config.get("stage2_seed_selection")))
    for family in seeds:
        domain.validate_family(family)
    initialized_count = len({domain.semantic_hash(family) for family in seeds})
    payload = {
        "setting": "lead_time_backlog",
        "contexts": list(domain.context_ids),
        "schemas": {row.context_id: list(row.case.schema.names) for row in domain.contexts},
        "case_suffixes": list(CASE_SUFFIXES), "cases": len(training_cases),
        "tasks_per_family": len(training_cases),
        "initial_families": [family.family_id for family in seeds],
        "initial_seed_count": len(seeds),
        "feature_init_seed_count": 0,
        "feature_init_variant_count": 0,
        "seed_import_audit": seed_audit,
        "iteration0_families": initialized_count,
        "iteration0_tasks": initialized_count * len(training_cases),
        "feature_init_protocol": FEATURE_INIT_PROTOCOL,
        "feature_init_max_variants_per_seed": FEATURE_INIT_VARIANTS_PER_FAMILY,
        "workers": args.workers, "fidelity": args.fidelity,
        "iterations": args.iterations,
        "reference_mode": reference_mode,
        "cyclic_cap_enabled": bool(config.get("cyclic_cap_enabled", False)),
        "mc_only": True, "exact_dp": False
    }
    if args.dry_run:
        return Stage3Setup(dry_run_payload=payload)
    worker = FamilyWorker(
        config_path=args.config,
        reference_cache_dir=args.output_dir / "reference_cache",
        reference_mode=reference_mode,
    )
    def prepare_references():
        # Called only after LLM configuration and resume identity validation.
        atomic_json(payload, write_path(args.output_dir, "SEED_SETUP.json"))
        # Warm each reference once, avoiding duplicate family lock waiters.
        tasks = [(args.config, worker.reference_cache_dir, context, case_id, args.fidelity)
                 for context in domain.context_ids for case_id in CASE_SUFFIXES]
        print(json.dumps({"phase": "references_started", "tasks": len(tasks),
                          "workers": min(args.workers, len(tasks))}), flush=True)
        references = []
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
            for future in as_completed([pool.submit(_prepare_reference_task, task) for task in tasks]):
                row = future.result()
                references.append(row)
                print(json.dumps({"phase": "reference_completed", "completed": len(references),
                                  "total": len(tasks), **row}), flush=True)
        atomic_json(sorted(references, key=lambda row: row["case"]), args.output_dir / "REFERENCES.json")
    return Stage3Setup(
        domain=domain, initial_families=seeds, case_ids=CASE_SUFFIXES,
        worker=worker,
        feature_init_families=(),
        before_search=prepare_references,
    )


SPEC = Stage3RunnerSpec(
    setting="lead_time_backlog", description=__doc__ or "Lead-time + backlog Stage III",
    default_output_dir=STAGE_DIR / "results/stage3_ml_md_run",
    default_config=STAGE_DIR / "configs/stage3.json",
    default_reference_mode="best_baseline", reference_modes=("best_baseline",),
    prepare=_prepare,
    configure_parser=_configure_parser,
)


def main() -> None:
    run_stage3(SPEC)


if __name__ == "__main__":
    main()
