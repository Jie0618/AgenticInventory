#!/usr/bin/env python3
"""Lead-time + perishable Stage-III family-search adapter."""

from __future__ import annotations

from sr.family.artifacts import write_path
import argparse
import json
from pathlib import Path

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

from experiments.lead_time_perishable.stage3_multi_context_sr.cases import CASE_SUFFIXES, CONTEXTS, cases
from experiments.lead_time_perishable.stage3_multi_context_sr.runtime.family_worker import FamilyWorker


from sr.family.transfer import import_stage2

STAGE_DIR = Path(__file__).resolve().parents[1]


def _configure_parser(parser):
    parser.add_argument("--stage2-checkpoint", type=Path, required=True, help="Import portable structures from the completed Stage2 frontier")
    parser.set_defaults(iterations=100)



def _prepare(args: argparse.Namespace, reference_mode: str) -> Stage3Setup:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_stage3_config_declarations(
        config,
        schema_version=2,
        setting="lead_time_perishable",
        training_contexts=CONTEXTS,
        reference="best_formula_baseline_empirical_mc_shared_final_bank",
        optimizer="two_fidelity_composed_mc",
    )
    training_cases = cases(int(config.get("state_action_cap", 20)))
    representatives = {
        f"L{lead}_M{shelf}": next(
            row for row in training_cases
            if row.name.startswith(f"L{lead}_M{shelf}__")
        )
        for lead, shelf in CONTEXTS
    }
    domain = InventoryFamilyDomain.from_cases(
        representatives, training_case_count=len(CASE_SUFFIXES),
        cyclic_cap=bool(config.get("cyclic_cap_enabled", False))
    )
    source_case = next(row.case for row in domain.contexts if row.case.lead_time == 3 and row.case.shelf_life == 3)
    seeds, import_audit = import_stage2(
        args.stage2_checkpoint, domain, source_case=source_case,
        selection=config.get("stage2_seed_selection"))
    for family in seeds:
        domain.validate_family(family)
    payload = {
        "seed_import_audit": import_audit,
        "setting": "lead_time_perishable",
        "contexts": list(domain.context_ids),
        "schemas": {row.context_id: list(row.case.schema.names) for row in domain.contexts},
        "case_suffixes": list(CASE_SUFFIXES), "cases": len(training_cases),
        "tasks_per_family": len(training_cases),
        "initial_families": [family.family_id for family in seeds],
        "initial_seed_count": len(seeds),
        "feature_init_seed_count": 0,
        "feature_init_variant_count": 0,
        "initial_family_count": len(seeds),
        "initial_task_count": len(seeds) * len(training_cases),
        "feature_init_protocol": FEATURE_INIT_PROTOCOL,
        "feature_init_max_variants_per_seed": FEATURE_INIT_VARIANTS_PER_FAMILY,
        "workers": args.workers, "fidelity": args.fidelity,
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
    def write_transfer_audit():
        from runtime.artifacts import atomic_json
        atomic_json({"source": str(args.stage2_checkpoint),
                     "rows": import_audit, "seeds": [s.family_id for s in seeds]},
                    write_path(args.output_dir, "STAGE2_TRANSFER_AUDIT.json"))
    return Stage3Setup(
        domain=domain, initial_families=seeds, case_ids=CASE_SUFFIXES,
        worker=worker,
        feature_init_families=(),
        before_search=write_transfer_audit,
    )


SPEC = Stage3RunnerSpec(
    setting="lead_time_perishable", description=__doc__ or "Lead-time + perishable Stage III",
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
