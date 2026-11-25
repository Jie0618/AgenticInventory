#!/usr/bin/env python3
"""Lead-time Stage-III family-search adapter."""

from __future__ import annotations

from sr.family.artifacts import write_path
import argparse
import json
from pathlib import Path
from typing import Any, Callable

from experiments.lead_time.stage3_multi_leadtime_sr.runtime.family_worker import (
    CanonicalFamilyWorker,
)
from sr.family.proposals import (
    FEATURE_INIT_PROTOCOL,
    FEATURE_INIT_VARIANTS_PER_FAMILY,
)
from sr.family.stage3_runner import Stage3RunnerSpec, Stage3Setup, run_stage3

from experiments.lead_time.paths import EXPERIMENT_ROOT, STAGE3_DATA


from experiments.lead_time.stage3_multi_leadtime_sr.cases import CASE_SUFFIXES


def _configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(iterations=100)
    parser.add_argument("--stage2-checkpoint", type=Path, required=True, help="Import completed Stage2 frontier as the actual Stage3 seeds")
    parser.add_argument(
        "--lead-times", default="3,5,8,10",
        help="Comma-separated active L values to optimize (default: 3,5,8,10)",
    )



def _lead_times(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(dict.fromkeys(
            int(value.strip()) for value in str(raw).split(",") if value.strip()
        ))
    except ValueError as exc:
        raise ValueError(f"--lead-times must be comma-separated integers: {exc}") from exc
    if not values or any(value < 2 or value > 10 for value in values):
        raise ValueError("--lead-times must be a nonempty subset of 2,...,10")
    return values


def make_spec(
    *,
    worker_factory: Callable[..., Any] = CanonicalFamilyWorker,
    default_output_dir: Path = STAGE3_DATA / "results" / "stage3_ml_md_run",
    default_config: Path = EXPERIMENT_ROOT / "stage3_multi_leadtime_sr/configs/stage3.json",
) -> Stage3RunnerSpec:
    """Build the Stage2-to-Stage3 transfer adapter."""

    def prepare(args: argparse.Namespace, reference_mode: str) -> Stage3Setup:
        lead_times = _lead_times(args.lead_times)
        worker = worker_factory(
            config_path=args.config,
            reference_cache_dir=args.output_dir / "reference_cache",
            reference_mode=reference_mode,
        )
        domain = worker.family_domain(
            lead_times, training_case_count=len(CASE_SUFFIXES)
        )
        from sr.family.transfer import import_stage2
        settings = json.loads(args.config.read_text())
        source_case = next(
            (row.case for row in domain.contexts if row.case.lead_time == 3), None
        )
        if source_case is None:
            source_case = worker.family_domain(
                (3,), training_case_count=len(CASE_SUFFIXES)
            ).contexts[0].case
        seeds, audit = import_stage2(
            args.stage2_checkpoint, domain,
            source_case=source_case,
            selection=settings.get("stage2_seed_selection"),
        )
        def write_transfer_audit():
            args.output_dir.mkdir(parents=True, exist_ok=True)
            write_path(args.output_dir, "STAGE2_TRANSFER_AUDIT.json").write_text(json.dumps({
                "source": str(args.stage2_checkpoint),
                "seeds": [s.family_id for s in seeds], "rows": audit
            }, indent=2) + "\n")
        if args.dry_run:
            return Stage3Setup(dry_run_payload={
                "stage": "ML-MD", "iterations": args.iterations,
                "workers": args.workers, "lead_times": list(lead_times),
                "tasks_per_family": len(lead_times) * len(CASE_SUFFIXES),
                "initial_families": [family.family_id for family in seeds],
                "stage2_checkpoint": str(args.stage2_checkpoint),
                "seed_import_audit": audit,
                "feature_init_protocol": FEATURE_INIT_PROTOCOL,
                "feature_init_max_variants_per_seed": FEATURE_INIT_VARIANTS_PER_FAMILY,
                "reference_mode": reference_mode
            })
        return Stage3Setup(
            domain=domain,
            initial_families=seeds,
            case_ids=CASE_SUFFIXES,
            worker=worker,
            before_search=write_transfer_audit,
            feature_init_families=(),
        )

    return Stage3RunnerSpec(
        setting="lead_time",
        description="Stage III ML-MD family search",
        default_output_dir=default_output_dir,
        default_config=default_config,
        default_reference_mode="fp3",
        reference_modes=("fp3", "best_baseline"),
        prepare=prepare,
        configure_parser=_configure_parser,
    )


def main() -> None:
    run_stage3(make_spec())


if __name__ == "__main__":
    main()
