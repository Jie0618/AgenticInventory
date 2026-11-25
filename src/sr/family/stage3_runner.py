"""Shared CLI orchestration for Stage-III family-search experiments."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from inventory.policy.family import PolicyFamily
from inventory.policy.family_domain import InventoryFamilyDomain

from .reference import validate_reference_mode
from .search import MultiContextFamilySearch
from .scoring import validate_score_protocol
from inventory.common.fitness import SCORE_PROTOCOL


def validate_stage3_config_declarations(
    config: Mapping[str, object],
    *,
    schema_version: int,
    setting: str,
    training_contexts: Sequence[object],
    reference: str,
    optimizer: str,
) -> None:
    """Fail when descriptive Stage-III config fields drift from executable inputs."""

    expected = {
        "schema_version": int(schema_version),
        "setting": str(setting),
        "training_contexts": json.loads(json.dumps(list(training_contexts))),
        "reference": str(reference),
        "optimizer": str(optimizer),
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "Stage-III config declaration mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )


@dataclass(frozen=True)
class Stage3Setup:
    """Setting-owned objects resolved after parsing the shared CLI."""

    domain: InventoryFamilyDomain | None = None
    initial_families: tuple[PolicyFamily, ...] = ()
    case_ids: tuple[str, ...] = ()
    worker: Any | None = None
    feature_init_families: tuple[PolicyFamily, ...] | None = None
    dry_run_payload: Mapping[str, object] | None = None
    before_search: Callable[[], None] | None = None


@dataclass(frozen=True)
class Stage3RunnerSpec:
    setting: str
    description: str
    default_output_dir: Path
    default_config: Path
    default_reference_mode: str
    reference_modes: tuple[str, ...]
    prepare: Callable[[argparse.Namespace, str], Stage3Setup]
    configure_parser: Callable[[argparse.ArgumentParser], None] | None = None


def build_parser(spec: Stage3RunnerSpec) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=spec.description)
    parser.add_argument("--output-dir", type=Path, default=spec.default_output_dir)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--workers", type=int, default=60)
    parser.add_argument("--config", type=Path, default=spec.default_config)
    parser.add_argument("--mutation-config", type=Path)
    parser.add_argument("--reflection-config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--feature-init-pass", action=argparse.BooleanOptionalAction, default=False,
        help="Opt into extra deterministic summary variants (outside the paper's default protocol)",
    )
    parser.add_argument(
        "--fidelity", choices=("fast", "standard", "certify"), default="standard"
    )
    parser.add_argument(
        "--reference-mode",
        choices=spec.reference_modes,
        default=spec.default_reference_mode,
    )
    if spec.configure_parser is not None:
        spec.configure_parser(parser)
    return parser


def run_stage3(spec: Stage3RunnerSpec, argv: Sequence[str] | None = None) -> None:
    """Run one setting adapter through the common Stage-III lifecycle."""

    parser = build_parser(spec)
    args = parser.parse_args(argv)
    if args.workers <= 0 or args.iterations < 0:
        parser.error("workers must be positive and iterations must be nonnegative")
    try:
        settings = json.loads(args.config.read_text(encoding="utf-8"))
        if settings.get("llm_template_init", True) is not True:
            raise ValueError("Stage III requires per-seed LLM template initialization")
        score_protocol = validate_score_protocol(settings.get("score_protocol", SCORE_PROTOCOL))
        reference_mode = validate_reference_mode(
            args.reference_mode, setting=spec.setting
        )
        setup = spec.prepare(args, reference_mode)
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        parser.error(str(exc))

    if args.dry_run:
        if setup.dry_run_payload is None:
            parser.error("setting did not provide a dry-run payload")
        print(json.dumps({**dict(setup.dry_run_payload), "score_protocol": score_protocol,
                          "llm_template_init": True,
                          "llm_configured": bool(args.mutation_config or args.reflection_config)}, indent=2))
        return

    if setup.domain is None or setup.worker is None:
        parser.error("setting did not provide a search domain and worker")
    if not setup.initial_families or not setup.case_ids:
        parser.error("setting did not provide initial families and training cases")

    proposal_fn, reflection_fn = _llm_functions(args, setup.domain)
    if proposal_fn is None:
        parser.error("LLM template initialization requires --mutation-config or --reflection-config")
    initialization_fn = proposal_fn.initialize
    search = MultiContextFamilySearch(
        domain=setup.domain,
        initial_families=setup.initial_families,
        case_ids=setup.case_ids,
        worker=setup.worker,
        output_dir=args.output_dir,
        workers=args.workers,
        iterations=args.iterations,
        proposal_fn=proposal_fn,
        reflection_fn=reflection_fn,
        resume=args.resume,
        fidelity=args.fidelity,
        feature_init_pass=args.feature_init_pass,
        feature_init_families=setup.feature_init_families,
        reference_mode=reference_mode,
        score_protocol=score_protocol,
        initialization_fn=initialization_fn,
        scheduling=settings.get("scheduling"),
    )
    # Reject invalid resumes before adapters warm references or replace audits.
    validate = getattr(search, "validate_run_identity", None)
    if validate is not None:
        validate()
    if setup.before_search is not None:
        setup.before_search()
    result = search.run()
    print(json.dumps({
        "status": "complete",
        "families": len(result),
        "best_score": min(
            (row.cross_context_score for row in result), default=None
        ),
    }, indent=2))


def _llm_functions(args, domain: InventoryFamilyDomain):
    if not (args.mutation_config or args.reflection_config):
        return None, None

    from sr.family.proposals import llm_proposal_batch_fn, llm_reflect_motifs
    from sr.llm import JSONChatClient, LLMConfig

    mutation_path = args.mutation_config or args.reflection_config
    reflection_path = args.reflection_config or mutation_path
    mutation_client = JSONChatClient(
        LLMConfig.from_json(mutation_path),
        audit_log=args.output_dir / "mutation_audit.jsonl",
        call_kind="stage3_family_mutation",
    )
    reflection_client = JSONChatClient(
        LLMConfig.from_json(reflection_path),
        audit_log=args.output_dir / "reflection_audit.jsonl",
        call_kind="stage3_family_reflection",
    )

    def proposal_fn(
        parents, iteration, evaluations=None, reflected_motifs=None,
        known_definition_keys=(),
    ):
        return llm_proposal_batch_fn(
            mutation_client, parents, iteration, None, count=20,
            parent_evaluations=evaluations,
            reflected_motifs=reflected_motifs,
            known_definition_keys=known_definition_keys,
            domain=domain,
        )

    def reflection_fn(families, iteration, evaluations=None):
        return llm_reflect_motifs(
            reflection_client, families, iteration, domain=domain, parent_evaluations=evaluations
        )

    def initialize(parents):
        from sr.family.template_init import initialize_templates
        settings = json.loads(args.config.read_text())
        return initialize_templates(mutation_client, parents, domain=domain,
            output_dir=args.output_dir,
            count=int(settings.get("llm_template_init_count_per_seed", 6)),
            minimum=int(settings.get("llm_template_init_min_per_seed", 5)))

    from .template_init import PROTOCOL
    initialize.protocol = PROTOCOL

    proposal_fn.initialize = initialize

    return proposal_fn, reflection_fn


def run_family_worker_cli(
    worker_factory: Callable[..., Any],
    *,
    default_reference_mode: str,
    reference_modes: tuple[str, ...],
    domain_for_task: Callable[[Any, Any], InventoryFamilyDomain] | None = None,
    argv: Sequence[str] | None = None,
) -> None:
    """Shared isolated-process protocol used by family workers."""

    from sr.family.models import FamilyTask
    from sr.family.proposals import family_from_json

    parser = argparse.ArgumentParser()
    parser.add_argument("--family", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path)
    parser.add_argument(
        "--reference-mode", choices=reference_modes,
        default=default_reference_mode,
    )
    args = parser.parse_args(argv)
    task = FamilyTask(**json.loads(args.task.read_text()))
    worker = worker_factory(
        config_path=args.config,
        reference_cache_dir=args.reference_cache,
        reference_mode=args.reference_mode,
    )
    domain = (
        worker.family_domain()
        if domain_for_task is None
        else domain_for_task(worker, task)
    )
    family = family_from_json(json.loads(args.family.read_text()), domain=domain)
    metric = worker(family, task)
    payload = json.dumps({
        "context_id": metric.context_id,
        "case_id": metric.case_id,
        "gap_pct": metric.gap_pct,
        "runtime_seconds": metric.runtime_seconds,
        "status": metric.status,
        "metadata": dict(metric.metadata),
    }, sort_keys=True) + "\n"
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(args.output)


__all__ = [
    "Stage3RunnerSpec",
    "Stage3Setup",
    "build_parser",
    "run_family_worker_cli",
    "run_stage3",
    "validate_stage3_config_declarations",
]
