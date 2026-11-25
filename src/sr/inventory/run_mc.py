"""MC-only adapters to the established single/multi distribution SR controllers."""
from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

from runtime.artifacts import atomic_json
from sr.analysis.frontier import initialization_signature
from sr.core.inventory import InventorySearchBindings
from sr.inventory.frontier import import_frontier_seeds
from sr.inventory.initialization import run_frontier_initialization
from sr.inventory.mc import MonteCarloScreen, MonteCarloSearchContext
from sr.inventory.multi import MultiDistributionSearch
from sr.inventory.multi_evaluation import CaseEvaluationInput
from sr.inventory.run_single import _config
from sr.inventory.single import SymbolicSearch
from sr.llm import JSONChatClient, LLMConfig
from sr.multi.models import MultiObjective

def _reference_one(payload):
    case, worker_type, config_path, cache_dir, fidelity = payload
    worker = worker_type(config_path=config_path, reference_cache_dir=cache_dir)
    cost, hit, kind = worker._reference(case, fidelity)
    return case.name, cost, kind, hit


def prepared_reference(path, case, settings, fidelity, seed_namespace):
    record = json.loads(path.read_text())
    expected = {"case_signature": case.case_signature, "evaluation": settings,
                "fidelity": fidelity, "seed_namespace": seed_namespace}
    if (record["identity"] != expected or not math.isfinite(float(record["cost"]))
            or float(record["cost"]) <= 0 or record["kind"] != "best_formula_baseline_empirical_mc"):
        raise ValueError(f"Stage1 preparation identity or reference differs: {path}")
    return record


def parser(stage, root, *, prepare_per_case=False):
    location = root / ("stage1_single_distribution_sr" if stage == 1 else "stage2_multi_distribution_sr")
    result = argparse.ArgumentParser(description=f"Inventory Stage {stage}: distribution SR (MC only)")
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--search-config", type=Path, default=location / "configs/search.json")
    result.add_argument("--evaluation-config", type=Path, default=(location / "configs/evaluation.json" if root.parent.name == "experiments" else root / "configs/mc.json"))
    result.add_argument("--reference-cache-dir", type=Path, required=not prepare_per_case,
                        help="Stage1 baseline reference cache")
    result.add_argument("--workers", type=int, default=60)
    result.add_argument("--fidelity", choices=("fast", "standard", "certify"), default="standard")
    result.add_argument("--mutation-config", type=Path)
    result.add_argument("--reflection-config", type=Path)
    result.add_argument("--dry-run", action="store_true")
    if stage == 1:
        result.add_argument("--case", help="one case name; omitted runs the complete training suite")
    else:
        result.add_argument("--checkpoint-root", type=Path, required=True)
        result.add_argument("--resume", action="store_true", help="Restore the existing Stage2 checkpoint")
        result.add_argument("--phase", choices=("all", "initialize", "search"), default="all")
        result.add_argument("--validate-inputs", action="store_true",
                            help="Validate completed Stage1 protocols, import frontiers and LLM configs even in dry-run")
    return result


def manifest(output, payload):
    path = output / "MC_PROTOCOL.json"
    if path.exists() and json.loads(path.read_text()) != payload:
        raise ValueError(f"run identity changed; use a new output directory: {output}")
    atomic_json(payload, path)


def clients(args, *, output_dir=None):
    audit_dir = args.output_dir if output_dir is None else output_dir
    def client(path, kind):
        return None if path is None else JSONChatClient(
            LLMConfig.from_json(path), audit_log=audit_dir / "llm_calls.jsonl", call_kind=kind)
    return client(args.mutation_config, "mutation"), client(args.reflection_config or args.mutation_config, "reflection")


def main(stage, *, root, case_provider, worker_type, argv=None, prepare_per_case=False):
    args = parser(stage, root, prepare_per_case=prepare_per_case).parse_args(argv)
    if prepare_per_case and args.reference_cache_dir is None:
        args.reference_cache_dir = (args.output_dir if stage == 1 else args.checkpoint_root) / ".reference_cache"
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    config = _config(args.search_config, args.workers)
    settings = json.loads(args.evaluation_config.read_text())
    from inventory.common.fitness import SCORE_PROTOCOL
    from sr.family.scoring import validate_score_protocol
    validate_score_protocol(settings.get("score_protocol", SCORE_PROTOCOL))
    objective = MultiObjective()
    selected = case_provider(int(settings.get("state_action_cap", 20)))
    if stage == 1 and args.case:
        selected = tuple(case for case in selected if case.name == args.case)
        if not selected:
            raise ValueError(f"unknown training case: {args.case}")
    seed_namespace = getattr(worker_type, "seed_namespace", "ltb")
    payload = {"stage": stage, "mc_only": True, "cases": [case.name for case in selected],
               "case_signatures": [case.case_signature for case in selected],
               "config": asdict(config), "evaluation": settings, "objective": objective.to_payload(), "fidelity": args.fidelity,
               "reference": "best_formula_baseline_empirical_mc", "screen": "fixed_bank_mc",
               "gap_definition": "100*(final_mc_cost/best_baseline_mc_cost-1)",
               "legacy_field_names": {"dp_cost": "baseline_mc_reference_cost", "bellman_regret_pct": "mc_screen_gap_pct"}}
    if seed_namespace != "ltb":
        payload["seed_namespace"] = seed_namespace
    bundle = None
    if stage == 2 and (not args.dry_run or args.validate_inputs):
        for case in selected:
            checkpoint_path = args.checkpoint_root / case.name / "checkpoint.json"
            checkpoint = json.loads(checkpoint_path.read_text())
            if int(checkpoint["iteration"]) < int(checkpoint["config"]["iterations"]):
                raise ValueError(f"Stage-1 search is incomplete: {checkpoint_path}")
            if args.validate_inputs or prepare_per_case:
                protocol = json.loads(checkpoint_path.with_name("MC_PROTOCOL.json").read_text())
                if (protocol["case_signatures"] != [case.case_signature]
                        or protocol["evaluation"] != settings or protocol["fidelity"] != args.fidelity
                        or protocol.get("seed_namespace", "ltb") != seed_namespace):
                    raise ValueError(f"Stage-1 evaluation protocol differs: {checkpoint_path}")
        if prepare_per_case:
            for case in selected:
                prepared_reference(args.checkpoint_root / case.name / "PREPARATION.json",
                                   case, settings, args.fidelity, seed_namespace)
        bundle = import_frontier_seeds(
            {case.name: args.checkpoint_root / case.name / "checkpoint.json" for case in selected},
            case=selected[0], config=config,
        )
        if args.validate_inputs:
            for path in (args.mutation_config, args.reflection_config):
                if path is not None:
                    LLMConfig.from_json(path)
            payload["source_checkpoints"] = dict(bundle.source_checkpoints)
            payload["initialization_candidates"] = len(bundle.candidates)
            payload["initialization_tasks"] = len(bundle.candidates) * len(selected)
            payload["objective"] = objective.to_payload()
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return
    if stage == 1:
        existing = [case.name for case in selected
                    if (args.output_dir / case.name / "checkpoint.json").exists()]
        if existing:
            raise ValueError(f"Existing Stage1 checkpoint; use a new output directory: {existing}")
        for case in selected:
            manifest(args.output_dir / case.name, {**payload,
                "cases": [case.name], "case_signatures": [case.case_signature]})
    else:
        manifest(args.output_dir, payload)
    cache_dir = args.reference_cache_dir
    tasks = [(case, worker_type, str(args.evaluation_config), str(cache_dir), args.fidelity) for case in selected]
    contexts = {}
    if prepare_per_case:
        tasks = []
    if stage == 2 and prepare_per_case:
        for case in selected:
            path = args.checkpoint_root / case.name / "PREPARATION.json"
            record = prepared_reference(path, case, settings, args.fidelity, seed_namespace)
            contexts[case.name] = MonteCarloSearchContext(case, int(settings["horizon"]), record["cost"], settings,
                                                        args.fidelity, record["kind"], seed_namespace=seed_namespace)
    with ProcessPoolExecutor(max_workers=max(1, min(int(args.workers), len(tasks)))) as pool:
        for case_name, cost, kind, hit in pool.map(_reference_one, tasks):
            case = next(row for row in selected if row.name == case_name)
            print(json.dumps({"phase": "reference", "case": case.name,
                              "reference_cache_hit": hit}), flush=True)
            contexts[case.name] = MonteCarloSearchContext(case, int(settings["horizon"]), cost, settings,
                                                        args.fidelity, kind, seed_namespace=seed_namespace)
    if stage == 1:
        for case in selected:
            if prepare_per_case:
                name, cost, kind, hit = _reference_one((case, worker_type, str(args.evaluation_config), str(cache_dir), args.fidelity))
                atomic_json({"identity": {"case_signature": case.case_signature, "evaluation": settings,
                                           "fidelity": args.fidelity, "seed_namespace": seed_namespace},
                             "cost": cost, "kind": kind}, args.output_dir / case.name / "PREPARATION.json")
                print(json.dumps({"phase": "reference", "case": name, "reference_cache_hit": hit}), flush=True)
                contexts[name] = MonteCarloSearchContext(case, int(settings["horizon"]), cost, settings,
                                                       args.fidelity, kind, seed_namespace=seed_namespace)
            context = contexts[case.name]
            mutation, reflection = clients(args, output_dir=args.output_dir / case.name)
            bindings = InventorySearchBindings.build_context_free(case, config)
            SymbolicSearch(
                case_name=case.name, context=context, diagnostic_json=None,
                output_dir=args.output_dir / case.name, config=config, bindings=bindings,
                samples=(), screen=MonteCarloScreen(context, context.motifs),
                mutation_client=mutation, reflection_client=reflection,
            ).run()
        return
    mutation, reflection = clients(args)
    assert bundle is not None
    signature = initialization_signature(bundle, case_names=tuple(contexts), horizon=int(settings["horizon"]),
                                         objective_payload=objective.to_payload(), config_payload=payload)
    inputs = {name: CaseEvaluationInput(name, context, (), config.random_seed) for name, context in contexts.items()}
    cache = args.output_dir / "initialization" / signature
    if args.phase in {"all", "initialize"}:
        print(json.dumps({"phase": "initialization_started", "candidates": len(bundle.candidates),
                          "tasks": len(bundle.candidates) * len(inputs), "cache_dir": str(cache)}), flush=True)
        initialized = run_frontier_initialization(bundle, inputs, config=config, objective=objective,
                                                  cache_dir=cache, initialization_signature=signature)
        print(json.dumps({"phase": "initialization_complete", "retained": len(initialized.individuals),
                          "dropped": len(initialized.dropped_candidate_hashes),
                          "scheduler_counts": dict(initialized.scheduler_counts)}), flush=True)
    if args.phase in {"all", "search"}:
        print(json.dumps({"phase": "search_started", "iterations": config.iterations,
                          "workers": args.workers, "islands": config.island_count}), flush=True)
        MultiDistributionSearch(
            case_inputs=inputs, diagnostic_paths={name: None for name in contexts},
            seed_bundle=bundle, initialization_cache_dir=cache, initialization_signature=signature,
            output_dir=args.output_dir, config=config, objective=objective,
            screens={name: MonteCarloScreen(context, bundle.motifs) for name, context in contexts.items()},
            mutation_client=mutation, reflection_client=reflection,
            resume=args.resume,
        ).run()
