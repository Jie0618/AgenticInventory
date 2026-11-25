#!/usr/bin/env python3
"""Fit one case-policy task, or assemble its complete paired-MC comparison."""

from __future__ import annotations

from runtime.artifacts import stable_key
import argparse
import json
import math
import os
import signal
import sys
import time
from importlib import import_module
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from runtime.artifacts import atomic_json

from benchmarks.inventory_benchmark_family.mc_protocol import BASELINE_TYPES, load_model, load_policy_set
from benchmarks.inventory_benchmark_family.mc_runtime import MCDemandBank
from optimization.models.types import (
    MonteCarloProposalConfig,
    OptimizationConfig,
)
from optimization.solvers.policy_cost import optimize_policy_constants


class PolicyTimeout(TimeoutError):
    pass


@contextmanager
def _policy_deadline(seconds):
    if seconds is None:
        yield
        return
    def expired(signum, frame):
        raise PolicyTimeout("computational baseline wall-time limit exceeded")
    previous = signal.signal(signal.SIGALRM, expired)
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def main() -> None:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--setting", default="lead_time")
    setting, _ = bootstrap.parse_known_args()
    if setting.setting in ("lead_time_backlog", "lead_time_perishable"):
        parser = argparse.ArgumentParser(description="Evaluate one joint-setting case-policy request")
        parser.add_argument("--setting", choices=("lead_time_backlog", "lead_time_perishable"), required=True)
        parser.add_argument("--model-module", required=True)
        parser.add_argument("--request", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        args = parser.parse_args()
        model = import_module(args.model_module)
        if model.SETTING != args.setting:
            raise ValueError("MC model does not match the requested setting")
        request = json.loads(args.request.read_text())
        settings = request["settings"]
        if sys.platform.startswith("linux") and settings["task_memory_gb"]:
            import resource
            limit = int(settings["task_memory_gb"] * 1024**3)
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        case = (next(model.cases(settings, only_name=request["case"]))
                if args.setting == "lead_time_backlog" else
                next(case for case in model.cases(settings) if case.name == request["case"]))
        atomic_json(model.evaluate(case, request["policy"], settings), args.output)
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", choices=("lead_time", "perishable"), default="lead_time")
    parser.add_argument("--model-module", required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--policy-set", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--computational-first", action="store_true")
    parser.add_argument("--policy-id")
    args = parser.parse_args()

    started = time.perf_counter()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    model = load_model(args.setting, model_module=args.model_module)
    case = next(row for row in model.cases(config) if row.name == args.instance)
    policy_set_name, definitions = load_policy_set(args.policy_set)
    if not any(row.type in BASELINE_TYPES for row in definitions):
        raise ValueError("policy set must include at least one registered or computational baseline")

    if args.policy_id is not None:
        definitions = tuple(row for row in definitions if row.id == args.policy_id)
        if not definitions:
            raise ValueError("Unknown policy ID")

    horizon = int(config["horizon"])
    tuning_seed = model.seed("mc_tuning", args.instance)
    evaluation_seed = model.seed("mc_evaluation", args.instance)
    numerics = model.make_numerics(case, config)
    tuning_bank = MCDemandBank.create(case.demand, horizon, int(config["tuning_paths"]), tuning_seed, randomized_policies=model.randomized_policies)
    proposal_paths = int(config["proposal_paths"])
    proposal_horizon = int(config.get("proposal_horizon", horizon))
    if not 0 < proposal_paths <= tuning_bank.num_paths:
        raise ValueError("proposal_paths must lie in [1, tuning_paths]")
    if not 0 < proposal_horizon <= horizon:
        raise ValueError("proposal_horizon must lie in [1, horizon]")
    proposal_bank = tuning_bank.prefix(proposal_paths, proposal_horizon)
    evaluation_bank = MCDemandBank.create(case.demand, horizon, int(config["evaluation_paths"]), evaluation_seed, randomized_policies=model.randomized_policies)
    strict_paths = int(config.get("strict_paths", config["tuning_paths"]))
    if not 0 < strict_paths <= tuning_bank.num_paths:
        raise ValueError("strict_paths must lie in [1, tuning_paths]")
    strict_bank = tuning_bank.prefix(strict_paths, horizon)

    policy_results: list[dict[str, object]] = []
    path_costs_by_policy: dict[str, np.ndarray] = {}
    execution_definitions = sorted(definitions, key=lambda row: row.type != "computational_baseline") if args.computational_first else definitions
    for definition in execution_definitions:
        policy_started = time.perf_counter()
        policy = model.resolve_policy(definition, getattr(case, model.context_field), case.n, case=case)
        computational_metadata = numerics.policy_metadata(policy)
        proposal_budget = config.get("proposal_budget")
        optimization_config = OptimizationConfig(
            mode="certify",
            objective="policy_cost",
            random_seed=model.seed("fit", case.name, policy.id),
            total_evaluations=(
                None if proposal_budget is None else int(proposal_budget)
            ),
            mc_proposal=MonteCarloProposalConfig(enabled=False, routing_mode="never"),
        )
        if model.fixed_computational and policy.type == "computational_baseline":
            if policy.parameter_specs:
                raise ValueError("fixed computational policies cannot have fitted parameters")
            best_params, convergence = {}, {}
            route, tuning_cost, count, completed, optimizer_seconds = "fixed", None, 0, 0, 0.0
        elif policy.execution:
            from benchmarks.baselines.cyclic_co import fit_rate_grid, schedule_for
            if proposal_horizon != horizon:
                raise ValueError("cyclic rate screening requires the full tuning horizon")
            best_params, tuning_cost, cyclic_fit = fit_rate_grid(
                schedule_for(policy), numerics.evaluator(policy, strict_bank),
                numerics.evaluator(policy, proposal_bank),
                top_k=int(config.get("cyclic_top_k", 64)),
            )
            convergence = {"proposal_count": cyclic_fit["screen_count"],
                           "exact_evaluations": cyclic_fit["strict_count"],
                           "cyclic_rate_search": cyclic_fit}
            route = cyclic_fit["route"]
            count, completed = cyclic_fit["candidate_count"], cyclic_fit["strict_count"]
            optimizer_seconds = cyclic_fit["optimizer_seconds"]
        else:
            low_evaluator = numerics.evaluator(policy, proposal_bank)
            high_evaluator = numerics.evaluator(policy, strict_bank)
            evaluator = numerics.fitting_evaluator(high_evaluator, low_evaluator, float(horizon) / proposal_horizon)
            optimized = optimize_policy_constants(policy.parameter_specs, evaluator, config=optimization_config)
            best_params = dict(optimized.best_params)
            convergence = dict(optimized.convergence_report)
            route, tuning_cost = optimized.route, float(optimized.best_cost)
            count, completed, optimizer_seconds = optimized.unique_parameter_points, optimized.completed_evaluations, optimized.elapsed_seconds
        timeout = config.get("computational_policy_timeout_seconds")
        limited = (model.setting == "perishable" and case.shelf_life >= 10
                   and policy.id in ("pb_chao", "db_chao", "b_zaw", "tb_zaw"))
        skip_tle = bool(config.get("skip_perishable_computational_baselines", False)) and limited
        try:
            if skip_tle:
                raise PolicyTimeout("user requested TLE without running")
            with _policy_deadline(float(timeout) if limited and timeout is not None else None):
                final_evaluator = numerics.evaluator(policy, evaluation_bank)
                evaluation_paths = numerics.path_costs(final_evaluator, best_params, evaluation_bank)
        except PolicyTimeout:
            elapsed = time.perf_counter() - policy_started
            policy_results.append({
                "policy_id": policy.id, "policy_type": policy.type,
                "formula": policy.formula, "constants": best_params,
                "status": "TLE" if skip_tle else "skipped_timeout", "route": route,
                "skip_reason": "user_requested_long_tail_skip" if skip_tle else "timeout",
                "evaluation_cost": None, "evaluation_se": None,
                "tuning_cost": tuning_cost, "tuning_se": None,
                "candidate_count": int(count), "optimizer_seconds": optimizer_seconds,
                "policy_wall_seconds": elapsed, "timeout_seconds": float(timeout) if timeout is not None else None,
                "computational_baseline": computational_metadata,
            })
            print(json.dumps({"event": "mc_policy_skipped", "instance_id": case.name,
                              "policy_id": policy.id, "reason": "user_requested_tle" if skip_tle else "timeout",
                              "timeout_seconds": timeout, "policy_wall_seconds": elapsed}), flush=True)
            continue
        evaluation_cost = float(np.mean(evaluation_paths))
        evaluation_se = _standard_error(evaluation_paths)
        path_costs_by_policy[policy.id] = np.asarray(evaluation_paths, dtype=float)
        policy_results.append(
            {
                "policy_id": policy.id,
                "policy_type": policy.type,
                "formula": policy.formula,
                "constants": best_params,
                "status": "empirical_mc_best_found",
                "route": route,
                "tuning_cost": tuning_cost,
                "tuning_se": None,
                "evaluation_cost": evaluation_cost,
                "evaluation_se": evaluation_se,
                "candidate_count": int(count),
                "screen_evaluation_count": int(convergence.get("proposal_count", 0)),
                "tuning_evaluation_count": int(completed),
                "strict_tuning_evaluation_count": int(
                    convergence.get("exact_evaluations", 0)
                ),
                "proposal_evaluator_call_count": int(
                    convergence.get("fast_evaluator_calls", 0)
                ),
                "search_stop_reason": convergence.get("search_stop_reason"),
                "certified_global": bool(convergence.get("certified_global", False)),
                "optimizer_seconds": optimizer_seconds,
                "policy_wall_seconds": time.perf_counter() - policy_started,
                "computational_baseline": computational_metadata,
            }
        )
        print(
            json.dumps(
                {
                    "event": "mc_policy_complete",
                    "instance_id": case.name,
                    "policy_id": policy.id,
                    "route": route,
                    "candidate_count": count,
                    "evaluation_cost": evaluation_cost,
                    "policy_wall_seconds": policy_results[-1]["policy_wall_seconds"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if policy.execution:
            policy_results[-1]["execution"] = policy.execution
            policy_results[-1]["cyclic_rate_search"] = cyclic_fit

    # Preserve canonical output order and resume compatibility after scheduling.
    policy_order = {definition.id: i for i, definition in enumerate(definitions)}
    policy_results.sort(key=lambda row: policy_order[row["policy_id"]])
    payload = {
        "schema_version": 2,
        "suite": model.suite,
        "reference_policy": "best_all_policies_per_instance" if config.get("reference_scope") == "all_policies" else "best_baseline_per_instance",
        "reference_selection_bank": "evaluation",
        "policy_set": policy_set_name,
        "instance_id": case.name,
        model.context_field: getattr(case, model.context_field),
        "distribution_id": case.demand.name,
        "distribution_family": case.kind,
        "h": case.h,
        "b": case.b,
        "b_over_h": case.b / case.h,
        "n": case.n,
        "horizon": horizon,
        "tuning_paths": tuning_bank.num_paths,
        "evaluation_paths": evaluation_bank.num_paths,
        "tuning_seed": tuning_seed,
        "evaluation_seed": evaluation_seed,

        "policies": policy_results,
        "total_task_seconds": time.perf_counter() - started,
        "pid": os.getpid(),
    }
    payload.update({name: getattr(case, name) for name in model.extra_case_fields})
    if config.get("computational_policy_timeout_seconds") is not None:
        payload["computational_policy_timeout_seconds"] = config["computational_policy_timeout_seconds"]
    if config.get("skip_perishable_computational_baselines", False):
        payload["skip_perishable_computational_baselines"] = True
    if args.policy_id is not None:
        payload["path_costs"] = {key: values.tolist() for key, values in path_costs_by_policy.items()}
        payload["task_kind"] = "case_policy"
        atomic_json(payload, args.output)
    else:
        _complete_case(payload, path_costs_by_policy, config, args.output)


def _complete_case(payload, path_costs_by_policy, config, output):
    policy_results = payload["policies"]
    best_baseline = _best_baseline_row(policy_results, include_sr=config.get("reference_scope") == "all_policies")
    reference_policy_id = str(best_baseline["policy_id"])
    reference_paths = path_costs_by_policy[reference_policy_id]
    reference_cost = float(best_baseline["evaluation_cost"])
    if not np.isfinite(reference_cost) or reference_cost <= 0.0:
        raise RuntimeError("best-baseline reference cost must be finite and positive")
    for row in policy_results:
        if row["status"] in ("skipped_timeout", "TLE"):
            row.update(reference_policy_id=reference_policy_id, reference_cost=reference_cost,
                       relative_gap_pct=None, paired_cost_difference=None,
                       paired_cost_difference_se=None, relative_gap_se_pct=None)
            continue
        policy_paths = path_costs_by_policy[str(row["policy_id"])]
        policy_cost = float(row["evaluation_cost"])
        row["reference_policy_id"] = reference_policy_id
        row["reference_cost"] = reference_cost
        row["relative_gap_pct"] = 100.0 * (policy_cost / reference_cost - 1.0)
        differences = policy_paths - reference_paths
        row["paired_cost_difference"] = float(np.mean(differences))
        row["paired_cost_difference_se"] = _standard_error(differences)
        row["relative_gap_se_pct"] = _ratio_standard_error_pct(
            policy_paths, reference_paths
        )

    payload.update(reference_policy_id=reference_policy_id, reference_cost=reference_cost)
    atomic_json(payload, output)


def part_path(directory, instance, policy_id):
    key = stable_key(f"{instance}|{policy_id}".encode())
    return directory / f"{key}.json"


def merge_case(parts_dir, instance, policy_set, config, output, method_ids):
    """Join independently fitted policies without changing paired statistics."""
    _, definitions = load_policy_set(policy_set)
    definitions = tuple(d for d in definitions if d.id in method_ids)
    payload, costs, rows = None, {}, []
    identity = None
    for definition in definitions:
        row = json.loads(part_path(parts_dir, instance, definition.id).read_text())
        shared = {k: v for k, v in row.items()
                  if k not in ("policies", "path_costs", "pid", "total_task_seconds", "task_kind")}
        if row.get("task_kind") != "case_policy" or row.get("instance_id") != instance:
            raise ValueError("Invalid case-policy shard")
        if identity is not None and shared != identity:
            raise ValueError("Case-policy demand banks or numerical settings differ")
        identity = shared
        if payload is None:
            payload = dict(shared, total_task_seconds=0.0, pid=os.getpid())
        if [p["policy_id"] for p in row["policies"]] != [definition.id]:
            raise ValueError("Case-policy membership mismatch")
        for key, values in row["path_costs"].items():
            if key != definition.id or len(values) != config["evaluation_paths"]:
                raise ValueError("Case-policy path costs mismatch")
            values = np.asarray(values, dtype=float)
            if not np.all(np.isfinite(values)):
                raise ValueError("Nonfinite path costs")
            costs[key] = values
        rows.extend(row["policies"])
        payload["total_task_seconds"] += row["total_task_seconds"]
    payload["policies"] = rows
    _complete_case(payload, costs, config, output)


def _best_baseline_row(rows: list[dict[str, object]], *, include_sr: bool = False) -> dict[str, object]:
    baselines = [row for row in rows if (include_sr or row.get("policy_type") in BASELINE_TYPES) and row.get("evaluation_cost") is not None and (not include_sr or row.get("status") == "empirical_mc_best_found") and math.isfinite(float(row["evaluation_cost"]))]
    if not baselines:
        raise ValueError("at least one registered or computational baseline is required")
    return min(
        baselines,
        key=lambda row: (float(row["evaluation_cost"]), str(row["policy_id"])),
    )




def _standard_error(values: np.ndarray) -> float:
    rows = np.asarray(values, dtype=float)
    return float(np.std(rows, ddof=1) / math.sqrt(len(rows))) if len(rows) > 1 else 0.0


def _ratio_standard_error_pct(numerator: np.ndarray, denominator: np.ndarray) -> float:
    x = np.asarray(numerator, dtype=float)
    y = np.asarray(denominator, dtype=float)
    mean_x = float(np.mean(x))
    mean_y = float(np.mean(y))
    influence = (x - mean_x) / mean_y - mean_x * (y - mean_y) / (mean_y**2)
    return 100.0 * _standard_error(influence)


if __name__ == "__main__":
    main()
