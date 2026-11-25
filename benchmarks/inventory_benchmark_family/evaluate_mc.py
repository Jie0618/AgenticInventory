#!/usr/bin/env python3
"""Single-machine MC campaign coordination with one task per case and policy."""

from __future__ import annotations

from runtime.artifacts import stable_key
import argparse
import json
import os
import subprocess
import sys
import time
import tempfile
from importlib import import_module
from dataclasses import asdict
from pathlib import Path

from benchmarks.inventory_benchmark_family.mc_protocol import BASELINE_TYPES, load_model, load_policy_set, _policy_payload
from runtime.artifacts import atomic_json
from runtime.task_queue import QueuedTask, RunningTask, run_task_queue
from inventory.policy.ast import ast_hash


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BENCHMARK = HERE
SUPPORTED_CONTEXTS = tuple(range(2, 129))


def run_isolated_tasks(pending, *, workers, settings, scratch,
                       setting, model_module, validate_cached, rows, output):
    """Run joint-setting numerical adapters on the shared local process queue."""
    failures = {}
    tasks = [(case.name, policy["id"], case, policy, path, model_module)
             for case, policy, path in pending]
    with tempfile.TemporaryDirectory(prefix="queue_", dir=scratch) as temporary:
        directory = Path(temporary)
        def launcher(queued):
            name, identifier, case, policy, path, _ = queued.task
            task_dir = directory / stable_key(f"{name}|{identifier}".encode())
            task_dir.mkdir(exist_ok=True)
            request, result = task_dir / "request.json", task_dir / "result.json"
            result.unlink(missing_ok=True)
            atomic_json(dict(case=name, policy=policy, settings=settings), request)
            stream = (task_dir / "worker.log").open("w")
            try:
                process = subprocess.Popen(
                    [sys.executable, "-m", "benchmarks.inventory_benchmark_family.run_one_mc",
                     "--setting", setting, "--model-module", model_module,
                     "--request", str(request), "--output", str(result)],
                    cwd=ROOT, env=_environment(), stdout=stream, stderr=subprocess.STDOUT)
            except BaseException:
                stream.close()
                raise
            return RunningTask(queued, process, stream, result,
                               deadline=time.monotonic() + settings["task_timeout_seconds"])

        def validator(running):
            name, identifier, case, policy, path, _ = running.queued.task
            try:
                if running.process.returncode != 0:
                    raise RuntimeError(f"Worker exit {running.process.returncode}")
                row = json.loads(running.destination.read_text())
                validate_cached(row, case, policy, settings)
                atomic_json(row, path)
                return True
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                log = running.destination.parent / "worker.log"
                detail = log.read_text()[-4000:] if log.exists() else ""
                failures[f"{name}/{identifier}"] = f"{type(exc).__name__}: {exc}\n{detail}"
                return False

        summary = run_task_queue(
            tasks, workers=workers, launcher=launcher, result_validator=validator,
            max_retries=int(settings.get("max_retries", 2)),
            retry_cooldown_seconds=float(settings.get("retry_cooldown_seconds", 10)),
            memory_safety_reserve_gb=float(settings.get("memory_safety_reserve_gb", 0)),
            estimated_task_memory_gb=max(float(settings.get("task_memory_gb", 1)), 0.01),
            emit=lambda value: print(value, flush=True))
    failed_keys = {(a, b) for a, b, _ in summary.failures}
    for name, policy, code in summary.failures:
        failures.setdefault(f"{name}/{policy}", f"Worker exit {code}")
    for case, policy, path in pending:
        if (case.name, policy["id"]) not in failed_keys:
            rows.append(json.loads(path.read_text()))
            failures.pop(f'{case.name}/{policy["id"]}', None)
    atomic_json(asdict(summary), output / "QUEUE_SUMMARY.json")
    atomic_json(dict(failures=failures), output / "FAILURES.json")
    if summary.failures:
        raise RuntimeError(f"{len(summary.failures)} tasks failed; see FAILURES.json and rerun to resume")
    return summary


def main(setting="lead_time", *, model_module=None, argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", choices=("lead_time", "perishable"), default=setting)
    parser.add_argument("--campaign-dir", type=Path)
    parser.add_argument("--model-module", default=model_module, required=model_module is None,
                        help="Importable numerical adapter exposing MODEL")
    parser.add_argument("--policy-set", "--policies", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--lead-times", nargs="+", type=int, choices=SUPPORTED_CONTEXTS)
    parser.add_argument("--shelf-lives", nargs="+", type=int, choices=SUPPORTED_CONTEXTS)
    parser.add_argument("--instances", nargs="+", default=None)
    parser.add_argument("--limit-instances", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--computational-first", action="store_true")
    parser.add_argument("--method-ids", nargs="+")
    parser.add_argument("--contexts", nargs="+", type=int, choices=SUPPORTED_CONTEXTS)
    args = parser.parse_args(argv)

    model = load_model(args.setting, model_module=args.model_module)
    campaign = args.campaign_dir or model.data_dir
    args.config = args.config or campaign / "configs" / model.default_config
    args.policy_set = args.policy_set or campaign / "policies" / "sets" / model.default_policies
    args.run_dir = args.run_dir or campaign / "runs" / "mc_final"
    config = _load_config(args.config)
    if args.workers is not None:
        config["workers"] = int(args.workers)
    workers = int(config["workers"])
    if workers <= 0:
        raise ValueError("workers must be positive")
    policy_set_name, definitions = load_policy_set(args.policy_set)
    if args.method_ids:
        requested = set(args.method_ids)
        if len(requested) != len(args.method_ids) or not requested <= {d.id for d in definitions}:
            raise ValueError("Unknown or duplicate method IDs")
        definitions = tuple(d for d in definitions if d.id in requested)
    policy_ids = [row.id for row in definitions]
    if (config.get("reference_scope") != "all_policies"
            and not any(row.type in BASELINE_TYPES for row in definitions)):
        raise ValueError(
            "MC benchmark policy set must include at least one registered or computational baseline"
        )

    state_caps = {int(key): int(value) for key, value in config["state_caps"].items()}
    if args.lead_times and model.context_field != "lead_time":
        raise ValueError("--lead-times belongs to the lead_time setting")
    if args.shelf_lives and model.context_field != "shelf_life":
        raise ValueError("--shelf-lives belongs to the perishable setting")
    if args.contexts and (args.lead_times or args.shelf_lives):
        raise ValueError("Use either --contexts or the setting-specific context option")
    all_cases = model.cases(config, args.contexts or args.lead_times or args.shelf_lives)
    requested_instances = None if args.instances is None else set(args.instances)
    selected_cases = [
        row for row in all_cases if requested_instances is None or row.name in requested_instances
    ]
    selected_names = {row.name for row in selected_cases}
    if requested_instances is not None and selected_names != requested_instances:
        raise ValueError(
            f"unknown instances: {sorted(requested_instances - selected_names)}"
        )
    if args.limit_instances is not None:
        if args.limit_instances <= 0:
            raise ValueError("limit-instances must be positive")
        selected_cases = selected_cases[: args.limit_instances]
    if not selected_cases:
        raise ValueError("at least one instance is required")

    signatures = []
    policy_formulas: dict[str, dict[str, str]] = {identifier: {} for identifier in policy_ids}
    for lead_time in sorted({getattr(row, model.context_field) for row in selected_cases}):
        for definition in definitions:
            resolved = model.resolve_policy(definition, lead_time, state_caps[lead_time])
            policy_formulas[resolved.id][str(lead_time)] = resolved.formula
            signatures.append(
                {
                    "id": resolved.id,
                    "type": resolved.type,
                    model.context_field: lead_time,
                    "ast_hash": ast_hash(resolved.ast),
                    "motifs": resolved.motifs,
                    "parameter_specs": [
                        {
                            "name": spec.name,
                            "kind": spec.kind,
                            "lower": spec.lower,
                            "upper": spec.upper,
                            "values": list(spec.values) if spec.values is not None else None,
                            "default": spec.default,
                            "monotone_group": spec.monotone_group,
                            "monotone_index": spec.monotone_index,
                        }
                        for spec in resolved.parameter_specs
                    ],
                }
            )
            if model.setting == "perishable" and resolved.type == "computational_baseline":
                signatures[-1]["implementation_version"] = resolved.computational_spec.implementation_version
            if getattr(resolved, "execution", None):
                signatures[-1]["execution"] = resolved.execution


    run_dir = args.run_dir.resolve()
    results_dir = run_dir / "results"
    effective_config = run_dir / "effective_config.json"
    stored_config = None
    if effective_config.exists():
        stored_config = json.loads(effective_config.read_text(encoding="utf-8"))
        if {k: v for k, v in stored_config.items() if k != "workers"} != {
            k: v for k, v in config.items() if k != "workers"
        }:
            raise ValueError(f"run directory effective config mismatch: {effective_config}")
    from benchmarks.inventory_benchmark_family.run_one_mc import part_path, merge_case
    execution_definitions = (
        sorted(definitions, key=lambda row: row.type != "computational_baseline")
        if args.computational_first else definitions
    )
    tasks = [
        (case.name, definition.id, str(results_dir), getattr(case, model.context_field), "empirical_mc", model.suite)
        for case in selected_cases
        for definition in execution_definitions
    ]
    manifest = {
        "schema_version": 2,
        "suite": model.suite,
        "policy_set": policy_set_name,
        "policy_set_path": str(args.policy_set.resolve()),
        "policy_definitions": _policy_payload(args.policy_set),
        "resolved_policies": signatures,
        "config_source": str(args.config.resolve()),
        "evaluation_config": {k: v for k, v in config.items() if k != "workers"},
        "instances": [row.name for row in selected_cases],
        model.contexts_key: sorted({getattr(row, model.context_field) for row in selected_cases}),
        "policies": policy_ids,
        "policy_formulas": policy_formulas,
        "task_count": len(tasks),
        "instance_count": len(selected_cases),
        "task_granularity": "case_policy",
        "workers": workers,
        "mode": "optimizer_pipeline_with_empirical_mc_cost",
        "reference_policy": "best_all_policies_per_instance" if config.get("reference_scope") == "all_policies" else "best_baseline_per_instance",
        "reference_selection_bank": "evaluation",
        "reference_eligible_policy_types": sorted(
            BASELINE_TYPES | {"formula_ast"} | (
                {"formula_family"} if any(row.type == "formula_family" for row in definitions) else set())
            if config.get("reference_scope") == "all_policies" else BASELINE_TYPES),
        "reference_method": (
            f"independent_{int(config['evaluation_paths'])}_path_mc_best_{'all_policy' if config.get('reference_scope') == 'all_policies' else 'baseline'}_envelope"
        ),
        "optimizer_restarts": 1,
        "proposal_paths": int(config["proposal_paths"]),
        "tuning_paths": int(config["tuning_paths"]),
        "evaluation_paths": int(config["evaluation_paths"]),
        "dry_run": bool(args.dry_run),
        "ordering": f"{model.context_field}_ascending_then_instance",
        "created_at": time.time(),
    }
    if args.computational_first:
        manifest["execution_order"] = "computational_baselines_first"
    print(
        json.dumps(
            {
                "event": "mc_benchmark_plan",
                "instances": len(selected_cases),
                "policies_per_instance": len(definitions),
                "tasks": len(tasks),
                "task_granularity": "case_policy",
                model.contexts_key: manifest[model.contexts_key],
                "evaluation_paths": config["evaluation_paths"],
                "workers": workers,
                "dry_run": args.dry_run,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if args.dry_run:
        return

    results_dir.mkdir(parents=True, exist_ok=True)
    if stored_config is None:
        atomic_json(config, effective_config)
    _validate_or_write_manifest(run_dir / "RUN_MANIFEST.json", manifest)
    atomic_json(manifest["policy_definitions"], run_dir / "POLICIES.json")
    parts_dir = run_dir / ".scratch" / "policies"
    parts_dir.mkdir(parents=True, exist_ok=True)
    environment = _environment()
    policy_set_path = args.policy_set.resolve()

    def complete_if_ready(instance_id):
        destination = results_dir / f"{instance_id}.json"
        if _result_is_valid(destination, instance_id, policy_ids, config, model.suite, model):
            for identifier in policy_ids:
                part = part_path(parts_dir, instance_id, identifier)
                part.unlink(missing_ok=True)
                part.with_suffix(".log").unlink(missing_ok=True)
            return True
        if all(_part_is_valid(part_path(parts_dir, instance_id, identifier), instance_id,
                              identifier, config, model) for identifier in policy_ids):
            merge_case(parts_dir, instance_id, policy_set_path, config, destination, policy_ids)
            for identifier in policy_ids:
                part = part_path(parts_dir, instance_id, identifier)
                part.unlink(missing_ok=True)
                part.with_suffix(".log").unlink(missing_ok=True)
            return True
        return False

    def launcher(queued: QueuedTask) -> RunningTask | None:
        instance_id, identifier = queued.task[:2]
        destination = part_path(parts_dir, instance_id, identifier)
        if complete_if_ready(instance_id) or _part_is_valid(
            destination, instance_id, identifier, config, model
        ):
            return None
        stream = destination.with_suffix(".log").open("w", encoding="utf-8")
        stream.write(f"[MC benchmark attempt {queued.retries + 1}] {instance_id}\n")
        stream.flush()
        command = [
            sys.executable,
            str(BENCHMARK / "run_one_mc.py"),
            "--setting",
            model.setting,
            "--model-module", args.model_module,
            "--instance",
            instance_id,
            "--policy-id",
            identifier,
            "--policy-set",
            str(policy_set_path),
            "--config",
            str(effective_config),
            "--output",
            str(destination),
        ]
        if args.computational_first:
            command.append("--computational-first")
        try:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        except OSError:
            stream.close()
            raise
        timeout = config.get("task_timeout_seconds")
        return RunningTask(queued, process, stream, destination,
                           deadline=None if timeout is None else time.monotonic() + float(timeout))

    def validator(running: RunningTask) -> bool:
        instance_id, identifier = running.queued.task[:2]
        if _result_is_valid(results_dir / f"{instance_id}.json", instance_id,
                            policy_ids, config, model.suite, model):
            return True
        valid = running.process.returncode == 0 and _part_is_valid(
            running.destination, instance_id, identifier, config, model)
        if valid:
            complete_if_ready(instance_id)
        return valid

    summary = run_task_queue(
        tasks,
        workers=workers,
        max_retries=int(config["max_retries"]),
        retry_cooldown_seconds=float(config["retry_cooldown_seconds"]),
        memory_safety_reserve_gb=float(config["memory_safety_reserve_gb"]),
        estimated_task_memory_gb=float(config["estimated_task_memory_gb"]),
        memory_ewma_alpha=float(config["memory_ewma_alpha"]),
        memory_safety_margin=float(config["memory_safety_margin"]),
        launcher=launcher,
        result_validator=validator,
        emit=lambda value: print(value, flush=True),
    )
    atomic_json(asdict(summary), run_dir / "QUEUE_SUMMARY.json")
    atomic_json(
        [
            {"instance_id": row[0], "task": row[1], "returncode": row[2]}
            for row in summary.failures
        ],
        run_dir / "failures.json",
    )
    if summary.failures:
        raise SystemExit(f"MC benchmark worker failures: {list(summary.failures)}")
    subprocess.run(
        [sys.executable, str(HERE / "tools" / "build_mc_report.py"), "--setting", model.setting, "--model-module", args.model_module, "--run-dir", str(run_dir)],
        cwd=ROOT,
        env=environment,
        check=True,
    )


def _load_config(path: Path) -> dict[str, object]:
    config = dict(json.loads(path.read_text(encoding="utf-8")))
    required = {
        "workers",
        "horizon",
        "tuning_paths",
        "evaluation_paths",
        "proposal_paths",
        "state_caps",
        "candidate_batch_size",
        "max_retries",
        "retry_cooldown_seconds",
        "memory_safety_reserve_gb",
        "estimated_task_memory_gb",
        "memory_ewma_alpha",
        "memory_safety_margin",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"MC config missing fields: {missing}")
    positive = (
        "horizon",
        "tuning_paths",
        "evaluation_paths",
        "candidate_batch_size",
    )
    if any(int(config[key]) <= 0 for key in positive):
        raise ValueError("MC path counts and optimizer budgets must be positive")
    policy_timeout = config.get("computational_policy_timeout_seconds")
    if not isinstance(config.get("skip_perishable_computational_baselines", False), bool):
        raise ValueError("skip_perishable_computational_baselines must be boolean")
    if policy_timeout is not None and (not isinstance(policy_timeout, (int, float)) or not 0 < policy_timeout < float("inf")):
        raise ValueError("computational_policy_timeout_seconds must be finite and positive")
    proposal_budget = config.get("proposal_budget")
    if (proposal_budget is not None and int(proposal_budget) < 0) or int(
        config.get("proposal_paths", 0)
    ) <= 0:
        raise ValueError("proposal budgets must be nonnegative")
    if int(config["proposal_paths"]) > int(config["tuning_paths"]):
        raise ValueError("proposal_paths must not exceed tuning_paths")
    caps = {int(key): int(value) for key, value in config["state_caps"].items()}
    if not set(caps).issubset(set(SUPPORTED_CONTEXTS)) or not caps or any(value <= 0 for value in caps.values()):
        raise ValueError("state_caps must define positive N for a nonempty subset of contexts 2,...,128")
    return config


def _result_is_valid(
    path: Path, instance_id: str, policy_ids: list[str], config: dict[str, object], suite: str, model
) -> bool:
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
        result_policy_ids = [value.get("policy_id") for value in row.get("policies", ())]
        if model.result_is_current is not None and not model.result_is_current(row):
            return False
        return (
            row.get("schema_version") == 2
            and row.get("suite") == suite
            and row.get("instance_id") == instance_id
            and row.get("reference_policy") == ("best_all_policies_per_instance" if config.get("reference_scope") == "all_policies" else "best_baseline_per_instance")
            and int(row.get("evaluation_paths", -1)) == int(config["evaluation_paths"])
            and int(row.get("tuning_paths", -1)) == int(config["tuning_paths"])
            and int(row.get("horizon", -1)) == int(config["horizon"])
            and result_policy_ids == policy_ids
            and row.get("computational_policy_timeout_seconds") == config.get("computational_policy_timeout_seconds")
            and row.get("skip_perishable_computational_baselines", False) == config.get("skip_perishable_computational_baselines", False)
        )
    except (AttributeError, OSError, ValueError, TypeError):
        return False


def _part_is_valid(path, instance_id, identifier, config, model):
    """A fitted shard is reusable only with its matching bank and path costs."""
    try:
        import numpy as np
        row = json.loads(path.read_text())
        if row.get("task_kind") != "case_policy":
            return False
        if not _result_is_valid(path, instance_id, [identifier], config, model.suite, model):
            return False
        if row.get("tuning_seed") != model.seed("mc_tuning", instance_id):
            return False
        if row.get("evaluation_seed") != model.seed("mc_evaluation", instance_id):
            return False
        policy = row["policies"][0]
        costs = row["path_costs"]
        if policy["status"] in ("skipped_timeout", "TLE"):
            return not costs and policy["evaluation_cost"] is None
        values = np.asarray(costs[identifier], dtype=float)
        return (set(costs) == {identifier} and values.shape == (config["evaluation_paths"],)
                and np.all(np.isfinite(values))
                and np.isclose(float(np.mean(values)), policy["evaluation_cost"], rtol=1e-12))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(ROOT))),
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": "",
        }
    )
    return environment


def _validate_or_write_manifest(path: Path, manifest: dict[str, object]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        immutable = (
            "suite",
            "policy_definitions",
            "resolved_policies",
            "evaluation_config",
            "instances",
            "policies",
            "reference_method",
            "reference_policy",
            "reference_eligible_policy_types",
        )
        if any(existing.get(key) != manifest.get(key) for key in immutable):
            raise ValueError(f"run directory manifest mismatch: {path}")
        return
    atomic_json(manifest, path)


if __name__ == "__main__":
    main()
