"""Context-neutral execution and caching for policy-family evaluation."""

from __future__ import annotations

from runtime.artifacts import stable_key
from .artifacts import write_path

import json
import math
import os
import subprocess
import sys
import tempfile
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Mapping, Sequence

from inventory.policy.family import PolicyFamily, semantic_family_hash
from inventory.policy.family_domain import InventoryFamilyDomain
from runtime.task_queue import RunningTask, run_task_queue
from runtime.artifacts import atomic_json
from sr.family.models import CaseMetric, FamilyEvaluation, FamilyTask, ScenarioScore
from sr.family.planner import plan_family_tasks
from sr.family.scoring import cross_context_score, scenario_score
from sr.family.scoring import validate_score_protocol
from inventory.common.fitness import SCORE_PROTOCOL


Worker = Callable[[PolicyFamily, FamilyTask], CaseMetric]
_EVALUATION_PROTOCOL = "family-context-mode-budget-two-fidelity"


def record_task_timeout(key, task, cache_path, *, seconds):
    """Persist a budget-bound rejection separately from successful metrics."""
    row = {"family_hash": key, "family_id": task.family_id,
           "context_id": task.context_id, "case_id": task.case_id,
           "timeout_seconds": seconds, "status": "timeout"}
    atomic_json(row, cache_path.with_suffix(".timeout.json"))
    return CaseMetric(task.context_id, task.case_id, float("inf"),
                      seconds, "timeout", row)


def family_hash(family: PolicyFamily) -> str:
    """Stable executable identity independent of the display family id."""
    return semantic_family_hash(family)


def _valid(row: Mapping[str, object], family_key: str, task: FamilyTask) -> bool:
    if not isinstance(row, Mapping):
        return False
    try:
        return (
            row.get("family_hash") == family_key
            and str(row.get("context_id")) == task.context_id
            and row.get("case_id") == task.case_id
            and row.get("fidelity") == task.fidelity
            and row.get("status") == "complete"
            and math.isfinite(float(row["gap_pct"]))
        )
    except (KeyError, TypeError, ValueError):
        return False


def _metric(row: Mapping[str, object]) -> CaseMetric:
    return CaseMetric(
        str(row["context_id"]), str(row["case_id"]), float(row["gap_pct"]),
        float(row.get("runtime_seconds", 0.0)), str(row.get("status", "complete")),
        dict(row.get("metadata", {})),
    )


def _payload(key: str, task: FamilyTask, metric: CaseMetric) -> dict[str, object]:
    return {
        "family_hash": key, "context_id": task.context_id,
        "case_id": task.case_id, "fidelity": task.fidelity,
        "gap_pct": float(metric.gap_pct),
        "runtime_seconds": float(metric.runtime_seconds),
        "status": metric.status, "metadata": dict(metric.metadata),
    }


def evaluate_families(
    families: Sequence[PolicyFamily], case_ids: Sequence[str], worker: Worker, *,
    domain: InventoryFamilyDomain, output_dir: Path,
    workers: int = 60, global_seed: int = 20260811,
    fidelity: str = "standard",
    score_protocol: str = SCORE_PROTOCOL,
    _task_filter: Callable[[FamilyTask], bool] | None = None,
    task_executor=None,
) -> tuple[FamilyEvaluation, ...]:
    """Evaluate complete tensors; audit and exclude budget-timeout families."""
    validate_score_protocol(score_protocol)
    if not families:
        return ()
    unique: dict[str, PolicyFamily] = {}
    for family in families:
        domain.validate_family(family)
        unique.setdefault(domain.semantic_hash(family), family)
    root = Path(output_dir)
    cache_root = root / "task_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    protocol = _EVALUATION_PROTOCOL + "\0" + str(
        getattr(worker, "evaluation_protocol_signature",
                getattr(worker, "reference_mode", "legacy"))
    )
    config_path = getattr(worker, "config_path", None)
    # A configured but unreadable file is an error, not permission to reuse a
    # generic cache. Workers without a file still bind their protocol identity.
    config_bytes = Path(config_path).read_bytes() if config_path is not None else b"no-config"
    config_signature = stable_key(protocol.encode() + b"\0" + config_bytes)[:16]
    compatible_signatures = []
    if config_path is not None:
        # A watchdog increase must retry timeouts without re-fitting successful
        # tasks. Keep exact config snapshots so compatibility never depends on
        # guessing old settings or ignoring changes to numerical budgets.
        snapshots = cache_root / "configurations"
        snapshots.mkdir(exist_ok=True)
        current = json.loads(config_bytes)
        current.pop("evaluation_hard_timeout_seconds", None)
        for snapshot in snapshots.glob("*.json"):
            previous_bytes = snapshot.read_bytes()
            previous = json.loads(previous_bytes)
            previous.pop("evaluation_hard_timeout_seconds", None)
            signature = stable_key(protocol.encode() + b"\0" + previous_bytes)[:16]
            if previous == current and signature != config_signature:
                compatible_signatures.append(signature)
        snapshot = snapshots / (stable_key(config_bytes) + ".json")
        if not snapshot.exists():
            # Distinct files per config; atomic replacement also supports
            # concurrent family coordinators taking the same snapshot.
            with tempfile.NamedTemporaryFile(dir=snapshots, delete=False) as stream:
                stream.write(config_bytes)
                temporary = Path(stream.name)
            temporary.replace(snapshot)

    tasks_by_family: dict[str, tuple[PolicyFamily, tuple[FamilyTask, ...]]] = {}
    pending: list[tuple[str, PolicyFamily, FamilyTask, Path]] = []
    metrics: dict[str, list[CaseMetric]] = {key: [] for key in unique}
    for key, family in unique.items():
        planned = plan_family_tasks(
            family, case_ids, contexts=domain.context_ids,
            global_seed=global_seed, fidelity=fidelity, evaluator="mc",
        )
        tasks = tuple(task for task in planned if _task_filter is None or _task_filter(task))
        tasks_by_family[key] = (family, tasks)
        for task in tasks:
            safe_context = task.context_id.replace("/", "_")
            safe_case = task.case_id.replace("/", "_")
            path = cache_root / key / (
                f"{safe_context}__{safe_case}__{task.fidelity}__"
                f"seed{task.seed}__cfg{config_signature}.json"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                row = {}
            if not _valid(row, key, task):
                for signature in compatible_signatures:
                    alternate = path.with_name(path.name.replace(
                        "cfg" + config_signature, "cfg" + signature))
                    try:
                        candidate = json.loads(alternate.read_text())
                    except (OSError, json.JSONDecodeError):
                        continue
                    if _valid(candidate, key, task):
                        row = candidate
                        atomic_json(row, path)
                        break
            if _valid(row, key, task):
                metrics[key].append(_metric(row))
            elif path.with_suffix(".timeout.json").exists():
                # The filename binds this rejection to the same seed, fidelity,
                # worker configuration and budget. Never reuse it as a score.
                rejected = json.loads(path.with_suffix(".timeout.json").read_text())
                metrics[key].append(CaseMetric(task.context_id, task.case_id,
                    float("inf"), float(rejected["timeout_seconds"]), "timeout", rejected))
            else:
                pending.append((key, family, task, path))

    worker_module = getattr(worker, "subprocess_module", None)
    if pending and task_executor is not None:
        # A persistent executor shared by all in-flight families enforces one
        # global task limit, rather than workers per family/batch.
        def shared_task(item):
            key, family, task, path = item
            if worker_module is not None:
                local = {key: []}
                _run_subprocess_tasks([item], worker=worker,
                    worker_module=str(worker_module), output_dir=root,
                    metrics=local, workers=1,
                    cancel_event=getattr(task_executor, "cancel_event", None))
                if not local[key]:
                    raise RuntimeError(f"Stage3 task failed: {family.family_id}/{task.context_id}/{task.case_id}")
                return key, local[key][0]
            result = worker(family, task)
            if (result.status != "complete" or not math.isfinite(result.gap_pct)
                    or result.context_id != task.context_id or result.case_id != task.case_id):
                raise ValueError("worker returned invalid task metric")
            path.write_text(json.dumps(_payload(key, task, result), sort_keys=True) + "\n")
            return key, result
        futures = [task_executor.submit(shared_task, item) for item in pending]
        for future in as_completed(futures):
            key, result = future.result()
            metrics[key].append(result)
        pending = []
    if pending and worker_module is None:
        def run_one(item):
            key, family, task, path = item
            result = worker(family, task)
            if result.context_id != task.context_id or result.case_id != task.case_id:
                raise ValueError("worker returned mismatched task identity")
            if result.status != "complete" or not math.isfinite(result.gap_pct):
                raise ValueError("worker returned incomplete or non-finite metric")
            path.write_text(json.dumps(_payload(key, task, result), sort_keys=True) + "\n")
            return key, result
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            futures = [pool.submit(run_one, item) for item in pending]
            for future in as_completed(futures):
                key, result = future.result()
                metrics[key].append(result)
        pending = []
    if pending:
        _run_subprocess_tasks(
            pending, worker=worker, worker_module=str(worker_module),
            output_dir=root, metrics=metrics, workers=workers,
        )

    results: list[FamilyEvaluation] = []
    for key, (family, tasks) in tasks_by_family.items():
        rows = tuple(sorted(metrics[key], key=lambda row: (row.context_id, row.case_id)))
        timeouts = [row for row in rows if row.status == "timeout"]
        if timeouts and len(rows) == len(tasks):
            # An incomplete tensor has no comparable score. Reject this family
            # explicitly, retaining all successful task caches for inspection.
            atomic_json({"family_id": family.family_id, "family_hash": key,
                         "status": "timeout", "expected_tasks": len(tasks),
                         "completed_tasks": sum(row.status == "complete" for row in rows),
                         "timeouts": [dict(row.metadata) for row in timeouts]},
                        root / "rejected_families" / f"{key}.json")
            continue
        if len(rows) != len(tasks):
            results.append(FamilyEvaluation(
                family, tasks, rows, (), float("inf"),
                status=f"failed:{len(tasks) - len(rows)}",
            ))
            continue
        per_context = tuple(
            scenario_score(rows, context_id, protocol=score_protocol)
            for context_id in domain.context_ids
        )
        results.append(FamilyEvaluation(
            family, tasks, rows, per_context,
            cross_context_score(per_context, protocol=score_protocol), status="complete",
        ))
    return tuple(results)


def _run_subprocess_tasks(
    pending, *, worker: Worker, worker_module: str, output_dir: Path,
    metrics: dict[str, list[CaseMetric]], workers: int, cancel_event=None,
) -> None:
    work_root = output_dir / "task_runs"
    work_root.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="family_batch_", dir=work_root))
    task_dir = workdir / "tasks"
    task_dir.mkdir()
    audit_dir = output_dir / "task_audit" / workdir.name
    audit_dir.mkdir(parents=True, exist_ok=True)
    family_paths: dict[str, Path] = {}
    queue_rows = []
    lookup = {}
    for index, (key, family, task, cache_path) in enumerate(pending):
        if key not in family_paths:
            family_path = workdir / f"{key}.json"
            family_path.write_text(json.dumps({
                "family_id": family.family_id, "ast": family.ast,
                "motifs": family.motifs,
                "features": [asdict(row) for row in family.features],
                "parameters": [asdict(row) for row in family.parameters],
                "description": family.description,
                "parent_family_id": family.parent_family_id,
                "parent_family_hash": family.parent_family_hash,
                "mutation_action": family.mutation_action,
            }))
            family_paths[key] = family_path
        task_path = task_dir / f"{index}.json"
        destination = task_dir / f"{index}.out.json"
        task_path.write_text(json.dumps(asdict(task)))
        (audit_dir / f"{index}.json").write_text(json.dumps(asdict(task), sort_keys=True) + "\n")
        queue_row = (
            task.case_id, key, str(output_dir), task.context_id,
            task.fidelity, "family-batch",
        )
        queue_rows.append(queue_row)
        lookup[queue_row] = (key, task, cache_path, destination, family_paths[key], task_path)
    spawned = []
    logs = []
    timers = []
    timed_out = set()
    timed_out_destinations = set()
    settings = getattr(worker, "config", {})
    timeout = float(settings.get("evaluation_hard_timeout_seconds", 2400))
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("evaluation_hard_timeout_seconds must be finite and positive")
    def launcher(queued):
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("family evaluation cancelled")
        raw = queued.task if hasattr(queued, "task") else queued
        key, task, cache_path, destination, family_path, task_path = lookup[raw]
        destination.unlink(missing_ok=True)
        attempt = int(getattr(queued, "retries", 0)) + 1
        log = destination.with_name(
            f"{destination.stem}.attempt{attempt}.log"
        ).open("w")
        logs.append(log)
        config_path = getattr(worker, "config_path", None)
        if not config_path:
            raise ValueError("subprocess family worker requires config_path")
        command = [
            sys.executable, "-m", worker_module, "--family", str(family_path),
            "--task", str(task_path), "--output", str(destination),
            "--config", str(config_path), "--reference-cache",
            str(getattr(worker, "run_reference_cache_dir",
                        getattr(worker, "reference_cache_dir", output_dir / "reference_cache"))),
        ]
        reference_mode = getattr(worker, "reference_mode", None)
        if reference_mode:
            command.extend(["--reference-mode", str(reference_mode)])
        environment = dict(os.environ)
        environment.update({
            "PYTHONPATH": os.pathsep.join((str(Path.cwd() / "src"), str(Path.cwd()))),
            "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": "",
        })
        try:
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT,
                env=environment, start_new_session=True,
            )
        except OSError:
            log.close()
            raise
        spawned.append(process)
        def expire():
            if process.poll() is None:
                timed_out.add(process.pid)
                timed_out_destinations.add(destination)
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                (audit_dir / f"{index_for_timeout(task)}.timeout.json").write_text(json.dumps({
                    "family_id": task.family_id, "context_id": task.context_id,
                    "case_id": task.case_id, "timeout_seconds": timeout,
                    "error": "evaluation_hard_timeout"}) + "\n")
        timer = threading.Timer(timeout, expire)
        timer.daemon = True
        timer.start()
        timers.append(timer)
        return RunningTask(queued, process, log, destination)
    def index_for_timeout(task):
        return stable_key(repr(task).encode())[:16]
    def interruptible_sleep(seconds):
        if cancel_event is None:
            time.sleep(seconds)
        elif cancel_event.wait(seconds):
            raise RuntimeError("family evaluation cancelled")
    expected_tasks = {values[3]: values[1] for values in lookup.values()}
    validated_outputs: set[Path] = set()
    def valid(running):
        validated_outputs.discard(running.destination)
        try:
            row = json.loads(running.destination.read_text())
            task = expected_tasks[running.destination]
            accepted = (
                running.process.returncode == 0 and row.get("status") == "complete"
                and str(row.get("context_id")) == task.context_id
                and row.get("case_id") == task.case_id
                and math.isfinite(float(row["gap_pct"]))
            )
            if accepted:
                validated_outputs.add(running.destination)
            return accepted
        except (OSError, ValueError, TypeError, KeyError, AttributeError, json.JSONDecodeError):
            return False
    try:
        summary = run_task_queue(
            queue_rows, workers=int(workers), max_retries=int(settings.get("max_retries", 2)),
            retryable_failure=lambda running, _returncode: running.process.pid not in timed_out,
            retry_launch_errors=True,
            estimated_task_memory_gb=float(settings.get("estimated_task_memory_gb", 1.0)),
            memory_safety_reserve_gb=float(settings.get("memory_safety_reserve_gb", 100.0)),
            launcher=launcher, result_validator=valid,
            sleep=interruptible_sleep,
        )
    finally:
        for timer in timers:
            timer.cancel()
            timer.join()
        for process in spawned:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait()
        for log in logs:
            log.close()
    (audit_dir / "QUEUE_SUMMARY.json").write_text(json.dumps(asdict(summary), sort_keys=True) + "\n")
    for key, task, cache_path, destination, _family_path, _task_path in lookup.values():
        if destination in timed_out_destinations:
            metrics[key].append(record_task_timeout(key, task, cache_path, seconds=timeout))
            continue
        if destination not in validated_outputs:
            continue
        try:
            result = _metric(json.loads(destination.read_text()))
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            continue
        if (result.status == "complete" and math.isfinite(result.gap_pct)
                and result.context_id == task.context_id and result.case_id == task.case_id):
            cache_path.write_text(json.dumps(_payload(key, task, result), sort_keys=True) + "\n")
            metrics[key].append(result)


def screen_families(
    families: Sequence[PolicyFamily], case_ids: Sequence[str], worker: Worker, *,
    domain: InventoryFamilyDomain, output_dir: Path,
    incumbent_score: float, workers: int = 60, global_seed: int = 20260811,
    fidelity: str = "standard", cases_per_context: int = 2,
    screening_factor: float = 2.0, catastrophic_factor: float = 5.0,
) -> tuple[set[str], dict[str, object]]:
    if not families or not math.isfinite(float(incumbent_score)):
        return set(), {"enabled": False, "reason": "no_finite_incumbent"}
    count = max(1, min(len(case_ids), int(cases_per_context)))
    selected = set(sorted(map(str, case_ids))[:count])
    partial = evaluate_families(
        families, case_ids, worker, domain=domain,
        output_dir=output_dir, workers=workers, global_seed=global_seed,
        fidelity=fidelity, _task_filter=lambda task: task.case_id in selected,
    )
    screened: set[str] = set()
    audit: dict[str, object] = {
        "enabled": True, "incumbent": float(incumbent_score),
        "sentinel_cases_per_context": count, "families": {},
    }
    catastrophic_limit = max(1, math.ceil(0.75 * len(domain.context_ids)))
    for result in partial:
        scores: list[ScenarioScore] = []
        catastrophic = 0
        for context in domain.context_ids:
            observed = [row for row in result.metrics if row.context_id == context]
            padded = observed + [
                CaseMetric(context, f"__optimistic_zero_{index}", 0.0)
                for index in range(max(0, len(case_ids) - len(observed)))
            ]
            scores.append(scenario_score(padded, context))
            if len(observed) >= count and all(row.gap_pct >= catastrophic_factor for row in observed):
                catastrophic += 1
        lower = cross_context_score(scores)
        reason = None
        if lower > incumbent_score * screening_factor:
            reason = "optimistic_lower_bound_exceeds_incumbent"
        elif catastrophic >= catastrophic_limit:
            reason = "catastrophic_sentinels_across_contexts"
        key = domain.semantic_hash(result.family)
        if reason:
            screened.add(key)
        audit["families"][key] = {
            "lower_bound": lower, "catastrophic_contexts": catastrophic,
            "screened": bool(reason), "reason": reason,
        }
    write_path(output_dir, "SCREENING_AUDIT.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return screened, audit


def evaluate_family(family, case_ids, worker, **kwargs) -> FamilyEvaluation:
    return evaluate_families((family,), case_ids, worker, **kwargs)[0]


__all__ = ["Worker", "evaluate_families", "evaluate_family", "family_hash", "screen_families"]
