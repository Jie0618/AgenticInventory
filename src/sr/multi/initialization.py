"""Durable no-LLM initialization of a multi-case SR population."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from runtime.artifacts import atomic_json
from sr.analysis.frontier import FrontierSeedBundle
from sr.multi.models import MultiSearchIndividual


class InitializationScheduler(Protocol):
    pending: object
    active: object
    completed: Mapping[str, MultiSearchIndividual]
    pair_failures: Mapping[str, str]

    def submit_group(self, candidate: object, case_inputs: Mapping[str, Any]) -> str: ...
    def poll(self) -> object: ...
    def counts(self) -> dict[str, int]: ...
    def shutdown(self) -> None: ...


SchedulerFactory = Callable[[Path, Path, Mapping[str, tuple[str, ...]]], InitializationScheduler]


@dataclass(frozen=True)
class FrontierInitializationResult:
    cache_dir: Path
    individuals: tuple[MultiSearchIndividual, ...]
    dropped_candidate_hashes: tuple[str, ...]
    elapsed_seconds: float
    scheduler_counts: Mapping[str, int]


def run_frontier_initialization(
    bundle: FrontierSeedBundle,
    case_inputs: Mapping[str, Any],
    *,
    cache_dir: Path,
    initialization_signature: str,
    scheduler_factory: SchedulerFactory,
    progress_interval_seconds: float = 5.0,
) -> FrontierInitializationResult:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "protocol": "multi-frontier-standard-initialization",
        "initialization_signature": str(initialization_signature),
        "case_names": list(case_inputs),
        "candidate_count": len(bundle.candidates),
        "pair_count": len(bundle.candidates) * len(case_inputs),
        "source_checkpoints": dict(sorted(bundle.source_checkpoints.items())),
    }
    _validate_or_write_manifest(cache_dir / "manifest.json", manifest)
    scheduler = scheduler_factory(
        cache_dir / "evaluations", cache_dir / "events.jsonl", bundle.provenance
    )
    previously_dropped = _load_dropped_candidates(cache_dir)
    known_hashes = {candidate.ast_hash for candidate in bundle.candidates}
    previously_dropped.intersection_update(known_hashes)
    started = time.perf_counter()
    try:
        for candidate in bundle.candidates:
            if candidate.ast_hash in previously_dropped:
                continue
            scheduler.submit_group(candidate, case_inputs)
        last_progress = 0.0
        while scheduler.pending or scheduler.active:
            scheduler.poll()
            now = time.monotonic()
            if now - last_progress >= max(0.1, float(progress_interval_seconds)):
                _write_progress(
                    cache_dir, scheduler, len(bundle.candidates), len(case_inputs), started
                )
                last_progress = now
            if scheduler.pending or scheduler.active:
                time.sleep(0.1)
        scheduler.poll()
        _write_progress(
            cache_dir, scheduler, len(bundle.candidates), len(case_inputs), started
        )
        failed = sorted(previously_dropped | {
            row.ast_hash
            for row in bundle.candidates
            if row.ast_hash not in scheduler.completed
        })
        individuals = tuple(
            scheduler.completed[row.ast_hash]
            for row in bundle.candidates
            if row.ast_hash in scheduler.completed
        )
        if not individuals:
            atomic_json(
                {
                    "schema_version": 1,
                    "status": "incomplete",
                    "initialization_signature": str(initialization_signature),
                    "failed_candidate_hashes": failed,
                    "pair_failures": dict(sorted(scheduler.pair_failures.items())),
                    "scheduler_counts": scheduler.counts(),
                },
                cache_dir / "INITIALIZATION_INCOMPLETE.json",
            )
            raise RuntimeError(
                "frontier initialization produced no eligible candidates"
            )
        elapsed = time.perf_counter() - started
        atomic_json(
            {
                "schema_version": 1,
                "status": "complete",
                "initialization_signature": str(initialization_signature),
                "requested_candidate_count": len(bundle.candidates),
                "candidate_count": len(individuals),
                "pair_count": len(individuals) * len(case_inputs),
                "dropped_candidate_hashes": failed,
                "elapsed_seconds": elapsed,
                "scheduler_counts": scheduler.counts(),
                "aggregates": [
                    {
                        "ast_hash": row.ast_hash,
                        "score": row.score,
                        "mean_gap_pct": row.mean_gap_pct,
                        "mean_top_k_gap_pct": row.mean_top_k_gap_pct,
                        "worst_gap_pct": row.worst_gap_pct,
                        "worst_case": row.worst_case,
                        "result_path": row.result_path,
                    }
                    for row in individuals
                ],
            },
            cache_dir / "INITIALIZATION_COMPLETE.json",
        )
        incomplete = cache_dir / "INITIALIZATION_INCOMPLETE.json"
        if incomplete.exists():
            incomplete.unlink()
        return FrontierInitializationResult(
            cache_dir=cache_dir,
            individuals=individuals,
            dropped_candidate_hashes=tuple(failed),
            elapsed_seconds=elapsed,
            scheduler_counts=scheduler.counts(),
        )
    except BaseException:
        scheduler.shutdown()
        raise


def _write_progress(
    cache_dir: Path,
    scheduler: InitializationScheduler,
    candidate_count: int,
    case_count: int,
    started: float,
) -> None:
    counts = scheduler.counts()
    payload = {
        "schema_version": 1,
        "time": time.time(),
        "elapsed_seconds": time.perf_counter() - started,
        "completed_pairs": counts.get("pair_completed", 0),
        "total_pairs": candidate_count * case_count,
        "completed_structures": counts.get("group_completed", 0),
        "total_structures": candidate_count,
        "running_pairs": counts.get("pair_running", 0),
        "pending_pairs": counts.get("pending_pairs", 0),
        "failed_pairs": counts.get("pair_failed", 0),
        "cache_hits": counts.get("cache_hits", 0),
        "scheduler_counts": counts,
    }
    atomic_json(payload, cache_dir / "progress.json")
    print(json.dumps({"progress": "multi_initialization", **payload}, sort_keys=True), flush=True)


def _validate_or_write_manifest(path: Path, payload: Mapping[str, object]) -> None:
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != dict(payload):
            raise ValueError(f"initialization cache manifest mismatch: {path}")
        return
    atomic_json(payload, path)


def _load_dropped_candidates(cache_dir: Path) -> set[str]:
    dropped: set[str] = set()
    for name in ("INITIALIZATION_COMPLETE.json", "INITIALIZATION_INCOMPLETE.json"):
        path = cache_dir / name
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = payload.get(
            "dropped_candidate_hashes",
            payload.get("failed_candidate_hashes", ()),
        )
        dropped.update(str(value) for value in values)
    return dropped


__all__ = [
    "FrontierInitializationResult",
    "InitializationScheduler",
    "run_frontier_initialization",
]
