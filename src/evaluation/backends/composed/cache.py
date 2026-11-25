"""Process-safe disk cache for compiled composed DP references."""

from __future__ import annotations

import fcntl
import os
import pickle
from pathlib import Path

from evaluation.dp import ComposedDPResult, solve_composed_dp
from inventory.core import InventoryCase


CACHE_SCHEMA_VERSION = 1


def _cache_path(cache_dir: Path, suite: str, case: InventoryCase, horizon: int) -> Path:
    token = case.case_signature[:16]
    return cache_dir / str(suite) / f"{case.name}__h{int(horizon)}__{token}.pickle"


def _valid(payload: object, case: InventoryCase, horizon: int) -> bool:
    if not isinstance(payload, dict):
        return False
    reference = payload.get("reference")
    return (
        payload.get("schema_version") == CACHE_SCHEMA_VERSION
        and payload.get("case_signature") == case.case_signature
        and int(payload.get("horizon", -1)) == int(horizon)
        and isinstance(reference, ComposedDPResult)
        and reference.metadata.get("case_signature") == case.case_signature
        and int(reference.horizon) == int(horizon)
    )


def load_or_create_composed_dp(
    case: InventoryCase,
    horizon: int,
    *,
    suite: str,
    cache_dir: Path,
) -> ComposedDPResult:
    """Load one signed compiled reference, creating it under an exclusive lock."""

    destination = _cache_path(Path(cache_dir), suite, case, horizon)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.with_suffix(destination.suffix + ".lock")
    with lock_path.open("a+b") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        if destination.exists():
            try:
                payload = pickle.loads(destination.read_bytes())
            except (OSError, EOFError, pickle.PickleError):
                payload = None
            if _valid(payload, case, horizon):
                return payload["reference"]
        reference = solve_composed_dp(case, horizon)
        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "case_signature": case.case_signature,
            "horizon": int(horizon),
            "reference": reference,
        }
        temporary = destination.with_name(
            f".{destination.name}.{os.getpid()}.tmp"
        )
        temporary.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
        temporary.replace(destination)
        return reference


__all__ = ["CACHE_SCHEMA_VERSION", "load_or_create_composed_dp"]
