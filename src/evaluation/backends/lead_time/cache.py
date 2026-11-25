"""Lead-time case-level read-only cache for reference DP tables.

The gap runner launches one numerical process per ``(case, policy)``.  The
reference DP is identical for all policies of a case, so this module writes it
once and lets every policy worker reopen the arrays as read-only NumPy
memmaps.  A small file lock protects the first writer when the seven workers
for a case are admitted together.
"""

from __future__ import annotations

from runtime.artifacts import stable_key
import json
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np

from evaluation.backends.lead_time.dp import DPResult, solve_dp


CACHE_VERSION = 1
DEFAULT_CACHE_MEMORY_MARGIN = 0.25
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def cache_case_dir(cache_dir: Path, suite: str, case_name: str) -> Path:
    """Return the private directory for one suite/case cache."""

    safe_suite = _SAFE_NAME.sub("_", str(suite)).strip("._") or "suite"
    safe_case = _SAFE_NAME.sub("_", str(case_name)).strip("._") or "case"
    return Path(cache_dir) / safe_suite / safe_case


def cache_is_ready(cache_dir: Path, suite: str, case_name: str) -> bool:
    """Return whether a complete cache artifact is visible to readers."""

    case_dir = cache_case_dir(Path(cache_dir), suite, case_name)
    return all(
        (case_dir / filename).is_file()
        for filename in ("metadata.json", "value_table.npy", "policy_table.npy")
    )


def estimate_dp_bytes(
    lead_time: int,
    n: int,
    horizon: int,
    *,
    margin: float = DEFAULT_CACHE_MEMORY_MARGIN,
) -> int:
    """Estimate one uncached dense DP allocation for admission reservation."""

    cells = (2 * int(horizon) + 1) * (int(n) + 1) ** (int(lead_time) + 1)
    # ``cells`` already counts both tables' period axes (2H+1); each native
    # value/action entry is 8 bytes on the supported Linux/macOS environments.
    # The margin covers Bellman temporaries.
    return int(cells * 8 * (1.0 + float(margin)))


def _signature(case: object, horizon: int) -> str:
    demand = getattr(case, "demand")
    payload = {
        "version": CACHE_VERSION,
        "lead_time": int(getattr(case, "lead_time")),
        "horizon": int(horizon),
        "n": int(getattr(case, "n")),
        "h": float(getattr(case, "h")),
        "b": float(getattr(case, "b")),
        "demand_values": np.asarray(demand.values, dtype=int).tolist(),
        "demand_probs": np.asarray(demand.probs, dtype=float).tolist(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return stable_key(encoded)


def _metadata(case: object, horizon: int) -> dict[str, object]:
    demand = getattr(case, "demand")
    lead_time = int(getattr(case, "lead_time"))
    n = int(getattr(case, "n"))
    return {
        "cache_version": CACHE_VERSION,
        "signature": _signature(case, horizon),
        "case_name": str(getattr(case, "name")),
        "lead_time": lead_time,
        "horizon": int(horizon),
        "n": n,
        "h": float(getattr(case, "h")),
        "b": float(getattr(case, "b")),
        "demand_values": np.asarray(demand.values, dtype=int).tolist(),
        "demand_probs": np.asarray(demand.probs, dtype=float).tolist(),
        "value_dtype": np.dtype(float).str,
        "policy_dtype": np.dtype(int).str,
        "value_shape": [horizon + 1] + [n + 1] * (lead_time + 1),
        "policy_shape": [horizon] + [n + 1] * (lead_time + 1),
    }


def _read_cached(case: object, horizon: int, case_dir: Path) -> DPResult | None:
    metadata_path = case_dir / "metadata.json"
    value_path = case_dir / "value_table.npy"
    policy_path = case_dir / "policy_table.npy"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = _metadata(case, horizon)
        if any(metadata.get(key) != expected[key] for key in (
            "cache_version",
            "signature",
            "lead_time",
            "horizon",
            "n",
            "h",
            "b",
            "demand_values",
            "demand_probs",
            "value_dtype",
            "policy_dtype",
            "value_shape",
            "policy_shape",
        )):
            return None
        value_table = np.load(value_path, mmap_mode="r", allow_pickle=False)
        policy_table = np.load(policy_path, mmap_mode="r", allow_pickle=False)
        if value_table.dtype.str != str(expected["value_dtype"]):
            return None
        if policy_table.dtype.str != str(expected["policy_dtype"]):
            return None
        if tuple(value_table.shape) != tuple(expected["value_shape"]):
            return None
        if tuple(policy_table.shape) != tuple(expected["policy_shape"]):
            return None
        value_table.setflags(write=False)
        policy_table.setflags(write=False)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None

    return DPResult(
        lead_time=int(getattr(case, "lead_time")),
        horizon=int(horizon),
        n=int(getattr(case, "n")),
        demand=getattr(case, "demand"),
        h=float(getattr(case, "h")),
        b=float(getattr(case, "b")),
        value_table=value_table,
        policy_table=policy_table,
    )


@contextmanager
def _case_lock(lock_path: Path) -> Iterator[None]:
    """Serialize cache creation on POSIX while remaining usable elsewhere."""

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+")
    try:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):
            # The project runners are POSIX, but a best-effort fallback keeps
            # direct development runs usable on platforms without flock.
            pass
        yield
    finally:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except (ImportError, OSError):
            pass
        handle.close()


def _atomic_save_array(array: np.ndarray, destination: Path) -> None:
    temporary = destination.with_name(destination.name + f".tmp-{os.getpid()}")
    try:
        np.save(temporary, array, allow_pickle=False)
        # np.save adds .npy when the supplied name does not already end with it.
        generated = Path(str(temporary) if str(temporary).endswith(".npy") else str(temporary) + ".npy")
        os.replace(generated, destination)
    finally:
        for candidate in (temporary, Path(str(temporary) + ".npy")):
            try:
                candidate.unlink()
            except FileNotFoundError:
                pass


def _atomic_write_json(payload: dict[str, object], destination: Path) -> None:
    temporary = destination.with_name(destination.name + f".tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def load_or_create_dp(
    case: object,
    horizon: int,
    *,
    suite: str,
    cache_dir: Path | None,
    _solver=None,
) -> DPResult:
    """Load a shared DP cache or create it once under a case lock.

    ``cache_dir=None`` retains the original direct-computation behavior for
    standalone workers and unit tests.  Cached arrays are read-only memmaps;
    callers must not mutate either table.
    """

    solver = solve_dp if _solver is None else _solver
    if cache_dir is None:
        return solver(
            int(getattr(case, "lead_time")),
            int(horizon),
            int(getattr(case, "n")),
            getattr(case, "demand"),
            float(getattr(case, "h")),
            float(getattr(case, "b")),
        )

    case_dir = cache_case_dir(Path(cache_dir), suite, str(getattr(case, "name")))
    case_dir.mkdir(parents=True, exist_ok=True)
    cached = _read_cached(case, horizon, case_dir)
    if cached is not None:
        return cached

    with _case_lock(case_dir / ".lock"):
        cached = _read_cached(case, horizon, case_dir)
        if cached is not None:
            return cached
        result = solver(
            int(getattr(case, "lead_time")),
            int(horizon),
            int(getattr(case, "n")),
            getattr(case, "demand"),
            float(getattr(case, "h")),
            float(getattr(case, "b")),
        )
        _atomic_save_array(result.value_table, case_dir / "value_table.npy")
        _atomic_save_array(result.policy_table, case_dir / "policy_table.npy")
        _atomic_write_json(_metadata(case, horizon), case_dir / "metadata.json")
        cached = _read_cached(case, horizon, case_dir)
        if cached is None:
            raise RuntimeError(f"DP cache was not readable after creation: {case_dir}")
        return cached


def remove_case_cache(cache_dir: Path, suite: str, case_name: str) -> None:
    """Remove only one completed case cache, never the cache root."""

    case_dir = cache_case_dir(Path(cache_dir), suite, case_name)
    if not case_dir.exists():
        return
    for path in case_dir.iterdir():
        if path.is_file() or path.is_symlink():
            path.unlink()
        elif path.is_dir():
            # Cache creation never makes nested directories; refuse to recurse
            # so cleanup cannot accidentally target an unrelated path.
            raise RuntimeError(f"unexpected nested directory in DP cache: {path}")
    case_dir.rmdir()
