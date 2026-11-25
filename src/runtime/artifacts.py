"""Atomic writers shared by experiment and library runtimes."""

from __future__ import annotations

import csv
import json
import sys
import zlib
from pathlib import Path
from typing import Any


def stable_key(value: Any) -> str:
    """Compact, deterministic label for cache paths and structural deduplication."""
    data = value if isinstance(value, bytes) else json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode()
    # Independent non-cryptographic accumulators keep labels short enough for
    # filenames. Callers retain the actual inputs in their cached records.
    return f"{zlib.crc32(data):08x}{zlib.adler32(data):08x}{len(data):x}"


def seed_from_text(value: bytes | str) -> int:
    """Derive a reproducible 32-bit RNG seed without Python's salted hash()."""
    return zlib.crc32(value.encode() if isinstance(value, str) else value)


_JSON_RESULT_FIELDS = {
    "convergence_report",
    "distribution_features",
    "distribution_pmf",
    "optimizer_metadata",
    "params",
}
_INTEGER_RESULT_FIELDS = {
    "horizon",
    "lead_time",
    "n",
    "nominal_exact_budget",
    "nominal_proposal_budget",
    "sample_count",
    "schema_version",
}
_FLOAT_RESULT_FIELDS = {
    "b",
    "cost",
    "dp_cost",
    "evaluator_setup_seconds",
    "fit_elapsed_seconds",
    "gap_pct",
    "h",
    "optimizer_wall_seconds",
    "reference_dp_seconds",
    "rolling_optimum_seconds",
    "rolling_reported_seconds",
    "total_task_seconds",
}
_BOOLEAN_RESULT_FIELDS = {"cuda", "single_worker"}


def atomic_json(payload: Any, path: Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def load_result_records(root: Path) -> list[dict[str, Any]]:
    """Load consolidated CSV results, with flat-JSON input support."""

    source = Path(root)
    csv_path = source if source.suffix.lower() == ".csv" else source / "RESULTS.csv"
    if csv_path.is_file():
        csv.field_size_limit(sys.maxsize)
        with csv_path.open(newline="", encoding="utf-8") as stream:
            return [_decode_csv_row(dict(row)) for row in csv.DictReader(stream)]

    json_root = source / "raw" if (source / "raw").is_dir() else source
    rows = []
    for path in sorted(json_root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and {"name", "policy", "cost", "dp_cost"} <= payload.keys():
            rows.append(payload)
    return rows


def _decode_csv_row(row: dict[str, str]) -> dict[str, Any]:
    decoded: dict[str, Any] = dict(row)
    for field in _JSON_RESULT_FIELDS:
        value = row.get(field)
        if value:
            decoded[field] = json.loads(value)
    for field in _INTEGER_RESULT_FIELDS:
        value = row.get(field)
        if value not in (None, ""):
            decoded[field] = int(value)
    for field in _FLOAT_RESULT_FIELDS:
        value = row.get(field)
        if value not in (None, ""):
            decoded[field] = float(value)
    for field in _BOOLEAN_RESULT_FIELDS:
        value = row.get(field)
        if value not in (None, ""):
            decoded[field] = str(value).lower() == "true"
    return decoded


__all__ = ["atomic_json", "load_result_records", "stable_key", "seed_from_text"]
