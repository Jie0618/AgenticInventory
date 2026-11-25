"""Deterministic selection helpers for independent SR certification."""

from __future__ import annotations

from runtime.artifacts import seed_from_text
from typing import Any, Mapping, Sequence


def certification_seed(base_seed: int, namespace: str, *identity: object) -> int:
    material = "|".join((str(base_seed), str(namespace), *(str(x) for x in identity)))
    return seed_from_text(material.encode("utf-8"))


def select_profile_candidates(
    checkpoint: Mapping[str, Any],
    *,
    profiles: Sequence[str] = ("parameter_profile", "complexity_profile"),
    max_per_profile: int,
) -> tuple[dict[str, Any], ...]:
    """Select deterministic, hash-deduplicated finalists from profile views."""

    selected: dict[str, dict[str, Any]] = {}
    for requested_key in profiles:
        key = requested_key
        if key not in checkpoint and key.endswith("_profile"):
            key = key.replace("_profile", "_frontier")
        ranked = sorted(
            checkpoint.get(key, ()),
            key=lambda row: (
                float(row["gap_pct"]),
                int(row["num_parameters"]),
                float(row["expanded_complexity"]),
                str(row["ast_hash"]),
            ),
        )
        for row in ranked[: max(1, int(max_per_profile))]:
            selected[str(row["ast_hash"])] = dict(row)
    return tuple(selected.values())


def select_best_profile_candidate(
    checkpoint: Mapping[str, Any], *, max_parameters: int
) -> dict[str, Any]:
    profile = checkpoint.get("parameter_profile", checkpoint.get("parameter_frontier", ()))
    eligible = [row for row in profile if int(row["num_parameters"]) <= int(max_parameters)]
    if not eligible:
        raise ValueError(f"checkpoint has no policy with <= {max_parameters} parameters")
    return dict(
        min(
            eligible,
            key=lambda row: (
                float(row["gap_pct"]),
                int(row["num_parameters"]),
                float(row["expanded_complexity"]),
                str(row["ast_hash"]),
            ),
        )
    )


__all__ = [
    "certification_seed",
    "select_best_profile_candidate",
    "select_profile_candidates",
]
