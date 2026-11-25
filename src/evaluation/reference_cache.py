"""Shared identity and validation helpers for reference-DP artifacts.

Serialization remains backend-specific (memmap for lead-time, compiled tables
for composed models), while identity checks are intentionally common.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ReferenceIdentity:
    case_signature: str
    horizon: int
    backend: str


def identity_for(reference: Any, *, case_signature: str) -> ReferenceIdentity:
    metadata = getattr(reference, "metadata", {})
    metadata = metadata if isinstance(metadata, Mapping) else {}
    backend = str(metadata.get("backend", getattr(reference, "backend", "unknown")))
    return ReferenceIdentity(
        case_signature=str(case_signature),
        horizon=int(getattr(reference, "horizon")),
        backend=backend,
    )


def validate_identity(
    reference: Any,
    *,
    case_signature: str,
    horizon: int,
    expected_backend: str | None = None,
) -> None:
    metadata = getattr(reference, "metadata", {})
    metadata = metadata if isinstance(metadata, Mapping) else {}
    signature = metadata.get("case_signature")
    if signature is not None and str(signature) != str(case_signature):
        raise ValueError("reference does not belong to the supplied case")
    if int(getattr(reference, "horizon")) != int(horizon):
        raise ValueError("reference horizon does not match the requested horizon")
    if expected_backend is not None:
        # Result classes expose a backend label that is more
        # specific than the routing decision (for example compiled vs scalar
        # composed DP).  Only enforce an explicit serialized backend marker;
        # class-level labels remain informational for compatibility.
        backend = str(metadata.get("backend", ""))
        if backend and backend != expected_backend:
            raise ValueError(
                f"reference backend {backend!r} does not match {expected_backend!r}"
            )


__all__ = ["ReferenceIdentity", "identity_for", "validate_identity"]
