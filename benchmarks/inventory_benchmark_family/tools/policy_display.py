"""Stable human-facing names for benchmark policies.

Policy IDs remain formula/source identifiers.  This module provides the
consistent labels used in reports so that the same policy is named the same
way across MC reports without rewriting result IDs.
"""

from __future__ import annotations

from typing import Mapping


_CANONICAL: dict[str, str] = {
    "base_stock": "Base stock",
    "capped_base_stock": "Capped BS",
    "vector_base_stock": "Vector BS",
    "tilted_cbs": "Tilted CBS",
    "tilted_pic": "Tilted PIC",
    "weighted_tilted_pic": "Weighted Tilted PIC",
    "aips_discounted_pipeline": "AIPS Discounted",
    "aips_hybrid": "AIPS Hybrid",
    "myopic": "Myopic",
    "myopic2": "Myopic-2",
    "pil": "PIL",
    "fp3": "FP3",
    "llm_or_projected_mv_capped": "LLM-OR Projected MV",
    "constant_order": "Constant order",
    "constant_order_cyclic": "Constant order cyclic",
    "dual_balancing": "Dual balancing",
}


def display_name(policy_id: str, manifest: Mapping[str, object] | None = None) -> str:
    """Return a consistent display label while preserving the source ID."""

    if policy_id in _CANONICAL:
        return _CANONICAL[policy_id]
    if manifest:
        names = manifest.get("policy_display_names")
        if isinstance(names, dict) and isinstance(names.get(policy_id), str):
            return str(names[policy_id])
    return policy_id
