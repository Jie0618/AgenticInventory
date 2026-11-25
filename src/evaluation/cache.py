"""Model-aware reference-DP cache entry point.

Callers provide a canonical :class:`InventoryCase`; backend selection and
serialization remain internal.  This keeps SR and experiment runtimes from
depending on either the lead-time memmap cache or the composed pickle cache.
"""

from __future__ import annotations

from pathlib import Path

from inventory.core import InventoryCase
from evaluation.routing import select_evaluation_backend


def load_or_create_inventory_dp(
    case: InventoryCase,
    horizon: int,
    *,
    suite: str,
    cache_dir: Path | None,
):
    """Load or create the correctly routed reference for ``case``.

    The optimized deterministic lead-time model retains its read-only memmap
    cache.  Every other composition uses the canonical composed cache.  A
    ``None`` cache directory means direct computation for either route.
    """

    if not isinstance(case, InventoryCase):
        raise TypeError("case must be an InventoryCase")
    if int(horizon) <= 0:
        raise ValueError("horizon must be positive")
    decision = select_evaluation_backend(case)
    if decision.optimized:
        from evaluation.backends.lead_time.cache import load_or_create_dp

        return load_or_create_dp(
            case,
            int(horizon),
            suite=str(suite),
            cache_dir=None if cache_dir is None else Path(cache_dir),
        )

    if cache_dir is None:
        from evaluation.routing import solve_inventory_dp

        return solve_inventory_dp(case, int(horizon))

    from evaluation.backends.composed.cache import load_or_create_composed_dp

    return load_or_create_composed_dp(
        case,
        int(horizon),
        suite=str(suite),
        cache_dir=Path(cache_dir),
    )


__all__ = ["load_or_create_inventory_dp"]
