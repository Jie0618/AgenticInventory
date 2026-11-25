"""Stage-III reference-mode validation."""

from __future__ import annotations

REFERENCE_MODES = frozenset({"fp3", "best_baseline"})


def validate_reference_mode(mode: str, *, setting: str) -> str:
    value = str(mode).strip().lower()
    if value not in REFERENCE_MODES:
        raise ValueError(
            f"unsupported reference_mode={mode!r}; choose fp3 or best_baseline"
        )
    if value == "fp3" and str(setting).strip().lower() != "lead_time":
        raise ValueError(
            "reference_mode='fp3' is only supported for setting='lead_time'"
        )
    return value


__all__ = ["REFERENCE_MODES", "validate_reference_mode"]
