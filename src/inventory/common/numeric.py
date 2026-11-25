"""Shared validation for integer-valued public inventory arguments."""

from __future__ import annotations

from typing import Any, Iterable, Optional, Tuple

import numpy as np


def integer_scalar(
    value: Any, *, name: str, minimum: Optional[int] = None
) -> int:
    """Return one integer-valued Python/NumPy scalar without truncation.

    Integral floating-point scalars remain accepted for compatibility with
    serialized experiment inputs. Boolean values, arrays (including 0-D and
    one-element arrays), non-finite values, and fractional values are rejected.
    """

    if (
        isinstance(value, (bool, np.bool_, np.ndarray))
        or not np.isscalar(value)
    ):
        raise ValueError(f"{name} must be an integer")
    try:
        normalized = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be an integer") from error
    try:
        is_integral = bool(value == normalized)
    except (TypeError, ValueError):
        is_integral = False
    if not is_integral:
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and normalized < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return normalized


def integer_tuple(
    values: Iterable[Any],
    *,
    name: str,
    minimum: Optional[int] = None,
    allow_empty: bool = False,
) -> Tuple[int, ...]:
    """Normalize a one-dimensional iterable of integer-valued scalars."""

    if values is None:  # type: ignore[comparison-overlap]
        raw = ()
    else:
        try:
            raw = tuple(values)
        except TypeError as error:
            raise ValueError(f"{name} must contain integers") from error
    if not allow_empty and not raw:
        raise ValueError(f"{name} must be nonempty")
    normalized = []
    for value in raw:
        try:
            normalized.append(
                integer_scalar(value, name=f"{name} value", minimum=minimum)
            )
        except ValueError as error:
            qualifier = "" if minimum is None else f" >= {minimum}"
            raise ValueError(
                f"{name} must contain integers{qualifier}"
            ) from error
    return tuple(normalized)


__all__ = ["integer_scalar", "integer_tuple"]
