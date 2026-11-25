"""Flat-array contracts internal to the deterministic lead-time backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


FlatState = tuple[int, ...]


@dataclass(frozen=True)
class FlatStateSchema:
    """Ordered coordinates used by vectorized lead-time arrays."""

    model_id: str
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must be nonempty")
        if not self.names or any(not str(name) for name in self.names):
            raise ValueError("state schema requires nonempty coordinate names")
        if len(set(self.names)) != len(self.names):
            raise ValueError("state coordinate names must be unique")
        if "remaining_horizon" in self.names:
            raise ValueError("remaining_horizon is supplied separately")

    @property
    def allowed_state_names(self) -> frozenset[str]:
        return frozenset((*self.names, "remaining_horizon"))

    def mapping(
        self, state: Sequence[int], remaining_horizon: int
    ) -> dict[str, float]:
        normalized = tuple(int(value) for value in state)
        if len(normalized) != len(self.names):
            raise ValueError(
                f"{self.model_id} expects {len(self.names)} state coordinates, "
                f"got {len(normalized)}"
            )
        return {
            **{name: float(value) for name, value in zip(self.names, normalized)},
            "remaining_horizon": float(remaining_horizon),
        }


@dataclass(frozen=True)
class FlatTransitionOutcome:
    """One demand-conditioned transition in the lead-time array encoding."""

    next_state: FlatState
    holding_cost: float
    shortage_cost: float
    expiry_cost: float = 0.0

    @property
    def total_cost(self) -> float:
        return float(self.holding_cost + self.shortage_cost + self.expiry_cost)


@dataclass
class RollingDPReference:
    """Table-free deterministic lead-time geometry for rolling exact methods."""

    lead_time: int
    horizon: int
    n: int
    demand: object
    h: float
    b: float
    continuation_horizon: int = 0
    continuation_values: np.ndarray | None = None

    def value(self, _t: int, _state: object) -> float:
        raise RuntimeError("rolling reference does not retain a value table")


__all__ = [
    "FlatState",
    "FlatStateSchema",
    "FlatTransitionOutcome",
    "RollingDPReference",
]
