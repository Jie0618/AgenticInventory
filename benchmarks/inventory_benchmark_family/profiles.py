"""Frozen setting-specific economic profiles for benchmark family."""

from __future__ import annotations

from dataclasses import dataclass

from inventory.core import TransitionCosts


@dataclass(frozen=True)
class EconomicProfile:
    profile_id: str
    shortage_over_holding: float
    expiry_over_holding: float = 0.0
    backlog_over_shortage: float = 0.0
    abandonment_over_shortage: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.shortage_over_holding,
            self.expiry_over_holding,
            self.backlog_over_shortage,
            self.abandonment_over_shortage,
        )
        if not self.profile_id or any(float(value) < 0.0 for value in values):
            raise ValueError("economic profiles require an id and nonnegative ratios")
        if float(self.shortage_over_holding) <= 0.0:
            raise ValueError("shortage_over_holding must be positive")

    def costs(self, *, holding: float = 1.0) -> TransitionCosts:
        shortage = float(holding) * float(self.shortage_over_holding)
        return TransitionCosts(
            holding=float(holding),
            shortage=shortage,
            expiry=float(holding) * float(self.expiry_over_holding),
            backlog=shortage * float(self.backlog_over_shortage),
            abandonment=shortage * float(self.abandonment_over_shortage),
        )


LEAD_TIME_PROFILES = tuple(
    EconomicProfile(f"bh{str(ratio).replace('.', 'p')}", ratio)
    for ratio in (1.0, 3.0, 9.0, 20.0)
)

# Eight-run balanced design over four levels of b/h and w/h. Every level
# appears twice. The paired extremes and middle levels avoid fixing expiry
# economics to one demand family while keeping the formal benchmark tractable.
PERISHABLE_PROFILES = tuple(
    EconomicProfile(
        f"bh{str(shortage).replace('.', 'p')}__wh{str(expiry).replace('.', 'p')}",
        shortage,
        expiry_over_holding=expiry,
    )
    for shortage, expiry in (
        (1.0, 0.25),
        (1.0, 16.0),
        (3.0, 1.0),
        (3.0, 4.0),
        (9.0, 0.25),
        (9.0, 16.0),
        (20.0, 1.0),
        (20.0, 4.0),
    )
)

# Complete 2 x 2 x 2 design. Ratios are relative so changing b/h preserves
# the intended backlog and abandonment economics.
LEAD_TIME_BACKLOG_PROFILES = tuple(
    EconomicProfile(
        "bh{}__gb{}__ab{}".format(
            str(shortage).replace(".", "p"),
            str(backlog).replace(".", "p"),
            str(abandonment).replace(".", "p"),
        ),
        shortage,
        backlog_over_shortage=backlog,
        abandonment_over_shortage=abandonment,
    )
    for shortage in (1.0, 20.0)
    for backlog in (0.05, 0.25)
    for abandonment in (0.25, 1.0)
)


__all__ = [
    "EconomicProfile",
    "LEAD_TIME_BACKLOG_PROFILES",
    "LEAD_TIME_PROFILES",
    "PERISHABLE_PROFILES",
]
