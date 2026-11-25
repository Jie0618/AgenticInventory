"""Small cases shared by the runnable examples."""

from pathlib import Path
import sys

# Make direct script execution work without installing the repository.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inventory.common.distributions import DiscreteDemand
from inventory.core import (
    DeterministicBacklog,
    DeterministicLeadTime,
    FixedShelfLife,
    LostSales,
    inventory_case,
)


SETTINGS = ("lead_time", "perishable", "lead_time_perishable", "lead_time_backlog")


def positive_int(value):
    import argparse

    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def make_case(setting):
    """Use deliberately small dimensions for quick finite-horizon evaluation."""
    if setting not in SETTINGS:
        raise ValueError(f"unknown setting: {setting}")
    perishable = setting in ("perishable", "lead_time_perishable")
    backlog = setting == "lead_time_backlog"
    return inventory_case(
        name=f"example_{setting}",
        demand=DiscreteDemand(values=[0, 1, 2], probs=[0.2, 0.5, 0.3]),
        n=3,
        capacity=6,
        costs={
            "holding": 1.0,
            "shortage": 0.0 if backlog else 3.0,
            "expiry": 1.0 if perishable else 0.0,
            "backlog": 1.0 if backlog else 0.0,
            "abandonment": 3.0 if backlog else 0.0,
        },
        lead_time=None if setting == "perishable" else DeterministicLeadTime(1),
        perishability=FixedShelfLife(2) if perishable else None,
        backlog=DeterministicBacklog(2) if backlog else LostSales(),
    )
