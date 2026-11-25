"""Unified training: only structural parameters differ from Stage1/2."""
from dataclasses import dataclass
from inventory.core import DeterministicLeadTime, FixedShelfLife, DeterministicBacklog
from inventory.core.factory import inventory_case
from experiments.perishable.configs import cases as anchor_cases

CONTEXTS = (3, 5, 8, 10)
CASE_SUFFIXES = tuple(c.name.split("_", 1)[1] for c in anchor_cases())

@dataclass(frozen=True)
class TrainingProfile:
    profile_id: str
    shortage_ratio: float
    expiry_ratio: float
    ordering_ratio: float

TRAINING_PROFILES = tuple(TrainingProfile(key, c.costs.shortage/c.costs.holding,
    c.costs.expiry/c.costs.holding, c.costs.ordering/c.costs.holding)
    for key, c in zip(CASE_SUFFIXES, anchor_cases()))

def cases(n: int = 22):
    output = []
    for shelf_life in CONTEXTS:
        for key, source in zip(CASE_SUFFIXES, anchor_cases()):
            output.append(inventory_case(
                name=f"M{shelf_life}__{key}",
                demand=source.demand, n=int(n), costs=source.costs,
                perishability=FixedShelfLife(shelf_life),
                backlog=source.config.backlog,
                lead_time=source.config.lead_time,
                capacity=int(n), kind=source.kind,
            ))
    return output
