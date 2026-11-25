"""Unified training: only structural parameters differ from Stage1/2."""
from dataclasses import dataclass
from inventory.core import DeterministicLeadTime, FixedShelfLife, DeterministicBacklog
from inventory.core.factory import inventory_case
from experiments.lead_time_backlog.configs import cases as anchor_cases

CONTEXTS = ((2, 2), (3, 3), (1, 4), (4, 1), (5, 3), (3, 5))
CASE_SUFFIXES = tuple(c.name.split("__", 1)[1] for c in anchor_cases())

@dataclass(frozen=True)
class TrainingProfile:
    profile_id: str
    shortage_ratio: float
    backlog_ratio: float
    abandonment_ratio: float

TRAINING_PROFILES = tuple(TrainingProfile(key, c.costs.shortage/c.costs.holding,
    c.costs.backlog/c.costs.shortage, c.costs.abandonment/c.costs.shortage)
    for key, c in zip(CASE_SUFFIXES, anchor_cases()))

def cases(n: int = 20):
    output = []
    for lead_time, patience in CONTEXTS:
        for key, source in zip(CASE_SUFFIXES, anchor_cases()):
            output.append(inventory_case(
                name=f"L{lead_time}_K{patience}__{key}",
                demand=source.demand, n=int(n), costs=source.costs,
                backlog=DeterministicBacklog(patience),
                perishability=source.config.perishability,
                lead_time=DeterministicLeadTime(lead_time),
                capacity=int(n), kind=source.kind,
            ))
    return output
