"""Repeat the exact Stage1/2 ten economic profiles across six contexts."""
from inventory.core import DeterministicLeadTime, FixedShelfLife
from inventory.core.factory import inventory_case
from experiments.lead_time_perishable.configs import cases as anchor_cases

CONTEXTS = ((2, 2), (3, 3), (1, 4), (4, 1), (5, 3), (3, 5))
CASE_SUFFIXES = tuple(c.name.split("__", 1)[1] for c in anchor_cases())


def cases(n: int = 20):
    anchors = anchor_cases(state_action_cap=n)
    output = []
    for lead_time, shelf_life in CONTEXTS:
        for suffix, source in zip(CASE_SUFFIXES, anchors):
            output.append(inventory_case(
                name=f"L{lead_time}_M{shelf_life}__{suffix}",
                demand=source.demand, n=source.n, costs=source.costs,
                lead_time=DeterministicLeadTime(lead_time),
                perishability=FixedShelfLife(shelf_life),
                backlog=source.config.backlog,
                capacity=source.config.capacity, kind=source.kind,
            ))
    return output
