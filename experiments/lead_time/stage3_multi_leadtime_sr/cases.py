"""Ten fixed economic environments repeated at each lead-time context."""
from inventory.core import DeterministicLeadTime
from inventory.core.factory import inventory_case
from experiments.lead_time.configs import cases as anchor_cases

CASE_SUFFIXES = tuple(c.name.split("_", 1)[1] for c in anchor_cases())


def cases(*, state_caps, lead_times):
    output = []
    for lead in lead_times:
        for key, source in zip(CASE_SUFFIXES, anchor_cases()):
            output.append(inventory_case(
                name=f"L{lead}__{key}", demand=source.demand, n=int(state_caps[lead]),
                costs=source.costs, lead_time=DeterministicLeadTime(lead),
                backlog=source.config.backlog, perishability=source.config.perishability,
                capacity=int(state_caps[lead]), kind=source.kind))
    return output
