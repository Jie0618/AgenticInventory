"""Inventory domains and policy fixtures shared by family-search tests."""
from functools import lru_cache

from inventory.common.distributions import DiscreteDemand
from inventory.core import DeterministicLeadTime, LostSales, TransitionCosts
from inventory.core.factory import inventory_case
from inventory.policy.family import FamilyParameter, PolicyFamily
from inventory.policy.family_domain import InventoryFamilyDomain


@lru_cache(maxsize=None)
def family_domain(lead_times: tuple[int, ...] = (3, 5, 8, 10)) -> InventoryFamilyDomain:
    contexts = {}
    for lead_time in lead_times:
        contexts[str(lead_time)] = inventory_case(
            name=f"domain-L{lead_time}",
            demand=DiscreteDemand.from_pmf({0: 0.5, 1: 0.5}, name="domain"),
            n=20,
            costs=TransitionCosts(holding=1.0, shortage=3.0),
            lead_time=DeterministicLeadTime(lead_time),
            backlog=LostSales(),
            capacity=20,
        )
    return InventoryFamilyDomain.from_cases(contexts, training_case_count=8)


def canonical_dual_shortfall_seed() -> PolicyFamily:
    """Build a fixed dual-shortfall fixture for family-search tests."""
    ip = {"op": "motif", "name": "inventory_position"}
    far = {"op": "motif", "name": "far_pipeline"}
    dual = {"op": "minimum", "left": {"op": "add", "args": [{"op": "parameter", "name": "S"}, {"op": "neg", "arg": ip}]}, "right": {"op": "add", "args": [{"op": "parameter", "name": "F"}, {"op": "neg", "arg": far}]}}
    ast = {"op": "minimum", "left": {"op": "mul", "args": [dual, {"op": "parameter", "name": "K"}]}, "right": {"op": "parameter", "name": "C"}}
    params = (
        FamilyParameter("S", "target", 0.0, 200.0),
        FamilyParameter("F", "target", 0.0, 200.0),
        FamilyParameter("K", "scale", 0.0, 3.0),
        FamilyParameter("C", "cap", 0.0, 200.0),
    )
    family = PolicyFamily("canonical_capped_dual_shortfall", ast, {}, params, "Initial seed generated from the canonical capped dual shortfall family.")
    family.validate()
    return family
