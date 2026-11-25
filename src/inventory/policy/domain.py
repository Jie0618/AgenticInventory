"""Policy-search vocabulary derived from canonical inventory semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from inventory.core import InventoryCase
from inventory.policy.ast.core import COST_FEATURES, DISTRIBUTION_FEATURES
from inventory.policy.features import feature_specs_for_config


@dataclass(frozen=True)
class InventoryPolicyDomain:
    name: str
    objective: str
    state_names: tuple[str, ...]
    components: tuple[str, ...]
    distribution_names: tuple[str, ...]
    cost_names: tuple[str, ...]
    rules: tuple[str, ...]
    parameter_roles: dict[str, str]
    features: Mapping[str, Mapping[str, object]]


def policy_domain_for_case(case: InventoryCase) -> InventoryPolicyDomain:
    """Describe policy-visible semantics without constructing an SR prompt."""

    if not isinstance(case, InventoryCase):
        raise TypeError("case must be an InventoryCase")
    config = case.config
    rules = [
        "The policy output is projected to the feasible nonnegative integer action domain.",
        "Prefer registered summaries over coordinate-specific branches.",
    ]
    if config.lead_time.enabled:
        rules.append("Pipeline coordinates are ordered by arrival time; p1 arrives first.")
    if config.perishability.enabled:
        rules.extend(
            (
                "Inventory ages are ordered from expiring first to freshest.",
                "Older usable inventory is consumed before fresher inventory, and expiry cost applies to units that perish.",
            )
        )
    if config.backlog.enabled:
        rules.append("Account for outstanding backlog before inventory serves new demand.")
    elif config.backlog.mode.value == "lost_sales":
        rules.append("Unmet current demand is lost rather than retained in state.")
    cost_names = ["h", "b"]
    if config.perishability.enabled:
        cost_names.extend(("w", "expiry"))
    if config.backlog.enabled:
        cost_names.append("backlog")
    if config.backlog.enabled and case.costs.abandonment != 0.0:
        cost_names.append("abandonment")
    if case.costs.ordering != 0.0:
        cost_names.append("ordering")
    features = feature_specs_for_config(config)
    return InventoryPolicyDomain(
        name=case.model_id,
        objective="minimize expected total inventory-system cost",
        state_names=tuple(case.schema.names),
        components=tuple(config.enabled_components),
        distribution_names=tuple(sorted(DISTRIBUTION_FEATURES)),
        cost_names=tuple(name for name in cost_names if name in COST_FEATURES),
        rules=tuple(rules),
        parameter_roles={
            "S/T/A/P/N/F/Q": "inventory, pipeline, age, backlog, or aggregate targets",
            "C/cap/r": "order caps or thresholds",
            "K/alpha/beta/gamma/w/W": "nonnegative slopes or weights",
        },
        features={
            name: {
                "kind": spec.kind,
                "window": spec.window,
                "weights": list(spec.weights),
                "weight_rule": spec.weight_rule,
            }
            for name, spec in sorted(features.items())
        },
    )


__all__ = ["InventoryPolicyDomain", "policy_domain_for_case"]
