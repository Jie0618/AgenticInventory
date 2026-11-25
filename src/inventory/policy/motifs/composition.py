"""Schema-aware composition of component-local and cross-component motifs."""

from __future__ import annotations

from inventory.core.config import InventoryModelConfig
from inventory.policy.ast import AST, add, cost, dist, safe_ratio, state, sub
from inventory.policy.motifs.backlog import backlog_motifs
from inventory.policy.motifs.lead_time import pipeline_motifs
from inventory.policy.motifs.perishable import perishable_motifs


def _sum(nodes: list[AST]) -> AST:
    return nodes[0] if len(nodes) == 1 else add(*nodes)


def _merge(target: dict[str, AST], incoming: dict[str, AST]) -> None:
    for name, ast in incoming.items():
        if name in target and target[name] != ast:
            raise ValueError(f"conflicting motif definition: {name}")
        target[name] = ast


def motifs_for_config(config: InventoryModelConfig) -> dict[str, AST]:
    """Build motifs whose raw state references match ``config`` exactly."""

    if not isinstance(config, InventoryModelConfig):
        raise TypeError("config must be an InventoryModelConfig")

    motifs: dict[str, AST] = {
        "demand_mean": dist("mu"),
        "demand_cv": dist("cv"),
        "cost_ratio": safe_ratio(cost("b"), cost("h")),
    }
    if config.perishability.enabled:
        _merge(
            motifs,
            perishable_motifs(
                config.perishability.max_shelf_life,
                decision_includes_freshest=config.lead_time.state_dimension > 0,
            ),
        )
        ages = [
            state(f"a{index}")
            for index in range(1, config.perishability.state_dimension + 1)
        ]
        on_hand = _sum(ages)
        effective_on_hand = sub(on_hand, ages[0])
    else:
        on_hand = state("x")
        effective_on_hand = on_hand
        motifs["total_inventory"] = on_hand

    pipeline = [
        state(f"p{index}")
        for index in range(1, config.lead_time.state_dimension + 1)
    ]
    if pipeline:
        _merge(motifs, pipeline_motifs(len(pipeline)))
        available = add(on_hand, pipeline[0])
        inventory_position = add(on_hand, *pipeline)
        effective_inventory_position = add(effective_on_hand, *pipeline)
        cumulative = on_hand
        for index, term in enumerate(pipeline, start=1):
            cumulative = add(cumulative, term)
            motifs[f"cumulative_inventory_{index}"] = cumulative
    else:
        available = on_hand
        inventory_position = on_hand
        effective_inventory_position = effective_on_hand

    motifs["available_inventory"] = available
    motifs["inventory_position"] = inventory_position
    if config.perishability.enabled:
        motifs["effective_inventory_position"] = effective_inventory_position

    backlog = backlog_motifs(config.backlog)
    _merge(motifs, backlog)
    if backlog:
        total_backlog = backlog["total_backlog"]
        motifs["net_inventory_position"] = sub(inventory_position, total_backlog)
        motifs["effective_net_inventory_position"] = sub(
            effective_inventory_position, total_backlog
        )
        motifs["available_after_backlog"] = sub(available, total_backlog)
        motifs["backlog_coverage"] = safe_ratio(available, total_backlog)

    return motifs


__all__ = ["motifs_for_config"]
