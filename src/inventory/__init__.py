"""Composable inventory state, case, policy-AST, and transition definitions."""

from inventory.core import (
    CanonicalStateSchema,
    InventoryCase,
    InventoryModelConfig,
    InventoryState,
    TransitionBranch,
    TransitionCosts,
    TransitionOutcome,
    build_transition_kernel,
    inventory_case,
)

CanonicalTransitionOutcome = TransitionOutcome
StateSchema = CanonicalStateSchema

__all__ = [
    "CanonicalTransitionOutcome",
    "CanonicalStateSchema",
    "InventoryCase",
    "inventory_case",
    "InventoryModelConfig",
    "InventoryState",
    "StateSchema",
    "TransitionBranch",
    "TransitionCosts",
    "TransitionOutcome",
    "build_transition_kernel",
]
