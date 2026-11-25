"""Cycle-safe public entry points for policy-parameter optimizers."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "fit_inventory_constants": ("optimization.solvers.api", "fit_inventory_constants"),
    "fit_inventory_policy": (
        "optimization.solvers.api",
        "fit_inventory_policy",
    ),
    "fit_inventory_policy_mc": (
        "optimization.solvers.api",
        "fit_inventory_policy_mc",
    ),
    "optimize_policy_constants": (
        "optimization.solvers.policy_cost",
        "optimize_policy_constants",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
