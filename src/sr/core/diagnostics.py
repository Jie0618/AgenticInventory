"""Probability-weighted parent evidence, with Bellman regret only when available."""
from __future__ import annotations

from runtime.artifacts import stable_key
import math
import json
from typing import Mapping, Sequence

DIAGNOSTIC_PROTOCOL = "dp-or-mc-parent-spread"


def action_signature(records) -> str:
    """Behavior identity on fixed states; neither formula nor parameter names enter."""
    return stable_key(json.dumps(records, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode())


def decision_records(bank: Sequence[Mapping], actions: Mapping[str, Sequence[int]], *,
                     case: str, limit: int = 10,
                     occupancy_source: str | None = None) -> dict[str, object]:
    """Rank EC.36 regrets per parent and EC.37 spread; never invent DP values."""
    if not bank:
        return {"case": case, "available": False, "reason": "no_dp_reference"}
    has_bellman = ["q_cost_advantage" in row for row in bank]
    if any(has_bellman) and not all(has_bellman):
        raise ValueError("diagnostic bank mixes records with and without Bellman values")
    dp_available = all(has_bellman)
    total = sum(float(row["probability"]) for row in bank)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("diagnostic bank must have positive finite probability")
    if any(len(values) != len(bank) for values in actions.values()):
        raise ValueError("diagnostic actions do not match fixed bank")
    regrets = {parent: [] for parent in actions}
    spreads = []
    for index, row in enumerate(bank):
        weight = float(row["probability"]) / total
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("invalid diagnostic probability")
        orders = {parent: int(values[index]) for parent, values in actions.items()}
        base = {"case": case, "state": row["state"], "period": row["period"],
                "probability": weight, "parent_actions": orders}
        advantages = row.get("q_cost_advantage")
        if advantages is not None:
            optimum = min(advantages)
            optimal_actions = [i for i, value in enumerate(advantages)
                               if math.isclose(value, optimum, abs_tol=1e-10, rel_tol=0)]
            for parent, action in orders.items():
                if not 0 <= action < len(advantages):
                    raise ValueError("parent action exceeds DP diagnostic action cap")
                difference = max(0.0, float(advantages[action]) - optimum)
                regrets[parent].append({**base, "parent_action": action,
                                        "optimal_actions": optimal_actions,
                                        "bellman_cost_difference": difference,
                                        "weighted_regret": weight * difference})
        if orders:
            spreads.append({**base, "weighted_action_spread":
                            weight * (max(orders.values()) - min(orders.values()))})
    result = {"case": case, "available": True,
              "occupancy_source": occupancy_source or ("dp_optimal" if dp_available else "mc_reference"),
              "bank_size": len(bank), "regret_available": dp_available,
              "action_spread": sorted(spreads, key=lambda r: -r["weighted_action_spread"])[:limit]}
    if dp_available:
        result["regret"] = {
            parent: sorted(rows, key=lambda r: -r["weighted_regret"])[:limit]
            for parent, rows in regrets.items()
        }
    else:
        result["regret_unavailable_reason"] = "no_dp_reference"
    return result
