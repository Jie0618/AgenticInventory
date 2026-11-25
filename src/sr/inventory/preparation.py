"""Prepare one search case's DP reference and occupancy screening samples."""
from __future__ import annotations

import fcntl
import json
from pathlib import Path

import numpy as np

from evaluation.cache import load_or_create_inventory_dp
from runtime.artifacts import atomic_json

SUITE = "stage1-reference"
DIAGNOSTIC_PROTOCOL = "dp-occupancy-16-periods-normalized"


def prepare_case(case, horizon, diagnostic_root, cache_dir, *, sample_count=16, prune_probability=1e-13):
    """Reuse signed inputs, or prepare just this case before its search starts."""
    path = Path(diagnostic_root) / f"{case.name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    identity = dict(case_signature=case.case_signature, horizon=horizon,
                    sample_count=sample_count, prune_probability=prune_probability, suite=SUITE,
                    diagnostic_protocol=DIAGNOSTIC_PROTOCOL)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if path.exists():
            payload = json.loads(path.read_text())
            if payload.get("identity") != identity or not payload.get("representative_states"):
                raise ValueError(f"Stage1 preparation identity differs or samples missing: {path}")
        reference = load_or_create_inventory_dp(case, horizon, suite=SUITE, cache_dir=cache_dir)
        if not path.exists():
            payload = diagnostic(case, reference, horizon, sample_count, prune_probability)
            atomic_json({**payload, "identity": identity}, path)
    return reference, path


def diagnostic(case, reference, horizon, sample_count=16, prune_probability=1e-13):
    periods = set(np.linspace(round(.2 * (horizon - 1)), round(.8 * (horizon - 1)),
                              min(sample_count, horizon)).round().astype(int).tolist())
    composed = hasattr(reference, "compiled_transition")
    transition = reference.compiled_transition if composed else None
    if composed and transition is None:
        raise ValueError("screening diagnostics require compiled transitions")
    initial = reference.state_index[case.initial_state] if composed else tuple(case.initial_state.coordinates)
    occupancy = {initial: 1.0}
    rows = []
    for period in range(max(periods) + 1):
        following = {}
        for key, mass in occupancy.items():
            if composed:
                state = reference.states[key]
                action = int(reference.policy_table[period, key])
                next_ids = transition.next_state_ids[key, action]
                probabilities = transition.demand_probabilities
                if period in periods:
                    q = (transition.costs[key] + reference.value_table[period + 1][transition.next_state_ids[key]]) @ probabilities
                    features = reference.schema.mapping(state, horizon - period)
                    coordinates = list(state.coordinates)
            else:
                action = int(reference.policy_table[(period,) + key])
                available = key[0] + key[1]
                next_ids = [(min(max(available - int(d), 0), reference.n),) + key[2:] + (action,)
                            for d in reference.demand.values]
                probabilities = reference.demand.probs
                if period in periods:
                    q = np.zeros(reference.n + 1)
                    for d, probability in zip(reference.demand.values, probabilities):
                        remaining = max(available - int(d), 0)
                        immediate = reference.h * remaining + reference.b * max(int(d) - available, 0)
                        q += probability * (immediate + reference.value_table[period + 1][
                            (min(remaining, reference.n),) + key[2:] + (slice(None),)])
                    features = {"x": key[0], **{f"p{i}": v for i, v in enumerate(key[1:], 1)},
                                "remaining_horizon": horizon - period}
                    coordinates = list(key)
            if period in periods:
                rows.append(dict(state=coordinates, state_features=features, period=period,
                                 probability=float(mass),
                                 dp_action=action, q_cost_advantage=(q - q.min()).tolist()))
            for next_key, probability in zip(next_ids, probabilities):
                if composed:
                    next_key = int(next_key)
                following[next_key] = following.get(next_key, 0.) + mass * float(probability)
        if period < max(periods):
            following = {k: v for k, v in following.items() if v >= prune_probability}
            total = sum(following.values())
            if total <= 0:
                raise ValueError("diagnostic pruning removed every state")
            occupancy = {k: v / total for k, v in following.items()}
    total = sum(row["probability"] for row in rows)
    for row in rows:
        row["probability"] /= total
    return dict(case=case.name, horizon=horizon, sample_periods=sorted(periods),
                occupancy_source="dp_optimal", diagnostic_protocol=DIAGNOSTIC_PROTOCOL,
                dp_average_cost=float(reference.value_table[
                    (0, reference.state_index[case.initial_state]) if composed
                    else (0,) + tuple(case.initial_state.coordinates)]) / horizon,
                representative_states=rows)
