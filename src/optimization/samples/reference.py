"""Generic supervised samples from any finite-horizon reference table."""

from __future__ import annotations

from itertools import product
from typing import Any, Callable

from optimization.models.types import SupervisedSample
from optimization.support.utils import _state_dict


def reference_action_samples(
    reference: Any,
    *,
    state_features: Callable[[Any, int], dict[str, object]] | None = None,
) -> list[SupervisedSample]:
    """Flatten a reference's policy using its public ``states/action`` contract."""

    if hasattr(reference, "schema"):
        feature_builder = state_features or (
            lambda state, remaining: reference.schema.mapping(state, remaining)
        )
        states = tuple(reference.states)
    elif hasattr(reference, "lead_time") and hasattr(reference, "n"):
        states = tuple(product(range(int(reference.n) + 1), repeat=int(reference.lead_time) + 1))
        feature_builder = state_features or (
            lambda state, remaining: _state_dict(tuple(state), remaining)
        )
    else:
        raise TypeError("reference does not expose a supported finite-state contract")
    samples: list[SupervisedSample] = []
    for period in range(int(reference.horizon)):
        remaining = int(reference.horizon) - period
        for state in states:
            samples.append(
                SupervisedSample(
                    state=feature_builder(state, remaining),
                    target_action=float(reference.action(period, state)),
                    weight=1.0,
                )
            )
    return samples


__all__ = ["reference_action_samples"]
