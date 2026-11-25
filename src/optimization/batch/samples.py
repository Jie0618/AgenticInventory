from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from optimization.models.types import SupervisedSample


@dataclass(frozen=True)
class _BatchSamples:
    """Vectorized view of supervised samples for fast MSE objectives."""

    features: Mapping[str, np.ndarray]
    targets: np.ndarray
    weights: np.ndarray
    size: int

    @classmethod
    def from_samples(cls, samples: Sequence[SupervisedSample]) -> "_BatchSamples":
        """Build a batch from arbitrary named scalar state coordinates.

        Lead-time samples use ``x`` plus either a packed ``p``
        sequence or explicit ``p1...pL`` fields.  Inventory model plugins also
        expose age coordinates such as ``a1...aM``.  Discovering scalar fields
        here keeps the vectorized AST layer model-neutral while preserving the
        packed-pipeline compatibility path.
        """

        size = len(samples)
        max_pipeline = 0
        scalar_names: set[str] = {"x", "remaining_horizon"}
        for sample in samples:
            state = sample.state
            if not isinstance(state, Mapping):
                continue
            max_pipeline = max(max_pipeline, len(state.get("p", ())))
            for name, value in state.items():
                text = str(name)
                if text.startswith("p") and text[1:].isdigit():
                    max_pipeline = max(max_pipeline, int(text[1:]))
                if text != "p" and np.isscalar(value):
                    scalar_names.add(text)
        scalar_names.update(f"p{idx}" for idx in range(1, max_pipeline + 1))
        ordered_names = (
            ["x"]
            + sorted(name for name in scalar_names if name not in {"x", "remaining_horizon"})
            + ["remaining_horizon"]
        )
        raw_features: dict[str, list[float]] = {name: [] for name in ordered_names}
        targets: list[float] = []
        weights: list[float] = []
        for sample in samples:
            state = sample.state
            pipeline = list(state.get("p", ())) if isinstance(state, Mapping) else []
            for name in ordered_names:
                default = 1.0 if name == "remaining_horizon" else 0.0
                if name.startswith("p") and name[1:].isdigit():
                    index = int(name[1:]) - 1
                    default = pipeline[index] if index < len(pipeline) else 0.0
                value = state.get(name, default) if isinstance(state, Mapping) else default
                raw_features[name].append(float(value))
            targets.append(float(sample.target_action))
            weights.append(float(sample.weight))
        return cls(
            features={name: np.asarray(values, dtype=float) for name, values in raw_features.items()},
            targets=np.asarray(targets, dtype=float),
            weights=np.asarray(weights, dtype=float),
            size=size,
        )

    @classmethod
    def from_state_array(
        cls,
        state_array: np.ndarray,
        remaining_horizon: float,
        state_names: Sequence[str] | None = None,
    ) -> "_BatchSamples":
        """Build batch features directly from a dense DP state table."""

        size = int(state_array.shape[0])
        width = int(state_array.shape[1])
        names = (
            tuple(state_names)
            if state_names is not None
            else ("x",) + tuple(f"p{idx}" for idx in range(1, width))
        )
        if len(names) != width or len(set(names)) != len(names):
            raise ValueError("state_names must be unique and match state_array width")
        features = {
            name: state_array[:, index].astype(float, copy=False)
            for index, name in enumerate(names)
        }
        features["remaining_horizon"] = np.full(size, float(remaining_horizon), dtype=float)
        return cls(
            features=features,
            targets=np.zeros(size, dtype=float),
            weights=np.ones(size, dtype=float),
            size=size,
        )

    def unique_by_features(self, names: Sequence[str]) -> tuple["_BatchSamples", np.ndarray]:
        """Return unique feature rows and an inverse map to this batch."""

        selected_names = tuple(name for name in names if name in self.features)
        if not selected_names:
            indexes = np.array([0], dtype=int)
            inverse = np.zeros(self.size, dtype=int)
        else:
            matrix = np.column_stack([self.features[name] for name in selected_names])
            _unique, indexes, inverse = np.unique(matrix, axis=0, return_index=True, return_inverse=True)
            indexes = np.asarray(indexes, dtype=int)
            inverse = np.asarray(inverse, dtype=int)
        compact = _BatchSamples(
            features={name: np.asarray(values)[indexes] for name, values in self.features.items()},
            targets=np.asarray(self.targets)[indexes],
            weights=np.asarray(self.weights)[indexes],
            size=len(indexes),
        )
        return compact, inverse


_BATCH_SAMPLE_CACHE: dict[int, tuple[Sequence[SupervisedSample], _BatchSamples]] = {}


def _batch_from_samples_cached(samples: Sequence[SupervisedSample]) -> _BatchSamples:
    """Return a cached vectorized view for a reusable sample sequence."""

    key = id(samples)
    cached = _BATCH_SAMPLE_CACHE.get(key)
    # Keep the source sequence alive and compare by identity.  Caching only
    # ``id(samples)`` and length allowed Python to recycle an id after a case
    # finished, returning (for example) an L=2 feature batch for new L=3
    # samples of the same length.  That silently triggered scalar fallback and
    # could also optimize constants against the wrong states.
    if cached is not None and cached[0] is samples:
        return cached[1]
    batch = _BatchSamples.from_samples(samples)
    if isinstance(samples, (tuple, list)):
        _BATCH_SAMPLE_CACHE[key] = (samples, batch)
    return batch
