"""Model-independent policy records and fixed demand banks for final MC."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from optimization.policies.policy_sets_json import ResolvedPolicy


@dataclass(frozen=True)
class MCPolicy(ResolvedPolicy):
    computational_spec: object = None
    execution: dict | None = None


@dataclass(frozen=True)
class MCDemandBank:
    demand_indices: np.ndarray
    demand_values: np.ndarray
    num_paths: int
    horizon: int
    seed: int
    policy_uniforms: np.ndarray | None = None

    @classmethod
    def create(cls, demand, horizon, paths, seed, *, randomized_policies=False):
        if paths <= 0 or horizon <= 0:
            raise ValueError("paths and horizon must be positive")
        values = np.asarray(demand.values, dtype=np.int32)
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(values), size=(paths, horizon), p=np.asarray(demand.probs, dtype=float)).astype(np.int32)
        uniforms = rng.random((paths, horizon)) if randomized_policies else None
        return cls(indices, values[indices], paths, horizon, seed, uniforms)

    def prefix(self, paths, horizon):
        if not (0 < paths <= self.num_paths and 0 < horizon <= self.horizon):
            raise ValueError("prefix must lie within the demand bank")
        uniforms = None if self.policy_uniforms is None else self.policy_uniforms[:paths, :horizon]
        return MCDemandBank(self.demand_indices[:paths, :horizon], self.demand_values[:paths, :horizon], paths, horizon, self.seed, uniforms)
