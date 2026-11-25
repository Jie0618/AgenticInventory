"""Shared discrete demand distributions for inventory models."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np

from inventory.common.numeric import integer_scalar, integer_tuple


@dataclass(frozen=True)
class DiscreteDemand:
    """Finite-support discrete demand distribution.

    DP in this project enumerates demand values exactly, so the demand object is
    intentionally finite and explicit:

    - `values[k]` is one possible demand realization.
    - `probs[k]` is its probability.

    Values are restricted to nonnegative integers for now because the inventory
    DP state grid is integer-valued.
    """

    values: np.ndarray
    probs: np.ndarray
    name: str = "custom"
    params: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate shape/probabilities and store normalized numpy arrays."""

        raw_probs = np.asarray(self.probs)
        normalized_values = integer_tuple(
            self.values,
            name="Demand values",
            minimum=0,
            allow_empty=False,
        )
        if raw_probs.ndim != 1:
            raise ValueError("Demand values and probabilities must be one-dimensional.")
        if len(normalized_values) != len(raw_probs):
            raise ValueError("Demand values and probabilities must have the same length.")
        values = np.asarray(normalized_values, dtype=int)
        try:
            probs = np.asarray(raw_probs, dtype=float)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("Demand probabilities must be numeric.") from error
        if not np.all(np.isfinite(probs)):
            raise ValueError("Demand probabilities must be finite.")
        if np.any(probs < 0):
            raise ValueError("Demand probabilities must be nonnegative.")

        total_mass = float(np.sum(probs))
        if total_mass <= 0.0:
            raise ValueError("Demand distribution must have positive probability mass.")

        order = np.argsort(values)
        sorted_values = values[order]
        sorted_probs = probs[order] / total_mass

        # Merge duplicate demand values so every support point appears once.
        unique_values, inverse = np.unique(sorted_values, return_inverse=True)
        merged_probs = np.zeros(len(unique_values), dtype=float)
        np.add.at(merged_probs, inverse, sorted_probs)

        object.__setattr__(self, "values", unique_values)
        object.__setattr__(self, "probs", merged_probs / float(np.sum(merged_probs)))
        object.__setattr__(self, "params", dict(self.params))

    @classmethod
    def from_pmf(cls, pmf: Mapping[int, float], name: str = "custom") -> "DiscreteDemand":
        """Build a distribution from `{demand_value: probability}`."""

        return cls(list(pmf.keys()), list(pmf.values()), name=name)

    @classmethod
    def poisson(cls, lam: float, d_max: int, tail_to_max: bool = True) -> "DiscreteDemand":
        """Build a truncated Poisson distribution on `{0, ..., d_max}`.

        If `tail_to_max=True`, all omitted tail mass `P(D>d_max)` is added to
        the last support point.  That keeps total probability exactly one while
        preserving the finite grid required by DP.
        """

        if lam <= 0.0:
            raise ValueError("Poisson lambda must be positive.")
        d_max = integer_scalar(d_max, name="d_max", minimum=0)

        values = np.arange(d_max + 1, dtype=int)
        probs = np.empty(d_max + 1, dtype=float)
        probs[0] = math.exp(-lam)
        for d in range(1, d_max + 1):
            probs[d] = probs[d - 1] * lam / d
        if tail_to_max:
            probs[-1] += max(0.0, 1.0 - float(np.sum(probs)))

        return cls(values, probs, name="poisson", params={"lam": float(lam), "d_max": float(d_max)})

    @classmethod
    def geometric(cls, p: float, d_max: int, tail_to_max: bool = True) -> "DiscreteDemand":
        """Build a truncated geometric distribution on `{0, ..., d_max}`.

        Parameterization:

            P(D=k) = p * (1-p)^k,  k = 0,1,2,...

        so the untruncated mean is `(1-p)/p`.  If `tail_to_max=True`, omitted
        tail mass `P(D>d_max)` is added to `d_max`, matching the finite-support
        convention used by the Poisson builder.
        """

        if not 0.0 < p <= 1.0:
            raise ValueError("Geometric p must be in (0, 1].")
        d_max = integer_scalar(d_max, name="d_max", minimum=0)

        values = np.arange(d_max + 1, dtype=int)
        probs = p * np.power(1.0 - p, values.astype(float))
        if tail_to_max:
            probs[-1] += max(0.0, 1.0 - float(np.sum(probs)))

        return cls(values, probs, name="geometric", params={"p": float(p), "d_max": float(d_max)})

    @classmethod
    def uniform(cls, low: int, high: int) -> "DiscreteDemand":
        """Build a discrete uniform distribution on `{low, ..., high}`."""

        low = integer_scalar(low, name="low", minimum=0)
        high = integer_scalar(high, name="high", minimum=0)
        if high < low:
            raise ValueError("Uniform support must satisfy 0 <= low <= high.")
        values = np.arange(low, high + 1, dtype=int)
        probs = np.ones(len(values), dtype=float) / len(values)
        return cls(values, probs, name="uniform", params={"low": float(low), "high": float(high)})

    @property
    def mean(self) -> float:
        """Return `E[D]`."""

        return float(np.dot(self.values, self.probs))

    @property
    def variance(self) -> float:
        """Return `Var(D)`."""

        centered = self.values.astype(float) - self.mean
        return float(np.dot(centered * centered, self.probs))

    @property
    def std(self) -> float:
        """Return the standard deviation of demand."""

        return float(math.sqrt(max(0.0, self.variance)))

    @property
    def support_min(self) -> int:
        """Return the smallest demand support point."""

        return int(self.values[0])

    @property
    def support_max(self) -> int:
        """Return the largest demand support point."""

        return int(self.values[-1])

    def quantile(self, level: float) -> int:
        """Return the left-continuous discrete quantile.

        This is the smallest demand value `d` such that `P(D <= d) >= level`.
        """

        if not 0.0 <= level <= 1.0:
            raise ValueError("Quantile level must be in [0, 1].")
        cdf = np.cumsum(self.probs)
        idx = int(np.searchsorted(cdf, level, side="left"))
        idx = min(idx, len(self.values) - 1)
        return int(self.values[idx])

    def expect(self, fn: Callable[[np.ndarray], np.ndarray | float]) -> float:
        """Compute an exact expectation over the finite support.

        `fn` can be vectorized over the full numpy array of support values, or it
        can return a scalar.  The result is always converted to `float`.
        """

        values = np.asarray(fn(self.values), dtype=float)
        if values.ndim == 0:
            return float(values)
        if values.shape != self.values.shape:
            raise ValueError("Expectation function must return a scalar or one value per demand point.")
        return float(np.dot(values, self.probs))

    def pmf_dict(self) -> dict[int, float]:
        """Return the probability mass function as a plain dictionary."""

        return {int(d): float(p) for d, p in zip(self.values, self.probs)}

    def feature_dict(self) -> dict[str, object]:
        """Return distribution features consumed by formulas and reports."""

        mu = self.mean
        sigma = self.std
        return {
            "values": self.values.astype(int).tolist(),
            "probs": self.probs.astype(float).tolist(),
            "mu": mu,
            "sigma": sigma,
            "variance": self.variance,
            "cv": sigma / mu if mu > 0.0 else 0.0,
            "q50": float(self.quantile(0.50)),
            "q80": float(self.quantile(0.80)),
            "q90": float(self.quantile(0.90)),
            "q95": float(self.quantile(0.95)),
            "support_min": float(self.support_min),
            "support_max": float(self.support_max),
            "name": self.name,
            "params": dict(self.params),
        }
