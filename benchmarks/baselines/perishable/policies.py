"""Exact finite-support expectations for four published balancing rules.

Track today's order as a tagged cohort. Future orders are fresher and therefore
cannot affect this cohort's FIFO consumption before it expires. Dynamic
probability propagation avoids enumerating exponentially many demand paths.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Mapping, Sequence

import numpy as np

from inventory.core import InventoryCase, InventoryState


NAMES = ("pb_chao", "db_chao", "b_zaw", "tb_zaw")


class PerishableBalancingPolicy:
    """Residual-preserving randomized integer variants for FIFO lost sales.

    Randomness is supplied by evaluators, never hidden inside cached actions.
    Only the accelerated randomized implementation is available in production.
    DB's critical level is unchanged; tagged costs stop at the finite horizon.
    This version does not claim the source papers' approximation guarantees.
    """

    def __init__(self, case: InventoryCase, name: str) -> None:
        if name not in NAMES:
            raise ValueError(f"unknown balancing policy: {name}")
        if case.composition_key != "perishability" or case.shelf_life is None:
            raise ValueError("balancing policies require pure fixed-shelf-life perishability")
        if case.config.perishability.shelf_life is None:
            raise ValueError("random shelf life is not supported")
        if case.shelf_life < 2:
            raise ValueError("balancing formulas require shelf life M >= 2")
        if (2 * (case.shelf_life - 1) * case.h + case.w) <= 0:
            raise ValueError("PB balancing coefficient requires positive holding or expiry cost")
        self.case = case
        self.name = name
        self.m = case.shelf_life
        self.n = case.n
        self.h, self.b, self.w = case.h, case.b, case.w
        self.demands = tuple(
            (int(d), float(p)) for d, p in zip(case.demand.values, case.demand.probs)
            if p > 0
        )
        self._critical_level = self._critical_stock_level()
        # Instance-owned caches: workers must not retain previous cases.
        self._expected_terms = lru_cache(maxsize=8192)(self._expected_terms)
        self.action_interval = lru_cache(maxsize=4096)(self.action_interval)
        self._batch_intervals = {}

    def _critical_stock_level(self) -> int:
        # The continuous F^{-1}(b/(b+h)) identity in the source document is
        # not the solution of this holding/shortage *cost-balancing* equation.
        max_demand = max(d for d, _ in self.demands)
        for stock in range(max_demand + 1):
            holding = self.h * sum(p * max(stock - d, 0) for d, p in self.demands)
            shortage = self.b * sum(p * max(d - stock, 0) for d, p in self.demands)
            if holding >= shortage - 1e-12:
                return stock
        return max_demand

    def _expected_terms(self, ages: tuple[int, ...], q: int,
                        remaining_horizon: int | None = None) -> tuple[float, float, float, float]:
        """Return (shortage, tagged holding, tagged expiry, current holding)."""
        total = sum(ages)
        shortage = self.b * sum(p * max(d - total - q, 0) for d, p in self.demands)
        current_holding = self.h * sum(p * max(total + q - d, 0) for d, p in self.demands)
        if q == 0:
            return shortage, 0.0, 0.0, current_holding
        # At time k all surviving old items form a FIFO suffix of ages[k:].
        # Given total stock s, tagged stock is min(s,q). Thus total alone is
        # sufficient: no branching over age vectors is necessary. Zero stock
        # is an absorbing state with zero tagged cost and can be omitted.
        size = total + q
        pmf = np.zeros(size + 1)
        for d, p in self.demands:
            if d <= size:
                pmf[d] += p
        distribution = np.zeros(size + 1)
        distribution[size] = 1.0
        stock = np.arange(size + 1)
        tagged = np.minimum(stock, q)
        tails = np.cumsum(np.asarray(ages[::-1], dtype=int))[::-1]
        tagged_holding = 0.0
        tagged_expiry = 0.0
        periods = self.m if remaining_horizon is None else min(self.m, remaining_horizon)
        for period in range(periods):
            remaining = np.correlate(distribution, pmf, mode="full")[size:]
            remaining[0] = 0.0
            expected_tagged = float(remaining @ tagged)
            tagged_holding += self.h * expected_tagged
            if period == self.m - 1:
                tagged_expiry = self.w * expected_tagged
                break
            future_old = int(tails[period + 1]) if period + 1 < len(tails) else 0
            after_expiry = stock - np.maximum(stock - q - future_old, 0)
            distribution = np.bincount(after_expiry, weights=remaining, minlength=size + 1)
            if not np.any(distribution):
                break
        return shortage, tagged_holding, tagged_expiry, current_holding

    def action_interval(self, ages: tuple[int, ...],
                        remaining_horizon: int | None = None) -> tuple[int, int, float]:
        """Adjacent feasible integers and P(upper), preserving mean residual.

        A versioned discrete variant, not an inherited approximation guarantee.
        Only holding and expiry incurred within the remaining horizon count.
        """
        periods = self.m if remaining_horizon is None else min(self.m, remaining_horizon)
        if periods < 1:
            raise ValueError("remaining horizon must be positive")

        def terms(q):
            if periods == self.m:
                return self._expected_terms(ages, q)
            return self._expected_terms(ages, q, periods)

        upper = self.n
        if self.name == "db_chao":
            upper = min(upper, max(0, self._critical_level - sum(ages)))
        lower = 0
        if self.name == "tb_zaw":
            lower = min(range(upper + 1), key=lambda q: (sum(terms(q)[:3]), q))
        beta = (self.m*self.h+self.w)/(2*(self.m-1)*self.h+self.w)

        def residual(q):
            p, h, w, current = terms(q)
            return (beta*(h+w) if self.name == "pb_chao" else current+w if self.name == "db_chao" else h+w)-p

        hi = lower
        rh = residual(hi)
        if rh >= -1e-12:
            return hi, hi, 0.0
        while hi < upper:
            rl = rh
            hi += 1
            rh = residual(hi)
            if rh >= -1e-12:
                return hi-1, hi, float(np.clip(-rl/(rh-rl), 0, 1))
        return upper, upper, 0.0

    def action_distributions(self, states, params, remaining_horizon):
        if params:
            raise ValueError("these balancing baselines have no fitted parameters")
        result = []
        for state in states:
            self.case.schema.validate(state)
            lo, hi, p = self.action_interval(tuple(state.inventory[:-1]), min(self.m, remaining_horizon))
            result.append(((lo, 1-p), (hi, p)) if hi != lo else ((lo, 1.0),))
        return result

    def batch_action_intervals(self, inventory, params, remaining_horizon):
        """Resolve each distinct age vector once for a batch of MC paths."""
        if params:
            raise ValueError("these balancing baselines have no fitted parameters")
        inventory = np.asarray(inventory)
        if (inventory.ndim != 2 or inventory.shape[1] != self.m
                or not np.issubdtype(inventory.dtype, np.integer)
                or np.any(inventory < 0)):
            raise ValueError("invalid batch inventory")
        # Byte keys avoid sorting M separate structured fields for long lives.
        ages = np.ascontiguousarray(inventory[:, :-1])
        keys = ages.view(np.dtype((np.void, ages.dtype.itemsize * ages.shape[1]))).ravel()
        _, indices, inverse = np.unique(keys, return_index=True, return_inverse=True)
        ages = ages[indices]
        periods = min(self.m, remaining_horizon)
        if periods < 1:
            raise ValueError("remaining horizon must be positive")
        cache_keys = [(periods, row.tobytes()) for row in ages]
        missing = [i for i, key in enumerate(cache_keys) if key not in self._batch_intervals]
        intervals = np.empty((len(ages), 3))
        for index in set(range(len(ages))) - set(missing):
            intervals[index] = self._batch_intervals[cache_keys[index]]
        # Bound working memory; all age vectors and candidate quantities inside
        # each chunk share one finite-support propagation, never scalar solves.
        for start in range(0, len(missing), 128):
            indices = missing[start:start+128]
            values = self._action_intervals_many(ages[indices], periods)
            intervals[indices] = values
            for index, value in zip(indices, values):
                if len(self._batch_intervals) >= 8192:
                    self._batch_intervals.pop(next(iter(self._batch_intervals)))
                self._batch_intervals[cache_keys[index]] = value
        return (intervals[inverse, 0].astype(np.int64),
                intervals[inverse, 1].astype(np.int64), intervals[inverse, 2])

    def _action_intervals_many(self, ages, periods):
        """Same integer balance roots, vectorized over states and all q."""
        count = len(ages)
        active = ages.sum(axis=1) < max(d for d, _ in self.demands)
        if not np.all(active):
            # If current stock covers every demand outcome, q=0 already
            # balances; all tagged marginal costs and shortage are zero.
            result = np.zeros((count, 3))
            if np.any(active):
                result[active] = self._action_intervals_many(ages[active], periods)
            return result
        total = ages.sum(axis=1)
        # No minimizer needs stock above max demand: shortage is already zero
        # there and tagged holding/expiry cannot improve by ordering more.
        bounds = np.minimum(self.n, max(d for d, _ in self.demands)-total)
        if self.name == "db_chao":
            bounds = np.minimum(bounds, np.maximum(self._critical_level-total, 0))
        quantities = np.arange(int(bounds.max()) + 1)
        width = len(quantities)
        sizes = (total[:, None] + quantities).reshape(-1)
        q = np.tile(quantities, count)
        stock = np.arange(int(sizes.max()) + 1)
        demand = np.asarray([d for d, _ in self.demands])
        probabilities = np.asarray([p for _, p in self.demands])
        shortage = self.b * (np.maximum(demand - sizes[:, None], 0) @ probabilities)
        current = self.h * (np.maximum(sizes[:, None] - demand, 0) @ probabilities)
        distribution = np.zeros((len(sizes), len(stock)))
        row = np.arange(len(sizes))
        distribution[row, sizes] = 1.
        tagged = np.minimum(stock, q[:, None])
        tails = np.cumsum(ages[:, ::-1], axis=1)[:, ::-1]
        holding = np.zeros(len(sizes))
        expiry = np.zeros(len(sizes))
        for period in range(periods):
            remaining = np.zeros_like(distribution)
            for d, probability in self.demands:
                if d < len(stock):
                    remaining[:, :len(stock)-d] += probability * distribution[:, d:]
            remaining[:, 0] = 0.
            expected = np.sum(remaining * tagged, axis=1)
            holding += self.h * expected
            if period == self.m - 1:
                expiry = self.w * expected
                break
            future_old = np.repeat(tails[:, period+1], width) if period+1 < ages.shape[1] else 0
            boundary = q + future_old
            above = stock > boundary[:, None]
            distribution = np.where(above, 0., remaining)
            distribution[row, boundary] += np.sum(np.where(above, remaining, 0.), axis=1)
        shape = (count, width)
        shortage, holding, expiry, current = (v.reshape(shape) for v in (shortage, holding, expiry, current))
        upper = bounds
        lower = (np.argmin(np.where(quantities <= upper[:, None], shortage+holding+expiry, np.inf), axis=1)
                 if self.name == "tb_zaw" else np.zeros(count, dtype=int))
        beta = (self.m*self.h+self.w)/(2*(self.m-1)*self.h+self.w)
        residual = (beta*(holding+expiry) if self.name == "pb_chao" else
                    current+expiry if self.name == "db_chao" else holding+expiry) - shortage
        crossed = ((residual >= -1e-12) & (quantities >= lower[:, None])
                   & (quantities <= upper[:, None]))
        has_crossing = crossed.any(axis=1)
        hi = np.where(has_crossing, np.argmax(crossed, axis=1), upper)
        lo = np.where(has_crossing, np.maximum(hi-1, lower), upper)
        rows = np.arange(count)
        probability = np.zeros(count)
        interpolate = hi != lo
        probability[interpolate] = np.clip(
            -residual[rows[interpolate], lo[interpolate]] /
            (residual[rows[interpolate], hi[interpolate]]-residual[rows[interpolate], lo[interpolate]]), 0, 1)
        return np.column_stack((lo, hi, probability))

    def actions(
        self,
        states: Sequence[InventoryState],
        params: Mapping[str, float],
        remaining_horizon: int,
    ) -> np.ndarray:
        raise ValueError("randomized policy requires action_distributions and explicit sampling")


__all__ = ["NAMES", "PerishableBalancingPolicy"]
