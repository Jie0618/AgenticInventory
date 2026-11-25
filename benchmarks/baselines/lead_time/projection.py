"""Exact finite-grid inventory projection for lead-time benchmark baselines."""

from __future__ import annotations

from functools import lru_cache

import numpy as np


class DiscreteInventoryProjector:
    """Project benchmark states through scheduled arrivals and iid demands.

    The transition follows the benchmark's finite-grid convention exactly:
    ending on-hand inventory is clipped to ``n`` after every demand.  The
    returned distribution is the inventory immediately before an order placed
    in the current period arrives, i.e. after ``L`` intervening demands.
    """

    def __init__(self, demand_values: np.ndarray, demand_probs: np.ndarray, n: int) -> None:
        values = np.asarray(demand_values, dtype=np.int64)
        probs = np.asarray(demand_probs, dtype=float)
        if values.ndim != 1 or probs.ndim != 1 or len(values) != len(probs):
            raise ValueError("demand support and probabilities must be aligned vectors")
        if np.any(values < 0) or not np.isclose(float(np.sum(probs)), 1.0):
            raise ValueError("projector requires a normalized nonnegative demand PMF")
        self.values = values
        self.probs = probs
        self.n = int(n)
        grid = np.arange(self.n + 1, dtype=np.int64)
        transitions = np.zeros((self.n + 1, self.n + 1, self.n + 1), dtype=float)
        for arriving in range(self.n + 1):
            ending = np.clip(
                grid[:, None] + arriving - self.values[None, :],
                0,
                self.n,
            )
            for inventory in range(self.n + 1):
                np.add.at(
                    transitions[arriving, inventory],
                    ending[inventory],
                    self.probs,
                )
        self._transitions = transitions
        # MC evaluation revisits the same finite-grid states across candidates
        # and periods.  The former batch path deduplicated only within one
        # period, then recomputed the same L-step projection in the next
        # period.  Keep a bounded exact cache across batch calls.  Each value
        # is immutable and contains only n+1 probabilities.
        # Integer state codes avoid constructing Python tuples for every row
        # in the hot batch path.  The code is lossless because each component
        # is in ``[0, n]`` and uses a base-(n+1) representation.
        self._state_code_base = self.n + 1
        self._state_code_powers: np.ndarray | None = None
        self._batch_distribution_cache: dict[int, np.ndarray] = {}
        self._batch_distribution_cache_limit = 50_000

    @lru_cache(maxsize=50_000)
    def distribution(self, state: tuple[int, ...]) -> np.ndarray:
        if len(state) < 2:
            raise ValueError("positive-lead-time state must contain x and pipeline")
        distribution = np.zeros(self.n + 1, dtype=float)
        distribution[int(state[0])] = 1.0
        inventory_grid = np.arange(self.n + 1, dtype=np.int64)
        for arriving in state[1:]:
            ending = np.clip(
                inventory_grid[:, None] + int(arriving) - self.values[None, :],
                0,
                self.n,
            )
            weights = distribution[:, None] * self.probs[None, :]
            next_distribution = np.bincount(
                ending.ravel(), weights=weights.ravel(), minlength=self.n + 1
            )
            distribution = next_distribution
        distribution /= float(np.sum(distribution))
        distribution.setflags(write=False)
        return distribution

    def batch(self, states: np.ndarray) -> np.ndarray:
        rows = np.asarray(states, dtype=np.int64)
        if rows.ndim != 2 or rows.shape[1] < 2:
            raise ValueError("positive-lead-time states must be a two-dimensional batch")
        if np.any(rows < 0) or np.any(rows > self.n):
            raise ValueError("projector state lies outside [0,n]")

        # ``np.unique(axis=0)`` is considerably more expensive than the same
        # operation on a one-dimensional integer key.  Encode each finite-grid
        # state losslessly and unique the keys instead; this preserves the
        # exact projection arithmetic below while removing repeated sorting and
        # tuple allocation overhead from PIL/FP3's period loop.
        width = rows.shape[1]
        # A base-(n+1) code is faster, but it eventually exceeds int64 as L
        # grows.  Fall back to exact row-wise deduplication rather than
        # allowing numpy's int64 conversion to overflow and abort evaluation.
        max_code = self._state_code_base ** (width - 1) * self.n
        if max_code <= np.iinfo(np.int64).max:
            if self._state_code_powers is None or len(self._state_code_powers) < width:
                self._state_code_powers = np.asarray(
                    [self._state_code_base**i for i in range(width)], dtype=np.int64
                )
            powers = self._state_code_powers[:width]
            codes = rows @ powers
            unique_codes, inverse = np.unique(codes, return_inverse=True)
            cache_key = lambda position: int(unique_codes[position])
        else:
            unique_rows, inverse = np.unique(rows, axis=0, return_inverse=True)
            unique_codes = np.arange(len(unique_rows), dtype=np.int64)
            cache_key = lambda position: tuple(int(value) for value in unique_rows[position])
        # Recover one representative row for every unique code without a
        # second axis-wise unique.  ``np.unique`` returns sorted codes, so use
        # inverse labels to place first occurrences directly.
        first_indices = np.full(len(unique_codes), len(rows), dtype=np.int64)
        np.minimum.at(first_indices, inverse, np.arange(len(rows), dtype=np.int64))
        unique_rows = rows[first_indices]

        cached_output = np.empty((len(unique_rows), self.n + 1), dtype=float)
        missing_positions: list[int] = []
        missing_rows: list[np.ndarray] = []
        for position, row in enumerate(unique_rows):
            key = cache_key(position)
            cached = self._batch_distribution_cache.get(key)
            if cached is None:
                missing_positions.append(position)
                missing_rows.append(row)
            else:
                cached_output[position] = cached

        if not missing_rows:
            return cached_output[inverse]

        rows_to_project = np.asarray(missing_rows, dtype=np.int64)

        # Each scheduled arrival selects one of only n+1 transition matrices.
        # Grouping rows by that arrival turns millions of Python-level scalar
        # recursions into at most L*(n+1) small dense matrix products. Candidate
        # policies frequently visit identical states, so project each unique
        # state once before restoring the caller's row order.
        distribution = np.zeros((len(rows_to_project), self.n + 1), dtype=float)
        distribution[np.arange(len(rows_to_project)), rows_to_project[:, 0]] = 1.0
        for column in range(1, rows_to_project.shape[1]):
            arriving = rows_to_project[:, column]
            next_distribution = np.empty_like(distribution)
            for order in np.unique(arriving):
                mask = arriving == order
                next_distribution[mask] = (
                    distribution[mask] @ self._transitions[int(order)]
                )
            distribution = next_distribution
        distribution /= np.sum(distribution, axis=1, keepdims=True)
        for source_position, unique_position in enumerate(missing_positions):
            value = distribution[source_position].copy()
            value.setflags(write=False)
            cached_output[unique_position] = value
            if len(self._batch_distribution_cache) < self._batch_distribution_cache_limit:
                key = cache_key(unique_position)
                self._batch_distribution_cache[key] = value
        return cached_output[inverse]

    def one_period_batch(
        self, on_hand: np.ndarray, arriving: np.ndarray
    ) -> np.ndarray:
        """Return next-inventory PMFs for aligned on-hand/arrival vectors."""

        inventory = np.asarray(on_hand, dtype=np.int64)
        arrivals = np.asarray(arriving, dtype=np.int64)
        if inventory.shape != arrivals.shape or inventory.ndim != 1:
            raise ValueError("on_hand and arriving must be aligned vectors")
        if (
            np.any(inventory < 0)
            or np.any(inventory > self.n)
            or np.any(arrivals < 0)
            or np.any(arrivals > self.n)
        ):
            raise ValueError("one-period projector input lies outside [0,n]")
        return self._transitions[arrivals, inventory]

    def batch_transition_products(self, arrivals: np.ndarray) -> np.ndarray:
        """Compose scheduled-arrival kernels for every row in one batch.

        ``output[path, x, r]`` is the probability of ending at ``r`` from
        starting inventory ``x`` after processing that path's arrival row.
        """

        rows = np.asarray(arrivals, dtype=np.int64)
        if rows.ndim != 2:
            raise ValueError("arrivals must be a two-dimensional batch")
        if np.any(rows < 0) or np.any(rows > self.n):
            raise ValueError("scheduled arrival lies outside [0,n]")
        paths = len(rows)
        identity = np.eye(self.n + 1, dtype=float)
        products = np.broadcast_to(identity, (paths, self.n + 1, self.n + 1)).copy()
        for column in range(rows.shape[1]):
            arriving = rows[:, column]
            next_products = np.empty_like(products)
            for order in np.unique(arriving):
                mask = arriving == order
                next_products[mask] = products[mask] @ self._transitions[int(order)]
            products = next_products
        return products

    def cost_after_each_arrival(self, cost_matrix: np.ndarray) -> np.ndarray:
        """Compose every possible final arrival with a downstream cost table."""

        costs = np.asarray(cost_matrix, dtype=float)
        if costs.ndim != 2 or costs.shape[0] != self.n + 1:
            raise ValueError("cost_matrix must have n+1 inventory rows")
        return self._transitions @ costs


__all__ = ["DiscreteInventoryProjector"]
