"""Paper-based computational policies for deterministic lead-time lost sales."""

from __future__ import annotations

from typing import Mapping, Protocol

import numpy as np

from benchmarks.baselines.lead_time.projection import (
    DiscreteInventoryProjector,
)


class ComputationalActionProvider(Protocol):
    horizon_dependent: bool

    def actions(
        self,
        states: np.ndarray,
        params: Mapping[str, float],
        remaining_horizon: int,
    ) -> np.ndarray:
        ...


class RandomizedActionProvider(ComputationalActionProvider, Protocol):
    def action_distribution(
        self,
        states: np.ndarray,
        params: Mapping[str, float],
        remaining_horizon: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return lower actions, upper actions, and upper-action probabilities."""

        ...


class ProjectionPolicyBase:
    horizon_dependent = False

    def __init__(
        self,
        demand_values: np.ndarray,
        demand_probs: np.ndarray,
        n: int,
        lead_time: int,
        h: float,
        b: float,
    ) -> None:
        self.values = np.asarray(demand_values, dtype=np.int64)
        self.probs = np.asarray(demand_probs, dtype=float)
        self.n = int(n)
        self.lead_time = int(lead_time)
        self.h = float(h)
        self.b = float(b)
        self.projector = DiscreteInventoryProjector(self.values, self.probs, self.n)
        self._inventory = np.arange(self.n + 1, dtype=float)
        self._orders = np.arange(self.n + 1, dtype=float)

    def _project(self, states: np.ndarray) -> np.ndarray:
        rows = np.asarray(states, dtype=np.int64)
        if rows.ndim != 2 or rows.shape[1] != self.lead_time + 1:
            raise ValueError("state batch has the wrong lead-time dimension")
        return self.projector.batch(rows)


class PILPolicy(ProjectionPolicyBase):
    """Projected Inventory Level policy of van Jaarsveld and Arts (2024)."""

    optimization_route_hint = "pil_scalar"

    def actions(
        self,
        states: np.ndarray,
        params: Mapping[str, float],
        remaining_horizon: int,
    ) -> np.ndarray:
        del remaining_horizon
        projected = self._project(states)
        expected_inventory = projected @ self._inventory
        raw = np.maximum(float(params["U"]) - expected_inventory, 0.0)
        return np.clip(np.rint(raw), 0, self.n).astype(np.int32)

    def actions_many(
        self,
        states: np.ndarray,
        params_batch: tuple[Mapping[str, float], ...],
        remaining_horizon: int,
    ) -> np.ndarray:
        """Vectorize PIL over candidate and trajectory dimensions."""

        del remaining_horizon
        rows = np.asarray(states, dtype=np.int64)
        candidates, paths, width = rows.shape
        projected = self._project(rows.reshape(candidates * paths, width))
        expected_inventory = (projected @ self._inventory).reshape(candidates, paths)
        targets = np.asarray(
            [float(params["U"]) for params in params_batch], dtype=float
        )[:, None]
        raw = np.maximum(targets - expected_inventory, 0.0)
        return np.clip(np.rint(raw), 0, self.n).astype(np.int32)


class ProjectedMeanVarianceCappedPolicy(ProjectionPolicyBase):
    """Baek's projected mean-variance capped lost-sales policy."""

    def _actions_from_projected(
        self,
        projected: np.ndarray,
        target: np.ndarray,
        gamma: np.ndarray,
        cap: np.ndarray,
    ) -> np.ndarray:
        mean = projected @ self._inventory
        second_moment = projected @ np.square(self._inventory)
        variance = np.maximum(second_moment - np.square(mean), 0.0)
        raw = np.minimum(
            cap,
            np.maximum(target - mean - gamma * np.sqrt(variance), 0.0),
        )
        return np.clip(np.rint(raw), 0, self.n).astype(np.int32)

    def actions(
        self,
        states: np.ndarray,
        params: Mapping[str, float],
        remaining_horizon: int,
    ) -> np.ndarray:
        del remaining_horizon
        projected = self._project(states)
        return self._actions_from_projected(
            projected,
            np.asarray(float(params["T"])),
            np.asarray(float(params["gamma"])),
            np.asarray(float(params["C"])),
        )

    def actions_many(
        self,
        states: np.ndarray,
        params_batch: tuple[Mapping[str, float], ...],
        remaining_horizon: int,
    ) -> np.ndarray:
        del remaining_horizon
        rows = np.asarray(states, dtype=np.int64)
        candidates, paths, width = rows.shape
        projected = self._project(rows.reshape(candidates * paths, width))
        target = np.repeat(
            np.asarray([float(params["T"]) for params in params_batch]), paths
        )
        gamma = np.repeat(
            np.asarray([float(params["gamma"]) for params in params_batch]), paths
        )
        cap = np.repeat(
            np.asarray([float(params["C"]) for params in params_batch]), paths
        )
        return self._actions_from_projected(projected, target, gamma, cap).reshape(
            candidates, paths
        )


class ArrivalMyopicPolicy(ProjectionPolicyBase):
    """Morton/Zipkin myopic policy minimizing cost in the arrival period."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        available = self._inventory[:, None, None] + self._orders[None, :, None]
        demand = self.values[None, None, :].astype(float)
        self._conditional_arrival_cost = np.sum(
            self.probs[None, None, :]
            * (
                self.h * np.maximum(available - demand, 0.0)
                + self.b * np.maximum(demand - available, 0.0)
            ),
            axis=2,
        )

    def actions(
        self,
        states: np.ndarray,
        params: Mapping[str, float],
        remaining_horizon: int,
    ) -> np.ndarray:
        del params, remaining_horizon
        projected = self._project(states)
        # cost[r, q] is the expected arrival-period cost conditional on the
        # pre-arrival inventory r.  np.argmin implements the smallest-q tie.
        costs = projected @ self._conditional_arrival_cost
        return np.argmin(costs, axis=1).astype(np.int32)


class Myopic2Policy(ProjectionPolicyBase):
    """Zipkin's arrival-aligned two-stage Myopic-2 policy.

    An order ``q`` placed in the current period first affects cost when it
    arrives after ``L`` periods.  After the current demand is observed, the
    next order ``q'`` is selected and first affects the following period.  The
    policy minimizes exactly those two affected-period costs, with ``q'``
    allowed to depend on the current demand but not on later demands.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        available = self._inventory[:, None, None] + self._orders[None, :, None]
        demand = self.values[None, None, :].astype(float)
        self._conditional_arrival_cost = np.sum(
            self.probs[None, None, :]
            * (
                self.h * np.maximum(available - demand, 0.0)
                + self.b * np.maximum(demand - available, 0.0)
            ),
            axis=2,
        )
        # If today's order is q, this table maps inventory immediately after
        # the fixed pipeline to the arrival-cost vector for tomorrow's order.
        self._cost_after_final_arrival = self.projector.cost_after_each_arrival(
            self._conditional_arrival_cost
        )
        self._action_cache: dict[tuple[int, ...], int] = {}
        self._action_cache_limit = 200_000

    def actions(
        self,
        states: np.ndarray,
        params: Mapping[str, float],
        remaining_horizon: int,
    ) -> np.ndarray:
        del params, remaining_horizon
        rows = np.asarray(states, dtype=np.int64)
        unique_rows, inverse = np.unique(rows, axis=0, return_inverse=True)
        unique_actions = np.empty(len(unique_rows), dtype=np.int32)
        missing_indexes: list[int] = []
        for index, row in enumerate(unique_rows):
            key = tuple(int(value) for value in row)
            cached = self._action_cache.get(key)
            if cached is None:
                missing_indexes.append(index)
            else:
                unique_actions[index] = cached
        if missing_indexes:
            missing = unique_rows[np.asarray(missing_indexes, dtype=int)]
            computed = self._actions_uncached(missing)
            if len(self._action_cache) + len(missing) > self._action_cache_limit:
                self._action_cache.clear()
            for index, row, action in zip(missing_indexes, missing, computed):
                unique_actions[index] = action
                self._action_cache[tuple(int(value) for value in row)] = int(action)
        return unique_actions[inverse]

    def _actions_uncached(self, rows: np.ndarray) -> np.ndarray:
        projected = self._project(rows)
        objective = projected @ self._conditional_arrival_cost
        next_x_distribution = self.projector.one_period_batch(rows[:, 0], rows[:, 1])
        # The expensive p2,...,pL projection is independent of today's order
        # and is therefore composed once.  Appending q is represented by the
        # precomputed T_q @ arrival-cost matrix above.
        pipeline_product = self.projector.batch_transition_products(rows[:, 2:])
        followup = np.zeros_like(objective)
        for next_x in range(self.n + 1):
            probabilities = next_x_distribution[:, next_x]
            active = probabilities > 0.0
            if not np.any(active):
                continue
            # Only inventory values that current demand can actually produce
            # are evaluated. This is exact and avoids materializing the much
            # larger path x starting-inventory x q x q' tensor.
            costs = np.einsum(
                "pr,qra->pqa",
                pipeline_product[active, next_x, :],
                self._cost_after_final_arrival,
                optimize=True,
            )
            followup[active] += probabilities[active, None] * np.min(costs, axis=2)
        objective += followup
        return np.argmin(objective, axis=1).astype(np.int32)


class FP3Policy(ProjectionPolicyBase):
    """Exact-discrete fixed non-stockout-probability policy.

    The paper writes the event as strictly positive ending inventory under its
    continuous-demand assumption.  For discrete demand, equality can carry
    substantial mass, so the faithful no-lost-sale extension is ``D <= r+q``.
    The two events coincide for continuous demand.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        available = self._inventory[:, None, None] + self._orders[None, :, None]
        self._service_given_r_q = np.sum(
            self.probs[None, None, :]
            * (self.values[None, None, :] <= available),
            axis=2,
        )

    def actions(
        self,
        states: np.ndarray,
        params: Mapping[str, float],
        remaining_horizon: int,
    ) -> np.ndarray:
        del remaining_horizon
        target = float(params["P3"])
        projected = self._project(states)
        # Under discrete demand, no stockout/lost sale includes exact depletion.
        service = projected @ self._service_given_r_q
        meets = service >= target - 1.0e-12
        first = np.argmax(meets, axis=1)
        return np.where(np.any(meets, axis=1), first, self.n).astype(np.int32)

    def actions_many(
        self,
        states: np.ndarray,
        params_batch: tuple[Mapping[str, float], ...],
        remaining_horizon: int,
    ) -> np.ndarray:
        """Vectorize FP3 over candidate and trajectory dimensions."""

        del remaining_horizon
        rows = np.asarray(states, dtype=np.int64)
        candidates, paths, width = rows.shape
        projected = self._project(rows.reshape(candidates * paths, width))
        service = (projected @ self._service_given_r_q).reshape(
            candidates, paths, self.n + 1
        )
        targets = np.asarray(
            [float(params["P3"]) for params in params_batch], dtype=float
        )[:, None, None]
        meets = service >= targets - 1.0e-12
        first = np.argmax(meets, axis=2)
        return np.where(np.any(meets, axis=2), first, self.n).astype(np.int32)


class RandomizedDualBalancingPolicy(ProjectionPolicyBase):
    """Integer Dual-Balancing policy of Levi et al. with exact randomization."""

    horizon_dependent = True

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        max_available = 2 * self.n
        self._cumulative_cap = max_available + 1
        one_period = np.zeros(self._cumulative_cap + 1, dtype=float)
        clipped = np.minimum(self.values, self._cumulative_cap)
        np.add.at(one_period, clipped, self.probs)
        self._one_period_capped = one_period
        initial_cumulative = np.zeros(self._cumulative_cap + 1, dtype=float)
        initial_cumulative[0] = 1.0
        self._cumulative_pmfs: list[np.ndarray] = [initial_cumulative]
        self._holding_prefix: list[np.ndarray] = [
            np.zeros((self.n + 1, self.n + 1), dtype=float)
        ]

        available = self._inventory[:, None, None] + self._orders[None, :, None]
        demand = self.values[None, None, :].astype(float)
        self._arrival_penalty = self.b * np.sum(
            self.probs[None, None, :] * np.maximum(demand - available, 0.0),
            axis=2,
        )

    def _ensure_holding_horizon(self, periods: int) -> None:
        while len(self._holding_prefix) <= periods:
            previous = self._cumulative_pmfs[-1]
            next_pmf = np.zeros_like(previous)
            for current, current_probability in enumerate(previous):
                if current_probability == 0.0:
                    continue
                targets = np.minimum(
                    current + np.arange(len(self._one_period_capped)),
                    self._cumulative_cap,
                )
                np.add.at(
                    next_pmf,
                    targets,
                    current_probability * self._one_period_capped,
                )
            self._cumulative_pmfs.append(next_pmf)
            cumulative_demand = np.arange(len(next_pmf), dtype=float)
            available = self._inventory[:, None, None] + self._orders[None, :, None]
            remaining_total = np.maximum(
                available - cumulative_demand[None, None, :], 0.0
            )
            marginal_units = np.minimum(remaining_total, self._orders[None, :, None])
            period_holding = self.h * np.sum(
                marginal_units * next_pmf[None, None, :], axis=2
            )
            self._holding_prefix.append(self._holding_prefix[-1] + period_holding)

    def actions(
        self,
        states: np.ndarray,
        params: Mapping[str, float],
        remaining_horizon: int,
    ) -> np.ndarray:
        lower, upper, upper_probability = self.action_distribution(
            states, params, remaining_horizon
        )
        # This deterministic representative is retained for diagnostics only.
        # The benchmark evaluator consumes ``action_distribution`` and exactly
        # propagates both action branches.
        return np.where(upper_probability > 0.5, upper, lower).astype(np.int32)

    def action_distribution(
        self,
        states: np.ndarray,
        params: Mapping[str, float],
        remaining_horizon: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        del params
        affected_periods = max(0, int(remaining_horizon) - self.lead_time)
        if affected_periods == 0:
            zeros = np.zeros(len(states), dtype=np.int32)
            return zeros, zeros.copy(), np.zeros(len(states), dtype=float)
        self._ensure_holding_horizon(affected_periods)
        projected = self._project(states)
        holding = projected @ self._holding_prefix[affected_periods]
        penalty = projected @ self._arrival_penalty
        difference = holding - penalty
        crosses = difference >= -1.0e-12
        has_crossing = np.any(crosses, axis=1)
        upper = np.where(has_crossing, np.argmax(crosses, axis=1), self.n).astype(
            np.int32
        )
        lower = np.maximum(upper - 1, 0).astype(np.int32)

        row = np.arange(len(states))
        lower_difference = difference[row, lower]
        upper_difference = difference[row, upper]
        denominator = upper_difference - lower_difference
        upper_probability = np.zeros(len(states), dtype=float)
        interpolate = (upper != lower) & (denominator > 1.0e-15)
        upper_probability[interpolate] = np.clip(
            -lower_difference[interpolate] / denominator[interpolate], 0.0, 1.0
        )
        # If no grid action crosses, the state/action cap forces q=n.  If the
        # lower action already balances exactly, no upper randomization is
        # required.  Both cases are handled by the defaults above.
        upper_probability[~has_crossing] = 1.0
        lower[~has_crossing] = upper[~has_crossing]
        return lower, upper, upper_probability


__all__ = [
    "ArrivalMyopicPolicy",
    "ComputationalActionProvider",
    "FP3Policy",
    "Myopic2Policy",
    "PILPolicy",
    "ProjectedMeanVarianceCappedPolicy",
    "RandomizedActionProvider",
    "RandomizedDualBalancingPolicy",
]
