"""Optimized DP backend for deterministic lead-time inventory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from inventory.common.distributions import DiscreteDemand


State = tuple[int, ...]


@dataclass(frozen=True)
class DPResult:
    """Container for a finite-horizon lost-sales DP solution.

    `value_table[t, state...]` stores the optimal expected cost from period `t`.
    `policy_table[t, state...]` stores the optimal order quantity `q`.

    Periods are zero-indexed in code:

    - `t=0` is the first decision period.
    - `t=horizon` is the terminal value table and has value zero.
    """

    lead_time: int
    horizon: int
    n: int
    demand: DiscreteDemand
    h: float
    b: float
    value_table: np.ndarray
    policy_table: np.ndarray
    backend: str = "lead_time_array"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def normalize_state(self, state: int | Sequence[int]) -> State:
        """Convert an int/list/tuple state into the canonical tuple format."""

        if self.lead_time == 0 and isinstance(state, int):
            normalized = (state,)
        else:
            normalized = tuple(int(v) for v in state)  # type: ignore[arg-type]
        expected_dim = self.lead_time + 1
        if len(normalized) != expected_dim:
            raise ValueError(f"Expected state dimension {expected_dim}, got {len(normalized)}.")
        if any(v < 0 or v > self.n for v in normalized):
            raise ValueError(f"State {normalized} is outside the truncated grid [0, {self.n}].")
        return normalized

    def value(self, t: int, state: int | Sequence[int]) -> float:
        """Read `V_t(state)` from the value table."""

        if not 0 <= t <= self.horizon:
            raise ValueError("Value period t must be in [0, horizon].")
        return float(self.value_table[(t,) + self.normalize_state(state)])

    def action(self, t: int, state: int | Sequence[int]) -> int:
        """Read the optimal integer order quantity `q_t(state)`."""

        if not 0 <= t < self.horizon:
            raise ValueError("Policy period t must be in [0, horizon-1].")
        return int(self.policy_table[(t,) + self.normalize_state(state)])


def solve_dp(lead_time: int, horizon: int, n: int, demand: DiscreteDemand, h: float, b: float) -> DPResult:
    """Dispatch to the correct DP solver for lead time `L`.

    `L=0` has immediate replenishment, so the action is an order-up-to level.
    `L>=1` has fixed pipeline delay, so the action affects future states only.
    """

    if lead_time < 0:
        raise ValueError("lead_time must be nonnegative.")
    if lead_time == 0:
        return solve_l0_dp(horizon=horizon, n=n, demand=demand, h=h, b=b)
    return solve_lead_time_dp(lead_time=lead_time, horizon=horizon, n=n, demand=demand, h=h, b=b)


def solve_l0_dp(horizon: int, n: int, demand: DiscreteDemand, h: float, b: float) -> DPResult:
    """Solve the finite-horizon lost-sales DP for `L=0`.

    State is current inventory `x`.  Because replenishment arrives immediately,
    the action can be represented as an order-up-to level `y >= x`, and the
    stored policy is the actual order quantity `q = y - x`.
    """

    _validate_dp_inputs(horizon, n, h, b)
    value_table = np.zeros((horizon + 1, n + 1), dtype=float)
    policy_table = np.zeros((horizon, n + 1), dtype=int)
    demand_values = demand.values.astype(int)
    demand_probs = demand.probs.astype(float)

    for t in range(horizon - 1, -1, -1):
        next_values = value_table[t + 1]
        for x in range(n + 1):
            best_cost = float("inf")
            best_y = x
            for y in range(x, n + 1):
                end_inventory = np.maximum(y - demand_values, 0)
                holding = h * end_inventory
                shortage = b * np.maximum(demand_values - y, 0)
                future = next_values[end_inventory]
                expected_cost = float(np.dot(holding + shortage + future, demand_probs))
                if expected_cost < best_cost:
                    best_cost = expected_cost
                    best_y = y
            value_table[t, x] = best_cost
            policy_table[t, x] = best_y - x

    return DPResult(
        lead_time=0,
        horizon=horizon,
        n=n,
        demand=demand,
        h=float(h),
        b=float(b),
        value_table=value_table,
        policy_table=policy_table,
    )


def solve_lead_time_dp(lead_time: int, horizon: int, n: int, demand: DiscreteDemand, h: float, b: float) -> DPResult:
    """Solve the finite-horizon lost-sales DP for a fixed positive lead time.

    State is `(x, p1, ..., pL)`:

    - `x` is inventory at the beginning of the period.
    - `p1` arrives immediately before demand.
    - `q` placed now becomes the last pipeline entry next period.

    All state components and actions are truncated to `{0, ..., n}`.  If
    `x+p1-D` exceeds `n`, next inventory is clipped to `n`; this is the standard
    small-grid approximation used before increasing `n`.
    """

    _validate_dp_inputs(horizon, n, h, b)
    if lead_time <= 0:
        raise ValueError("solve_lead_time_dp requires lead_time >= 1.")

    table_shape = (horizon + 1,) + (n + 1,) * (lead_time + 1)
    policy_shape = (horizon,) + (n + 1,) * (lead_time + 1)
    value_table = np.zeros(table_shape, dtype=float)
    policy_table = np.zeros(policy_shape, dtype=int)
    demand_values = demand.values.astype(int)
    demand_probs = demand.probs.astype(float)
    grid_size = n + 1

    # The Bellman transition depends on `(x, p1)` only through the available
    # inventory `x + p1`.  Compute each equivalent state class once, then
    # expand it back to the public full-grid tables.  This is exact: states
    # with the same available inventory and `(p2, ..., pL)` have identical
    # current costs and identical next states for every `(demand, action)`.
    available_count = 2 * n + 1
    holding_by_available = np.zeros((available_count, len(demand_values)), dtype=float)
    shortage_by_available = np.zeros_like(holding_by_available)
    next_x_by_available = np.zeros((available_count, len(demand_values)), dtype=int)
    for available in range(available_count):
        holding_by_available[available] = h * np.maximum(available - demand_values, 0)
        shortage_by_available[available] = b * np.maximum(demand_values - available, 0)
        next_x_by_available[available] = np.minimum(np.maximum(available - demand_values, 0), n)

    # `available_grid[x, p1] == x + p1`.  Indexing the compressed result with
    # this grid restores the original `(x, p1, ..., pL)` table layout.
    state_axis = np.arange(grid_size, dtype=int)
    available_grid = np.add.outer(state_axis, state_axis)
    tail_slices = (slice(None),) * (lead_time - 1)
    next_plane_slices = (slice(None),) * lead_time
    compressed_shape = (available_count,) + (grid_size,) * (lead_time - 1)
    action_shape = (grid_size,) * lead_time

    for t in range(horizon - 1, -1, -1):
        next_values = value_table[t + 1]
        compressed_values = np.empty(compressed_shape, dtype=float)
        compressed_actions = np.empty(compressed_shape, dtype=int)

        for available in range(available_count):
            # Axes are `(p2, ..., pL, q)`.  Keeping the demand loop in its
            # original order preserves the Bellman summation order, while
            # NumPy evaluates every downstream-pipeline/action combination
            # together instead of entering Python once per state and action.
            action_costs = np.zeros(action_shape, dtype=float)
            for demand_index, demand_prob in enumerate(demand_probs):
                next_x = int(next_x_by_available[available, demand_index])
                next_plane = next_values[(next_x,) + next_plane_slices]
                immediate_cost = (
                    float(holding_by_available[available, demand_index])
                    + float(shortage_by_available[available, demand_index])
                )
                action_costs += float(demand_prob) * (immediate_cost + next_plane)

            compressed_actions[available] = np.argmin(action_costs, axis=-1)
            compressed_values[available] = np.min(action_costs, axis=-1)

        expanded_index = (available_grid,) + tail_slices
        value_table[t] = compressed_values[expanded_index]
        policy_table[t] = compressed_actions[expanded_index]

    return DPResult(
        lead_time=lead_time,
        horizon=horizon,
        n=n,
        demand=demand,
        h=float(h),
        b=float(b),
        value_table=value_table,
        policy_table=policy_table,
    )


def verify_l0_base_stock(result: DPResult) -> dict[str, object]:
    """Check whether an `L=0` DP policy has exact base-stock structure.

    For each period, base-stock means there exists `S_t` such that
    `q_t(x) = max(S_t - x, 0)` for every inventory state `x`.
    """

    if result.lead_time != 0:
        raise ValueError("Base-stock verification is only defined for L=0 results.")

    base_stock_levels: list[int] = []
    max_error = 0
    for t in range(result.horizon):
        orders = result.policy_table[t]
        s_t = int(orders[0])
        expected = np.maximum(s_t - np.arange(result.n + 1), 0)
        error = int(np.max(np.abs(orders - expected)))
        max_error = max(max_error, error)
        base_stock_levels.append(s_t)

    return {
        "ok": max_error == 0,
        "max_error": max_error,
        "base_stock_levels": base_stock_levels,
    }


def _validate_dp_inputs(horizon: int, n: int, h: float, b: float) -> None:
    """Validate shared DP dimensions and cost parameters."""

    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if n < 0:
        raise ValueError("n must be nonnegative.")
    if h < 0.0 or b < 0.0:
        raise ValueError("Holding and shortage costs must be nonnegative.")
