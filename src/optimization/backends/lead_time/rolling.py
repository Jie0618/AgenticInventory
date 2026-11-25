"""Lead-time rolling optimization and supervised-sample utilities.

This module contains only state transitions and the active simple AST batch
evaluator. It intentionally has no expectation/FP/PIL AST operations.
"""

from __future__ import annotations

import time

import numpy as np

from evaluation.backends.lead_time.models import RollingDPReference
from evaluation.backends.lead_time.dp import DPResult
from optimization.models.types import SupervisedSample


DEFAULT_ROLLING_MEMORY_MARGIN = 0.25
BLOCKED_WORKSPACE_TARGET_BYTES = 64 * 1024**2
ROLLING_BACKENDS = {"blocked_matrix", "compressed", "expanded_dense"}


def rolling_reference(case: object) -> RollingDPReference:
    """Build table-free DP geometry for one benchmark case."""

    return RollingDPReference(
        lead_time=int(getattr(case, "lead_time")),
        horizon=0,
        n=int(getattr(case, "n")),
        demand=getattr(case, "demand"),
        h=float(getattr(case, "h")),
        b=float(getattr(case, "b")),
    )


def estimate_rolling_dp_bytes(
    lead_time: int,
    n: int,
    *,
    margin: float = DEFAULT_ROLLING_MEMORY_MARGIN,
    backend: str = "blocked_matrix",
) -> int:
    """Conservative peak estimate for the selected rolling Bellman kernel."""

    lead_time = int(lead_time)
    n = int(n)
    if lead_time <= 0 or n < 0 or margin < 0.0:
        raise ValueError("rolling DP requires lead_time > 0, n >= 0, and margin >= 0")
    if backend not in ROLLING_BACKENDS:
        raise ValueError(
            "backend must be 'blocked_matrix', 'compressed', or 'expanded_dense'"
        )
    base = n + 1
    available_count = 2 * n + 1
    compressed = available_count * base ** max(0, lead_time - 1)
    action_surface = base**lead_time
    if backend == "compressed":
        cells = 2 * compressed + action_surface
    elif backend == "blocked_matrix":
        states = base ** (lead_time + 1)
        # The blocked kernel retains the expanded value layer but bounds its
        # matrix workspace independently of the full action surface.
        tail_count = base ** max(0, lead_time - 1)
        workspace_cells = min(
            BLOCKED_WORKSPACE_TARGET_BYTES // 8,
            (base + available_count) * tail_count * base,
        )
        cells = 2 * states + compressed + workspace_cells
    else:
        states = base ** (lead_time + 1)
        cells = 2 * states + compressed + action_surface
    return int(cells * 8 * (1.0 + float(margin)))


def solve_optimal_cost_rolling(
    dp: DPResult | RollingDPReference,
    horizon: int,
    *,
    sample_size: int = 5000,
    random_seed: int = 0,
    backend: str = "blocked_matrix",
    continuation_bound_periods: int = 0,
) -> tuple[float, float, tuple[SupervisedSample, ...]]:
    """Exact Bellman recursion plus a bounded sample of optimal actions.

    The full finite-horizon policy table is intentionally not retained.  At
    each Bellman step we sample a few uniformly distributed states and record
    their exact minimizing actions while the relevant action vector is already
    resident in memory.  This keeps memory independent of ``horizon`` and gives
    the raw-MSE constant fitter a representative DP imitation set.

    ``expanded_dense`` uses contiguous NumPy planes and remains available as
    an independent exact implementation. ``compressed`` is an
    exact, memory-constrained backend for runs whose expanded layers would
    otherwise exceed the available memory.
    """

    if dp.lead_time <= 0:
        raise ValueError("large-grid rolling solver requires positive lead time")
    if backend not in ROLLING_BACKENDS:
        raise ValueError(
            "backend must be 'blocked_matrix', 'compressed', or 'expanded_dense'"
        )
    continuation_bound_periods = int(continuation_bound_periods)
    if continuation_bound_periods < 0 or continuation_bound_periods > int(horizon):
        raise ValueError("continuation_bound_periods must lie in [0, horizon]")
    if isinstance(dp, RollingDPReference):
        dp.continuation_horizon = 0
        dp.continuation_values = None
    if backend == "blocked_matrix":
        return _solve_optimal_cost_rolling_blocked(
            dp,
            horizon,
            sample_size=sample_size,
            random_seed=random_seed,
            continuation_bound_periods=continuation_bound_periods,
        )
    if backend == "expanded_dense":
        return _solve_optimal_cost_rolling_expanded(
            dp,
            horizon,
            sample_size=sample_size,
            random_seed=random_seed,
            continuation_bound_periods=continuation_bound_periods,
        )

    started = time.perf_counter()
    n = dp.n
    base = n + 1
    lead_time = dp.lead_time
    # ``next_values[a, p2, ..., pL]`` is exactly equal to every full-grid
    # value ``V(x, p1, p2, ..., pL)`` for which ``x + p1 == a``.  Keeping this
    # quotient state throughout the recursion avoids constructing a dense
    # ``(x, p1, ..., pL)`` layer after every Bellman step.
    compressed_shape = (2 * n + 1,) + (base,) * (lead_time - 1)
    next_values = np.zeros(compressed_shape, dtype=float)
    demand_values = dp.demand.values.astype(int)
    demand_probs = dp.demand.probs.astype(float)
    available_count = 2 * n + 1
    next_x = np.empty((available_count, len(demand_values)), dtype=int)
    immediate = np.empty((available_count, len(demand_values)), dtype=float)
    for available in range(available_count):
        next_x[available] = np.clip(available - demand_values, 0, n)
        immediate[available] = (
            dp.h * np.maximum(available - demand_values, 0)
            + dp.b * np.maximum(demand_values - available, 0)
        )
    axis = np.arange(base, dtype=int)
    next_tail_slices = (slice(None),) * (lead_time - 1)
    action_shape = (base,) * lead_time
    rng = np.random.default_rng(int(random_seed))
    samples_per_step = max(1, int(np.ceil(max(1, sample_size) / horizon)))
    samples: list[SupervisedSample] = []
    for step in range(horizon):
        sampled_states = rng.integers(0, base, size=(samples_per_step, lead_time + 1))
        sampled_available = sampled_states[:, 0] + sampled_states[:, 1]
        sampled_actions = np.empty(samples_per_step, dtype=int)
        compressed = np.empty(compressed_shape, dtype=float)
        for available in range(available_count):
            action_costs = np.zeros(action_shape, dtype=float)
            for demand_index, probability in enumerate(demand_probs):
                # Axes of ``plane`` are ``(p2, ..., pL, q)``.  In the next
                # state the compressed available coordinate is
                # ``clip(available-D, 0, n) + p2`` and the remaining pipeline
                # is ``(p3, ..., pL, q)``.
                next_available = int(next_x[available, demand_index]) + axis
                plane = next_values[(next_available,) + next_tail_slices]
                action_costs += float(probability) * (
                    float(immediate[available, demand_index]) + plane
                )
            compressed[available] = np.min(action_costs, axis=-1)
            matching = np.flatnonzero(sampled_available == available)
            for sample_index in matching:
                tail = tuple(int(value) for value in sampled_states[sample_index, 2:])
                sampled_actions[sample_index] = int(np.argmin(action_costs[tail]))
        _retain_continuation_layer(dp, step + 1, continuation_bound_periods, compressed)
        next_values = compressed
        for state_row, action in zip(sampled_states, sampled_actions):
            state = {
                "x": int(state_row[0]),
                "p": tuple(int(value) for value in state_row[1:]),
                "remaining_horizon": int(step + 1),
            }
            samples.append(SupervisedSample(state=state, target_action=float(action)))
    if len(samples) > sample_size:
        indexes = np.linspace(0, len(samples) - 1, sample_size, dtype=int)
        samples = [samples[int(index)] for index in indexes]
    return (
        float(next_values[(0,) * lead_time]),
        time.perf_counter() - started,
        tuple(samples),
    )


def _solve_optimal_cost_rolling_blocked(
    dp: DPResult | RollingDPReference,
    horizon: int,
    *,
    sample_size: int,
    random_seed: int,
    continuation_bound_periods: int,
) -> tuple[float, float, tuple[SupervisedSample, ...]]:
    """Exact expanded recursion using a bounded transition-matrix workspace.

    For fixed available inventory, demand only selects the next on-hand axis.
    We therefore collapse the demand loop into a stochastic transition matrix
    and apply it to contiguous blocks of ``(p2, ..., pL, q)`` states.  The
    Bellman equation, finite grid, action set, and float64 value representation
    are unchanged; blocking only changes the order in which independent cells
    are evaluated.
    """

    started = time.perf_counter()
    n = int(dp.n)
    base = n + 1
    lead_time = int(dp.lead_time)
    available_count = 2 * n + 1
    demand_values = dp.demand.values.astype(int)
    demand_probs = dp.demand.probs.astype(float)

    transition = np.zeros((available_count, base), dtype=float)
    expected_immediate = np.zeros(available_count, dtype=float)
    for available in range(available_count):
        for demand, probability in zip(demand_values, demand_probs):
            next_x = int(np.clip(available - int(demand), 0, n))
            transition[available, next_x] += float(probability)
            expected_immediate[available] += float(probability) * (
                dp.h * max(available - int(demand), 0)
                + dp.b * max(int(demand) - available, 0)
            )

    tail_count = base ** max(0, lead_time - 1)
    # A block materializes both a contiguous source panel and the available ×
    # tail × action result. Keep their combined nominal size near 64 MiB.
    bytes_per_tail = max(1, (base + available_count) * base * 8)
    block_tail_count = max(1, BLOCKED_WORKSPACE_TARGET_BYTES // bytes_per_tail)
    block_tail_count = min(tail_count, int(block_tail_count))

    next_values = np.zeros((base,) * (lead_time + 1), dtype=float)
    available_grid = np.add.outer(
        np.arange(base, dtype=int),
        np.arange(base, dtype=int),
    )
    tail_slices = (slice(None),) * (lead_time - 1)
    compressed_shape = (available_count,) + (base,) * (lead_time - 1)
    rng = np.random.default_rng(int(random_seed))
    samples_per_step = max(1, int(np.ceil(max(1, sample_size) / horizon)))
    samples: list[SupervisedSample] = []

    for step in range(horizon):
        sampled_states = rng.integers(0, base, size=(samples_per_step, lead_time + 1))
        sampled_available = sampled_states[:, 0] + sampled_states[:, 1]
        sampled_actions = np.empty(samples_per_step, dtype=int)
        if lead_time == 1:
            sampled_tail = np.zeros(samples_per_step, dtype=int)
        else:
            sampled_tail = np.ravel_multi_index(
                tuple(sampled_states[:, axis] for axis in range(2, lead_time + 1)),
                (base,) * (lead_time - 1),
            )

        values = next_values.reshape(base, tail_count, base)
        compressed_flat = np.empty((available_count, tail_count), dtype=float)
        for start in range(0, tail_count, block_tail_count):
            stop = min(tail_count, start + block_tail_count)
            width = stop - start
            source = np.ascontiguousarray(values[:, start:stop, :]).reshape(base, width * base)
            action_costs = (transition @ source).reshape(available_count, width, base)
            action_costs += expected_immediate[:, None, None]
            compressed_flat[:, start:stop] = np.min(action_costs, axis=-1)

            matching = np.flatnonzero((sampled_tail >= start) & (sampled_tail < stop))
            if len(matching):
                sampled_actions[matching] = np.argmin(
                    action_costs[
                        sampled_available[matching],
                        sampled_tail[matching] - start,
                        :,
                    ],
                    axis=-1,
                )

        compressed = compressed_flat.reshape(compressed_shape)
        _retain_continuation_layer(dp, step + 1, continuation_bound_periods, compressed)
        next_values = compressed[(available_grid,) + tail_slices]
        for state_row, action in zip(sampled_states, sampled_actions):
            samples.append(
                SupervisedSample(
                    state={
                        "x": int(state_row[0]),
                        "p": tuple(int(value) for value in state_row[1:]),
                        "remaining_horizon": int(step + 1),
                    },
                    target_action=float(action),
                )
            )

    if len(samples) > sample_size:
        indexes = np.linspace(0, len(samples) - 1, sample_size, dtype=int)
        samples = [samples[int(index)] for index in indexes]
    return (
        float(next_values[(0,) * (lead_time + 1)]),
        time.perf_counter() - started,
        tuple(samples),
    )


def _solve_optimal_cost_rolling_expanded(
    dp: DPResult | RollingDPReference,
    horizon: int,
    *,
    sample_size: int,
    random_seed: int,
    continuation_bound_periods: int,
) -> tuple[float, float, tuple[SupervisedSample, ...]]:
    """Previous rolling implementation retained for exact A/B validation."""

    started = time.perf_counter()
    n = dp.n
    base = n + 1
    lead_time = dp.lead_time
    next_values = np.zeros((base,) * (lead_time + 1), dtype=float)
    demand_values = dp.demand.values.astype(int)
    demand_probs = dp.demand.probs.astype(float)
    available_count = 2 * n + 1
    next_x = np.empty((available_count, len(demand_values)), dtype=int)
    immediate = np.empty((available_count, len(demand_values)), dtype=float)
    for available in range(available_count):
        next_x[available] = np.clip(available - demand_values, 0, n)
        immediate[available] = (
            dp.h * np.maximum(available - demand_values, 0)
            + dp.b * np.maximum(demand_values - available, 0)
        )
    axis = np.arange(base, dtype=int)
    available_grid = np.add.outer(axis, axis)
    tail_slices = (slice(None),) * (lead_time - 1)
    plane_slices = (slice(None),) * lead_time
    compressed_shape = (available_count,) + (base,) * (lead_time - 1)
    action_shape = (base,) * lead_time
    rng = np.random.default_rng(int(random_seed))
    samples_per_step = max(1, int(np.ceil(max(1, sample_size) / horizon)))
    samples: list[SupervisedSample] = []
    for step in range(horizon):
        sampled_states = rng.integers(0, base, size=(samples_per_step, lead_time + 1))
        sampled_available = sampled_states[:, 0] + sampled_states[:, 1]
        sampled_actions = np.empty(samples_per_step, dtype=int)
        compressed = np.empty(compressed_shape, dtype=float)
        for available in range(available_count):
            action_costs = np.zeros(action_shape, dtype=float)
            for demand_index, probability in enumerate(demand_probs):
                plane = next_values[(int(next_x[available, demand_index]),) + plane_slices]
                action_costs += float(probability) * (
                    float(immediate[available, demand_index]) + plane
                )
            compressed[available] = np.min(action_costs, axis=-1)
            matching = np.flatnonzero(sampled_available == available)
            for sample_index in matching:
                tail = tuple(int(value) for value in sampled_states[sample_index, 2:])
                sampled_actions[sample_index] = int(np.argmin(action_costs[tail]))
        _retain_continuation_layer(dp, step + 1, continuation_bound_periods, compressed)
        next_values = compressed[(available_grid,) + tail_slices]
        for state_row, action in zip(sampled_states, sampled_actions):
            state = {
                "x": int(state_row[0]),
                "p": tuple(int(value) for value in state_row[1:]),
                "remaining_horizon": int(step + 1),
            }
            samples.append(SupervisedSample(state=state, target_action=float(action)))
    if len(samples) > sample_size:
        indexes = np.linspace(0, len(samples) - 1, sample_size, dtype=int)
        samples = [samples[int(index)] for index in indexes]
    return (
        float(next_values[(0,) * (lead_time + 1)]),
        time.perf_counter() - started,
        tuple(samples),
    )


def _retain_continuation_layer(
    dp: DPResult | RollingDPReference,
    completed_periods: int,
    requested_periods: int,
    compressed: np.ndarray,
) -> None:
    """Retain one exact compressed V_K layer on a rolling reference."""

    if (
        requested_periods <= 0
        or completed_periods != requested_periods
        or not isinstance(dp, RollingDPReference)
    ):
        return
    dp.continuation_horizon = int(completed_periods)
    dp.continuation_values = np.array(compressed, dtype=float, copy=True, order="C")
