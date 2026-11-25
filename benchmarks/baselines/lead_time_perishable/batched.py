"""Exact batched projections and MC for deterministic lead-time perishables.

The scalar provider remains the independent reference. This module never calls
its actions or scalar projection: demand branches, states, candidates and paths
are evaluated as arrays. Python cache lookups and result formatting remain;
neither trajectories nor candidate simulations run one at a time.
"""
from collections import OrderedDict
from math import comb
import time

import numpy as np
from scipy.sparse import csr_matrix
from .native import reduced_batch, inventory_forward, LongProjector

from optimization.evaluators.composed import ComposedMonteCarloPolicyCostEvaluator
from optimization.models.types import EvaluationResult


def unique_rows(rows):
    """Lossless byte keys avoid mixed-radix overflow at L=M=10."""
    rows = np.ascontiguousarray(rows, dtype=np.int32)
    keys = rows.view(np.dtype((np.void, rows.shape[1] * rows.dtype.itemsize))).ravel()
    _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    return rows[first], inverse


class ExactBatchProjector:
    """Backward expectation on a deduplicated, batched zero-order state DAG.

    A node stores expected terminal inventory and cumulative expiry. Sharing
    child nodes across roots avoids propagating separate Python distributions
    for each trajectory. No pruning, sampled projections or approximations.
    """

    def __init__(self, case, *, cache_limit=200_000, chunk_size=256, quantity='both',
                 cached_inventory=True, age_state_limit=1024, native_long=True):
        if (case.lead_time < 1 or case.shelf_life is None
                or case.config.transition_mode.value != 'deterministic'
                or case.schema.backlog_dimension):
            raise ValueError('Batched projections require deterministic L/M and lost sales')
        self.case = case
        if quantity not in {'both', 'inventory', 'waste'}:
            raise ValueError('Unknown projection quantity')
        self.quantity = quantity
        self.L, self.M = case.lead_time, case.shelf_life
        positive = np.asarray(case.demand.probs) > 0
        self.values = np.asarray(case.demand.values, dtype=np.int32)[positive]
        self.probs = np.asarray(case.demand.probs, dtype=float)[positive]
        order = np.argsort(self.values)
        self.values, self.probs = self.values[order], self.probs[order]
        self.caches = [OrderedDict() for _ in range(self.L + 1)]
        self.layer_limit = max(1, cache_limit // self.L)
        self.chunk_size = chunk_size
        self.age_states = None
        capacity = case.config.capacity
        self.cached_inventory = cached_inventory
        self.native_long = native_long
        self._long_projector = None
        # The newest bin is zero after every canonical aging step. Enumerating
        # only the remaining bins gives 231 states at M=3, cap=20, rather than
        # growing a separate demand tree for every pipeline/trajectory.
        # Large tables compute every age state for each arrival suffix and
        # thrash the bounded suffix cache (10,626 states at M=5, cap=20).
        # Keep those projections on the reachable-state native kernels.
        if capacity is not None and comb(int(capacity) + self.M - 1, self.M - 1) <= age_state_limit:
            cap = int(capacity)
            def compositions(width, budget):
                if width == 0:
                    yield ()
                else:
                    for first in range(budget + 1):
                        for rest in compositions(width - 1, budget - first):
                            yield (first,) + rest
            self.age_states = np.asarray(list(compositions(self.M - 1, cap)), dtype=np.int32).reshape(-1, self.M - 1) if self.M > 1 else np.empty((1, 0), dtype=np.int32)
            self.age_powers = (cap + 1) ** np.arange(self.M - 1, dtype=np.int64)
            self.age_lookup = np.full((cap + 1) ** (self.M - 1), -1, dtype=np.int32)
            self.age_lookup[self.age_states @ self.age_powers] = np.arange(len(self.age_states))
            self.age_operators = {}
            self.age_suffixes = OrderedDict()
            self.age_suffix_limit = max(16, (64 * 1024**2) // (len(self.age_states) * 16))
        self.stats = dict(batch_calls=0, cache_hits=0, expanded_states=0,
                          transition_batches=0, peak_branch_rows=0, reduced_batches=0,
                          suffix_hits=0, suffix_nodes=0, age_operator_builds=0)

    def batch(self, rows):
        rows = np.asarray(rows)
        if (rows.ndim != 2 or rows.shape[1] != self.M + self.L
                or np.any(rows < 0) or np.any(rows != np.rint(rows))):
            raise ValueError('Invalid projection state matrix')
        capacity = self.case.config.capacity
        if capacity is not None and (np.any(rows[:, :self.M].sum(axis=1) > capacity)
                                     or np.any(rows[:, self.M:] > capacity)):
            raise ValueError('Projection state exceeds canonical capacity')
        self.stats['batch_calls'] += 1
        if (self.native_long and self.age_states is None and self.M >= 5
                and self.L >= self.M and capacity is not None and capacity <= 100):
            if self._long_projector is None:
                self._long_projector = LongProjector(self.M, self.L, int(capacity),
                    self.quantity, self.case.w > 0, self.cached_inventory,
                    max(self.layer_limit, 200_000), self.values, self.probs)
            output, stats = self._long_projector.batch(rows)
            self.stats.update(stats)
            self.stats['expanded_states'] = stats['native_expanded_states']
            self.stats['cache_hits'] = stats['native_cache_hits']
            self.stats['reduced_batches'] = stats['native_reduced_roots']
            self.stats['transition_batches'] = (stats['native_expanded_states']
                                                - stats['native_reduced_roots'])
            return output
        return self._values(np.asarray(rows, dtype=np.int32), self.L)

    def _values(self, rows, remaining):
        if remaining == 0:
            return np.column_stack((rows[:, :self.M].sum(axis=1), np.zeros(len(rows))))
        if self.age_states is not None:
            return self._finite_age_values(rows, remaining)
        if self.quantity == 'waste' and remaining <= self.M:
            # FIFO serves original stock first; youngest-first truncation
            # cannot discard it. No future receipt expires in this window.
            # Therefore arrivals cannot affect waste from the original stock.
            # At remaining==M retain only the first receipt, which can expire.
            rows = rows.copy()
            rows[:, self.M + int(remaining == self.M):] = 0
        unique, inverse = unique_rows(rows)
        output = np.empty((len(unique), 2), dtype=float)
        cache = self.caches[remaining]
        missing = []
        # Python operations here are bounded cache lookups, not transitions or
        # policy evaluations. All cache misses below are expanded together.
        for i, row in enumerate(unique):
            key = row.tobytes()
            value = cache.get(key)
            if value is None:
                missing.append(i)
            else:
                output[i] = value
                cache.move_to_end(key)
                self.stats['cache_hits'] += 1
        for start in range(0, len(missing), self.chunk_size):
            positions = missing[start:start + self.chunk_size]
            parents = unique[positions]
            if (remaining < self.M or (remaining == self.M and self.quantity == 'waste')) and self.case.config.capacity is not None:
                results = self._no_new_expiry(parents, remaining)
            else:
                children, expiry = self._branches(parents)
                next_values = self._values(children.reshape(-1, children.shape[-1]), remaining - 1)
                next_values = next_values.reshape(len(parents), len(self.values), 2)
                if self.case.w > 0:
                    next_values[:, :, 1] += expiry
                results = np.einsum('sdk,d->sk', next_values, self.probs)
            output[positions] = results
            for row, result in zip(parents, results):
                key = row.tobytes()
                cache[key] = tuple(result)
                cache.move_to_end(key)
                if len(cache) > self.layer_limit:
                    cache.popitem(last=False)
        return output[inverse]

    def _no_new_expiry(self, parents, periods):
        """Exact reduced law, with a specialized waste-only boundary case.

        The initial stock is always older than future receipts. Let W be
        initial units already consumed/expired, N its original quantity, and
        E_t the cumulative quantity whose original life ends by t. Then
        W'=max(min(N,W+D),E_t). Freshest-first capacity discards never remove
        initial units (their quantity is <=capacity), so only total inventory
        I is additionally needed. No distribution over age profiles is needed.
        Waste-only propagation omits I; at periods==M it additionally accounts
        for the first receipt's possible discard and expiry in the native code.
        """
        if self.quantity == 'inventory' and self.cached_inventory:
            results = inventory_forward(parents, periods, self.M,
                int(self.case.config.capacity), self.values, self.probs)
            self.stats['reduced_batches'] += 1
            self.stats['expanded_states'] += len(parents)
            return results
        results = reduced_batch(parents, periods, self.M, int(self.case.config.capacity),
                                self.values, self.probs, self.case.w > 0 and self.quantity != 'inventory',
                                self.quantity != 'waste')
        self.stats['reduced_batches'] += 1
        self.stats['expanded_states'] += len(parents)
        return results

    def _finite_age_values(self, rows, remaining):
        canonical = rows[:, self.M - 1] == 0
        result = np.empty((len(rows), 2))
        if np.any(~canonical):
            children, expiry = self._branches(rows[~canonical])
            future = self._values(children.reshape(-1, children.shape[-1]), remaining - 1)
            future = future.reshape(len(children), len(self.values), 2)
            if self.case.w > 0:
                future[:, :, 1] += expiry
            result[~canonical] = np.einsum('sdk,d->sk', future, self.probs)
        roots = rows[canonical]
        if not len(roots):
            return result
        arrivals, inverse = unique_rows(roots[:, self.M:self.M + remaining])
        state_index = self.age_lookup[roots[:, :self.M - 1] @ self.age_powers]
        values = self._age_suffix_values(arrivals)
        result[canonical] = values[inverse, state_index]
        return result

    def _age_suffix_values(self, arrivals):
        size = len(self.age_states)
        if arrivals.shape[1] == 0:
            terminal = np.column_stack((self.age_states.sum(axis=1), np.zeros(size)))
            return np.broadcast_to(terminal, (len(arrivals), size, 2))
        result = np.empty((len(arrivals), size, 2))
        missing = []
        for i, row in enumerate(arrivals):
            cached = self.age_suffixes.get(row.tobytes())
            if cached is None:
                missing.append(i)
            else:
                result[i] = cached
                self.age_suffixes.move_to_end(row.tobytes())
                self.stats['suffix_hits'] += 1
        for start in range(0, len(missing), 64):
            positions = np.asarray(missing[start:start+64])
            suffixes, inverse = unique_rows(arrivals[positions, 1:]) if arrivals.shape[1] > 1 else (np.empty((1, 0), dtype=np.int32), np.zeros(len(positions), dtype=int))
            future = self._age_suffix_values(suffixes)[inverse]
            for arrival in np.unique(arrivals[positions, 0]):
                local = np.flatnonzero(arrivals[positions, 0] == arrival)
                operator, reward = self._age_operator(int(arrival))
                children = future[local].transpose(1, 0, 2).reshape(size, -1)
                computed = operator.dot(children).reshape(size, len(local), 2).transpose(1, 0, 2)
                if self.case.w > 0:
                    computed[:, :, 1] += reward
                result[positions[local]] = computed
            for index in positions:
                self.age_suffixes[arrivals[index].tobytes()] = result[index].copy()
                if len(self.age_suffixes) > self.age_suffix_limit:
                    self.age_suffixes.popitem(last=False)
                self.stats['suffix_nodes'] += 1
        return result

    def _age_operator(self, arrival):
        cached = self.age_operators.get(arrival)
        if cached is not None:
            return cached
        size, demands = len(self.age_states), len(self.values)
        inventory = np.zeros((size, demands, self.M), dtype=np.int32)
        inventory[:, :, :self.M - 1] = self.age_states[:, None, :]
        inventory[:, :, -1] = arrival
        need = np.broadcast_to(self.values, (size, demands)).copy()
        for age in range(self.M):
            used = np.minimum(inventory[:, :, age], need)
            inventory[:, :, age] -= used
            need -= used
        expiry = inventory[:, :, 0].copy()
        inventory[:, :, :-1] = inventory[:, :, 1:]
        inventory[:, :, -1] = 0
        ComposedMonteCarloPolicyCostEvaluator._project_many(inventory, self.case.config.capacity, freshest_first=True)
        targets = self.age_lookup[inventory[:, :, :-1] @ self.age_powers]
        weights = np.broadcast_to(self.probs, targets.shape)
        origins = np.broadcast_to(np.arange(size)[:, None], targets.shape)
        operator = csr_matrix((weights.ravel(), (origins.ravel(), targets.ravel())), shape=(size, size))
        operator.sum_duplicates()
        operator.eliminate_zeros()
        reward = np.sum(weights * expiry, axis=1)
        self.age_operators[arrival] = operator, reward
        self.stats['age_operator_builds'] += 1
        return operator, reward

    def _branches(self, parents):
        """Zero-order receive/FIFO/expire/youngest-discard over all PMF branches."""
        s, d = len(parents), len(self.values)
        children = np.broadcast_to(parents[:, None, :], (s, d, self.M + self.L)).copy()
        inventory = children[:, :, :self.M]
        inventory[:, :, -1] += parents[:, self.M, None]
        need = np.broadcast_to(self.values, (s, d)).copy()
        for age in range(self.M):
            used = np.minimum(inventory[:, :, age], need)
            inventory[:, :, age] -= used
            need -= used
        expiry = inventory[:, :, 0].copy()
        inventory[:, :, :-1] = inventory[:, :, 1:]
        inventory[:, :, -1] = 0
        ComposedMonteCarloPolicyCostEvaluator._project_many(
            inventory, self.case.config.capacity, freshest_first=True)
        children[:, :, self.M:-1] = children[:, :, self.M + 1:]
        children[:, :, -1] = 0
        self.stats['expanded_states'] += s
        self.stats['transition_batches'] += 1
        self.stats['peak_branch_rows'] = max(self.stats['peak_branch_rows'], s * d)
        return children, expiry


class MeanDemandBatchProjector:
    """Paper-native deterministic forecast; batch every candidate/path together.

    Demand remains fractional when its mean is fractional. Only actual orders
    are rounded. The L-step forecast receives existing pipeline, serves FIFO,
    expires oldest stock, and applies the canonical youngest-first capacity
    projection. Neither candidate nor path is advanced in a Python loop.
    """

    def __init__(self, case):
        if (case.lead_time < 1 or case.shelf_life is None
                or case.config.transition_mode.value != 'deterministic'
                or case.schema.backlog_dimension):
            raise ValueError('Mean-demand projections require deterministic L/M and lost sales')
        self.case = case
        self.L, self.M = case.lead_time, case.shelf_life
        self.mean = float(np.dot(case.demand.values, case.demand.probs))
        self.stats = dict(batch_calls=0, projected_rows=0, transition_batches=0, max_batch_rows=0)

    def batch(self, rows):
        # Float forecasts must not be passed to the integer-state exact projector.
        rows = np.asarray(rows, dtype=float)
        if rows.ndim != 2 or rows.shape[1] != self.M + self.L or np.any(rows < 0) or not np.isfinite(rows).all():
            raise ValueError('Invalid mean-demand projection states')
        inventory = rows[:, :self.M].copy()
        waste = np.zeros(len(rows))
        for period in range(self.L):
            inventory[:, -1] += rows[:, self.M + period]
            need = np.full(len(rows), self.mean)
            for age in range(self.M):
                used = np.minimum(inventory[:, age], need)
                inventory[:, age] -= used
                need -= used
            waste += inventory[:, 0]
            inventory[:, :-1] = inventory[:, 1:]
            inventory[:, -1] = 0
            ComposedMonteCarloPolicyCostEvaluator._project_many(
                inventory[None, :, :], self.case.config.capacity, freshest_first=True)
        self.stats['batch_calls'] += 1
        self.stats['projected_rows'] += len(rows)
        self.stats['transition_batches'] += self.L
        self.stats['max_batch_rows'] = max(self.stats['max_batch_rows'], len(rows))
        return np.column_stack((inventory.sum(axis=1), waste))


class BatchedLeadTimePerishableProvider:
    """Broadcast policy parameters over candidates; project all states together."""

    def __init__(self, case, name, *, certified_projection=True):
        if name not in {'pil', 'ewa', 'bsp_low_ew', 'apil', 'sqmax_ew'}:
            raise ValueError(f'Unknown computational baseline {name}')
        self.case, self.name = case, name
        self.projector = (MeanDemandBatchProjector(case) if name in {'apil', 'sqmax_ew'} else
                          ExactBatchProjector(case, quantity='inventory' if name == 'pil' else 'waste'))
        self._rounding_projector = None
        self.rounding_rechecks = 0
        self.projection_error = 0.0
        self.certified_projection = certified_projection
        if name == 'pil' and certified_projection and case.config.capacity is not None:
            # A sub-probability projection sums retained demand branches only.
            # Terminal inventory is in [0,cap], so union bound gives a rigorous
            # omitted contribution <=cap*L*omitted_PMF_mass. Never renormalize.
            cap = int(case.config.capacity)
            budget = .005 / max(1, cap * case.lead_time)
            order = np.argsort(self.projector.probs)
            keep = np.ones(len(order), dtype=bool)
            omitted = 0.0
            for index in order:
                probability = float(self.projector.probs[index])
                if omitted + probability > budget:
                    break
                keep[index] = False
                omitted += probability
            if omitted > 0:
                self.projector.values = self.projector.values[keep]
                self.projector.probs = self.projector.probs[keep]
                self.projection_error = cap * case.lead_time * omitted

    @property
    def optimization_route_hint(self):
        return 'pil_scalar' if self.name == 'pil' else None

    def actions_many(self, states, params_batch, *, pil_targets=None, pil_intervals=False):
        candidates, paths, width = states.shape
        ip = states.sum(axis=2)

        def p(name):
            if name == 'U' and pil_targets is not None:
                return pil_targets
            return np.asarray([params[name] for params in params_batch], dtype=float)[:, None]

        if self.name in {'apil', 'sqmax_ew'}:
            projection = self.projector.batch(states.reshape(-1, width)).reshape(candidates, paths, 2)
            if self.name == 'apil':
                raw = np.maximum(p('U') - projection[:, :, 0], 0)
            else:
                raw = np.minimum(p('Qmax'), np.maximum(p('S') - ip + projection[:, :, 1], 0))
            return np.clip(np.rint(raw), 0, self.case.n).astype(np.int32)

        # Prove zero/saturated actions from rigorous projection bounds before
        # doing any stochastic projection. This is especially important for
        # wide optimizer domains: evaluating U=400 cannot require a projection
        # when terminal inventory <=20 and every possible order clips to20.
        cap = self.case.config.capacity
        if self.name == 'pil':
            target = np.broadcast_to(p('U'), (candidates, paths))
            zero = target <= 0
            saturated = target >= (cap + self.case.n) if cap is not None else np.zeros_like(zero)
        elif self.name == 'ewa':
            zero = np.broadcast_to(p('s') <= 0, (candidates, paths))
            saturated = p('s') - ip >= self.case.n
        else:
            s1, s2, tau = p('S1'), p('S2'), np.maximum(1, p('tau'))
            base_action = np.where(ip < tau, s1 - (1 - (s2 - s1) / tau) * ip, s2 - ip)
            zero = base_action + ip <= 0  # projected waste <= all existing units
            saturated = base_action >= self.case.n
        needed = ~(zero | saturated)
        fixed = None
        if self.name != 'pil':
            # Only stock whose expiry deadline is within the projection can
            # contribute waste. Demand/capacity discards can only reduce it.
            m, l = self.case.shelf_life, self.case.lead_time
            max_waste = states[:, :, :min(l, m)].sum(axis=2)
            if l >= m:
                max_waste = max_waste + states[:, :, m:m + l - m + 1].sum(axis=2)
            if self.case.w <= 0:
                max_waste = np.zeros_like(max_waste)
            # Keep shortcuts conservative at integer/half-integer boundaries
            # despite roundoff in summed probability expectations.
            max_waste = max_waste * (1 + 64 * np.finfo(float).eps)
            if self.name == 'ewa':
                batch = np.maximum(1, np.rint(p('Q')))
                lower = np.ceil(np.maximum(p('s') - ip, 0) / batch) * batch
                upper = np.ceil(np.maximum(p('s') - ip + max_waste, 0) / batch) * batch
            else:
                lower = np.rint(np.maximum(base_action, 0))
                upper = np.rint(np.maximum(base_action + max_waste, 0))
            lower = np.clip(lower, 0, self.case.n)
            upper = np.clip(upper, 0, self.case.n)
            fixed = lower == upper
            needed &= ~fixed
        projection = np.zeros((candidates, paths, 2))
        if np.any(needed):
            projection[needed] = self.projector.batch(states[needed])

        if self.name == 'pil':
            raw = np.maximum(p('U') - projection[:, :, 0], 0)
            errors = np.full((candidates, paths), self.projection_error)
            if self.projection_error or (self.projector.cached_inventory and self.projector.age_states is None):
                # A different summation order can perturb a half-integer tie.
                # Recheck only this narrow band using the original forward law
                # so the acceleration preserves the previous rounded actions.
                tolerance = 512 * np.finfo(float).eps * (1 + (cap or self.case.n)) * self.case.lead_time
                # Both endpoints must imply exactly the same clipped action.
                low = np.clip(np.rint(np.maximum(p('U') - projection[:, :, 0] - errors - tolerance, 0)), 0, self.case.n)
                high = np.clip(np.rint(np.maximum(p('U') - projection[:, :, 0] + tolerance, 0)), 0, self.case.n)
                uncertain = needed & (low != high)
                if np.any(uncertain):
                    if self._rounding_projector is None:
                        self._rounding_projector = ExactBatchProjector(
                            self.case, quantity='inventory', cached_inventory=False,
                            age_state_limit=12000, native_long=False)
                    projection[uncertain] = self._rounding_projector.batch(states[uncertain])
                    errors[uncertain] = 0.0
                    self.rounding_rechecks += int(uncertain.sum())
                    raw = np.maximum(p('U') - projection[:, :, 0], 0)
            raw = np.rint(raw)
        elif self.name == 'ewa':
            batch = np.maximum(1, np.rint(p('Q')))
            raw = np.ceil(np.maximum(p('s') - ip + projection[:, :, 1], 0) / batch) * batch
        else:
            s1, s2, tau = p('S1'), p('S2'), np.maximum(1, p('tau'))
            raw = np.where(ip < tau, s1 - (1 - (s2 - s1) / tau) * ip, s2 - ip)
            raw = np.rint(np.maximum(raw + projection[:, :, 1], 0))
        if self.name != 'pil' and self.projector.age_states is None:
            # Preserve ceil/half-integer decisions when the equivalent
            # native summation order could move a value across a boundary.
            tolerance = 512 * np.finfo(float).eps * (1 + self.case.n) * self.case.lead_time
            waste = projection[:, :, 1]
            if self.name == 'ewa':
                value = (p('s') - ip + waste) / batch
                rounded_low = np.ceil(np.maximum(value - tolerance, 0)) * batch
                rounded_high = np.ceil(np.maximum(value + tolerance, 0)) * batch
            else:
                value = np.where(ip < tau, s1 - (1 - (s2 - s1) / tau) * ip, s2 - ip) + waste
                rounded_low = np.rint(np.maximum(value - tolerance, 0))
                rounded_high = np.rint(np.maximum(value + tolerance, 0))
            uncertain = needed & (waste > 0) & (np.clip(rounded_low, 0, self.case.n) != np.clip(rounded_high, 0, self.case.n))
            if np.any(uncertain):
                if self._rounding_projector is None:
                    self._rounding_projector = ExactBatchProjector(
                        self.case, quantity='waste', cached_inventory=False, age_state_limit=12000,
                        native_long=False)
                waste[uncertain] = self._rounding_projector.batch(states[uncertain])[:, 1]
                self.rounding_rechecks += int(uncertain.sum())
                if self.name == 'ewa':
                    raw = np.ceil(np.maximum(p('s') - ip + waste, 0) / batch) * batch
                else:
                    base = np.where(ip < tau, s1 - (1 - (s2 - s1) / tau) * ip, s2 - ip)
                    raw = np.rint(np.maximum(base + waste, 0))
        raw = np.where(zero, 0, np.where(saturated, self.case.n, raw))
        actions = np.clip(raw, 0, self.case.n).astype(np.int32)
        if fixed is not None:
            actions = np.where(fixed, lower, actions).astype(np.int32)
        if pil_intervals:
            if self.name != 'pil':
                raise ValueError('Only PIL admits scalar target certificates')
            expected = projection[:, :, 0]
            # Open intervals avoid ties-to-even rounding boundaries. A generous
            # floating-point guard also excludes subtraction/addition roundoff.
            guard = 64 * np.finfo(float).eps * (1 + np.abs(expected) + np.abs(p('U')) + self.case.n)
            guard = np.maximum(guard, 512 * np.finfo(float).eps
                               * (1 + (cap or self.case.n)) * self.case.lead_time)
            lower = np.where(actions > 0, expected + errors + actions - .5 + guard, -np.inf)
            upper = np.where(actions < self.case.n, expected + actions + .5 - guard, np.inf)
            # Bounds that bypassed projection need their own conservative proof.
            lower = np.where(saturated, cap + self.case.n + guard, lower) if cap is not None else lower
            upper = np.where(zero, -guard, upper)
            return actions, lower, upper
        return actions


class BatchedLeadTimePerishableMC(ComposedMonteCarloPolicyCostEvaluator):
    """Candidate x trajectory MC with exact batched projected policy actions."""

    def __init__(self, case, horizon, provider, *, paths, seed, batch_size=64,
                 reuse_pil_paths=True):
        super().__init__(ast={'op': 'const', 'value': 0}, case=case, horizon=horizon,
                         paths=paths, seed=seed, batch_size=batch_size)
        self.provider = (provider if isinstance(provider, BatchedLeadTimePerishableProvider)
                         else BatchedLeadTimePerishableProvider(case, provider.name))
        self.reuse_pil_paths = reuse_pil_paths
        self.stats = dict(simulation_batches=0, candidates_simulated=0, max_candidate_batch=0)
        self.stats.update(pil_paths_reused=0, pil_paths_simulated=0)
        # Certificates belong to this evaluator's demand bank, never the shared
        # provider. Bound storage to 32 MiB (including strict/final path counts).
        self._pil_certificates = OrderedDict()
        self._pil_certificate_limit = max(1, min(2048, (32 * 1024**2) // (24 * self.paths)))

    def clear_cache(self):
        super().clear_cache()
        if hasattr(self, '_pil_certificates'):
            self._pil_certificates.clear()

    @property
    def optimization_route_hint(self):
        return self.provider.optimization_route_hint

    def evaluate_many(self, params_batch, incumbent=None):
        # Canonicalize only policies proven identical for *every* legal state.
        # The optimizer still sees its original parameter values and budget.
        # In particular all U>=capacity+order_cap mean constant maximum order.
        unique, inverse, positions = [], [], {}
        cap = self.exact_evaluator.case.config.capacity
        for params in params_batch:
            effective = dict(params)
            if self.provider.name == 'pil' and cap is not None:
                effective['U'] = min(max(float(effective['U']), 0.0),
                                     cap + self.exact_evaluator.case.n)
            elif self.provider.name == 'ewa':
                # Once Q>=n, any positive batch clips to n, independently
                # of state. Large Q values are globally identical policies.
                n = self.exact_evaluator.case.n
                effective['Q'] = min(max(1, round(float(effective['Q']))), n)
                if float(effective['s']) <= 0:
                    effective.update(s=0, Q=1)
                elif cap is not None:
                    max_ip = (self.exact_evaluator.case.lead_time + 1) * cap
                    if float(effective['s']) >= max_ip + n:
                        effective.update(s=max_ip + n, Q=n)
            elif self.provider.name == 'bsp_low_ew':
                s1, s2 = float(effective['S1']), float(effective['S2'])
                # When the two levels coincide, tau is algebraically unused:
                # both branches of the BSP rule return S1 - inventory.
                # Canonicalizing it avoids evaluating the same policy once
                # for every tau value in the discrete proposal grid.
                if s1 == s2:
                    effective['tau'] = 1
                if max(s1, s2) <= 0:
                    effective.update(S1=0, S2=0, tau=1)
                elif cap is not None:
                    max_ip = (self.exact_evaluator.case.lead_time + 1) * cap
                    n = self.exact_evaluator.case.n
                    if min(s1, s2) >= max_ip + n:
                        effective.update(S1=max_ip + n, S2=max_ip + n, tau=1)
            key = tuple(sorted(effective.items()))
            if key not in positions:
                positions[key] = len(unique)
                unique.append(effective)
            inverse.append(positions[key])
        results = super().evaluate_many(unique, incumbent=incumbent)
        return [results[i] for i in inverse]

    def _evaluate_canonical_many(self, params_batch, *, pathwise=False):
        if self.provider.name == 'pil' and self.reuse_pil_paths:
            return self._evaluate_pil_certified_many(params_batch, pathwise=pathwise)
        started = time.perf_counter()
        case, horizon = self.exact_evaluator.case, self.exact_evaluator.horizon
        initial = np.asarray(self.exact_evaluator.initial_state.flatten(case.schema), dtype=np.int32)
        candidates = len(params_batch)
        states = np.broadcast_to(initial, (candidates, self.paths, len(initial))).copy()
        costs = np.zeros((candidates, self.paths))
        m = case.shelf_life
        inventory, pipeline = states[:, :, :m], states[:, :, m:]
        for period in range(horizon):
            actions = self.provider.actions_many(states, params_batch)
            incoming = pipeline[:, :, 0].copy()
            pipeline[:, :, :-1] = pipeline[:, :, 1:]
            pipeline[:, :, -1] = actions
            if case.config.capacity is not None:
                np.minimum(pipeline[:, :, -1], case.config.capacity, out=pipeline[:, :, -1])
            inventory[:, :, -1] += incoming
            unmet = self._consume_many(inventory, self._demands[None, :, period])
            costs += case.costs.holding * inventory.sum(axis=2)
            costs += case.costs.shortage * unmet
            costs += case.costs.ordering * actions
            costs += case.costs.expiry * inventory[:, :, 0]
            inventory[:, :, :-1] = inventory[:, :, 1:]
            inventory[:, :, -1] = 0
            self._project_many(inventory, case.config.capacity, freshest_first=True)
        self.stats['simulation_batches'] += 1
        self.stats['candidates_simulated'] += candidates
        self.stats['max_candidate_batch'] = max(self.stats['max_candidate_batch'], candidates)
        if pathwise:
            return costs
        return [EvaluationResult(cost=float(np.mean(row)), completed=True,
            visited_states=self.paths * horizon,
            metadata={'fidelity': 'mc_fixed', 'backend': ('ltp_mean_demand_batched_mc'
                      if self.provider.name in {'apil', 'sqmax_ew'} else 'ltp_exact_projection_batched_mc'),
                      'candidate_batch_size': candidates, 'paths': self.paths, 'horizon': horizon,
                      'seed': self.seed, 'proposal_mc_seconds': time.perf_counter() - started}) for row in costs]

    def _evaluate_pil_certified_many(self, params_batch, *, pathwise=False):
        """Reuse path costs only when every rounded action is certified equal.

        For a fixed state, q=clip(rint(U-E[I]),0,n) is constant on an open
        interval. Intersect those intervals along a path. Induction then proves
        that any target inside the intersection has the same states and cost
        under this demand path. No monotonic-cost assumption or interpolation.
        """
        started = time.perf_counter()
        case, horizon = self.exact_evaluator.case, self.exact_evaluator.horizon
        initial = np.asarray(self.exact_evaluator.initial_state.flatten(case.schema), dtype=np.int32)
        costs = np.empty((len(params_batch), self.paths))
        # Seed one small wave only for an initially empty, densely spaced batch.
        # Then pack all remaining misses together; repeatedly running a whole
        # horizon for tiny waves would erase the benefit of trajectory reuse.
        all_targets = np.asarray([float(p['U']) for p in params_batch])
        boundaries = [0, len(params_batch)]
        if (not self._pil_certificates and len(params_batch) > 8
                and np.ptp(all_targets) < .25):
            boundaries.insert(1, 8)
        for start, stop in zip(boundaries, boundaries[1:]):
            params = params_batch[start:stop]
            targets = np.asarray([float(p['U']) for p in params])
            lower = np.full((len(params), self.paths), -np.inf)
            upper = np.full_like(lower, np.inf)
            result = np.empty_like(lower)
            known = np.zeros_like(lower, dtype=bool)
            keys = np.asarray(sorted(self._pil_certificates))
            if len(keys):
                indices = np.searchsorted(keys, targets)
                # Neighboring representative targets are sufficient for useful
                # hits; any missed certificate merely causes exact recomputation.
                for offset in (-1, 0):
                    for row, index in enumerate(indices + offset):
                        if index < 0 or index >= len(keys):
                            continue
                        lo, hi, value = self._pil_certificates[float(keys[index])]
                        hit = ~known[row] & (targets[row] > lo) & (targets[row] < hi)
                        result[row, hit] = value[hit]
                        lower[row, hit], upper[row, hit] = lo[hit], hi[hit]
                        known[row, hit] = True
            self.stats['pil_paths_reused'] += int(known.sum())
            candidate_ids, path_ids = np.nonzero(~known)
            active = len(path_ids)
            self.stats['pil_paths_simulated'] += active
            if active:
                # Flatten only missing candidate/path pairs, keeping the whole
                # active set in a single batched transition/projection kernel.
                states = np.broadcast_to(initial, (active, 1, len(initial))).copy()
                inventory, pipeline = states[:, :, :case.shelf_life], states[:, :, case.shelf_life:]
                active_targets = targets[candidate_ids, None]
                totals = np.zeros((active, 1))
                lo = np.full((active, 1), -np.inf)
                hi = np.full_like(lo, np.inf)
                for period in range(horizon):
                    actions, step_lo, step_hi = self.provider.actions_many(
                        states, (), pil_targets=active_targets, pil_intervals=True)
                    np.maximum(lo, step_lo, out=lo)
                    np.minimum(hi, step_hi, out=hi)
                    incoming = pipeline[:, :, 0].copy()
                    pipeline[:, :, :-1] = pipeline[:, :, 1:]
                    pipeline[:, :, -1] = actions
                    if case.config.capacity is not None:
                        np.minimum(pipeline[:, :, -1], case.config.capacity, out=pipeline[:, :, -1])
                    inventory[:, :, -1] += incoming
                    unmet = self._consume_many(inventory, self._demands[path_ids, period, None])
                    totals += case.costs.holding * inventory.sum(axis=2)
                    totals += case.costs.shortage * unmet
                    totals += case.costs.ordering * actions
                    totals += case.costs.expiry * inventory[:, :, 0]
                    inventory[:, :, :-1] = inventory[:, :, 1:]
                    inventory[:, :, -1] = 0
                    self._project_many(inventory, case.config.capacity, freshest_first=True)
                result[candidate_ids, path_ids] = totals[:, 0]
                lower[candidate_ids, path_ids], upper[candidate_ids, path_ids] = lo[:, 0], hi[:, 0]
            for row, target in enumerate(targets):
                self._pil_certificates[float(target)] = (lower[row].copy(), upper[row].copy(), result[row].copy())
                self._pil_certificates.move_to_end(float(target))
                while len(self._pil_certificates) > self._pil_certificate_limit:
                    self._pil_certificates.popitem(last=False)
            costs[start:start + len(params)] = result
        self.stats['simulation_batches'] += 1
        self.stats['candidates_simulated'] += len(params_batch)
        self.stats['max_candidate_batch'] = max(self.stats['max_candidate_batch'], len(params_batch))
        if pathwise:
            return costs
        return [EvaluationResult(cost=float(np.mean(row)), completed=True,
            visited_states=self.paths * horizon,
            metadata={'fidelity': 'mc_fixed', 'backend': 'ltp_exact_projection_batched_mc',
                      'candidate_batch_size': len(params_batch), 'paths': self.paths, 'horizon': horizon,
                      'seed': self.seed, 'proposal_mc_seconds': time.perf_counter() - started}) for row in costs]
