"""Batched fixed-lead-time myopic; independent inner MC, exact arrival demand.

Bu and Li (2026), section 5 equation (5), specialized to fixed L and K.
The transition below is deliberately restricted to nonperishable finite-K
cases and independently checked against the canonical scalar transition.
No scalar path/action evaluation or access to outer future demand is used.
"""

import numpy as np

BACKEND = 'ltb_myopic_path_x_inner_x_action_batch_no_entry_shortage'


def advance(inventory, backlog, incoming, demand, capacity, costs):
    """One canonical FIFO step on arbitrary leading batch axes; no mutation."""
    available = inventory + incoming
    older = np.cumsum(backlog[..., ::-1], axis=-1)[..., ::-1] - backlog
    residual = np.maximum(backlog - np.maximum(available[..., None]-older, 0), 0)
    free = np.maximum(available-backlog.sum(axis=-1), 0)
    unmet = np.maximum(demand-free, 0)
    ending = np.maximum(free-demand, 0)
    aged = np.concatenate((unmet[..., None], residual[..., :-1]), axis=-1)
    # Capacity discards youngest cohorts first, without abandonment charges.
    older_aged = np.cumsum(aged[..., ::-1], axis=-1)[..., ::-1] - aged
    aged = np.minimum(aged, np.maximum(capacity-older_aged, 0))
    cost = (costs.holding*ending
            + costs.abandonment*residual[..., -1] + costs.backlog*aged.sum(axis=-1))
    return np.minimum(ending, capacity), aged, cost


class Myopic:
    def __init__(self, case, settings, *, inner_seed):
        if (case.config.lead_time.mode.value != 'deterministic' or case.lead_time < 1
                or case.config.perishability.mode.value != 'disabled'
                or case.config.backlog.mode.value != 'deterministic'
                or case.config.capacity != case.n or case.costs.ordering != 0):
            raise ValueError('myopic requires fixed positive L/K, nonperishable, capacity=n, zero ordering cost')
        self.case, self.settings = case, settings
        self.inner_seed = inner_seed
        rng = np.random.default_rng(self.inner_seed)
        self.prefix_demands = rng.choice(case.demand.values,
            size=(settings['myopic_inner_paths'], case.lead_time), p=case.demand.probs)
        # F(z)=E[(D-z)+]. This table integrates arrival-period demand exactly.
        thresholds = np.arange(3*case.n+1)
        self.shortage = np.maximum(case.demand.values[None, :]-thresholds[:, None], 0) @ case.demand.probs
        self.mean_demand = float(case.demand.values @ case.demand.probs)
        self.statistics = dict(decision_calls=0, max_path_batch=0,
                               inner_paths=len(self.prefix_demands), max_action_batch=0)

    def objectives(self, inventory, pipeline, backlog):
        """Return (paths, n+1) expected arrival costs, using one fixed inner bank."""
        c = self.case
        count, inner = len(inventory), len(self.prefix_demands)
        inv = np.broadcast_to(inventory[:, None], (count, inner)).copy()
        back = np.broadcast_to(backlog[:, None, :], (count, inner, backlog.shape[-1])).copy()
        # q arrives at t+L. Existing p1..pL arrive at t..t+L-1.
        # Hence this prefix is identical for every candidate q.
        for period in range(c.lead_time):
            inv, back, _ = advance(inv, back, pipeline[:, period, None],
                self.prefix_demands[None, :, period], c.n, c.costs)
        result = np.empty((count, c.n+1), dtype=float)
        for start in range(0, c.n+1, self.settings['myopic_action_batch_size']):
            actions = np.arange(start, min(c.n+1, start+self.settings['myopic_action_batch_size']))
            available = inv[..., None] + actions
            total = back.sum(axis=-1)[..., None]
            free = np.maximum(available-total, 0)
            oldest = np.maximum(back[..., -1, None]-available, 0)
            retained = np.maximum(total-available, 0)-oldest
            shortage = self.shortage[free]
            carry = retained + shortage - self.shortage[free+c.n-retained]
            expected = (c.h*(free-self.mean_demand+shortage)
                        + c.g*carry + c.a*oldest)
            result[:, start:start+len(actions)] = expected.mean(axis=1)
            self.statistics['max_action_batch'] = max(self.statistics['max_action_batch'], len(actions))
        self.statistics['decision_calls'] += 1
        self.statistics['max_path_batch'] = max(self.statistics['max_path_batch'], count)
        return result

    def actions(self, inventory, pipeline, backlog):
        result = np.empty(len(inventory), dtype=np.int32)
        for start in range(0, len(inventory), self.settings['myopic_path_batch_size']):
            stop = start+self.settings['myopic_path_batch_size']
            values = self.objectives(inventory[start:stop], pipeline[start:stop], backlog[start:stop])
            # Ascending integer actions; first argmin implements minimum-action ties.
            result[start:stop] = np.argmin(values, axis=1)
        return result

    def simulate(self, demands):
        c = self.case
        paths, horizon = demands.shape
        inventory = np.full(paths, c.initial_state.inventory[0], dtype=np.int64)
        pipeline = np.broadcast_to(c.initial_state.pipeline, (paths, c.lead_time)).copy()
        backlog = np.broadcast_to(c.initial_state.backlog, (paths, c.config.backlog.patience)).copy()
        costs = np.zeros(paths)
        for period in range(horizon):
            actions = self.actions(inventory, pipeline, backlog)
            inventory, backlog, step_cost = advance(inventory, backlog, pipeline[:, 0],
                demands[:, period], c.n, c.costs)
            pipeline[:, :-1] = pipeline[:, 1:]
            pipeline[:, -1] = actions
            costs += step_cost
        return costs
