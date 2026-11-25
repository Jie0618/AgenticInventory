"""Computational baselines for fixed positive lead-time perishables.

Independent scalar references, not production MC kernels. PIL/EWA use exact
finite-support expectations; APIL/SQmax-EW use the papers' deterministic mean
demand forecasts with the same receipt/FIFO/expiry/capacity event order.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Mapping, Sequence

import numpy as np

from inventory.core import InventoryCase, InventoryState


class LeadTimePerishableProvider:
    horizon_dependent = False

    def __init__(self, case: InventoryCase, name: str) -> None:
        if case.lead_time < 1 or case.shelf_life is None:
            raise ValueError("lead-time/perishable providers require L>=1 and fixed shelf life")
        if case.config.perishability.stochastic or case.config.lead_time.stochastic:
            raise ValueError("this catalog is for deterministic L and fixed shelf life")
        if name not in {"pil", "ewa", "bsp_low_ew", "apil", "sqmax_ew"}:
            raise ValueError(f"unknown computational baseline {name}")
        self.case = case
        self.name = name
        self.L = int(case.lead_time)
        self.n = int(case.n)
        self.demands = tuple(zip(case.demand.values.astype(int), case.demand.probs.astype(float)))
        self.mean_demand = float(np.dot(case.demand.values, case.demand.probs))

    @lru_cache(maxsize=200_000)
    def _project_distribution(self, state_tuple: tuple[int, ...], periods: int):
        state = InventoryState.from_flat(state_tuple, self.case.schema)
        distribution: dict[InventoryState, float] = {state: 1.0}
        expected_expiry = 0.0
        for _ in range(max(0, int(periods))):
            following: dict[InventoryState, float] = {}
            for current, mass in distribution.items():
                for demand, probability in self.demands:
                    for outcome in self.case.kernel.outcomes(current, 0, int(demand)):
                        weight = mass * float(probability) * float(outcome.probability)
                        following[outcome.next_state] = following.get(outcome.next_state, 0.0) + weight
                        if self.case.w > 0.0:
                            expected_expiry += weight * float(outcome.outcome.expiry_cost) / self.case.w
            distribution = following
        expected_inventory = sum(mass * state.total_inventory for state, mass in distribution.items())
        return float(expected_inventory), float(expected_expiry)

    def _projected(self, state: InventoryState) -> tuple[float, float]:
        return self._project_distribution(tuple(state.flatten(self.case.schema)), self.L)

    def _waste(self, state: InventoryState) -> float:
        # EWA's R=1 specialization: expected outdating through the next
        # arrival window.  The current order is not included in the forecast.
        return self._project_distribution(tuple(state.flatten(self.case.schema)), self.L)[1]

    def _project_mean(self, state: InventoryState) -> tuple[float, float]:
        """Independent scalar reference only; production uses the array projector."""
        m = self.case.shelf_life
        row = state.flatten(self.case.schema)
        stock = list(map(float, row[:m]))
        waste = 0.0
        for incoming in row[m:]:
            stock[-1] += incoming
            need = self.mean_demand
            for j in range(m):
                used = min(stock[j], need)
                stock[j] -= used
                need -= used
            waste += stock[0]
            stock = stock[1:] + [0.0]
            if self.case.config.capacity is not None:
                excess = max(sum(stock) - self.case.config.capacity, 0)
                for j in reversed(range(m)):
                    removed = min(stock[j], excess)
                    stock[j] -= removed
                    excess -= removed
        return sum(stock), waste

    def actions(self, states: Sequence[InventoryState], params: Mapping[str, float], remaining_horizon: int) -> np.ndarray:
        del remaining_horizon
        output: list[int] = []
        for state in states:
            self.case.schema.validate(state)
            ip = float(state.total_inventory + state.total_pipeline)
            if self.name in {"apil", "sqmax_ew"}:
                projected, waste = self._project_mean(state)
                raw = (max(float(params['U']) - projected, 0.0) if self.name == 'apil' else
                       min(float(params['Qmax']), max(float(params['S']) - ip + waste, 0.0)))
                output.append(int(np.clip(np.rint(raw), 0, self.n)))
            elif self.name == "pil":
                raw = max(float(params["U"]) - self._projected(state)[0], 0.0)
                output.append(int(np.clip(np.rint(raw), 0, self.n)))
            elif self.name == "ewa":
                s = float(params["s"]); batch = max(1, int(round(float(params["Q"]))))
                raw = max(s - ip + self._waste(state), 0.0)
                output.append(int(np.clip(np.ceil(raw / batch) * batch, 0, self.n)))
            else:
                s1 = float(params["S1"]); s2 = float(params["S2"]); tau = max(1.0, float(params["tau"]))
                alpha = 1.0 - (s2 - s1) / tau
                raw = s1 - alpha * ip if ip < tau else s2 - ip
                output.append(int(np.clip(np.rint(max(raw + self._waste(state), 0.0)), 0, self.n)))
        return np.asarray(output, dtype=np.int32)


__all__ = ["LeadTimePerishableProvider"]
