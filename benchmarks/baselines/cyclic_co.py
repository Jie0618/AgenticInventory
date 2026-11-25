"""Benchmark-only deterministic integer orders at a fractional average rate.

Benchmark resolvers use these adapters for the existing constant_order policy.
Shared Stage1/2/3 baseline implementations retain their original semantics.
The adapters reuse existing transition kernels, replacing only action generation.
"""

from dataclasses import dataclass
from numbers import Integral
import time

import numpy as np

from inventory.policy.ast import parameter
from optimization.backends.lead_time.mc_evaluator import MonteCarloPolicyCostEvaluator
from optimization.evaluators.composed import (
    ComposedExactPolicyCostEvaluator, ComposedMonteCarloPolicyCostEvaluator,
    _ComposedASTActions,
)
from optimization.models.types import ParameterSpec


VERSION = "cyclic_constant_order"


@dataclass(frozen=True)
class CyclicOrder:
    parameter_name: str
    cap: int
    denominator: int = 100
    phase_ticks: int = 0

    def __post_init__(self):
        if any(not isinstance(v, Integral) or isinstance(v, bool) for v in (self.cap, self.denominator, self.phase_ticks)):
            raise ValueError("cap, denominator and phase_ticks must be integers")
        if self.cap < 0 or not 1 <= self.denominator <= 10000:
            raise ValueError("invalid cyclic cap/denominator")
        if not 0 <= self.phase_ticks < self.denominator:
            raise ValueError("phase_ticks must lie in [0, denominator)")

    @property
    def metadata(self):
        return {"version": VERSION, "parameter_name": self.parameter_name,
                "denominator": self.denominator, "phase_ticks": self.phase_ticks,
                "action_cap": self.cap, "initialization": "reset_at_bank_start"}

    @property
    def parameter_specs(self):
        return (ParameterSpec(self.parameter_name, kind="grid", lower=0,
                              upper=self.cap,
                              values=tuple(j / self.denominator for j in
                                           range(self.cap * self.denominator + 1)),
                              default=min(self.cap, max(1, self.cap // 3))),)

    @property
    def formula(self):
        return ("q_t=floor(t*r+phi)-floor((t-1)*r+phi); "
                f"r={self.parameter_name}, phi={self.phase_ticks}/{self.denominator}, "
                f"r=j/{self.denominator}, 0<=r<={self.cap}")

    def ticks(self, rates):
        rates = np.asarray(rates, dtype=float)
        scaled = rates * self.denominator
        ticks = np.rint(scaled)
        if (not np.all(np.isfinite(rates)) or np.any(rates < 0)
                or np.any(rates > self.cap)
                or not np.allclose(scaled, ticks, rtol=0, atol=1e-8)):
            raise ValueError("cyclic rate must lie on the declared rate grid")
        return ticks.astype(np.int64)

    def action(self, rates, period):
        """Zero-based period; integer arithmetic makes all boundary ties exact."""
        if period < 0:
            raise ValueError("period must be nonnegative")
        ticks = self.ticks(rates)
        return (((period + 1) * ticks + self.phase_ticks) // self.denominator
                - (period * ticks + self.phase_ticks) // self.denominator).astype(np.int32)

    def sequence(self, params, horizon):
        ticks = self.ticks(params[self.parameter_name])
        periods = np.arange(horizon + 1, dtype=np.int64)
        return np.diff((periods * ticks + self.phase_ticks) // self.denominator).astype(np.int32)


def resolve_cyclic(identifier, setting, n, *, denominator=100, phase_ticks=0):
    from benchmarks.inventory_benchmark_family.mc_runtime import MCPolicy
    schedule = CyclicOrder("r" if setting == "lead_time" else "Q", n,
                           denominator, phase_ticks)
    return MCPolicy(identifier, "registered_baseline", parameter(schedule.parameter_name),
                    schedule.parameter_specs, {}, schedule.formula,
                    execution=schedule.metadata)


def schedule_for(policy):
    data = dict(policy.execution)
    version = data.pop("version")
    data.pop("initialization")
    if version != VERSION:
        raise ValueError(f"unsupported cyclic implementation: {version}")
    data["cap"] = data.pop("action_cap")
    return CyclicOrder(**data)


class CyclicLeadTimeMC(MonteCarloPolicyCostEvaluator):
    def __init__(self, *args, schedule, **kwargs):
        self.schedule = schedule
        super().__init__(*args, **kwargs)

    def _policy_actions(self, states, params, remaining_horizon):
        action = self.schedule.action(params[self.schedule.parameter_name],
                                      self.horizon - remaining_horizon)
        return np.full(len(states), action, dtype=np.int32)

    def _policy_actions_many_prepared(self, prepared, remaining_horizon):
        rates = prepared.parameter_arrays[self.schedule.parameter_name].reshape(prepared.candidates, prepared.paths)[:, 0]
        actions = self.schedule.action(rates, self.horizon - remaining_horizon)
        return np.broadcast_to(actions[:, None], (prepared.candidates, prepared.paths))

    def action_signature(self, params):
        return self.schedule.sequence(params, self.horizon)

    def exact_action_signature_key(self, params):
        return self.action_signature(params).tobytes()


class _CyclicComposedActions(_ComposedASTActions):
    def __init__(self, case, horizon, schedule):
        super().__init__(parameter(schedule.parameter_name), case, {},
                         rounding="round", action_cap=case.n)
        self.horizon, self.schedule = horizon, schedule

    def actions_array(self, matrix, params, remaining_horizon):
        actions = self.schedule.action(params[self.schedule.parameter_name],
                                      self.horizon - remaining_horizon)
        return np.broadcast_to(actions, (len(matrix),)).astype(np.int32)


class CyclicPerishableMC(ComposedMonteCarloPolicyCostEvaluator):
    def __init__(self, case, bank, schedule, *, batch_size=64):
        # The shared MC backend takes a model/action descriptor. Construction
        # does not solve a DP or evaluate an Exact benchmark.
        exact = ComposedExactPolicyCostEvaluator(
            parameter(schedule.parameter_name), case, bank.horizon)
        exact._actions = _CyclicComposedActions(case, bank.horizon, schedule)
        exact._horizon_dependent = True
        exact.schedule = schedule
        super().__init__(exact, paths=bank.num_paths, seed=bank.seed,
                         batch_size=batch_size)
        self.mc_only = True  # Construction above does not solve a DP.
        self.schedule = schedule
        self.bind_demand_paths(bank.demand_values)

    def _canonical_actions_many(self, prepared, remaining_horizon):
        rates = prepared.parameter_arrays[self.schedule.parameter_name].reshape(prepared.candidates, prepared.paths)[:, 0]
        actions = self.schedule.action(rates,
                                      self.exact_evaluator.horizon - remaining_horizon)
        return np.broadcast_to(actions[:, None], (prepared.candidates, prepared.paths))

    def action_signature(self, params):
        return self.schedule.sequence(params, self.exact_evaluator.horizon)

    def exact_action_signature_key(self, params):
        return self.action_signature(params).tobytes()


def fit_rate_grid(schedule, high, low, *, top_k=64):
    """Screen every declared rate; arbitrate top K plus ALL integer rates.

    Proposal paths screen rates; strict paths select the fit. This remains
    empirical best-found; independent final paths never select rates.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    started = time.perf_counter()
    rates = np.arange(schedule.cap * schedule.denominator + 1) / schedule.denominator
    params = [{schedule.parameter_name: float(r)} for r in rates]
    screen = low.evaluate_many(params)
    ranking = sorted(range(len(params)), key=lambda i: (screen[i].cost, rates[i]))
    promoted = sorted(set(ranking[:top_k]) | {i for i, r in enumerate(rates) if r.is_integer()})
    screen_count = len(params)
    results = high.evaluate_many([params[i] for i in promoted])
    best = min(range(len(results)), key=lambda j: (results[j].cost, rates[promoted[j]]))
    return params[promoted[best]], float(results[best].cost), {
        "route": "cyclic_rate_grid_mc_screened",
        "candidate_count": len(params), "screen_count": screen_count,
        "strict_count": len(promoted), "optimizer_seconds": time.perf_counter() - started,
        "global_on_declared_grid": len(promoted) == len(params),
        "execution": schedule.metadata,
    }
