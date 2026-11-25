"""Benchmark registrations and parameter domains for lead-time/perishable baselines."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

from inventory.core import InventoryCase
from inventory.policy.ast import add, const, indicator_gt, maximum, motif, mul, parameter, safe_ratio, state, sub
from optimization.policies.baselines.lead_time_perishable import baselines_for_lead_time_perishable
from optimization.models.types import ParameterSpec

from .policies import LeadTimePerishableProvider


@dataclass(frozen=True)
class ComputationalLeadTimePerishableSpec:
    name: str
    display_name: str
    formula: str
    parameter_specs: tuple[ParameterSpec, ...]
    principle: str
    implementation_version: str

    def make_provider(self, case: InventoryCase) -> LeadTimePerishableProvider:
        return LeadTimePerishableProvider(case, self.name)


def computational_baselines(lead_time: int, shelf_life: int, n: int) -> tuple[ComputationalLeadTimePerishableSpec, ...]:
    if lead_time < 1 or shelf_life < 1:
        raise ValueError("positive lead time and shelf life are required")
    upper = (lead_time + shelf_life) * n
    return (
        ComputationalLeadTimePerishableSpec(
            "ewa", "Estimated Withdrawal and Aging (EWA)",
            "q=Q ceil([s-IP+EW]^+/Q)",
            (ParameterSpec("s", kind="integer", lower=0, upper=upper, default=n),
             ParameterSpec("Q", kind="integer", lower=1, upper=n, default=max(1, n // 3))),
            "Use projected waste to correct an (s,nQ) order rule.", "ewa_transition",
        ),
        ComputationalLeadTimePerishableSpec(
            "bsp_low_ew", "BSP-low-EW", "piecewise_affine(IP)+EW",
            (ParameterSpec("S1", kind="integer", lower=0, upper=upper, default=n),
             ParameterSpec("S2", kind="integer", lower=0, upper=upper, default=n),
             ParameterSpec("tau", kind="integer", lower=1, upper=upper, default=max(1, n // 2))),
            "Add expected waste correction to BSP-low.", "bsp_low_ew_transition",
        ),
        ComputationalLeadTimePerishableSpec(
            "pil", "Projected Inventory Level (PIL)", "q=[U-E(I at arrival)]^+",
            (ParameterSpec("U", kind="continuous", lower=0.0, upper=float(upper), default=float(n)),),
            "Target expected usable inventory when the order arrives.", "pil_transition",
        ),
        ComputationalLeadTimePerishableSpec(
            "apil", "Approximate Projected Inventory Level (APIL)",
            "q=[U-I_at_arrival(D_future=mean_D)]^+",
            (ParameterSpec("U", kind="continuous", lower=0.0, upper=float(upper), default=float(n)),),
            "Bu, Gong and Yin (2026), section 6: project with mean demand, including lost sales.",
            "apil_mean_demand_batched",
        ),
        ComputationalLeadTimePerishableSpec(
            "sqmax_ew", "SQmax-EW", "q=min(Qmax,[S-IP+EW(D_future=mean_D)]^+)",
            (ParameterSpec("S", kind="integer", lower=0, upper=upper, default=n),
             ParameterSpec("Qmax", kind="integer", lower=0, upper=n, default=max(1, n // 3))),
            "Haijema and Minner (2019), eq. (12): capped base stock with mean-demand waste.",
            "sqmax_ew_mean_demand_batched",
        ),
    )


def baseline_specs(case):
    specs = baselines_for_lead_time_perishable(case.lead_time, case.shelf_life, case.n,
                                             include_extended=True, include_age_split=True)
    # Preserve the existing monotone-weight policy class, but remove omega_M:
    # a_M is zero at all decision epochs reachable from the empty initial state.
    # This changes neither attainable actions nor the remaining weight domain.
    result = []
    for spec in specs:
        if spec.name == "bsp_low":
            # safe_ratio(x,y)=x/(1+abs(y)); use tau-1 with tau>=1
            # to implement the paper's ordinary division by tau exactly.
            ip = motif("inventory_position")
            low = add(sub(parameter("S1"), ip), mul(
                safe_ratio(sub(parameter("S2"), parameter("S1")), sub(parameter("tau"), const(1.0))), ip))
            high = sub(parameter("S2"), ip)
            use_high = indicator_gt(ip, sub(parameter("tau"), const(1.0)))
            spec = replace(spec, ast=add(low, mul(use_high, sub(high, low))),
                parameter_specs=tuple(replace(p, lower=1) if p.name == "tau" else p
                                      for p in spec.parameter_specs))
        if spec.name in {"s_s", "s_s_qmin_qmax"}:
            groups = {"s": ("reorder_levels", 0), "S": ("reorder_levels", 1),
                      "Qmin": ("batch_bounds", 0), "Qmax": ("batch_bounds", 1)}
            spec = replace(spec, parameter_specs=tuple(
                replace(p, monotone_group=groups[p.name][0], monotone_index=groups[p.name][1])
                if p.name in groups else p for p in spec.parameter_specs))
        if spec.name == "weighted_base_stock":
            terms = [mul(parameter(f"omega{j}"), state(f"a{j}")) for j in range(1, case.shelf_life)]
            weighted = const(0.0) if not terms else terms[0] if len(terms) == 1 else add(*terms)
            spec = replace(spec,
                ast=maximum(sub(parameter("S"), add(weighted, motif("total_pipeline"))), const(0.0)),
                parameter_specs=tuple(p for p in spec.parameter_specs if p.name != f"omega{case.shelf_life}"),
                formula="q=[S-sum_{j<M}(omega_j*a_j)-sum(p_l)]^+; monotone age weights",
                display_name="Weighted BSP (monotone variant)")
        result.append(spec)
    return tuple(result)


__all__ = ["ComputationalLeadTimePerishableSpec", "computational_baselines", "baseline_specs"]
