"""Catalog and paper metadata for lead-time computational baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from evaluation.backends.lead_time.dp import DPResult
from optimization import ParameterSpec

from benchmarks.baselines.lead_time.policies import (
    ArrivalMyopicPolicy,
    ComputationalActionProvider,
    FP3Policy,
    Myopic2Policy,
    PILPolicy,
    ProjectedMeanVarianceCappedPolicy,
    RandomizedDualBalancingPolicy,
)


ProviderFactory = Callable[[DPResult], ComputationalActionProvider]


@dataclass(frozen=True)
class ComputationalBaselineSpec:
    name: str
    display_name: str
    formula: str
    parameter_specs: tuple[ParameterSpec, ...]
    principle: str
    paper_title: str
    authors: str
    year: int
    reference_url: str
    doi: str | None
    implementation_note: str
    horizon_dependent: bool
    implementation_version: str
    provider_factory: ProviderFactory

    def make_provider(self, reference: DPResult) -> ComputationalActionProvider:
        return self.provider_factory(reference)


def _factory(policy_class: type) -> ProviderFactory:
    def build(reference: DPResult) -> ComputationalActionProvider:
        return policy_class(
            reference.demand.values,
            reference.demand.probs,
            reference.n,
            reference.lead_time,
            reference.h,
            reference.b,
        )

    return build


def computational_baselines(lead_time: int, n: int) -> tuple[ComputationalBaselineSpec, ...]:
    if lead_time < 1:
        raise ValueError("computational baselines require positive lead time")
    return (
        ComputationalBaselineSpec(
            name="myopic",
            display_name="Arrival-aligned Myopic",
            formula="q=argmin_u E[h J_arrival(u)+b Lost_arrival(u)|state]",
            parameter_specs=(),
            principle="Minimize expected holding plus lost-sales cost in the period when today's order arrives.",
            paper_title="The Near-Myopic Nature of the Lagged-Proportional-Cost Inventory Problem with Lost Sales",
            authors="Thomas E. Morton",
            year=1971,
            reference_url="https://doi.org/10.1287/opre.19.7.1708",
            doi="10.1287/opre.19.7.1708",
            implementation_note=(
                "Exact finite-grid arrival-inventory projection; choose the smallest integer minimizer and clip to n. "
                "The policy originates with Morton (1971) and is implemented here using the arrival-period formulation "
                "reported in Zipkin (2008), DOI 10.1287/opre.1070.0471."
            ),
            horizon_dependent=False,
            implementation_version="computational-baseline",
            provider_factory=_factory(ArrivalMyopicPolicy),
        ),
        ComputationalBaselineSpec(
            name="myopic2",
            display_name="Myopic-2",
            formula="q=argmin_u E[C_arrival(u)+min_{u'(D0)} C_next-arrival(u,u')|state]",
            parameter_specs=(),
            principle=(
                "Minimize the two costs first affected by today's and next period's orders, "
                "with the follow-up order chosen after observing current demand."
            ),
            paper_title="Old and New Methods for Lost-Sales Inventory Systems",
            authors="Paul Zipkin",
            year=2008,
            reference_url="https://doi.org/10.1287/opre.1070.0471",
            doi="10.1287/opre.1070.0471",
            implementation_note=(
                "Exact finite-support arrival-aligned two-stage stochastic lookahead for arbitrary positive "
                "lead time. The follow-up order observes current demand but is fixed before subsequent demand "
                "outcomes, matching the information timing of Zipkin's Myopic-2 policy."
            ),
            horizon_dependent=False,
            implementation_version="computational-baseline-exact-lookahead",
            provider_factory=_factory(Myopic2Policy),
        ),
        ComputationalBaselineSpec(
            name="pil",
            display_name="Projected Inventory Level (PIL)",
            formula="q=[U-E(J immediately before q arrives|state)]^+",
            parameter_specs=(
                ParameterSpec("U", kind="continuous", lower=0, upper=2 * n, default=n / 2),
            ),
            principle="Raise expected inventory at the order-arrival epoch to a fitted projected level U.",
            paper_title="Projected Inventory Level Policies for Lost Sales Inventory Systems: Asymptotic Optimality in Two Regimes",
            authors="Willem van Jaarsveld and Joachim Arts",
            year=2024,
            reference_url="https://doi.org/10.1287/opre.2021.0032",
            doi="10.1287/opre.2021.0032",
            implementation_note=(
                "Exact discrete finite-grid projection with a continuously fitted PIL level U. The paper permits "
                "continuous orders; this integer-inventory benchmark rounds each resulting order to the nearest "
                "integer and clips it to [0,n]."
            ),
            horizon_dependent=False,
            implementation_version="computational-baseline-continuous-u",
            provider_factory=_factory(PILPolicy),
        ),
        ComputationalBaselineSpec(
            name="fp3",
            display_name="Fixed Non-stockout Probability (FP3)",
            formula="q=min{u>=0: P(D_arrival<=J_prearrival+u|state)>=P3}",
            parameter_specs=(
                ParameterSpec(
                    "P3",
                    kind="grid",
                    lower=0.0,
                    upper=0.999,
                    values=(*(index / 200 for index in range(200)), 0.999),
                    default=0.8,
                ),
            ),
            principle="Choose the smallest order attaining a fitted probability of no lost sale in its arrival period.",
            paper_title="Fixed Non-Stockout-Probability Policies for the Single-Item Lost-Sales Model",
            authors="Ton de Kok and Mirjam S. Meijer",
            year=2023,
            reference_url="https://arxiv.org/abs/2304.06936",
            doi=None,
            implementation_note=(
                "Exact discrete-PMF recursion rather than the paper's continuous-demand moment fit. For discrete demand, "
                "no stockout is D<=available, including exact depletion; this equals the paper's strict-positive event "
                "under its continuous-demand assumption. P3 uses a 0.005 grid plus 0.999 and is exhaustively fitted."
            ),
            horizon_dependent=False,
            implementation_version="computational-baseline-discrete-no-lost-sale",
            provider_factory=_factory(FP3Policy),
        ),
        ComputationalBaselineSpec(
            name="llm_or_projected_mv_capped",
            display_name="LLM-OR Projected Mean-Variance Capped",
            formula="q=min(C,[T-E(J)-gamma*sqrt(Var(J))]^+)",
            parameter_specs=(
                ParameterSpec("T", kind="continuous", lower=0.0, upper=2 * n, default=n / 2),
                ParameterSpec("gamma", kind="continuous", lower=0.0, upper=3.0, default=1.0),
                ParameterSpec("C", kind="integer", lower=0.0, upper=n, default=n),
            ),
            principle="Apply a capped order-up-to response to risk-adjusted projected usable inventory.",
            paper_title="LLMs Can Design Near-Optimal OR Algorithms",
            authors="J. Baek",
            year=2026,
            reference_url="https://doi.org/10.48550/arXiv.2608.27296",
            doi="10.48550/arXiv.2608.27296",
            implementation_note=(
                "Exact bounded-discrete projection computes the pre-arrival inventory mean and variance. "
                "The resulting real action is rounded and clipped using the benchmark's common policy boundary."
            ),
            horizon_dependent=False,
            implementation_version="computational-baseline-exact-projected-moments",
            provider_factory=_factory(ProjectedMeanVarianceCappedPolicy),
        ),
        ComputationalBaselineSpec(
            name="dual_balancing",
            display_name="Dual-Balancing (randomized integer)",
            formula="randomize q in {q1,q2} so E[H_marginal(q)-LostPenalty_arrival(q)|state]=0",
            parameter_specs=(),
            principle="Balance expected lifetime marginal holding cost of today's units against arrival-period lost-sales penalty.",
            paper_title="A 2-Approximation Algorithm for Stochastic Inventory Control Models with Lost Sales",
            authors="Retsef Levi, Ganesh Janakiraman, and Mahesh Nagarajan",
            year=2008,
            reference_url="https://doi.org/10.1287/moor.1070.0285",
            doi="10.1287/moor.1070.0285",
            implementation_note=(
                "Finite-horizon, exact discrete-PMF cost calculation. The adjacent integer actions and their mixing "
                "probability follow the paper's randomized integer construction. Both action branches are propagated "
                "exactly by the benchmark evaluator."
            ),
            horizon_dependent=True,
            implementation_version="computational-baseline-randomized-integer",
            provider_factory=_factory(RandomizedDualBalancingPolicy),
        ),
    )


def computational_baseline(name: str, lead_time: int, n: int) -> ComputationalBaselineSpec:
    available = {row.name: row for row in computational_baselines(lead_time, n)}
    if name not in available:
        raise ValueError(f"unknown computational baseline: {name}")
    return available[name]


__all__ = [
    "ComputationalBaselineSpec",
    "computational_baseline",
    "computational_baselines",
]
