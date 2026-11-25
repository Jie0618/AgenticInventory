"""SOF baselines for deterministic lead time with finite-patience backlog."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from inventory.core import DeterministicBacklog, DeterministicLeadTime, InventoryModelConfig
from inventory.policy.ast import AST, add, const, indicator_gt, maximum, minimum, motif, mul, parameter, state, sub
from inventory.policy.motifs.composition import motifs_for_config
from optimization.models.types import ParameterSpec


@dataclass(frozen=True)
class LeadTimeBacklogBaselineSpec:
    name: str
    category: str
    lead_time: int
    patience: int
    ast: AST
    parameter_specs: tuple[ParameterSpec, ...]
    motifs: Mapping[str, AST]
    formula: str
    principle: str
    reference: str
    display_name: str


def _integer(name: str, upper: int, default: int) -> ParameterSpec:
    return ParameterSpec(name, kind="integer", lower=0, upper=max(0, int(upper)),
                         default=max(0, min(int(default), int(upper))))


def _motifs(lead_time: int, patience: int) -> Mapping[str, AST]:
    return motifs_for_config(InventoryModelConfig(
        lead_time=DeterministicLeadTime(int(lead_time)),
        backlog=DeterministicBacklog(int(patience)),
    ))


def _spec(name, lead_time, patience, ast, specs, formula, principle, reference, display):
    return LeadTimeBacklogBaselineSpec(
        name=name, category="simple_formula", lead_time=int(lead_time),
        patience=int(patience), ast=ast, parameter_specs=tuple(specs),
        motifs=_motifs(lead_time, patience), formula=formula,
        principle=principle, reference=reference, display_name=display,
    )


def _positive_gap(position: AST) -> AST:
    return maximum(sub(parameter("S"), position), const(0.0))


def related_model_baselines(lead_time: int, patience: int, n: int):
    """Published rules transferred unchanged to finite-patience decision epochs.

    Kept separate from Stage1/2/3 defaults. SQ is Lim--Wei--Zhang's
    stockout-compensation policy, NOT a fixed-lot (s,Q) policy. QR is the
    single-fixed-lot periodic decision-epoch transfer, not continuous review
    or the multiple-lot (r,nQ) variant.
    """
    upper = (lead_time + 1) * n
    gross = motif("inventory_position")
    net = motif("net_inventory_position")
    empty = sub(const(1.0), indicator_gt(state("x"), const(0.0)))
    sq = maximum(sub(add(parameter("s"), mul(parameter("q"), empty)), gross), const(0.0))
    qr = mul(parameter("Q"), sub(const(1.0), indicator_gt(net, parameter("r"))))
    return (
        _spec("sq", lead_time, patience, sq,
              (_integer("s", upper, n), _integer("q", upper, n)),
              "u=[s+q*1{x=0}-x-pipeline]^+",
              "Original backlog-unobserved stockout compensation; geometric-to-fixed patience transfer.",
              "Lim, Wei and Zhang (2024), arXiv:2404.00046, eq. (18)", "SQ"),
        _spec("qr", lead_time, patience, qr,
              (ParameterSpec("r", kind="integer", lower=-n, upper=upper, default=n),
               ParameterSpec("Q", kind="integer", lower=1, upper=n, default=max(1, n//3))),
              "u=Q*1{netIP<=r}",
              "Fixed-lot reorder-point rule, checked once per period; different abandonment mechanism.",
              "Kim and Park (1985), DOI 10.1057/jors.1985.41; periodic decision-epoch transfer", "QR"),
    )


def baselines_for_lead_time_backlog(lead_time: int, patience: int, n: int) -> list[LeadTimeBacklogBaselineSpec]:
    if int(lead_time) < 1:
        raise ValueError("lead_time must be positive")
    if int(patience) < 1:
        raise ValueError("patience must be positive")
    if int(n) < 0:
        raise ValueError("n must be nonnegative")
    # On-hand contributes at most n, each of the L pipeline coordinates at
    # most n, and deterministic backlog is total-capped at n independently of
    # the number of age buckets. Target levels therefore scale with L+1, not
    # L+K; using patience in the bound creates increasingly large unreachable
    # search regions as K grows.
    target_upper = (int(lead_time) + 1) * int(n)
    gross = motif("inventory_position")
    net = motif("net_inventory_position")
    oldest = motif("imminent_abandonment_if_unserved")
    weighted_terms = [
        mul(parameter(f"omega{index}"), state(f"b{index}"))
        for index in range(1, patience + 1)
    ]
    weighted_backlog = weighted_terms[0] if len(weighted_terms) == 1 else add(*weighted_terms)
    weighted_position = sub(gross, weighted_backlog)
    net_gap = _positive_gap(net)
    active = sub(const(1.0), indicator_gt(net, parameter("s")))
    return [
        _spec("constant_order", lead_time, patience, parameter("Q"),
              (_integer("Q", n, max(1, n // 3)),), "q=Q",
              "State-independent replenishment sanity baseline.",
              "Goldberg et al. (2016), transferred diagnostic", "Constant Order"),
        _spec("gross_base_stock", lead_time, patience, _positive_gap(gross),
              (_integer("S", target_upper, n),), "q=[S-(x+sum p)]^+",
              "Ignore backlog and replenish gross inventory position.",
              "Lost-sales base-stock transfer", "Gross Base Stock"),
        _spec("net_base_stock", lead_time, patience, net_gap,
              (_integer("S", target_upper, n),), "q=[S-(x+sum p-B)]^+",
              "Credit outstanding backlog fully in net inventory position.",
              "Clark and Scarf (1960); Zipkin (2000)", "Net Base Stock"),
        _spec("net_capped_base_stock", lead_time, patience,
              minimum(net_gap, parameter("Qmax")),
              (_integer("S", target_upper, n), _integer("Qmax", n, max(1, n // 3))),
              "q=min(Qmax,[S-netIP]^+)",
              "Cap one-period recovery of the net inventory-position gap.",
              "Base-stock diagnostic extension", "Net Capped Base Stock"),
        _spec("net_s_s", lead_time, patience, mul(net_gap, active),
              (ParameterSpec(
                  "s", kind="integer", lower=0, upper=target_upper,
                  default=n // 2, monotone_group="net_s_s_thresholds",
                  monotone_index=0,
              ), ParameterSpec(
                  "S", kind="integer", lower=0, upper=target_upper,
                  default=n, monotone_group="net_s_s_thresholds",
                  monotone_index=1,
              )),
              "q=[S-netIP]^+ 1{netIP<=s}",
              "Add a transparent reorder dead zone on net inventory position.",
              "Zipkin (2000), transferred family", "Net (s,S)"),
        _spec("weighted_backlog_base_stock", lead_time, patience,
              _positive_gap(weighted_position),
              (_integer("S", target_upper, n), *(ParameterSpec(
                  f"omega{index}", kind="continuous", lower=0.0, upper=1.0,
                  default=index / patience, monotone_group="backlog_urgency_weights",
                  monotone_index=index - 1,
              ) for index in range(1, patience + 1))),
              "q=[S-(x+sum p-sum omega_j b_j)]^+",
              "Fit monotone urgency credits from youngest to oldest backlog.",
              "Our interpretable finite-patience candidate", "Weighted-Backlog Base Stock"),
        _spec("urgency_boosted_base_stock", lead_time, patience,
              minimum(parameter("Qmax"), add(net_gap, mul(parameter("alpha"), oldest))),
              (_integer("S", target_upper, n), _integer("Qmax", n, max(1, n // 2)),
               ParameterSpec("alpha", kind="continuous", lower=0.0, upper=2.0, default=1.0)),
              "q=min(Qmax,[S-netIP]^+ + alpha*b_oldest)",
              "Respond explicitly to backlog that will abandon if unserved.",
              "Our interpretable finite-patience candidate", "Urgency-Boosted Base Stock"),
    ]


__all__ = ["LeadTimeBacklogBaselineSpec", "baselines_for_lead_time_backlog"]
