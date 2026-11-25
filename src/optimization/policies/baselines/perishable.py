"""Simple operational-formula baselines for pure perishable inventory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from inventory.policy.ast import (
    AST,
    add,
    const,
    indicator_gt,
    maximum,
    minimum,
    motif,
    mul,
    neg,
    parameter,
    safe_ratio,
    state,
    sub,
)
from inventory.policy.motifs.perishable import perishable_motifs
from optimization.models.types import ParameterSpec


@dataclass(frozen=True)
class PerishableBaselineSpec:
    """One transparent fixed-shelf-life replenishment family."""

    name: str
    category: str
    shelf_life: int
    ast: AST
    parameter_specs: tuple[ParameterSpec, ...]
    motifs: Mapping[str, AST]
    formula: str
    principle: str
    reference: str
    display_name: str
    evidence_tier: str = "literature"


def baselines_for_perishable(
    shelf_life: int,
    n: int,
    *,
    include_extended: bool = False,
) -> list[PerishableBaselineSpec]:
    """Return the literature-backed SOF catalog for the pure-perishable model."""

    shelf_life = int(shelf_life)
    n = int(n)
    if shelf_life < 1:
        raise ValueError("shelf_life must be positive")
    if n < 0:
        raise ValueError("n must be nonnegative")
    rows = [
        constant_order(shelf_life, n),
        base_stock(shelf_life, n),
        reorder_point_order_up_to(shelf_life, n),
        capped_base_stock(shelf_life, n),
        bsp_low(shelf_life, n),
        weighted_base_stock(shelf_life, n),
    ]
    if include_extended:
        rows.append(bounded_reorder_point_order_up_to(shelf_life, n))
    return rows


def constant_order(shelf_life: int, n: int) -> PerishableBaselineSpec:
    return _spec(
        "constant_order",
        shelf_life,
        parameter("Q"),
        (_integer("Q", n, max(1, n // 3)),),
        "q=Q",
        "Use a state-independent replenishment quantity.",
        "Haijema and Minner (2016)",
        "Constant Order (COP)",
    )


def base_stock(shelf_life: int, n: int) -> PerishableBaselineSpec:
    total = motif("decision_inventory")
    return _spec(
        "base_stock",
        shelf_life,
        _positive(sub(parameter("S"), total)),
        (_integer("S", shelf_life * n, n),),
        "q=[S-X]^+",
        "Raise total decision-time inventory to a fitted target.",
        "Haijema and Minner (2016); Bu, Gong, and Chao (2023)",
        "Base Stock (BSP)",
    )


def reorder_point_order_up_to(
    shelf_life: int, n: int
) -> PerishableBaselineSpec:
    total = motif("decision_inventory")
    active = sub(const(1.0), indicator_gt(total, parameter("s")))
    return _spec(
        "s_s",
        shelf_life,
        mul(_positive(sub(parameter("S"), total)), active),
        (
            _integer("s", shelf_life * n, max(0, n // 2), "s_s_levels", 0),
            _integer("S", shelf_life * n, n, "s_s_levels", 1),
        ),
        "q=[S-X]^+ 1{X<=s}",
        "Add a reorder dead zone to a scalar order-up-to response.",
        "Haijema (2013); Liu and Lian (1999)",
        "(s,S)",
    )


def capped_base_stock(shelf_life: int, n: int) -> PerishableBaselineSpec:
    total = motif("decision_inventory")
    return _spec(
        "capped_base_stock",
        shelf_life,
        minimum(_positive(sub(parameter("S"), total)), parameter("Qmax")),
        (
            _integer("S", shelf_life * n, n),
            _integer("Qmax", n, max(1, n // 3)),
        ),
        "q=min(Qmax,[S-X]^+)",
        "Limit one-period recovery of a total-inventory gap.",
        "Haijema and Minner (2016)",
        "Capped Base Stock (SQmax)",
    )


def bounded_reorder_point_order_up_to(
    shelf_life: int, n: int
) -> PerishableBaselineSpec:
    total = motif("decision_inventory")
    active = sub(const(1.0), indicator_gt(total, parameter("s")))
    bounded = minimum(
        parameter("Qmax"),
        add(
            parameter("Qmin"),
            _positive(sub(sub(parameter("S"), total), parameter("Qmin"))),
        ),
    )
    return _spec(
        "s_s_qmin_qmax",
        shelf_life,
        mul(active, bounded),
        (
            _integer("s", shelf_life * n, max(0, n // 2), "s_s_bounded_levels", 0),
            _integer("S", shelf_life * n, n, "s_s_bounded_levels", 1),
            _integer("Qmin", n, 0, "s_s_order_bounds", 0),
            _integer("Qmax", n, n, "s_s_order_bounds", 1),
        ),
        "q=1{X<=s} min(Qmax,max(Qmin,S-X))",
        "Combine a reorder point, order-up-to target, and batch bounds.",
        "Haijema (2013)",
        "(s,S,Qmin,Qmax)",
    )


def bsp_low(shelf_life: int, n: int) -> PerishableBaselineSpec:
    total = motif("decision_inventory")
    low_branch = _positive(
        add(
            sub(parameter("S1"), total),
            safe_ratio(
                sub(
                    mul(total, parameter("S2")),
                    mul(total, parameter("S1")),
                ),
                parameter("tau"),
            ),
        )
    )
    high_branch = _positive(sub(parameter("S2"), total))
    use_low = indicator_gt(parameter("tau"), total)
    return _spec(
        "bsp_low",
        shelf_life,
        add(mul(use_low, low_branch), mul(sub(const(1.0), use_low), high_branch)),
        (
            _integer("S1", shelf_life * n, n),
            _integer("S2", shelf_life * n, n),
            ParameterSpec("tau", kind="integer", lower=1, upper=max(1, shelf_life * n), default=max(1, n // 2)),
        ),
        "q=[S1-(1-(S2-S1)/tau)X]^+ if X<tau else [S2-X]^+",
        "Use a thresholded piecewise-affine response to total inventory.",
        "Haijema and Minner (2016), L=0 specialization",
        "BSP-low",
    )


def weighted_base_stock(shelf_life: int, n: int) -> PerishableBaselineSpec:
    # Under immediate replenishment, a_M is a temporary within-period receipt
    # bucket.  The pre-order decision state contains the M-1 carry-over ages.
    active_ages = max(0, int(shelf_life) - 1)
    terms = [
        mul(parameter(f"omega{index}"), state(f"a{index}"))
        for index in range(1, active_ages + 1)
    ]
    effective: AST = const(0.0) if not terms else terms[0] if len(terms) == 1 else add(*terms)
    specs = [ParameterSpec("S", kind="integer", lower=0, upper=shelf_life * n, default=n)]
    specs.extend(
        ParameterSpec(
            f"omega{index}",
            kind="continuous",
            lower=0.0,
            upper=1.0,
            default=index / max(1, active_ages),
            monotone_group="perishable_age_weights",
            monotone_index=index - 1,
        )
        for index in range(1, active_ages + 1)
    )
    return _spec(
        "weighted_base_stock",
        shelf_life,
        _positive(sub(parameter("S"), effective)),
        tuple(specs),
        "q=[S-sum_j omega_j a_j]^+, 0<=omega_1<=...<=omega_{M-1}<=1",
        "Discount older inventory when computing the effective order-up-to position.",
        "Haijema and Minner (2019)",
        "Weighted Order-Up-To",
    )


def _spec(
    name: str,
    shelf_life: int,
    ast: AST,
    specs: tuple[ParameterSpec, ...],
    formula: str,
    principle: str,
    reference: str,
    display_name: str,
) -> PerishableBaselineSpec:
    return PerishableBaselineSpec(
        name=name,
        category="simple_formula",
        shelf_life=int(shelf_life),
        ast=ast,
        parameter_specs=specs,
        motifs=perishable_motifs(int(shelf_life)),
        formula=formula,
        principle=principle,
        reference=reference,
        display_name=display_name,
    )


def _positive(ast: AST) -> AST:
    return maximum(ast, const(0.0))


def _integer(
    name: str,
    upper: int,
    default: int,
    monotone_group: str | None = None,
    monotone_index: int | None = None,
) -> ParameterSpec:
    return ParameterSpec(
        name,
        kind="integer",
        lower=0,
        upper=max(0, int(upper)),
        default=max(0, min(int(default), int(upper))),
        monotone_group=monotone_group,
        monotone_index=monotone_index,
    )


__all__ = [
    "PerishableBaselineSpec",
    "base_stock",
    "baselines_for_perishable",
    "bounded_reorder_point_order_up_to",
    "bsp_low",
    "capped_base_stock",
    "constant_order",
    "reorder_point_order_up_to",
    "weighted_base_stock",
]
