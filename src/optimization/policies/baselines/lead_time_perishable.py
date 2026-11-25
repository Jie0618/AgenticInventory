"""SOF baselines for deterministic positive-lead-time perishables.

The formulas use the canonical composition motifs, so the same definitions
work for any fixed shelf life ``M`` and deterministic lead time ``L``.
"""

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
    parameter,
    sub,
    safe_ratio,
    state,
)
from inventory.policy.motifs.composition import motifs_for_config
from inventory.core.config import DeterministicLeadTime, FixedShelfLife, InventoryModelConfig
from optimization.models.types import ParameterSpec


@dataclass(frozen=True)
class LeadTimePerishableBaselineSpec:
    name: str
    category: str
    lead_time: int
    shelf_life: int
    ast: AST
    parameter_specs: tuple[ParameterSpec, ...]
    motifs: Mapping[str, AST]
    formula: str
    principle: str
    reference: str
    display_name: str
    evidence_tier: str = "literature"


def baselines_for_lead_time_perishable(
    lead_time: int, shelf_life: int, n: int, *, include_extended: bool = False,
    include_age_split: bool = False,
) -> list[LeadTimePerishableBaselineSpec]:
    if int(lead_time) < 1:
        raise ValueError("lead_time must be positive")
    if int(shelf_life) < 1:
        raise ValueError("shelf_life must be positive")
    if int(n) < 0:
        raise ValueError("n must be nonnegative")
    rows = [
        constant_order(lead_time, shelf_life, n),
        base_stock(lead_time, shelf_life, n),
        reorder_point_order_up_to(lead_time, shelf_life, n),
        copsq(lead_time, shelf_life, n),
        capped_base_stock(lead_time, shelf_life, n),
        bsp_low(lead_time, shelf_life, n),
        weighted_base_stock(lead_time, shelf_life, n),
    ]
    if include_extended:
        rows.append(bounded_reorder_point_order_up_to(lead_time, shelf_life, n))
    if include_age_split:
        rows.extend((bsp_2d(lead_time, shelf_life, n), bsp_oir(lead_time, shelf_life, n)))
    return rows


def _motifs(lead_time: int, shelf_life: int) -> Mapping[str, AST]:
    # Constructing through the canonical config prevents accidental use of
    # pure-perishable decision_inventory in a positive-lead-time formula.
    config = InventoryModelConfig(
        lead_time=DeterministicLeadTime(int(lead_time)),
        perishability=FixedShelfLife(int(shelf_life)),
    )
    return motifs_for_config(config)


def _gap() -> AST:
    return maximum(sub(parameter("S"), motif("inventory_position")), const(0.0))


def _spec(name, lead_time, shelf_life, ast, specs, formula, principle, reference, display):
    return LeadTimePerishableBaselineSpec(
        name=name, category="simple_formula", lead_time=int(lead_time),
        shelf_life=int(shelf_life), ast=ast, parameter_specs=tuple(specs),
        motifs=_motifs(lead_time, shelf_life), formula=formula,
        principle=principle, reference=reference, display_name=display,
    )


def _integer(name: str, upper: int, default: int, group=None, index=None) -> ParameterSpec:
    return ParameterSpec(name, kind="integer", lower=0, upper=max(0, int(upper)),
                         default=max(0, min(int(default), int(upper))),
                         monotone_group=group, monotone_index=index)


def constant_order(L, M, n):
    return _spec("constant_order", L, M, parameter("Q"), (_integer("Q", n, max(1, n // 3)),),
                 "q=Q", "State-independent replenishment", "Haijema and Minner (2016)", "Constant Order (COP)")


def base_stock(L, M, n):
    return _spec("base_stock", L, M, _gap(), (_integer("S", (L + M) * n, n),),
                 "q=[S-IP]^+", "Inventory-position base stock", "Haijema and Minner (2016)", "Base Stock (BSP)")


def reorder_point_order_up_to(L, M, n):
    ip = motif("inventory_position")
    active = sub(const(1.0), indicator_gt(ip, parameter("s")))
    return _spec("s_s", L, M, mul(_gap(), active),
                 (_integer("s", (L + M) * n, n // 2), _integer("S", (L + M) * n, n)),
                 "q=[S-IP]^+ 1{IP<=s}", "Reorder dead zone", "Haijema (2013); Liu and Lian (1999)", "(s,S)")


def copsq(L, M, n):
    ip = motif("inventory_position")
    active = sub(const(1.0), indicator_gt(ip, parameter("s")))
    return _spec("copsq", L, M, mul(parameter("Q"), active),
                 (_integer("s", (L + M) * n, n // 2), _integer("Q", n, max(1, n // 3))),
                 "q=Q 1{IP<=s}", "Constant order below a reorder point", "Haijema and Minner (2016)", "COPsQ")


def capped_base_stock(L, M, n):
    return _spec("capped_base_stock", L, M, minimum(_gap(), parameter("Qmax")),
                 (_integer("S", (L + M) * n, n), _integer("Qmax", n, max(1, n // 3))),
                 "q=min(Qmax,[S-IP]^+)", "Capped inventory-position recovery", "Haijema and Minner (2016)", "Capped BSP")


def bsp_low(L, M, n):
    ip = motif("inventory_position")
    # alpha = 1 - (S2-S1)/tau; tau is constrained to be at least one.
    # Algebraically rewrite S1-(1-(S2-S1)/tau)IP as
    # S1-IP + ((S2-S1)/tau)IP.  This is exactly equivalent and keeps the AST
    # within the canonical depth limit.
    low = add(sub(parameter("S1"), ip),
              mul(safe_ratio(sub(parameter("S2"), parameter("S1")), parameter("tau")), ip))
    high = sub(parameter("S2"), ip)
    # Inventory and tau are integer-valued; IP >= tau is encoded as
    # IP > tau-1 because the AST exposes a strict indicator_gt primitive.
    use_high = indicator_gt(ip, sub(parameter("tau"), const(1.0)))
    return _spec("bsp_low", L, M, add(low, mul(use_high, sub(high, low))),
                 (_integer("S1", (L + M) * n, n), _integer("S2", (L + M) * n, n),
                  _integer("tau", (L + M) * n, max(1, n // 2))),
                 "piecewise [S1-alpha IP]^+ / [S2-IP]^+", "Piecewise-affine inventory-position response", "Haijema and Minner (2016)", "BSP-low")


def bounded_reorder_point_order_up_to(L, M, n):
    ip = motif("inventory_position")
    active = sub(const(1.0), indicator_gt(ip, parameter("s")))
    bounded = minimum(parameter("Qmax"), maximum(parameter("Qmin"), sub(parameter("S"), ip)))
    return _spec("s_s_qmin_qmax", L, M, mul(active, bounded),
                 (_integer("s", (L + M) * n, n // 2), _integer("S", (L + M) * n, n),
                  _integer("Qmin", n, 0), _integer("Qmax", n, n)),
                 "q=1{IP<=s}min(Qmax,max(Qmin,S-IP))", "Bounded reorder-point policy", "Haijema (2013)", "(s,S,Qmin,Qmax)")


def weighted_base_stock(L, M, n):
    ages = [mul(parameter(f"omega{i}"), state(f"a{i}")) for i in range(1, M + 1)]
    weighted = ages[0] if len(ages) == 1 else add(*ages)
    weighted_position = add(weighted, motif("total_pipeline"))
    specs = [_integer("S", (L + M) * n, n)]
    specs.extend(ParameterSpec(f"omega{i}", kind="continuous", lower=0.0, upper=1.0,
                                default=i / max(1, M), monotone_group="perishable_age_weights", monotone_index=i - 1)
                  for i in range(1, M + 1))
    return _spec("weighted_base_stock", L, M, maximum(sub(parameter("S"), weighted_position), const(0.0)),
                 specs, "q=[S-sum(omega_j a_j)-sum(p_l)]^+", "Age-aware weighted stock position", "Haijema and Minner (2019)", "Weighted BSP")


def _old_inventory(M):
    """Nontrivial old/young splits of the M-1 pre-receipt age classes.

    ``k`` counts the oldest remaining-life buckets. For M=3 the only split
    is old=a1 and young=a2+pipeline, so k is fixed rather than fitted.
    All-young splits reduce to BSP and are already represented by the catalog.
    """
    upper = max(0, M - 2)
    if upper == 0:
        return const(0.0), ()
    if upper == 1:
        return state("a1"), ()
    terms = [mul(state(f"a{j}"), indicator_gt(parameter("k"), const(j - 1)))
             for j in range(1, upper + 1)]
    return add(*terms), (ParameterSpec("k", kind="integer", lower=1, upper=upper, default=1),)


def bsp_ws3(L, M, n):
    """Haijema/Minner (2019), eqs. (8)-(9), in remaining-life coordinates.

    a1 is oldest; aM is zero at reachable pre-receipt decision epochs.
    Only the three oldest extant bins are weighted, independently in [0,1].
    Younger inventory and every pipeline bin retain unit weight.
    """
    count = min(3, M - 1)
    terms = [mul(parameter(f"w{j}"), state(f"a{j}")) if j <= count else state(f"a{j}")
             for j in range(1, M)]
    position = add(motif("total_pipeline"), *terms) if terms else motif("total_pipeline")
    specs = [_integer("S", (L + M) * n, n)]
    specs.extend(ParameterSpec(f"w{j}", kind="continuous", lower=0.0, upper=1.0,
                               default=1.0) for j in range(1, count + 1))
    return _spec("bsp_ws3", L, M, maximum(sub(parameter("S"), position), const(0.0)),
                 specs, "q=[S-pipeline-sum_{j<=3}(w_j*a_j)-sum_{j>3}(a_j)]^+",
                 "Independent weights on the three oldest on-hand age classes",
                 "Haijema and Minner (2019), eqs. (8)-(9)", "BSP-WS3")


def bsp_2d(L, M, n):
    old, partition_specs = _old_inventory(M)
    young = sub(motif("inventory_position"), old)
    ast = maximum(maximum(const(0.0), sub(parameter("ST"), motif("inventory_position"))),
                  sub(parameter("SY"), young))
    return _spec("bsp_2d", L, M, ast,
                 (_integer("ST", (L + M) * n, n), _integer("SY", (L + M) * n, n), *partition_specs),
                 "q=max(0,ST-IP,SY-young); young=IP-old",
                 "Maintain separate total-stock and young-stock targets",
                 "Haijema et al. (2007), as defined in Haijema and Minner (2019), eq. (6)", "BSP-2D")


def bsp_oir(L, M, n):
    old, partition_specs = _old_inventory(M)
    ip = motif("inventory_position")
    # Compare old >= delta*IP directly. The AST's safe_ratio is a smoothed
    # ratio x/(1+abs(y)), not ordinary division. Cross multiplication keeps
    # the paper's threshold and equality exact, including at empty stock.
    replace_old = sub(const(1.0), indicator_gt(mul(parameter("delta"), ip), old))
    ast = maximum(const(0.0), add(sub(parameter("S"), ip), mul(old, replace_old)))
    return _spec("bsp_oir", L, M, ast,
                 (_integer("S", (L + M) * n, n),
                  ParameterSpec("delta", kind="continuous", lower=0.0, upper=1.0, default=0.5),
                  *partition_specs),
                 "q=[S-IP+old*1{old/IP>=delta}]^+",
                 "Replace old stock when its share of the inventory position is large",
                 "Duan and Liao (2013), as defined in Haijema and Minner (2019), eq. (7)", "BSP-OIR")


__all__ = ["LeadTimePerishableBaselineSpec", "baselines_for_lead_time_perishable"]
