"""Operational policy families for lead-time inventory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from inventory.policy.ast import AST, add, const, indicator_gt, maximum, minimum, motif, mul, neg, parameter, state
from inventory.policy.motifs.lead_time import lead_time_motifs
from optimization.models.types import ParameterSpec


@dataclass(frozen=True)
class BaselineSpec:
    """A transparent, distribution-free operational policy family."""

    name: str
    category: str
    lead_time: int
    ast: AST
    parameter_specs: tuple[ParameterSpec, ...]
    motifs: Mapping[str, AST]
    formula: str
    principle: str
    reference: str
    display_name: str = ""
    year: int | None = None
    evidence_tier: str = "literature"
    implementation_note: str = ""


def baselines_for_lead_time(lead_time: int, n: int) -> list[BaselineSpec]:
    """Return only policies admitted to the Simple Operational Formula class."""

    if lead_time < 0:
        raise ValueError("lead_time must be nonnegative")
    if lead_time == 0:
        return [base_stock(lead_time=0, n=n)]
    return [
        constant_order(lead_time, n),
        base_stock(lead_time, n),
        capped_base_stock(lead_time, n),
        vector_base_stock(lead_time, n),
        tilted_cbs(lead_time, n),
        tilted_pic(lead_time, n),
        weighted_tilted_pic(lead_time, n),
        aips_discounted_pipeline(lead_time, n),
        aips_hybrid(lead_time, n),
    ]


def lead_time_baselines(lead_time: int, n: int) -> list[BaselineSpec]:
    if lead_time < 1:
        raise ValueError("lead_time_baselines requires lead_time >= 1")
    return baselines_for_lead_time(lead_time, n)


def constant_order(lead_time: int, n: int) -> BaselineSpec:
    return BaselineSpec(
        name="constant_order",
        category="simple_formula",
        lead_time=lead_time,
        ast=parameter("r"),
        parameter_specs=(_integer_parameter("r", n, initial=max(1.0, n / 3.0)),),
        motifs=_motifs(lead_time),
        formula="q = r",
        principle="Use a state-independent replenishment rate.",
        reference="Goldberg et al. (2016)",
        display_name="Constant-Order (CO)",
        year=2016,
    )


def base_stock(lead_time: int, n: int) -> BaselineSpec:
    inventory = state("x") if lead_time == 0 else motif("inventory_position")
    high = n if lead_time == 0 else (lead_time + 1) * n
    return BaselineSpec(
        name="base_stock",
        category="simple_formula",
        lead_time=lead_time,
        ast=_nonnegative(add(parameter("S"), neg(inventory))),
        parameter_specs=(_integer_parameter("S", high, initial=float(n)),),
        motifs=_motifs(lead_time),
        formula="q = [S - IP]^+" if lead_time else "q = [S - x]^+",
        principle="Replenish the inventory-position gap to a fitted target.",
        reference="Clark and Scarf (1960); Zipkin (2000)",
        display_name="Base-Stock (BS)",
        year=1960,
    )


def capped_base_stock(lead_time: int, n: int) -> BaselineSpec:
    gap = _inventory_position_gap()
    return BaselineSpec(
        name="capped_base_stock",
        category="simple_formula",
        lead_time=lead_time,
        ast=minimum(gap, parameter("C")),
        parameter_specs=(
            _integer_parameter("S", (lead_time + 1) * n, initial=float(n)),
            _integer_parameter("C", n, initial=max(1.0, n / 3.0)),
        ),
        motifs=_motifs(lead_time),
        formula="q = min([S - IP]^+, C)",
        principle="Limit one-period recovery of an inventory-position gap.",
        reference="Xin (2021)",
        display_name="Capped Base-Stock (CBS)",
        year=2021,
    )


def vector_base_stock(lead_time: int, n: int) -> BaselineSpec:
    """Classic cumulative-pipeline constraints with independently fitted targets."""

    # Keep the exact cumulative-constraint formula while avoiding an O(L)-deep
    # AST: prefixes are reusable motifs and the final minimum is balanced.
    motifs = dict(_motifs(lead_time))
    gaps: list[AST] = []
    for index in range(lead_time + 1):
        cumulative = state("x") if index == 0 else motif(f"cumulative_inventory_{index}")
        gaps.append(add(parameter(f"S{index}"), neg(cumulative)))
    envelope = _minimum_all(gaps)
    specs = tuple(
        _integer_parameter(
            f"S{index}",
            (lead_time + 1) * n,
            initial=float(n),
            monotone_group="vector_base_stock_targets",
            monotone_index=index,
        )
        for index in range(lead_time + 1)
    )
    return BaselineSpec(
        name="vector_base_stock",
        category="simple_formula",
        lead_time=lead_time,
        ast=_nonnegative(envelope),
        parameter_specs=specs,
        motifs=motifs,
        formula="q=[min_l(S_l-u_l)]^+",
        principle="Impose separate targets on cumulative pipeline states.",
        reference="Morton/Zipkin vector base-stock policy",
        display_name="Vector Base-Stock (VBS)",
        evidence_tier="literature",
    )


def tilted_cbs(lead_time: int, n: int) -> BaselineSpec:
    gap = _inventory_position_gap()
    elastic_cap = add(parameter("C"), mul(parameter("alpha"), gap))
    return BaselineSpec(
        name="tilted_cbs",
        category="simple_formula",
        lead_time=lead_time,
        ast=minimum(gap, elastic_cap),
        parameter_specs=_tilted_parameter_specs(lead_time, n, include_k=False),
        motifs=_motifs(lead_time),
        formula="q = min(Delta, C + alpha*Delta), Delta=[S-IP]^+",
        principle="Use an order cap that increases linearly with the gap.",
        reference="Huang et al. (2026), InvEvolve preprint",
        display_name="Tilted-CBS",
        year=2026,
        evidence_tier="preprint",
    )


def tilted_pic(lead_time: int, n: int) -> BaselineSpec:
    gap = _inventory_position_gap()
    return BaselineSpec(
        name="tilted_pic",
        category="simple_formula",
        lead_time=lead_time,
        ast=minimum(
            mul(parameter("K"), gap),
            add(parameter("C"), mul(parameter("alpha"), gap)),
        ),
        parameter_specs=_tilted_parameter_specs(lead_time, n, include_k=True),
        motifs=_motifs(lead_time),
        formula="q = min(K*Delta, C + alpha*Delta), Delta=[S-IP]^+",
        principle="Apply proportional gap damping and an elastic recovery cap.",
        reference="Huang et al. (2026), InvEvolve preprint",
        display_name="Tilted-PIC",
        year=2026,
        evidence_tier="preprint",
    )


def weighted_tilted_pic(lead_time: int, n: int) -> BaselineSpec:
    """Pipeline-aware SOF skeleton recommended by the new research report."""

    weighted_inventory = add(
        state("x"),
        *(mul(parameter(f"w{index}"), state(f"p{index}")) for index in range(1, lead_time + 1)),
    )
    gap = _nonnegative(add(parameter("S"), neg(weighted_inventory)))
    specs = list(_tilted_parameter_specs(lead_time, n, include_k=True))
    specs.extend(
        ParameterSpec(f"w{index}", kind="continuous", lower=0.0, upper=1.0, default=1.0)
        for index in range(1, lead_time + 1)
    )
    return BaselineSpec(
        name="weighted_tilted_pic",
        category="candidate_simple_formula",
        lead_time=lead_time,
        ast=minimum(
            mul(parameter("K"), gap),
            add(parameter("C"), mul(parameter("alpha"), gap)),
        ),
        parameter_specs=tuple(specs),
        motifs=_motifs(lead_time),
        formula="q=min(K*Delta_w,C+alpha*Delta_w), Delta_w=[S-x-sum(w_l p_l)]^+",
        principle="Retain pipeline timing while using a transparent two-slope response.",
        reference="Candidate family specified in docs/research/simple_operational_formula_lost_sales_report.md",
        display_name="Weighted Tilted-PIC (candidate)",
        evidence_tier="candidate",
    )


def aips_discounted_pipeline(lead_time: int, n: int) -> BaselineSpec:
    """AIPS transferable policy with shared pipeline credit and a dead zone."""

    weighted_inventory = add(
        state("x"), mul(parameter("omega"), motif("total_pipeline"))
    )
    gap = add(parameter("S"), neg(weighted_inventory))
    positive_gap = _nonnegative(gap)
    return BaselineSpec(
        name="aips_discounted_pipeline",
        category="llm_discovered_simple_formula",
        lead_time=lead_time,
        ast=mul(
            parameter("rho"),
            positive_gap,
            indicator_gt(gap, parameter("tau")),
        ),
        parameter_specs=(
            _integer_parameter("S", (lead_time + 1) * n, initial=float(n)),
            ParameterSpec("omega", kind="continuous", lower=0.0, upper=1.0, default=1.0),
            ParameterSpec("rho", kind="continuous", lower=0.0, upper=1.0, default=1.0),
            _integer_parameter("tau", (lead_time + 1) * n, initial=0.0),
        ),
        motifs=_motifs(lead_time),
        formula="q=rho*[S-x-omega*sum(p_l)]^+*1{S-x-omega*sum(p_l)>tau}",
        principle="Discount all pipeline inventory uniformly, then apply proportional recovery outside a dead zone.",
        reference="Yang et al. (2026), AIPS, arXiv:2609.08071",
        display_name="AIPS Discounted-Pipeline",
        year=2026,
        evidence_tier="preprint",
    )


def aips_hybrid(lead_time: int, n: int) -> BaselineSpec:
    """Behavior-equivalent AIPS hybrid parameterization using effective weights."""

    weighted_inventory = add(
        state("x"),
        *(
            mul(parameter(f"v{index}"), state(f"p{index}"))
            for index in range(1, lead_time + 1)
        ),
    )
    affine_order = add(
        mul(
            parameter("rho"),
            add(parameter("S"), neg(weighted_inventory)),
        ),
        mul(
            add(const(1.0), neg(parameter("rho"))),
            parameter("delta"),
        ),
    )
    specs: list[ParameterSpec] = [
        _integer_parameter("S", (lead_time + 1) * n, initial=float(n)),
    ]
    specs.extend(
        ParameterSpec(
            f"v{index}", kind="continuous", lower=-1.0, upper=1.0, default=1.0
        )
        for index in range(1, lead_time + 1)
    )
    specs.extend(
        (
            ParameterSpec("rho", kind="continuous", lower=0.0, upper=1.0, default=1.0),
            _integer_parameter("delta", n, initial=max(1.0, n / 3.0)),
            _integer_parameter("a_min", n, initial=0.0),
        )
    )
    return BaselineSpec(
        name="aips_hybrid",
        category="llm_discovered_simple_formula",
        lead_time=lead_time,
        ast=maximum(parameter("a_min"), affine_order),
        parameter_specs=tuple(specs),
        motifs=_motifs(lead_time),
        formula="q=max(a_min,rho*(S-x-sum(v_l*p_l))+(1-rho)*delta)",
        principle="Combine position-specific pipeline credit, partial adjustment, constant-order mixing, and lower clipping.",
        reference="Yang et al. (2026), AIPS, arXiv:2609.08071",
        display_name="AIPS Hybrid",
        year=2026,
        evidence_tier="preprint",
        implementation_note=(
            "Uses the paper's behavior-equivalent effective weights v_l=omega_l-alpha*beta_l. "
            "The fitted bounds v_l in [-1,1] are explicit and the original decomposed parameters are not reported."
        ),
    )
def _inventory_position_gap() -> AST:
    return _nonnegative(add(parameter("S"), neg(motif("inventory_position"))))


def _nonnegative(node: AST) -> AST:
    """Canonical `[node]^+ = max(node, 0)` representation in the minimal grammar."""

    return maximum(node, const(0.0))


def _minimum_all(nodes: list[AST]) -> AST:
    """Build a balanced minimum tree without changing the formula value."""

    if not nodes:
        raise ValueError("minimum requires at least one node")
    level = list(nodes)
    while len(level) > 1:
        next_level: list[AST] = []
        for index in range(0, len(level), 2):
            if index + 1 < len(level):
                next_level.append(minimum(level[index], level[index + 1]))
            else:
                next_level.append(level[index])
        level = next_level
    return level[0]


def _integer_parameter(
    name: str,
    high: int,
    initial: float,
    *,
    monotone_group: str | None = None,
    monotone_index: int | None = None,
) -> ParameterSpec:
    return ParameterSpec(
        name,
        kind="integer",
        lower=0.0,
        upper=float(high),
        default=initial,
        monotone_group=monotone_group,
        monotone_index=monotone_index,
    )


def _tilted_parameter_specs(
    lead_time: int, n: int, *, include_k: bool
) -> tuple[ParameterSpec, ...]:
    specs: list[ParameterSpec] = [
        _integer_parameter("S", (lead_time + 1) * n, initial=float(n)),
        _integer_parameter("C", n, initial=max(1.0, n / 3.0)),
        ParameterSpec("alpha", kind="continuous", lower=0.0, upper=1.0, default=0.25),
    ]
    if include_k:
        specs.insert(
            1,
            ParameterSpec("K", kind="continuous", lower=0.0, upper=1.5, default=1.0),
        )
    return tuple(specs)


def _motifs(lead_time: int) -> Mapping[str, AST]:
    return {} if lead_time == 0 else lead_time_motifs(lead_time)
