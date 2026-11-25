"""Composable state-transition kernels for the canonical inventory model.

The public state contract is deliberately small: :class:`InventoryState`
contains an on-hand branch, a countdown pipeline branch, and a backlog
branch.  This module supplies the numerical semantics for those branches and
keeps deterministic and known-event stochastic transitions on the same code
path.  In particular, a random lead time is sampled when an order is placed;
there is no hidden-ETA/hazard representation here.

The event order is the one recorded by :class:`~inventory.core.config.TransitionSpec`:
order/event realization, due receipts, backlog service, demand service, cost,
then aging and exits.  ``outcomes`` enumerates the finite random support for
exact solvers, while ``sample`` uses the same deterministic application
routine for simulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from inventory.common.numeric import integer_scalar
from inventory.core.components.backlog import (
    advance_backlog,
    binomial_branches as _binomial_branches,
    binomial_sample as _binomial_sample,
    classify_unmet,
    serve_backlog,
)
from inventory.core.components.lead_time import (
    lead_time_choices,
    pipeline_and_arrival,
)
from inventory.core.components.perishability import (
    age_and_project_inventory,
    consume_inventory,
    lifetime_choices,
    receive_inventory,
)
from inventory.core.config import (
    BacklogMode,
    DEFAULT_EVENT_ORDER,
    DEFAULT_TRANSITION_VERSION,
    InventoryModelConfig,
)
from inventory.core.state import (
    CanonicalStateSchema,
    InventoryState,
    state_schema_for_config,
)


@dataclass(frozen=True)
class TransitionCosts:
    """Unit costs charged by a canonical transition.

    ``shortage`` is charged only for current demand lost without entering
    backlog. Entering backlog carries no one-time shortage charge.
    ``backlog`` is an optional per-unit carry cost charged on
    backlog retained into the next state after survival and any explicit
    finite-grid projection.  It defaults to zero so existing lost-sales and
    perishable cost conventions are preserved.
    """

    holding: float = 0.0
    shortage: float = 0.0
    expiry: float = 0.0
    backlog: float = 0.0
    abandonment: float = 0.0
    ordering: float = 0.0

    def __post_init__(self) -> None:
        names = ("holding", "shortage", "expiry", "backlog", "abandonment", "ordering")
        for name in names:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} cost must be finite and nonnegative")
            object.__setattr__(self, name, value)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "TransitionCosts":
        """Build costs from a mapping, accepting common short aliases."""

        aliases = {
            "h": "holding",
            "b": "shortage",
            "w": "expiry",
            "g": "backlog",
            "a": "abandonment",
            "k": "ordering",
            "holding_cost": "holding",
            "shortage_cost": "shortage",
            "expiry_cost": "expiry",
            "backlog_cost": "backlog",
            "abandonment_cost": "abandonment",
            "order_cost": "ordering",
            "ordering_cost": "ordering",
        }
        payload: dict[str, Any] = {}
        for key, value in values.items():
            payload[aliases.get(str(key), str(key))] = value
        allowed = {
            "holding",
            "shortage",
            "expiry",
            "backlog",
            "abandonment",
            "ordering",
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"unknown transition cost fields: {', '.join(unknown)}")
        return cls(**payload)


@dataclass(frozen=True)
class ExogenousEvent:
    """Realized random events supplied to one transition application.

    ``lead_time`` is the realized ETA of the newly placed order, ``lifetime``
    is the realized life of the due receipt, and ``abandoned`` is the number
    of eligible geometric-backlog units that abandon at the end of the
    period.  ``None`` means that the kernel should sample/derive the value.
    Mapping inputs with the aliases ``eta`` and ``shelf_life`` are accepted by
    :meth:`from_value`.
    """

    lead_time: Optional[int] = None
    lifetime: Optional[int] = None
    abandoned: Optional[int] = None

    @classmethod
    def from_value(cls, value: Any) -> "ExogenousEvent":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("random_event must be an ExogenousEvent or mapping")
        aliases = {
            "eta": "lead_time",
            "realized_lead_time": "lead_time",
            "shelf_life": "lifetime",
        }
        payload = {aliases.get(str(key), str(key)): item for key, item in value.items()}
        allowed = {"lead_time", "lifetime", "abandoned"}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"unknown exogenous event fields: {', '.join(unknown)}")
        return cls(**payload)

    def to_dict(self) -> dict[str, Optional[int]]:
        return {
            "lead_time": self.lead_time,
            "lifetime": self.lifetime,
            "abandoned": self.abandoned,
        }


@dataclass(frozen=True)
class TransitionOutcome:
    """One realized transition and its diagnostic quantities."""

    next_state: InventoryState
    holding_cost: float = 0.0
    shortage_cost: float = 0.0
    expiry_cost: float = 0.0
    backlog_cost: float = 0.0
    abandonment_cost: float = 0.0
    ordering_cost: float = 0.0
    action: int = 0
    demand: int = 0
    arrivals: int = 0
    served_backlog: int = 0
    served_demand: int = 0
    lost_sales: int = 0
    new_backlog: int = 0
    expired: int = 0
    abandoned: int = 0
    discarded: int = 0
    event: ExogenousEvent = field(default_factory=ExogenousEvent)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def total_cost(self) -> float:
        return float(
            self.holding_cost
            + self.shortage_cost
            + self.expiry_cost
            + self.backlog_cost
            + self.abandonment_cost
            + self.ordering_cost
        )

    @property
    def shortage_quantity(self) -> int:
        return int(self.lost_sales + self.new_backlog)

    @property
    def backlog_served(self) -> int:
        return int(self.served_backlog)

    @property
    def expiry_quantity(self) -> int:
        return int(self.expired)

    def flatten(self, schema: CanonicalStateSchema) -> tuple[int, ...]:
        return self.next_state.flatten(schema)


@dataclass(frozen=True)
class TransitionBranch:
    """A finite-support stochastic transition branch."""

    probability: float
    outcome: TransitionOutcome

    def __post_init__(self) -> None:
        probability = float(self.probability)
        if not math.isfinite(probability) or probability < 0.0:
            raise ValueError("branch probability must be finite and nonnegative")
        object.__setattr__(self, "probability", probability)

    @property
    def next_state(self) -> InventoryState:
        return self.outcome.next_state

    @property
    def total_cost(self) -> float:
        return self.outcome.total_cost


class TransitionEnumerationError(RuntimeError):
    """Raised when exact finite-support enumeration exceeds its safety bound."""


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    return integer_scalar(value, name=name, minimum=minimum)


def _weighted_choice(
    rng: Any, values: Sequence[int], probabilities: Sequence[float]
) -> int:
    if hasattr(rng, "choice"):
        try:
            return int(
                rng.choice(
                    np.asarray(values, dtype=int),
                    p=np.asarray(probabilities, dtype=float),
                )
            )
        except TypeError:
            pass
    draw = float(rng.random())
    cumulative = 0.0
    for value, probability in zip(values, probabilities):
        cumulative += float(probability)
        if draw < cumulative:
            return int(value)
    return int(values[-1])


class ComposedTransitionKernel:
    """Transition kernel compiled from one :class:`InventoryModelConfig`.

    Parameters ``h``, ``b`` and ``w`` are convenience aliases for the first
    three fields of :class:`TransitionCosts`; callers may instead pass a
    ``TransitionCosts`` instance or a mapping.  The kernel accepts either a
    structured :class:`InventoryState` or a flat tuple in schema order and
    always returns a structured state in its outcome.
    """

    def __init__(
        self,
        config: InventoryModelConfig,
        costs: Optional[TransitionCosts | Mapping[str, Any]] = None,
        *,
        h: Optional[float] = None,
        b: Optional[float] = None,
        w: Optional[float] = None,
        backlog_cost: Optional[float] = None,
        abandonment_cost: Optional[float] = None,
        ordering_cost: Optional[float] = None,
        schema: Optional[CanonicalStateSchema] = None,
        action_cap: Optional[int] = None,
        max_outcomes: int = 10000,
    ) -> None:
        if not isinstance(config, InventoryModelConfig):
            raise TypeError("config must be an InventoryModelConfig")
        if config.transition.version != DEFAULT_TRANSITION_VERSION:
            raise ValueError(
                "unsupported transition version "
                f"{config.transition.version!r}; expected "
                f"{DEFAULT_TRANSITION_VERSION!r}"
            )
        if config.transition.event_order != DEFAULT_EVENT_ORDER:
            raise ValueError("transition event order is not supported by this kernel")
        if costs is None:
            base = TransitionCosts()
        elif isinstance(costs, TransitionCosts):
            base = costs
        elif isinstance(costs, Mapping):
            base = TransitionCosts.from_mapping(costs)
        else:
            raise TypeError("costs must be TransitionCosts, mapping, or None")
        overrides = {
            "holding": h,
            "shortage": b,
            "expiry": w,
            "backlog": backlog_cost,
            "abandonment": abandonment_cost,
            "ordering": ordering_cost,
        }
        payload = {name: getattr(base, name) for name in overrides}
        payload.update(
            {name: value for name, value in overrides.items() if value is not None}
        )
        self.config = config
        self.schema = state_schema_for_config(config) if schema is None else schema
        if not isinstance(self.schema, CanonicalStateSchema):
            raise TypeError("schema must be a CanonicalStateSchema")
        expected_schema = state_schema_for_config(config, model_id=self.schema.model_id)
        if self.schema != expected_schema:
            raise ValueError("schema does not match the transition configuration")
        self.costs = TransitionCosts(**payload)
        self.action_cap = (
            None if action_cap is None else _integer(action_cap, name="action_cap")
        )
        self.max_outcomes = _integer(max_outcomes, name="max_outcomes", minimum=1)

    def _state(self, state: InventoryState | Sequence[int]) -> InventoryState:
        if isinstance(state, InventoryState):
            self.schema.validate(state)
            return state
        return InventoryState.from_flat(state, self.schema)

    def _validate_action_demand(self, action: int, demand: int) -> tuple[int, int]:
        action_i = _integer(action, name="action")
        demand_i = _integer(demand, name="demand")
        if self.action_cap is not None and action_i > self.action_cap:
            raise ValueError("action exceeds configured action_cap")
        return action_i, demand_i

    def _lead_choices(
        self, action: int, event: ExogenousEvent
    ) -> tuple[tuple[Optional[int], float], ...]:
        return lead_time_choices(self.config, action, event.lead_time)

    def _lifetime_choices(
        self, incoming: int, event: ExogenousEvent
    ) -> tuple[tuple[Optional[int], float], ...]:
        return lifetime_choices(self.config, incoming, event.lifetime)

    def _pipeline_and_arrival(
        self, state: InventoryState, action: int, eta: int
    ) -> tuple[tuple[int, ...], int, int]:
        return pipeline_and_arrival(self.config, state, action, eta)

    def _apply(
        self,
        state: InventoryState,
        action: int,
        demand: int,
        *,
        event: ExogenousEvent,
        eta: int,
        lifetime: Optional[int],
        abandoned: int = 0,
    ) -> tuple[TransitionOutcome, int]:
        """Apply one fully realized event and return (outcome, eligible B)."""

        next_pipeline, due_arrivals, pipeline_discarded = self._pipeline_and_arrival(
            state, action, eta
        )
        incoming = int(due_arrivals)
        discarded = int(pipeline_discarded)

        # One due pipeline coordinate is one aggregate receipt batch, so all
        # received units share the realized lifetime in transition.
        inventory_tuple = receive_inventory(
            self.config, state.inventory, incoming, lifetime
        )

        def consume(values: tuple[int, ...], quantity: int):
            return consume_inventory(self.config, values, quantity)

        inventory_tuple, backlog, served_backlog = serve_backlog(
            inventory_tuple,
            state.backlog,
            consume,
            scalar_total=self.config.backlog.uses_scalar_total,
        )

        inventory_tuple, served_demand, unmet = consume(inventory_tuple, demand)
        new_backlog, lost_sales = classify_unmet(self.config.backlog.mode, unmet)

        next_inventory, ending_inventory, expired, inventory_discard = (
            age_and_project_inventory(self.config, inventory_tuple)
        )
        discarded += int(inventory_discard)

        next_backlog, eligible_abandonment, abandoned_i, back_discard = (
            advance_backlog(
                self.config,
                backlog,
                new_backlog,
                abandoned,
            )
        )
        discarded += int(back_discard)

        # When no inventory is discarded, I_end = sum(next_inventory) + expired.
        # Thus h*I_end + w_code*expired = h*sum(next_inventory) + (w_code+h)*expired,
        # equivalent to the paper's model with w_paper = w_code + h.
        holding_cost = self.costs.holding * ending_inventory
        shortage_cost = self.costs.shortage * lost_sales
        expiry_cost = self.costs.expiry * expired
        backlog_cost = self.costs.backlog * sum(next_backlog)
        abandonment_cost = self.costs.abandonment * abandoned_i
        ordering_cost = self.costs.ordering * action
        realized_event = ExogenousEvent(
            lead_time=int(eta),
            lifetime=None if lifetime is None else int(lifetime),
            abandoned=int(abandoned_i),
        )
        outcome = TransitionOutcome(
            next_state=InventoryState(next_inventory, next_pipeline, next_backlog),
            holding_cost=float(holding_cost),
            shortage_cost=float(shortage_cost),
            expiry_cost=float(expiry_cost),
            backlog_cost=float(backlog_cost),
            abandonment_cost=float(abandonment_cost),
            ordering_cost=float(ordering_cost),
            action=int(action),
            demand=int(demand),
            arrivals=int(incoming),
            served_backlog=int(served_backlog),
            served_demand=int(served_demand),
            lost_sales=int(lost_sales),
            new_backlog=int(new_backlog),
            expired=int(expired),
            abandoned=int(abandoned_i),
            discarded=int(discarded),
            event=realized_event,
            metadata={
                "ending_inventory": ending_inventory,
                "eligible_geometric_abandonment": int(eligible_abandonment),
            },
        )
        return outcome, int(eligible_abandonment)

    def outcomes(
        self,
        state: InventoryState | Sequence[int],
        action: int,
        demand: int,
        *,
        random_event: ExogenousEvent | Mapping[str, Any] | None = None,
    ) -> tuple[TransitionBranch, ...]:
        """Enumerate all finite-support branches for one demand realization."""

        canonical = self._state(state)
        action_i, demand_i = self._validate_action_demand(action, demand)
        event = ExogenousEvent.from_value(random_event)
        branches: list[TransitionBranch] = []
        for (eta, lead_probability) in self._lead_choices(action_i, event):
            # Existing due receipts determine whether a random lifetime is
            # sampled.  The helper is called once with a provisional lifetime
            # only to discover the incoming quantity.
            next_pipeline, due_arrivals, _ = self._pipeline_and_arrival(
                canonical, action_i, int(eta or 0)
            )
            del next_pipeline
            for lifetime, life_probability in self._lifetime_choices(
                due_arrivals, event
            ):
                draft, eligible = self._apply(
                    canonical,
                    action_i,
                    demand_i,
                    event=event,
                    eta=int(eta or 0),
                    lifetime=lifetime,
                    abandoned=0,
                )
                if self.config.backlog.mode is BacklogMode.GEOMETRIC:
                    if event.abandoned is not None:
                        abandoned_i = _integer(event.abandoned, name="event.abandoned")
                        if abandoned_i > eligible:
                            raise ValueError(
                                "event.abandoned exceeds eligible geometric backlog"
                            )
                        abandon_choices = ((abandoned_i, 1.0),)
                    else:
                        survival = float(
                            self.config.backlog.survival_probability or 0.0
                        )
                        binomial_support_size = (
                            1 if survival in (0.0, 1.0) else eligible + 1
                        )
                        required_branches = len(branches) + binomial_support_size
                        if required_branches > self.max_outcomes:
                            raise TransitionEnumerationError(
                                "transition requires at least "
                                f"{required_branches} branches; "
                                f"max_outcomes={self.max_outcomes}"
                            )
                        abandon_choices = _binomial_branches(eligible, survival)
                else:
                    abandon_choices = ((0, 1.0),)
                for abandoned, abandon_probability in abandon_choices:
                    outcome, _ = self._apply(
                        canonical,
                        action_i,
                        demand_i,
                        event=event,
                        eta=int(eta or 0),
                        lifetime=lifetime,
                        abandoned=int(abandoned),
                    )
                    branches.append(
                        TransitionBranch(
                            float(lead_probability)
                            * float(life_probability)
                            * float(abandon_probability),
                            outcome,
                        )
                    )
        if len(branches) > self.max_outcomes:
            raise TransitionEnumerationError(
                f"transition has {len(branches)} branches; max_outcomes={self.max_outcomes}"
            )
        total = float(sum(branch.probability for branch in branches))
        if total <= 0.0 or not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-12):
            raise RuntimeError(f"transition branch probabilities sum to {total}")
        # Collapse exact duplicate branches (e.g. zero orders or degenerate
        # support) while retaining deterministic insertion order.
        collapsed: dict[tuple[Any, ...], float] = {}
        exemplar: dict[tuple[Any, ...], TransitionOutcome] = {}
        for branch in branches:
            key = (
                branch.outcome.next_state,
                branch.outcome.event,
                branch.outcome.abandoned,
            )
            collapsed[key] = collapsed.get(key, 0.0) + branch.probability
            exemplar[key] = branch.outcome
        return tuple(
            TransitionBranch(probability, exemplar[key])
            for key, probability in collapsed.items()
        )

    def sample(
        self,
        state: InventoryState | Sequence[int],
        action: int,
        demand: int,
        *,
        rng: Any = None,
        random_event: ExogenousEvent | Mapping[str, Any] | None = None,
    ) -> TransitionOutcome:
        """Sample one transition using a NumPy or Python-compatible RNG."""

        canonical = self._state(state)
        action_i, demand_i = self._validate_action_demand(action, demand)
        event = ExogenousEvent.from_value(random_event)
        if rng is None:
            rng = np.random.default_rng()
        lead_choices = self._lead_choices(action_i, event)
        eta = _weighted_choice(
            rng,
            [int(value or 0) for value, _ in lead_choices],
            [probability for _, probability in lead_choices],
        )
        _, due_arrivals, _ = self._pipeline_and_arrival(canonical, action_i, eta)
        life_choices = self._lifetime_choices(due_arrivals, event)
        lifetime = _weighted_choice(
            rng,
            [int(value) if value is not None else 0 for value, _ in life_choices],
            [probability for _, probability in life_choices],
        )
        lifetime_value = (
            None
            if not self.config.perishability.enabled or due_arrivals == 0
            else lifetime
        )
        provisional, eligible = self._apply(
            canonical,
            action_i,
            demand_i,
            event=event,
            eta=eta,
            lifetime=lifetime_value,
            abandoned=0,
        )
        del provisional
        if self.config.backlog.mode is BacklogMode.GEOMETRIC:
            if event.abandoned is not None:
                abandoned = _integer(event.abandoned, name="event.abandoned")
                if abandoned > eligible:
                    raise ValueError(
                        "event.abandoned exceeds eligible geometric backlog"
                    )
            else:
                abandoned = _binomial_sample(
                    rng,
                    eligible,
                    1.0 - float(self.config.backlog.survival_probability or 0.0),
                )
        else:
            abandoned = 0
        outcome, _ = self._apply(
            canonical,
            action_i,
            demand_i,
            event=event,
            eta=eta,
            lifetime=lifetime_value,
            abandoned=abandoned,
        )
        return outcome

    def transition(
        self,
        state: InventoryState | Sequence[int],
        action: int,
        demand: int,
        *,
        rng: Any = None,
        random_event: ExogenousEvent | Mapping[str, Any] | None = None,
    ) -> TransitionOutcome:
        """Alias for :meth:`sample`, convenient for simulation callers."""

        return self.sample(state, action, demand, rng=rng, random_event=random_event)

    __call__ = transition


class DeterministicTransitionKernel(ComposedTransitionKernel):
    """Kernel specialization that rejects stochastic component configs."""

    def __init__(self, config: InventoryModelConfig, *args: Any, **kwargs: Any) -> None:
        if config.transition_mode.value != "deterministic":
            raise ValueError(
                "DeterministicTransitionKernel requires a deterministic config"
            )
        super().__init__(config, *args, **kwargs)

    def transition(
        self,
        state: InventoryState | Sequence[int],
        action: int,
        demand: int,
        **kwargs: Any,
    ) -> TransitionOutcome:
        branches = self.outcomes(
            state, action, demand, random_event=kwargs.get("random_event")
        )
        if len(branches) != 1:
            raise RuntimeError("deterministic kernel produced multiple branches")
        return branches[0].outcome

    __call__ = transition


class KnownEventStochasticTransitionKernel(ComposedTransitionKernel):
    """Explicit name for a kernel with known-at-event random components."""

    def __init__(self, config: InventoryModelConfig, *args: Any, **kwargs: Any) -> None:
        if config.transition_mode.value != "known_event_stochastic":
            raise ValueError(
                "KnownEventStochasticTransitionKernel requires a stochastic config"
            )
        super().__init__(config, *args, **kwargs)


def build_transition_kernel(
    config: InventoryModelConfig,
    costs: Optional[TransitionCosts | Mapping[str, Any]] = None,
    **kwargs: Any,
) -> ComposedTransitionKernel:
    """Construct the deterministic or known-event stochastic kernel."""

    kernel_type = (
        KnownEventStochasticTransitionKernel
        if config.transition_mode.value == "known_event_stochastic"
        else DeterministicTransitionKernel
    )
    return kernel_type(config, costs, **kwargs)


def transition_outcomes(
    config: InventoryModelConfig,
    state: InventoryState | Sequence[int],
    action: int,
    demand: int,
    *,
    costs: Optional[TransitionCosts | Mapping[str, Any]] = None,
    random_event: ExogenousEvent | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> tuple[TransitionBranch, ...]:
    """Functional wrapper around :meth:`ComposedTransitionKernel.outcomes`."""

    return build_transition_kernel(config, costs, **kwargs).outcomes(
        state, action, demand, random_event=random_event
    )


def deterministic_transition(
    config: InventoryModelConfig,
    state: InventoryState | Sequence[int],
    action: int,
    demand: int,
    *,
    costs: Optional[TransitionCosts | Mapping[str, Any]] = None,
    **kwargs: Any,
) -> TransitionOutcome:
    """Functional deterministic transition helper."""

    kernel = DeterministicTransitionKernel(config, costs, **kwargs)
    return kernel.transition(state, action, demand)


def sample_transition(
    config: InventoryModelConfig,
    state: InventoryState | Sequence[int],
    action: int,
    demand: int,
    *,
    costs: Optional[TransitionCosts | Mapping[str, Any]] = None,
    rng: Any = None,
    random_event: ExogenousEvent | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> TransitionOutcome:
    """Functional transition helper for MC simulation."""

    return build_transition_kernel(config, costs, **kwargs).sample(
        state, action, demand, rng=rng, random_event=random_event
    )


# Readable aliases used by callers that prefer "composed" terminology.
ComposedTransitionOutcome = TransitionOutcome
TransitionResult = TransitionOutcome
KnownEventTransitionKernel = KnownEventStochasticTransitionKernel


__all__ = [
    "ComposedTransitionKernel",
    "ComposedTransitionOutcome",
    "DeterministicTransitionKernel",
    "ExogenousEvent",
    "KnownEventStochasticTransitionKernel",
    "KnownEventTransitionKernel",
    "TransitionBranch",
    "TransitionCosts",
    "TransitionEnumerationError",
    "TransitionOutcome",
    "TransitionResult",
    "build_transition_kernel",
    "deterministic_transition",
    "sample_transition",
    "transition_outcomes",
]
