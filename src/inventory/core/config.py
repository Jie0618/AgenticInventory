"""Canonical, compositional inventory-model configuration contracts.

This module deliberately contains no numerical transition code.  It defines
the mathematical *shape* of a model (lead time, perishability, and backlog)
and the metadata required to identify that shape in caches and experiment
manifests.  Every inventory case uses these immutable component objects;
there are no model-combination config classes.

The lead-time stochastic mode implemented here is the known-ETA model: a
realized lead time is sampled when an order is placed and can immediately be
inserted into the corresponding countdown bucket.  An unknown-ETA/hazard
model is intentionally not represented by this contract.
"""

from __future__ import annotations

from runtime.artifacts import stable_key
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Sequence, Tuple

from inventory.common.numeric import integer_scalar, integer_tuple


IntTuple = Tuple[int, ...]
FloatTuple = Tuple[float, ...]


class _StringEnum(str, Enum):
    """Enum whose values serialize naturally as strings."""

    def __str__(self) -> str:  # pragma: no cover - convenience for callers
        return self.value


class LeadTimeMode(_StringEnum):
    DISABLED = "disabled"
    DETERMINISTIC = "deterministic"
    KNOWN_ETA_STOCHASTIC = "known_eta_stochastic"

    # Short aliases are useful when loading concise JSON configs.
    KNOWN_ETA = "known_eta_stochastic"
    STOCHASTIC_KNOWN_ETA = "known_eta_stochastic"


class PerishabilityMode(_StringEnum):
    DISABLED = "disabled"
    DETERMINISTIC = "deterministic"
    KNOWN_ARRIVAL_STOCHASTIC = "known_arrival_stochastic"

    KNOWN_ARRIVAL = "known_arrival_stochastic"


class BacklogMode(_StringEnum):
    DISABLED = "disabled"
    LOST_SALES = "lost_sales"
    DETERMINISTIC = "deterministic"
    GEOMETRIC = "geometric"
    FULL = "full"


class TransitionMode(_StringEnum):
    """Whether a transition has exogenous random outcomes.

    ``KNOWN_EVENT_STOCHASTIC`` covers random quantities whose realization is
    available at the event that inserts them into state (in particular known
    ETA lead time).  It does not mean the unknown-ETA hazard model.
    """

    DETERMINISTIC = "deterministic"
    KNOWN_EVENT_STOCHASTIC = "known_event_stochastic"

    # A readable alias used by some callers.
    STOCHASTIC_KNOWN_EVENT = "known_event_stochastic"


DEFAULT_TRANSITION_VERSION = "inventory_transition"


DEFAULT_EVENT_ORDER: Tuple[str, ...] = (
    "observe_state",
    "choose_action",
    "sample_order_lead_time",
    "place_order",
    "receive_due",
    "sample_receipt_lifetime",
    "serve_backlog",
    "realize_demand",
    "serve_demand",
    "measure_holding_and_shortage",
    "expire_inventory",
    "age_inventory",
    "abandon_backlog",
    "age_backlog",
    "advance_pipeline",
    "project_finite_grid",
    "measure_exit_and_carry_costs",
)


def _as_mode(value: Any, enum_type: type[_StringEnum]) -> _StringEnum:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value))
    except ValueError as error:
        choices = ", ".join(item.value for item in enum_type)
        raise ValueError(
            f"invalid {enum_type.__name__} {value!r}; expected one of {choices}"
        ) from error


def _ints(values: Sequence[int], *, name: str, allow_empty: bool = False) -> IntTuple:
    """Normalize an integer sequence without silently truncating values."""

    return integer_tuple(values, name=name, allow_empty=allow_empty)


def _probabilities(values: Sequence[float], *, name: str) -> FloatTuple:
    result = tuple(float(value) for value in values)
    if not result:
        raise ValueError(f"{name} must be nonempty")
    if any(not math.isfinite(value) or value < 0.0 for value in result):
        raise ValueError(f"{name} must contain finite nonnegative values")
    total = sum(result)
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"{name} must sum to 1 (got {total!r})")
    # Normalize tiny floating-point noise, while retaining the caller's
    # distribution to enough precision for reproducible signatures.
    if total != 1.0:
        result = tuple(value / total for value in result)
    return result


def _support_and_probs(
    support: Sequence[int], probabilities: Sequence[float], *, name: str, minimum: int
) -> tuple[IntTuple, FloatTuple]:
    normalized_support = _ints(support, name=f"{name}.support")
    if any(value < minimum for value in normalized_support):
        raise ValueError(f"{name}.support values must be >= {minimum}")
    if len(set(normalized_support)) != len(normalized_support):
        raise ValueError(f"{name}.support values must be unique")
    if tuple(sorted(normalized_support)) != normalized_support:
        raise ValueError(f"{name}.support values must be sorted")
    normalized_probabilities = _probabilities(
        probabilities, name=f"{name}.probabilities"
    )
    if len(normalized_support) != len(normalized_probabilities):
        raise ValueError(
            f"{name}.support and {name}.probabilities must have equal lengths"
        )
    # A zero-mass point is not part of a distribution's mathematical support.
    # Removing it here also prevents impossible extrema from enlarging the
    # canonical state schema or changing an otherwise equivalent identity.
    positive = tuple(
        (value, probability)
        for value, probability in zip(normalized_support, normalized_probabilities)
        if probability > 0.0
    )
    return (
        tuple(value for value, _ in positive),
        tuple(probability for _, probability in positive),
    )


@dataclass(frozen=True, init=False)
class LeadTimeConfig:
    """Lead-time component of a canonical inventory model.

    ``mode='deterministic'`` uses ``periods``.  ``mode='known_eta_stochastic'``
    uses a finite ``support`` and matching ``probabilities``; the sampled ETA
    is known at order placement.  ``disabled`` means the pipeline branch is
    empty.  A deterministic ``periods=0`` configuration is an explicit
    immediate-replenishment component and is distinct from disabled.
    """

    mode: LeadTimeMode
    periods: Optional[int]
    support: IntTuple
    probabilities: FloatTuple

    def __init__(
        self,
        mode: Any = LeadTimeMode.DISABLED,
        periods: Optional[int] = None,
        support: Sequence[int] = (),
        probabilities: Sequence[float] = (),
    ) -> None:
        normalized_mode = _as_mode(mode, LeadTimeMode)
        support = tuple(support)
        probabilities = tuple(probabilities)
        normalized_periods: Optional[int]
        normalized_support: IntTuple = ()
        normalized_probabilities: FloatTuple = ()

        if normalized_mode is LeadTimeMode.DISABLED:
            if periods is not None or tuple(support) or tuple(probabilities):
                raise ValueError("disabled lead time cannot carry timing parameters")
            normalized_periods = None
        elif normalized_mode is LeadTimeMode.DETERMINISTIC:
            if periods is None:
                raise ValueError("deterministic lead time requires periods")
            normalized_periods = integer_scalar(
                periods, name="lead-time periods", minimum=0
            )
            if tuple(support) or tuple(probabilities):
                raise ValueError("deterministic lead time cannot carry a distribution")
        else:
            if periods is not None:
                raise ValueError("known-ETA stochastic lead time cannot carry periods")
            normalized_support, normalized_probabilities = _support_and_probs(
                support, probabilities, name="lead_time", minimum=0
            )
            normalized_periods = None

        object.__setattr__(self, "mode", normalized_mode)
        object.__setattr__(self, "periods", normalized_periods)
        object.__setattr__(self, "support", normalized_support)
        object.__setattr__(self, "probabilities", normalized_probabilities)

    @classmethod
    def disabled(cls) -> "LeadTimeConfig":
        return LeadTimeConfig()

    @classmethod
    def deterministic(cls, periods: int) -> "LeadTimeConfig":
        return LeadTimeConfig(LeadTimeMode.DETERMINISTIC, periods=periods)

    @classmethod
    def known_eta(
        cls, support: Sequence[int], probabilities: Sequence[float]
    ) -> "LeadTimeConfig":
        return LeadTimeConfig(
            LeadTimeMode.KNOWN_ETA_STOCHASTIC,
            support=support,
            probabilities=probabilities,
        )

    @property
    def enabled(self) -> bool:
        return self.mode is not LeadTimeMode.DISABLED

    @property
    def stochastic(self) -> bool:
        return self.mode is LeadTimeMode.KNOWN_ETA_STOCHASTIC

    @property
    def max_periods(self) -> int:
        if self.mode is LeadTimeMode.DETERMINISTIC:
            return int(self.periods or 0)
        if self.mode is LeadTimeMode.KNOWN_ETA_STOCHASTIC:
            return max(self.support)
        return 0

    @property
    def state_dimension(self) -> int:
        # ETA=0 is represented as immediate arrival, so there is no p0 bucket.
        return self.max_periods

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"mode": self.mode.value}
        if self.mode is LeadTimeMode.DETERMINISTIC:
            payload["periods"] = int(self.periods or 0)
        elif self.mode is LeadTimeMode.KNOWN_ETA_STOCHASTIC:
            payload["support"] = list(self.support)
            payload["probabilities"] = list(self.probabilities)
        return payload


@dataclass(frozen=True, init=False)
class DeterministicLeadTime(LeadTimeConfig):
    """Typed convenience constructor for a fixed lead time."""

    def __init__(self, periods: int) -> None:
        base = LeadTimeConfig.deterministic(periods)
        for field_name in ("mode", "periods", "support", "probabilities"):
            object.__setattr__(self, field_name, getattr(base, field_name))


@dataclass(frozen=True, init=False)
class KnownETALeadTime(LeadTimeConfig):
    """Typed convenience constructor for known-ETA random lead time."""

    def __init__(self, support: Sequence[int], probabilities: Sequence[float]) -> None:
        base = LeadTimeConfig.known_eta(support, probabilities)
        for field_name in ("mode", "periods", "support", "probabilities"):
            object.__setattr__(self, field_name, getattr(base, field_name))


KnownEtaLeadTime = KnownETALeadTime


@dataclass(frozen=True, init=False)
class NoLeadTime(LeadTimeConfig):
    """Typed convenience constructor for a disabled pipeline branch."""

    def __init__(self) -> None:
        base = LeadTimeConfig.disabled()
        for field_name in ("mode", "periods", "support", "probabilities"):
            object.__setattr__(self, field_name, getattr(base, field_name))


class ImmediateReplenishment(DeterministicLeadTime):
    """Typed convenience constructor for a zero-period lead time."""

    def __init__(self) -> None:
        super().__init__(0)


@dataclass(frozen=True, init=False)
class PerishabilityConfig:
    """Perishability component and age-bucket entry semantics."""

    mode: PerishabilityMode
    shelf_life: Optional[int]
    lifetime_support: IntTuple
    lifetime_probabilities: FloatTuple

    def __init__(
        self,
        mode: Any = PerishabilityMode.DISABLED,
        shelf_life: Optional[int] = None,
        lifetime_support: Sequence[int] = (),
        lifetime_probabilities: Sequence[float] = (),
    ) -> None:
        normalized_mode = _as_mode(mode, PerishabilityMode)
        lifetime_support = tuple(lifetime_support)
        lifetime_probabilities = tuple(lifetime_probabilities)
        normalized_shelf_life: Optional[int]
        normalized_support: IntTuple = ()
        normalized_probabilities: FloatTuple = ()
        if normalized_mode is PerishabilityMode.DISABLED:
            if (
                shelf_life is not None
                or tuple(lifetime_support)
                or tuple(lifetime_probabilities)
            ):
                raise ValueError(
                    "disabled perishability cannot carry lifetime parameters"
                )
            normalized_shelf_life = None
        elif normalized_mode is PerishabilityMode.DETERMINISTIC:
            if shelf_life is None:
                raise ValueError("deterministic perishability requires shelf_life")
            normalized_shelf_life = integer_scalar(
                shelf_life, name="shelf_life", minimum=1
            )
            if tuple(lifetime_support) or tuple(lifetime_probabilities):
                raise ValueError(
                    "deterministic perishability cannot carry a distribution"
                )
        else:
            if shelf_life is not None:
                raise ValueError(
                    "known-arrival stochastic perishability cannot carry shelf_life"
                )
            normalized_support, normalized_probabilities = _support_and_probs(
                lifetime_support,
                lifetime_probabilities,
                name="perishability",
                minimum=1,
            )
            normalized_shelf_life = None

        object.__setattr__(self, "mode", normalized_mode)
        object.__setattr__(self, "shelf_life", normalized_shelf_life)
        object.__setattr__(self, "lifetime_support", normalized_support)
        object.__setattr__(self, "lifetime_probabilities", normalized_probabilities)

    @classmethod
    def disabled(cls) -> "PerishabilityConfig":
        return PerishabilityConfig()

    @classmethod
    def deterministic(cls, shelf_life: int) -> "PerishabilityConfig":
        return PerishabilityConfig(
            PerishabilityMode.DETERMINISTIC, shelf_life=shelf_life
        )

    @classmethod
    def known_arrival(
        cls, support: Sequence[int], probabilities: Sequence[float]
    ) -> "PerishabilityConfig":
        return PerishabilityConfig(
            PerishabilityMode.KNOWN_ARRIVAL_STOCHASTIC,
            lifetime_support=support,
            lifetime_probabilities=probabilities,
        )

    @property
    def enabled(self) -> bool:
        return self.mode is not PerishabilityMode.DISABLED

    @property
    def stochastic(self) -> bool:
        return self.mode is PerishabilityMode.KNOWN_ARRIVAL_STOCHASTIC

    @property
    def max_shelf_life(self) -> int:
        if self.mode is PerishabilityMode.DETERMINISTIC:
            return int(self.shelf_life or 0)
        if self.mode is PerishabilityMode.KNOWN_ARRIVAL_STOCHASTIC:
            return max(self.lifetime_support)
        return 0

    @property
    def state_dimension(self) -> int:
        return self.max_shelf_life

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"mode": self.mode.value}
        if self.mode is PerishabilityMode.DETERMINISTIC:
            payload["shelf_life"] = int(self.shelf_life or 0)
        elif self.mode is PerishabilityMode.KNOWN_ARRIVAL_STOCHASTIC:
            payload["lifetime_support"] = list(self.lifetime_support)
            payload["lifetime_probabilities"] = list(self.lifetime_probabilities)
        return payload


@dataclass(frozen=True, init=False)
class FixedShelfLife(PerishabilityConfig):
    """Typed convenience constructor for deterministic shelf life."""

    def __init__(self, shelf_life: int) -> None:
        base = PerishabilityConfig.deterministic(shelf_life)
        for field_name in (
            "mode",
            "shelf_life",
            "lifetime_support",
            "lifetime_probabilities",
        ):
            object.__setattr__(self, field_name, getattr(base, field_name))


@dataclass(frozen=True, init=False)
class KnownArrivalShelfLife(PerishabilityConfig):
    """Typed convenience constructor for random life known on receipt."""

    def __init__(self, support: Sequence[int], probabilities: Sequence[float]) -> None:
        base = PerishabilityConfig.known_arrival(support, probabilities)
        for field_name in (
            "mode",
            "shelf_life",
            "lifetime_support",
            "lifetime_probabilities",
        ):
            object.__setattr__(self, field_name, getattr(base, field_name))


class NoPerishability(PerishabilityConfig):
    """Typed convenience constructor for a non-perishable branch."""

    def __init__(self) -> None:
        base = PerishabilityConfig.disabled()
        for field_name in (
            "mode",
            "shelf_life",
            "lifetime_support",
            "lifetime_probabilities",
        ):
            object.__setattr__(self, field_name, getattr(base, field_name))


@dataclass(frozen=True, init=False)
class BacklogConfig:
    """Backlog/patience component.

    ``GEOMETRIC`` and ``FULL`` use one scalar ``B`` state coordinate because
    their continuation law is memoryless (or infinite).  A deterministic
    patience value uses age buckets ``b1..bK``.  ``DISABLED`` and
    ``LOST_SALES`` both expose an empty backlog branch but remain distinct in
    the identity so a cache cannot confuse their semantics.
    """

    mode: BacklogMode
    patience: Optional[int]
    survival_probability: Optional[float]

    def __init__(
        self,
        mode: Any = BacklogMode.DISABLED,
        patience: Optional[int] = None,
        survival_probability: Optional[float] = None,
    ) -> None:
        normalized_mode = _as_mode(mode, BacklogMode)
        normalized_patience: Optional[int] = None
        normalized_survival: Optional[float] = None
        if normalized_mode in (
            BacklogMode.DISABLED,
            BacklogMode.LOST_SALES,
            BacklogMode.FULL,
        ):
            if patience is not None or survival_probability is not None:
                raise ValueError(
                    f"{normalized_mode.value} backlog cannot carry parameters"
                )
        elif normalized_mode is BacklogMode.DETERMINISTIC:
            if patience is None:
                raise ValueError("deterministic backlog patience is required")
            normalized_patience = integer_scalar(
                patience, name="deterministic backlog patience", minimum=1
            )
            if survival_probability is not None:
                raise ValueError(
                    "deterministic backlog cannot carry survival_probability"
                )
        else:
            if survival_probability is None:
                raise ValueError("geometric backlog requires survival_probability")
            if (
                not math.isfinite(float(survival_probability))
                or not 0.0 <= float(survival_probability) <= 1.0
            ):
                raise ValueError("survival_probability must lie in [0, 1]")
            if patience is not None:
                raise ValueError("geometric backlog cannot carry patience")
            normalized_survival = float(survival_probability)

        object.__setattr__(self, "mode", normalized_mode)
        object.__setattr__(self, "patience", normalized_patience)
        object.__setattr__(self, "survival_probability", normalized_survival)

    @classmethod
    def disabled(cls) -> "BacklogConfig":
        return BacklogConfig()

    @classmethod
    def lost_sales(cls) -> "BacklogConfig":
        return BacklogConfig(BacklogMode.LOST_SALES)

    @classmethod
    def deterministic(cls, patience: int) -> "BacklogConfig":
        return BacklogConfig(BacklogMode.DETERMINISTIC, patience=patience)

    @classmethod
    def geometric(cls, survival_probability: float) -> "BacklogConfig":
        return BacklogConfig(
            BacklogMode.GEOMETRIC, survival_probability=survival_probability
        )

    @classmethod
    def full(cls) -> "BacklogConfig":
        return BacklogConfig(BacklogMode.FULL)

    @property
    def enabled(self) -> bool:
        return self.mode not in (BacklogMode.DISABLED, BacklogMode.LOST_SALES)

    @property
    def stochastic(self) -> bool:
        return self.mode is BacklogMode.GEOMETRIC

    @property
    def state_dimension(self) -> int:
        if self.mode is BacklogMode.DETERMINISTIC:
            return int(self.patience or 0)
        if self.mode in (BacklogMode.GEOMETRIC, BacklogMode.FULL):
            return 1
        return 0

    @property
    def uses_scalar_total(self) -> bool:
        return self.mode in (BacklogMode.GEOMETRIC, BacklogMode.FULL)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"mode": self.mode.value}
        if self.mode is BacklogMode.DETERMINISTIC:
            payload["patience"] = int(self.patience or 0)
        elif self.mode is BacklogMode.GEOMETRIC:
            payload["survival_probability"] = float(self.survival_probability or 0.0)
        return payload


@dataclass(frozen=True, init=False)
class DeterministicBacklog(BacklogConfig):
    """Typed convenience constructor for finite deterministic patience."""

    def __init__(self, patience: int) -> None:
        base = BacklogConfig.deterministic(patience)
        for field_name in ("mode", "patience", "survival_probability"):
            object.__setattr__(self, field_name, getattr(base, field_name))


@dataclass(frozen=True, init=False)
class GeometricBacklog(BacklogConfig):
    """Typed convenience constructor for memoryless partial backlog."""

    def __init__(self, survival_probability: float) -> None:
        base = BacklogConfig.geometric(survival_probability)
        for field_name in ("mode", "patience", "survival_probability"):
            object.__setattr__(self, field_name, getattr(base, field_name))


@dataclass(frozen=True, init=False)
class NoBacklog(BacklogConfig):
    """Typed convenience constructor for a disabled backlog branch."""

    def __init__(self) -> None:
        base = BacklogConfig.disabled()
        for field_name in ("mode", "patience", "survival_probability"):
            object.__setattr__(self, field_name, getattr(base, field_name))


class LostSales(BacklogConfig):
    """Typed convenience constructor for lost-sales demand."""

    def __init__(self) -> None:
        base = BacklogConfig.lost_sales()
        for field_name in ("mode", "patience", "survival_probability"):
            object.__setattr__(self, field_name, getattr(base, field_name))


class FullBacklog(BacklogConfig):
    """Typed convenience constructor for infinite-patience backlog."""

    def __init__(self) -> None:
        base = BacklogConfig.full()
        for field_name in ("mode", "patience", "survival_probability"):
            object.__setattr__(self, field_name, getattr(base, field_name))


@dataclass(frozen=True)
class TransitionSpec:
    """Versioned transition metadata shared by exact and MC backends."""

    mode: TransitionMode
    event_order: Tuple[str, ...] = DEFAULT_EVENT_ORDER
    version: str = DEFAULT_TRANSITION_VERSION

    def __post_init__(self) -> None:
        mode = _as_mode(self.mode, TransitionMode)
        events = tuple(str(item) for item in self.event_order)
        if not events or any(not item for item in events):
            raise ValueError("event_order must contain nonempty names")
        if len(set(events)) != len(events):
            raise ValueError("event_order entries must be unique")
        if not self.version:
            raise ValueError("transition version must be nonempty")
        if self.version == DEFAULT_TRANSITION_VERSION and events != DEFAULT_EVENT_ORDER:
            raise ValueError(f"{DEFAULT_TRANSITION_VERSION} has a fixed event order")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "event_order", events)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "event_order": list(self.event_order),
            "version": self.version,
        }


@dataclass(frozen=True, init=False)
class InventoryModelConfig:
    """A compositional model selected by component parameters.

    The three branches always exist in the canonical state contract.  Passing
    ``None`` for a component is normalized to its explicit disabled config.
    ``capacity`` is an optional, explicit finite-grid projection bound for
    inventory totals, individual pipeline buckets, and backlog totals.  It is
    not the action cap.  Omitting it leaves transition quantities unprojected;
    finite-grid DP callers must opt in and sensitivity-check this numerical
    approximation.  The bound is included in model identity when supplied.
    """

    lead_time: LeadTimeConfig
    perishability: PerishabilityConfig
    backlog: BacklogConfig
    capacity: Optional[int]
    model_version: str
    transition: TransitionSpec

    def __init__(
        self,
        lead_time: Optional[LeadTimeConfig] = None,
        perishability: Optional[PerishabilityConfig] = None,
        backlog: Optional[BacklogConfig] = None,
        capacity: Optional[int] = None,
        model_version: str = "inventory_composition",
        transition: Optional[TransitionSpec] = None,
        event_order: Optional[Sequence[str]] = None,
    ) -> None:
        lead = LeadTimeConfig.disabled() if lead_time is None else lead_time
        perish = (
            PerishabilityConfig.disabled() if perishability is None else perishability
        )
        back = BacklogConfig.disabled() if backlog is None else backlog
        if not isinstance(lead, LeadTimeConfig):
            raise TypeError("lead_time must be a LeadTimeConfig")
        if not isinstance(perish, PerishabilityConfig):
            raise TypeError("perishability must be a PerishabilityConfig")
        if not isinstance(back, BacklogConfig):
            raise TypeError("backlog must be a BacklogConfig")
        # Store semantic base value objects.  Convenience constructors are
        # subclasses for readable call sites, but subclass-sensitive dataclass
        # equality must not make identical serialized configurations unequal.
        lead = LeadTimeConfig(
            lead.mode,
            periods=lead.periods,
            support=lead.support,
            probabilities=lead.probabilities,
        )
        perish = PerishabilityConfig(
            perish.mode,
            shelf_life=perish.shelf_life,
            lifetime_support=perish.lifetime_support,
            lifetime_probabilities=perish.lifetime_probabilities,
        )
        back = BacklogConfig(
            back.mode,
            patience=back.patience,
            survival_probability=back.survival_probability,
        )
        normalized_capacity = (
            None
            if capacity is None
            else integer_scalar(capacity, name="capacity", minimum=0)
        )
        if not model_version:
            raise ValueError("model_version must be nonempty")
        random_mode = lead.stochastic or perish.stochastic or back.stochastic
        expected_mode = (
            TransitionMode.KNOWN_EVENT_STOCHASTIC
            if random_mode
            else TransitionMode.DETERMINISTIC
        )
        if transition is not None and not isinstance(transition, TransitionSpec):
            raise TypeError("transition must be a TransitionSpec")
        if transition is None:
            transition = TransitionSpec(
                expected_mode,
                event_order=DEFAULT_EVENT_ORDER
                if event_order is None
                else tuple(event_order),
            )
        elif transition.mode is not expected_mode:
            raise ValueError(
                "transition mode does not match enabled component randomness: "
                f"expected {expected_mode.value}, got {transition.mode.value}"
            )
        elif event_order is not None and tuple(event_order) != transition.event_order:
            raise ValueError("event_order conflicts with transition.event_order")

        object.__setattr__(self, "lead_time", lead)
        object.__setattr__(self, "perishability", perish)
        object.__setattr__(self, "backlog", back)
        object.__setattr__(self, "capacity", normalized_capacity)
        object.__setattr__(self, "model_version", str(model_version))
        object.__setattr__(self, "transition", transition)

    @property
    def transition_mode(self) -> TransitionMode:
        return self.transition.mode

    @property
    def enabled_components(self) -> Tuple[str, ...]:
        result = []
        if self.lead_time.enabled:
            result.append("lead_time")
        if self.perishability.enabled:
            result.append("perishability")
        if self.backlog.enabled:
            result.append("backlog")
        return tuple(result)

    @property
    def composition_key(self) -> str:
        return "+".join(self.enabled_components) or "base"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "model_version": self.model_version,
            "lead_time": self.lead_time.to_dict(),
            "perishability": self.perishability.to_dict(),
            "backlog": self.backlog.to_dict(),
            "capacity": self.capacity,
            "transition": self.transition.to_dict(),
        }

    @property
    def composition_hash(self) -> str:
        return _composition_key(self.to_dict())

    @property
    def composition_id(self) -> str:
        return f"inventory-{self.composition_key}-{self.composition_hash[:20]}"


def _canonicalize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _canonicalize(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (tuple, list)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("identity payload cannot contain non-finite floats")
        return float(format(value, ".17g"))
    return value


def canonical_json(value: Any) -> str:
    """Serialize a payload deterministically for manifests and cache keys."""

    return json.dumps(
        _canonicalize(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def _composition_key(value: Any) -> str:
    return stable_key(canonical_json(value).encode("utf-8"))


def composition_identity(config: InventoryModelConfig) -> str:
    """Return a stable cache label for a model composition."""

    if not isinstance(config, InventoryModelConfig):
        raise TypeError("composition_identity expects InventoryModelConfig")
    return config.composition_hash


__all__ = [
    "BacklogConfig",
    "BacklogMode",
    "DEFAULT_EVENT_ORDER",
    "DEFAULT_TRANSITION_VERSION",
    "DeterministicBacklog",
    "DeterministicLeadTime",
    "FixedShelfLife",
    "FullBacklog",
    "GeometricBacklog",
    "ImmediateReplenishment",
    "InventoryModelConfig",
    "KnownETALeadTime",
    "KnownEtaLeadTime",
    "KnownArrivalShelfLife",
    "LeadTimeConfig",
    "LeadTimeMode",
    "NoBacklog",
    "NoLeadTime",
    "NoPerishability",
    "PerishabilityConfig",
    "PerishabilityMode",
    "LostSales",
    "TransitionMode",
    "TransitionSpec",
    "canonical_json",
    "composition_identity",
]
