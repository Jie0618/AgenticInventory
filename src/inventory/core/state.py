"""Canonical compositional inventory state and schema.

The public state always has three branches, in this order:

``inventory, pipeline, backlog``.

The component configuration chooses the length and interpretation of each
branch.  This keeps a single mathematical state contract while allowing a
backend to use a compact tuple for a particular case.  The module is
independent of evaluator-specific flat array encodings;
``CanonicalStateSchema`` exposes the same ``allowed_state_names`` and
``mapping`` conveniences used by the AST layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

from inventory.common.numeric import integer_scalar, integer_tuple
from inventory.core.config import (
    BacklogConfig,
    BacklogMode,
    InventoryModelConfig,
    LeadTimeMode,
    PerishabilityMode,
)


IntTuple = Tuple[int, ...]


def _normalize_branch(values: Sequence[int], *, name: str) -> IntTuple:
    return integer_tuple(values, name=name, minimum=0, allow_empty=True)


@dataclass(frozen=True, init=False)
class InventoryState:
    """Immutable state with the three canonical component branches.

    ``inventory`` is ``(x,)`` for a non-perishable case and ``(a1, ..., aM)``
    for a perishable case.  ``pipeline`` is ``(p1, ..., pLmax)`` and
    ``backlog`` is empty, scalar ``(B,)``, or ``(b1, ..., bK)`` according to
    the component configuration.  Branches may be empty, but the object
    itself always carries all three fields so generic code need not branch on
    a model id.
    """

    inventory: IntTuple
    pipeline: IntTuple
    backlog: IntTuple

    def __init__(
        self,
        inventory: Sequence[int] = (),
        pipeline: Sequence[int] = (),
        backlog: Sequence[int] = (),
    ) -> None:
        object.__setattr__(
            self, "inventory", _normalize_branch(inventory, name="inventory")
        )
        object.__setattr__(
            self, "pipeline", _normalize_branch(pipeline, name="pipeline")
        )
        object.__setattr__(self, "backlog", _normalize_branch(backlog, name="backlog"))

    @classmethod
    def zeros(cls, schema: "CanonicalStateSchema") -> "InventoryState":
        """Construct the all-zero state for ``schema``."""

        if not isinstance(schema, CanonicalStateSchema):
            raise TypeError("zeros expects a CanonicalStateSchema")
        return cls(
            (0,) * schema.inventory_dimension,
            (0,) * schema.pipeline_dimension,
            (0,) * schema.backlog_dimension,
        )

    @classmethod
    def from_flat(
        cls, values: Sequence[int], schema: "CanonicalStateSchema"
    ) -> "InventoryState":
        """Decode a flat backend tuple using the schema's canonical order."""

        if not isinstance(schema, CanonicalStateSchema):
            raise TypeError("from_flat expects a CanonicalStateSchema")
        normalized = tuple(values)
        if len(normalized) != schema.dimension:
            raise ValueError(
                f"{schema.model_id} expects {schema.dimension} coordinates, "
                f"got {len(normalized)}"
            )
        state = cls(
            normalized[: schema.inventory_dimension],
            normalized[
                schema.inventory_dimension : schema.inventory_dimension
                + schema.pipeline_dimension
            ],
            normalized[schema.inventory_dimension + schema.pipeline_dimension :],
        )
        schema.validate(state)
        return state

    def flatten(self, schema: "CanonicalStateSchema") -> IntTuple:
        """Encode this state in the schema's canonical flat order."""

        if not isinstance(schema, CanonicalStateSchema):
            raise TypeError("flatten expects a CanonicalStateSchema")
        schema.validate(self)
        return self.inventory + self.pipeline + self.backlog

    def as_dict(self, schema: "CanonicalStateSchema") -> dict[str, int]:
        """Return a coordinate-name dictionary, excluding horizon metadata."""

        flat = self.flatten(schema)
        return {name: int(value) for name, value in zip(schema.names, flat)}

    @property
    def coordinates(self) -> IntTuple:
        """Return coordinates in the invariant public branch order."""

        return self.inventory + self.pipeline + self.backlog

    def __iter__(self):
        """Return coordinates in the configuration-derived canonical order."""

        return iter(self.coordinates)

    def __len__(self) -> int:
        return len(self.coordinates)

    def __getitem__(self, index):
        return self.coordinates[index]

    @property
    def total_inventory(self) -> int:
        return int(sum(self.inventory))

    @property
    def total_pipeline(self) -> int:
        return int(sum(self.pipeline))

    @property
    def total_backlog(self) -> int:
        return int(sum(self.backlog))


@dataclass(frozen=True)
class CanonicalStateSchema:
    """Configuration-derived coordinate schema for :class:`InventoryState`."""

    model_id: str
    names: Tuple[str, ...]
    inventory_names: Tuple[str, ...]
    pipeline_names: Tuple[str, ...]
    backlog_names: Tuple[str, ...]
    capacity: Optional[int] = None
    lead_time_mode: LeadTimeMode = LeadTimeMode.DISABLED
    perishability_mode: PerishabilityMode = PerishabilityMode.DISABLED
    backlog_mode: BacklogMode = BacklogMode.DISABLED
    lead_time_support: Tuple[int, ...] = ()
    lifetime_support: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must be nonempty")
        lead_time_mode = self.lead_time_mode
        if not isinstance(lead_time_mode, LeadTimeMode):
            try:
                lead_time_mode = LeadTimeMode(str(lead_time_mode))
            except ValueError as error:
                raise ValueError("invalid lead_time_mode") from error
        perishability_mode = self.perishability_mode
        if not isinstance(perishability_mode, PerishabilityMode):
            try:
                perishability_mode = PerishabilityMode(str(perishability_mode))
            except ValueError as error:
                raise ValueError("invalid perishability_mode") from error
        backlog_mode = self.backlog_mode
        if not isinstance(backlog_mode, BacklogMode):
            try:
                backlog_mode = BacklogMode(str(backlog_mode))
            except ValueError as error:
                raise ValueError("invalid backlog_mode") from error
        names = tuple(str(name) for name in self.names)
        inventory_names = tuple(str(name) for name in self.inventory_names)
        pipeline_names = tuple(str(name) for name in self.pipeline_names)
        backlog_names = tuple(str(name) for name in self.backlog_names)
        if names != inventory_names + pipeline_names + backlog_names:
            raise ValueError(
                "schema names must be inventory + pipeline + backlog in that order"
            )
        if not names or any(not name for name in names):
            raise ValueError(
                "canonical state schema requires nonempty coordinate names"
            )
        if len(set(names)) != len(names):
            raise ValueError("canonical state coordinate names must be unique")
        if "remaining_horizon" in names:
            raise ValueError("remaining_horizon is supplied separately")
        if self.capacity is not None:
            object.__setattr__(
                self,
                "capacity",
                integer_scalar(self.capacity, name="capacity", minimum=0),
            )
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "inventory_names", inventory_names)
        object.__setattr__(self, "pipeline_names", pipeline_names)
        object.__setattr__(self, "backlog_names", backlog_names)
        object.__setattr__(self, "lead_time_mode", lead_time_mode)
        object.__setattr__(self, "perishability_mode", perishability_mode)
        object.__setattr__(self, "backlog_mode", backlog_mode)
        object.__setattr__(
            self,
            "lead_time_support",
            integer_tuple(
                self.lead_time_support,
                name="lead_time_support",
                minimum=0,
                allow_empty=True,
            ),
        )
        object.__setattr__(
            self,
            "lifetime_support",
            integer_tuple(
                self.lifetime_support,
                name="lifetime_support",
                minimum=1,
                allow_empty=True,
            ),
        )

    @property
    def dimension(self) -> int:
        return len(self.names)

    @property
    def inventory_dimension(self) -> int:
        return len(self.inventory_names)

    @property
    def pipeline_dimension(self) -> int:
        return len(self.pipeline_names)

    @property
    def backlog_dimension(self) -> int:
        return len(self.backlog_names)

    @property
    def allowed_state_names(self) -> frozenset[str]:
        return frozenset((*self.names, "remaining_horizon"))

    @property
    def component_names(self) -> Mapping[str, Tuple[str, ...]]:
        return {
            "inventory": self.inventory_names,
            "pipeline": self.pipeline_names,
            "backlog": self.backlog_names,
        }

    def mapping(
        self, state: Sequence[int] | InventoryState, remaining_horizon: int
    ) -> dict[str, float]:
        """Map a state to the numeric environment consumed by AST policies."""

        canonical = (
            state
            if isinstance(state, InventoryState)
            else InventoryState.from_flat(state, self)
        )
        values = canonical.flatten(self)
        return {
            **{name: float(value) for name, value in zip(self.names, values)},
            "remaining_horizon": float(remaining_horizon),
        }

    def validate(self, state: InventoryState) -> None:
        if not isinstance(state, InventoryState):
            raise TypeError("state must be an InventoryState")
        expected = {
            "inventory": self.inventory_dimension,
            "pipeline": self.pipeline_dimension,
            "backlog": self.backlog_dimension,
        }
        actual = {
            "inventory": len(state.inventory),
            "pipeline": len(state.pipeline),
            "backlog": len(state.backlog),
        }
        for branch in ("inventory", "pipeline", "backlog"):
            if actual[branch] != expected[branch]:
                raise ValueError(
                    f"{self.model_id} expects {expected[branch]} {branch} coordinates, "
                    f"got {actual[branch]}"
                )
        if self.capacity is not None:
            if sum(state.inventory) > int(self.capacity):
                raise ValueError(
                    f"inventory total {sum(state.inventory)} exceeds capacity {self.capacity}"
                )
            if any(value > int(self.capacity) for value in state.pipeline):
                raise ValueError("pipeline coordinate exceeds capacity")
            if sum(state.backlog) > int(self.capacity):
                raise ValueError(
                    f"backlog total {sum(state.backlog)} exceeds capacity {self.capacity}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "names": list(self.names),
            "inventory_names": list(self.inventory_names),
            "pipeline_names": list(self.pipeline_names),
            "backlog_names": list(self.backlog_names),
            "capacity": self.capacity,
            "lead_time_mode": self.lead_time_mode.value,
            "perishability_mode": self.perishability_mode.value,
            "backlog_mode": self.backlog_mode.value,
            "lead_time_support": list(self.lead_time_support),
            "lifetime_support": list(self.lifetime_support),
        }


def _coordinate_range(prefix: str, count: int) -> Tuple[str, ...]:
    return tuple(f"{prefix}{index}" for index in range(1, int(count) + 1))


def state_schema_for_config(
    config: InventoryModelConfig, *, model_id: Optional[str] = None
) -> CanonicalStateSchema:
    """Compile one canonical schema from a compositional configuration.

    Coordinate order is stable across all combinations:

    ``x`` or ``a1..aM``, then ``p1..pLmax``, then ``B`` or ``b1..bK``.

    A known ETA support containing zero has no ``p0`` coordinate; zero-ETA
    orders are immediate.  Intermediate pipeline buckets are retained even
    when a sparse support omits a particular lead time because countdown
    transitions need those locations.
    """

    if not isinstance(config, InventoryModelConfig):
        raise TypeError("state_schema_for_config expects InventoryModelConfig")

    if config.perishability.enabled:
        inventory_names = _coordinate_range("a", config.perishability.state_dimension)
    else:
        # A scalar on-hand coordinate remains part of the common state even
        # when the perishable branch is disabled.
        inventory_names = ("x",)

    pipeline_names = _coordinate_range("p", config.lead_time.state_dimension)

    if config.backlog.mode is BacklogMode.DETERMINISTIC:
        backlog_names = _coordinate_range("b", config.backlog.state_dimension)
    elif config.backlog.uses_scalar_total:
        backlog_names = ("B",)
    else:
        backlog_names = ()

    names = inventory_names + pipeline_names + backlog_names
    return CanonicalStateSchema(
        model_id=str(model_id or config.composition_key),
        names=names,
        inventory_names=inventory_names,
        pipeline_names=pipeline_names,
        backlog_names=backlog_names,
        capacity=config.capacity,
        lead_time_mode=config.lead_time.mode,
        perishability_mode=config.perishability.mode,
        backlog_mode=config.backlog.mode,
        lead_time_support=config.lead_time.support,
        lifetime_support=config.perishability.lifetime_support,
    )


# Names that make call sites read naturally while preserving one implementation.
InventoryStateSchema = CanonicalStateSchema
ComposedStateSchema = CanonicalStateSchema
StateSchema = CanonicalStateSchema
build_state_schema = state_schema_for_config


__all__ = [
    "CanonicalStateSchema",
    "ComposedStateSchema",
    "InventoryState",
    "InventoryStateSchema",
    "StateSchema",
    "build_state_schema",
    "state_schema_for_config",
]
