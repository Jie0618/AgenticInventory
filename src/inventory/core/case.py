"""Canonical case contract for compositional inventory models.

``InventoryCase`` is the model-neutral input object used by every runner.  It
keeps the state shape in :class:`InventoryModelConfig`, the
unit economics in :class:`TransitionCosts`, and the finite demand support in
the shared :class:`DiscreteDemand` contract.  There are no model-specific case
classes: component selection is entirely represented by the configuration.
"""

from __future__ import annotations

from runtime.artifacts import stable_key
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from inventory.common.distributions import DiscreteDemand
from inventory.common.numeric import integer_scalar
from inventory.core.config import (
    DeterministicLeadTime,
    FixedShelfLife,
    InventoryModelConfig,
    LostSales,
    NoLeadTime,
    NoPerishability,
    canonical_json,
)
from inventory.core.state import (
    CanonicalStateSchema,
    InventoryState,
    state_schema_for_config,
)
from inventory.core.transition import TransitionCosts, build_transition_kernel


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    return integer_scalar(value, name=name, minimum=minimum)


def _cost_payload(costs: TransitionCosts) -> dict[str, float]:
    return {
        "holding": float(costs.holding),
        "shortage": float(costs.shortage),
        "expiry": float(costs.expiry),
        "backlog": float(costs.backlog),
        "abandonment": float(costs.abandonment),
        "ordering": float(costs.ordering),
    }


@dataclass(frozen=True, init=False)
class InventoryCase:
    """One finite-horizon inventory environment selected by component config.

    Parameters are intentionally explicit and model-neutral.  ``initial_state``
    may be either a flat tuple in canonical schema order or an
    :class:`InventoryState`; omitted values default to the all-zero state.
    ``n`` is the finite action cap.  ``config.capacity`` is independent and
    optional: when present it requests a finite-state projection for small-
    grid DP work; when absent, Exact/MC transitions retain all inventory,
    pipeline, and backlog quantities.
    """

    name: str
    config: InventoryModelConfig
    costs: TransitionCosts
    demand: DiscreteDemand
    n: int
    kind: str
    initial_state: InventoryState
    schema: CanonicalStateSchema = field(compare=True)

    def __init__(
        self,
        name: str,
        config: InventoryModelConfig,
        costs: TransitionCosts | Mapping[str, Any],
        demand: DiscreteDemand,
        n: int,
        kind: str = "custom",
        initial_state: InventoryState | Sequence[int] | None = None,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("case name must be nonempty")
        if not isinstance(config, InventoryModelConfig):
            raise TypeError("config must be an InventoryModelConfig")
        if isinstance(costs, TransitionCosts):
            normalized_costs = costs
        elif isinstance(costs, Mapping):
            normalized_costs = TransitionCosts.from_mapping(costs)
        else:
            raise TypeError("costs must be TransitionCosts or a mapping")
        if not isinstance(demand, DiscreteDemand):
            raise TypeError("demand must be a DiscreteDemand")
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError("case kind must be nonempty")
        normalized_n = _integer(n, name="n")
        schema = state_schema_for_config(config, model_id=config.composition_key)
        if initial_state is None:
            normalized_initial = InventoryState.zeros(schema)
        elif isinstance(initial_state, InventoryState):
            schema.validate(initial_state)
            normalized_initial = initial_state
        else:
            normalized_initial = InventoryState.from_flat(initial_state, schema)
        object.__setattr__(self, "name", name.strip())
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "costs", normalized_costs)
        object.__setattr__(self, "demand", demand)
        object.__setattr__(self, "n", normalized_n)
        object.__setattr__(self, "kind", kind.strip())
        object.__setattr__(self, "initial_state", normalized_initial)
        object.__setattr__(self, "schema", schema)

    @property
    def model_id(self) -> str:
        """Readable model family id, derived from enabled components."""

        return self.config.composition_key

    @property
    def model_version(self) -> str:
        return self.config.model_version

    @property
    def model_identity(self) -> str:
        return f"{self.model_id}@{self.model_version}"

    @property
    def composition_key(self) -> str:
        return self.config.composition_key

    @property
    def composition_hash(self) -> str:
        return self.config.composition_hash

    @property
    def composition_id(self) -> str:
        return self.config.composition_id

    @property
    def composition_identity(self) -> str:
        """Alias for the full configuration hash used in cache identities."""

        return self.config.composition_hash

    @property
    def transition_spec(self):
        return self.config.transition

    @property
    def transition_mode(self):
        return self.config.transition_mode

    @property
    def event_order(self) -> tuple[str, ...]:
        return self.config.transition.event_order

    @property
    def state_schema(self) -> CanonicalStateSchema:
        """Compatibility spelling for callers that use ``state_schema``."""

        return self.schema

    @property
    def kernel(self):
        """Build the canonical transition kernel for this case."""

        return build_transition_kernel(
            self.config,
            self.costs,
            schema=self.schema,
            action_cap=self.n,
        )

    @property
    def action_cap(self) -> int:
        return self.n

    def feasible_actions(self, state: InventoryState | Sequence[int]) -> range:
        """Return the canonical state-independent integer order domain."""

        normalized = (
            state
            if isinstance(state, InventoryState)
            else InventoryState.from_flat(state, self.schema)
        )
        self.schema.validate(normalized)
        return range(self.action_cap + 1)

    # Concise cost spellings used by the established lead-time algorithms.
    @property
    def h(self) -> float:
        return float(self.costs.holding)

    @property
    def b(self) -> float:
        return float(self.costs.shortage)

    @property
    def w(self) -> float:
        return float(self.costs.expiry)

    @property
    def g(self) -> float:
        """Backlog carry cost per unit retained into the next state."""

        return float(self.costs.backlog)

    @property
    def a(self) -> float:
        """Backlog abandonment cost per unit leaving without service."""

        return float(self.costs.abandonment)

    @property
    def k(self) -> float:
        """Ordering cost per unit ordered."""

        return float(self.costs.ordering)

    @property
    def lead_time(self) -> int:
        """Maximum deterministic/known ETA represented by the pipeline."""
        lead = self.config.lead_time
        if lead.periods is not None:
            return int(lead.periods)
        return max((int(value) for value in lead.support), default=0)

    @property
    def shelf_life(self) -> int | None:
        perishability = self.config.perishability
        if perishability.shelf_life is not None:
            return int(perishability.shelf_life)
        if perishability.lifetime_support:
            return max(int(value) for value in perishability.lifetime_support)
        return None

    @property
    def initial_flat_state(self) -> tuple[int, ...]:
        return self.initial_state.flatten(self.schema)

    def identity_payload(self) -> dict[str, object]:
        """Return a JSON-safe payload covering all case-defining inputs."""

        payload = {
            "signature_schema_version": 1,
            "case_name": self.name,
            "kind": self.kind,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "composition_hash": self.composition_hash,
            "config": self.config.to_dict(),
            "schema": self.schema.to_dict(),
            "state_schema": list(self.schema.names),
            "costs": _cost_payload(self.costs),
            "n": self.n,
            "demand": {
                "name": str(self.demand.name),
                "values": [int(value) for value in self.demand.values],
                "probs": [float(probability) for probability in self.demand.probs],
                "params": dict(self.demand.params),
            },
            "initial_state": list(self.initial_flat_state),
        }
        if self.config.backlog.enabled:
            payload["cost_semantics"] = "backlog_no_entry_shortage"
        return payload

    def to_dict(self) -> dict[str, object]:
        """Serialize the case contract without NumPy scalar objects."""

        return self.identity_payload()

    @property
    def case_signature(self) -> str:
        payload = canonical_json(self.identity_payload()).encode("utf-8")
        return stable_key(payload)

    @property
    def signature(self) -> str:
        """Short spelling retained for generic cache/manifest code."""

        return self.case_signature


__all__ = ["InventoryCase"]
