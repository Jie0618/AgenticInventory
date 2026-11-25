"""Setting-specific benchmark suites over the shared demand catalog."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from inventory.core import (
    DEFAULT_TRANSITION_VERSION,
    DeterministicBacklog,
    DeterministicLeadTime,
    FixedShelfLife,
    LostSales,
    inventory_case,
)

from .demand_catalogs.inventory_50 import load_demands
from .profiles import (
    LEAD_TIME_BACKLOG_PROFILES,
    LEAD_TIME_PROFILES,
    PERISHABLE_PROFILES,
    EconomicProfile,
)


SETTING_LEAD_TIME = "lead_time"
SETTING_PERISHABLE = "perishable"
SETTING_LEAD_TIME_PERISHABLE = "lead_time_perishable"
SETTING_LEAD_TIME_BACKLOG = "lead_time_backlog"
SETTINGS = (
    SETTING_LEAD_TIME,
    SETTING_PERISHABLE,
    SETTING_LEAD_TIME_PERISHABLE,
    SETTING_LEAD_TIME_BACKLOG,
)

# Frozen lead-time execution geometry for consistent case IDs and MC banks.
LEAD_TIME_STATE_CAPS = {1: 50, 2: 40, 3: 22, 4: 14, 5: 8, 6: 5, 7: 4, 8: 3}
SEED = 20260819
HORIZON = 1000
FIT_SAMPLE_SIZE = 5000


@dataclass(frozen=True)
class StructuralContext:
    context_id: str
    lead_time: int = 0
    shelf_life: int = 0
    backlog_patience: int = 0

    def __post_init__(self) -> None:
        values = (self.lead_time, self.shelf_life, self.backlog_patience)
        if not self.context_id or any(int(value) < 0 for value in values):
            raise ValueError("structural contexts require an id and nonnegative dimensions")


DEFAULT_CONTEXTS = {
    SETTING_LEAD_TIME: tuple(
        StructuralContext(f"L{lead_time}", lead_time=lead_time)
        for lead_time in range(1, 9)
    ),
    SETTING_PERISHABLE: tuple(
        StructuralContext(f"M{shelf_life}", shelf_life=shelf_life)
        for shelf_life in (2, 3, 4, 5)
    ),
    SETTING_LEAD_TIME_PERISHABLE: tuple(
        StructuralContext(f"L{lead}_M{shelf}", lead_time=lead, shelf_life=shelf)
        for lead, shelf in ((2, 2), (3, 3), (1, 4), (4, 1), (5, 3), (3, 5))
    ),
    SETTING_LEAD_TIME_BACKLOG: tuple(
        StructuralContext(f"L{lead}_K{patience}", lead_time=lead, backlog_patience=patience)
        for lead, patience in ((2, 2), (3, 3), (1, 4), (4, 1), (5, 3), (3, 5))
    ),
}


def profiles_for(setting: str) -> tuple[EconomicProfile, ...]:
    if setting == SETTING_LEAD_TIME:
        return LEAD_TIME_PROFILES
    if setting in {SETTING_PERISHABLE, SETTING_LEAD_TIME_PERISHABLE}:
        return PERISHABLE_PROFILES
    if setting == SETTING_LEAD_TIME_BACKLOG:
        return LEAD_TIME_BACKLOG_PROFILES
    raise ValueError(f"unknown benchmark setting: {setting}")


def _validate_profiles(setting: str, profiles: Sequence[EconomicProfile]) -> None:
    if not profiles or len({row.profile_id for row in profiles}) != len(profiles):
        raise ValueError("economic profiles must be unique and nonempty")
    for profile in profiles:
        if setting == SETTING_LEAD_TIME:
            valid = (
                profile.expiry_over_holding == 0.0
                and profile.backlog_over_shortage == 0.0
                and profile.abandonment_over_shortage == 0.0
            )
        elif setting in {SETTING_PERISHABLE, SETTING_LEAD_TIME_PERISHABLE}:
            valid = (
                profile.expiry_over_holding > 0.0
                and profile.backlog_over_shortage == 0.0
                and profile.abandonment_over_shortage == 0.0
            )
        elif setting == SETTING_LEAD_TIME_BACKLOG:
            valid = (
                profile.expiry_over_holding == 0.0
                and profile.backlog_over_shortage > 0.0
                and profile.abandonment_over_shortage > 0.0
            )
        else:
            raise ValueError(f"unknown benchmark setting: {setting}")
        if not valid:
            raise ValueError(
                f"economic profile {profile.profile_id!r} is invalid for {setting}"
            )


def _validate_context(setting: str, context: StructuralContext) -> None:
    if setting == SETTING_LEAD_TIME:
        valid = context.lead_time >= 1 and context.shelf_life == 0 and context.backlog_patience == 0
    elif setting == SETTING_PERISHABLE:
        valid = context.lead_time == 0 and context.shelf_life >= 1 and context.backlog_patience == 0
    elif setting == SETTING_LEAD_TIME_PERISHABLE:
        valid = context.lead_time >= 1 and context.shelf_life >= 1 and context.backlog_patience == 0
    elif setting == SETTING_LEAD_TIME_BACKLOG:
        valid = context.lead_time >= 1 and context.shelf_life == 0 and context.backlog_patience >= 1
    else:
        raise ValueError(f"unknown benchmark setting: {setting}")
    if not valid:
        raise ValueError(f"context {context.context_id!r} is invalid for {setting}")


def cases(
    setting: str,
    *,
    contexts: Sequence[StructuralContext] | None = None,
    n: int | None = None,
    state_caps: dict[int, int] | None = None,
):
    """Materialize one independently setting-specific suite."""

    if setting not in SETTINGS:
        raise ValueError(f"unknown benchmark setting: {setting}")
    if n is not None and int(n) <= 0:
        raise ValueError("n must be positive")
    if state_caps is not None and any(int(key) <= 0 or int(value) <= 0 for key, value in state_caps.items()):
        raise ValueError("state_caps must contain positive lead times and caps")
    selected_contexts = tuple(DEFAULT_CONTEXTS[setting] if contexts is None else contexts)
    if not selected_contexts or len({row.context_id for row in selected_contexts}) != len(selected_contexts):
        raise ValueError("contexts must be unique and nonempty")
    for context in selected_contexts:
        _validate_context(setting, context)

    profiles = profiles_for(setting)
    _validate_profiles(setting, profiles)
    output = []
    for context in selected_contexts:
        if n is not None:
            case_n = int(n)
        elif setting == SETTING_LEAD_TIME:
            caps = LEAD_TIME_STATE_CAPS if state_caps is None else {int(k): int(v) for k, v in state_caps.items()}
            if context.lead_time not in caps:
                raise ValueError(f"missing state cap for lead time {context.lead_time}")
            case_n = caps[context.lead_time]
        else:
            case_n = 20
        for demand_record in load_demands():
            for profile in profiles:
                kwargs = {}
                if context.lead_time:
                    kwargs["lead_time"] = DeterministicLeadTime(context.lead_time)
                if context.shelf_life:
                    kwargs["perishability"] = FixedShelfLife(context.shelf_life)
                if context.backlog_patience:
                    kwargs["backlog"] = DeterministicBacklog(context.backlog_patience)
                else:
                    kwargs["backlog"] = LostSales()
                if setting == SETTING_LEAD_TIME:
                    shortage_label = format(profile.shortage_over_holding, "g")
                    case_name = (
                        f"L{context.lead_time}__{demand_record.demand_id}__"
                        f"h1_b{shortage_label}"
                    )
                    case_kind = demand_record.family
                else:
                    case_name = (
                        f"inventory_benchmark_family__{setting}__{context.context_id}__"
                        f"{demand_record.demand_id}__{profile.profile_id}"
                    )
                    case_kind = f"benchmark:{setting}:{demand_record.family}"
                output.append(inventory_case(
                    name=case_name,
                    demand=demand_record.demand,
                    n=case_n,
                    costs=profile.costs(),
                    capacity=case_n,
                    kind=case_kind,
                    **kwargs,
                ))
    expected = len(selected_contexts) * 50 * len(profiles)
    if len(output) != expected or len({row.name for row in output}) != expected:
        raise RuntimeError(f"expected {expected} unique {setting} cases, found {len(output)}")
    return output


def suite_manifest(setting: str) -> dict[str, object]:
    contexts = DEFAULT_CONTEXTS.get(setting)
    if contexts is None:
        raise ValueError(f"unknown benchmark setting: {setting}")
    profiles = profiles_for(setting)
    _validate_profiles(setting, profiles)
    return {
        "schema_version": 1,
        "benchmark_family": "inventory_benchmark_family",
        "suite_id": f"inventory_benchmark_family__{setting}",
        "setting": setting,
        "demand_catalog_count": 50,
        "transition_version": DEFAULT_TRANSITION_VERSION,
        "contexts": [
            {
                "context_id": row.context_id,
                "lead_time": row.lead_time,
                "shelf_life": row.shelf_life,
                "backlog_patience": row.backlog_patience,
            }
            for row in contexts
        ],
        "context_count": len(contexts),
        "economic_profiles": [
            {
                "profile_id": row.profile_id,
                "shortage_over_holding": row.shortage_over_holding,
                "expiry_over_holding": row.expiry_over_holding,
                "backlog_over_shortage": row.backlog_over_shortage,
                "abandonment_over_shortage": row.abandonment_over_shortage,
            }
            for row in profiles
        ],
        "economic_profile_count": len(profiles),
        "case_count": len(contexts) * 50 * len(profiles),
        "holding_cost": 1.0,
        "default_state_caps": (
            {str(key): value for key, value in LEAD_TIME_STATE_CAPS.items()}
            if setting == SETTING_LEAD_TIME
            else None
        ),
        "cost_fields": {
            SETTING_LEAD_TIME: ["holding", "shortage"],
            SETTING_PERISHABLE: ["holding", "shortage", "expiry"],
            SETTING_LEAD_TIME_PERISHABLE: ["holding", "shortage", "expiry"],
            SETTING_LEAD_TIME_BACKLOG: ["holding", "shortage", "backlog", "abandonment"],
        }[setting],
    }


__all__ = [
    "DEFAULT_CONTEXTS",
    "SETTINGS",
    "SETTING_LEAD_TIME",
    "SETTING_LEAD_TIME_BACKLOG",
    "SETTING_LEAD_TIME_PERISHABLE",
    "SETTING_PERISHABLE",
    "LEAD_TIME_STATE_CAPS",
    "SEED",
    "HORIZON",
    "FIT_SAMPLE_SIZE",
    "StructuralContext",
    "cases",
    "profiles_for",
    "suite_manifest",
]
