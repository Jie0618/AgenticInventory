"""Versioned benchmark family for the four canonical inventory settings."""

from .demand_catalogs.inventory_50 import DemandRecord, load_demands
from .profiles import (
    LEAD_TIME_PROFILES,
    LEAD_TIME_BACKLOG_PROFILES,
    PERISHABLE_PROFILES,
    EconomicProfile,
)
from .suites import (
    DEFAULT_CONTEXTS,
    LEAD_TIME_STATE_CAPS,
    SETTING_LEAD_TIME,
    SETTING_LEAD_TIME_BACKLOG,
    SETTING_LEAD_TIME_PERISHABLE,
    SETTING_PERISHABLE,
    StructuralContext,
    cases,
    suite_manifest,
)

__all__ = [
    "DEFAULT_CONTEXTS",
    "DemandRecord",
    "EconomicProfile",
    "LEAD_TIME_BACKLOG_PROFILES",
    "LEAD_TIME_PROFILES",
    "LEAD_TIME_STATE_CAPS",
    "PERISHABLE_PROFILES",
    "SETTING_LEAD_TIME",
    "SETTING_LEAD_TIME_BACKLOG",
    "SETTING_LEAD_TIME_PERISHABLE",
    "SETTING_PERISHABLE",
    "StructuralContext",
    "cases",
    "load_demands",
    "suite_manifest",
]
