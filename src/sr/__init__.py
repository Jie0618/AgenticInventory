"""Public symbolic-regression API with cycle-safe lazy exports."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "SearchConfig": ("sr.core.models", "SearchConfig"),
    "SearchCandidate": ("sr.core.models", "SearchCandidate"),
    "ScreenedCandidate": ("sr.core.models", "ScreenedCandidate"),
    "SearchIndividual": ("sr.core.models", "SearchIndividual"),
    "ReflectionMotif": ("sr.core.models", "ReflectionMotif"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
