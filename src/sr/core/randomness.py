"""Deterministic random namespaces for symbolic search."""

from __future__ import annotations



from runtime.artifacts import seed_from_text
def derived_seed(seed: int, *tokens: object) -> int:
    encoded = ":".join((str(seed), *(str(token) for token in tokens))).encode("utf-8")
    return seed_from_text(encoded)


__all__ = ["derived_seed"]
