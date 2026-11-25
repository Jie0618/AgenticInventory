"""Stable identities for Stage-III experiment inputs."""

from __future__ import annotations



from runtime.artifacts import stable_key
def case_suite_signature(cases) -> str:
    """Hash every named case so caches cannot outlive their environments."""

    rows = sorted(
        (str(case.name), str(case.case_signature))
        for case in cases
    )
    if not rows:
        raise ValueError("case suite must be nonempty")
    payload = "\n".join(f"{name}:{signature}" for name, signature in rows)
    return stable_key(payload.encode())[:16]


__all__ = ["case_suite_signature"]
