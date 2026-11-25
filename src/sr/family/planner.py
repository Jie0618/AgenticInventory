"""Deterministic context/case task planning for family evaluation."""

from __future__ import annotations

from runtime.artifacts import seed_from_text, stable_key
import json
from typing import Iterable, Sequence

from sr.family.models import FamilyTask


def family_identity(family: object) -> str:
    payload = {
        "ast": getattr(family, "ast"),
        "motifs": getattr(family, "motifs"),
        "features": [getattr(item, "__dict__", item) for item in getattr(family, "features", ())],
        "parameters": [getattr(item, "__dict__", item) for item in getattr(family, "parameters", ())],
    }
    return stable_key(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


def task_seed(global_seed: int, identity: str, context_id: str, case_id: str, fidelity: str) -> int:
    token = f"{global_seed}:{identity}:{context_id}:{case_id}:{fidelity}".encode()
    return seed_from_text(token)


def plan_family_tasks(
    family: object,
    case_ids: Sequence[str],
    *,
    contexts: Iterable[str],
    global_seed: int = 20260811,
    fidelity: str = "standard",
    evaluator: str = "mc",
) -> tuple[FamilyTask, ...]:
    context_ids = tuple(dict.fromkeys(str(value) for value in contexts))
    if not context_ids:
        raise ValueError("contexts must be nonempty")
    if not case_ids:
        raise ValueError("case_ids must be nonempty")
    identity = family_identity(family)
    return tuple(
        FamilyTask(
            str(getattr(family, "family_id")),
            context_id,
            str(case_id),
            str(evaluator),
            str(fidelity),
            task_seed(global_seed, identity, context_id, str(case_id), fidelity),
        )
        for context_id in context_ids
        for case_id in case_ids
    )


__all__ = ["family_identity", "plan_family_tasks", "task_seed"]
