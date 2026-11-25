"""Validated LLM structural proposal parsing and execution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from inventory.policy.ast import AST, ASTLimits, ast_check, maximum, minimum, parameter


ALLOWED_ACTIONS = {"minimum_crossover", "maximum_crossover", "cap", "replace_root",
                   "replace_subtree", "replace_summary", "change_cap_type"}


@dataclass(frozen=True)
class StructuralProposal:
    action: str
    parent_hash: str | None
    other_parent_hash: str | None
    ast: AST | None
    rationale: str
    path: tuple[str | int, ...] = ()
    mode: str | None = None
    bound: str = "upper"


def parse_structural_proposals(text: str) -> tuple[StructuralProposal, ...]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines)
    payload = json.loads(stripped)
    rows = payload.get("proposals") if isinstance(payload, Mapping) else None
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("LLM response requires a proposals list")
    proposals: list[StructuralProposal] = []
    for row in rows:
        if isinstance(row, str):
            try:
                row = json.loads(row)
            except json.JSONDecodeError as exc:
                raise ValueError("string proposal is not a JSON object") from exc
        if not isinstance(row, Mapping):
            raise ValueError("each proposal must be an object")
        action = str(row.get("action", ""))
        if action not in ALLOWED_ACTIONS:
            raise ValueError(f"unsupported action: {action}")
        ast = dict(row["ast"]) if isinstance(row.get("ast"), Mapping) else None
        proposals.append(
            StructuralProposal(
                action=action,
                parent_hash=str(row["parent_hash"]) if row.get("parent_hash") else None,
                other_parent_hash=str(row["other_parent_hash"]) if row.get("other_parent_hash") else None,
                ast=ast,
                rationale=str(row.get("rationale", "")),
                path=tuple(row.get("path", ())),
                mode=row.get("mode"),
                bound=str(row.get("bound", "upper")),
            )
        )
    if not proposals:
        raise ValueError("LLM response has no proposals")
    return tuple(proposals)


def execute_structural_proposal(
    proposal: StructuralProposal,
    parents: Mapping[str, AST],
    motifs: Mapping[str, AST],
    *,
    limits: ASTLimits | None = None,
    allowed_state_names: Sequence[str] | None = None,
) -> AST:
    if proposal.action == "replace_root":
        if proposal.ast is None:
            raise ValueError("replace_root requires ast")
        candidate = proposal.ast
    else:
        if proposal.parent_hash not in parents:
            raise ValueError("unknown parent_hash")
        left = parents[str(proposal.parent_hash)]
        if proposal.action == "cap":
            if proposal.bound not in {"upper", "lower"}:
                raise ValueError("cap bound must be upper or lower")
            candidate = (minimum if proposal.bound == "upper" else maximum)(left, parameter("C"))
        elif proposal.action in {"replace_subtree", "replace_summary", "change_cap_type"}:
            from sr.core.tree import iter_subtrees, replace_subtree
            target = next((ref.subtree for ref in iter_subtrees(left)
                           if tuple(ref.path) == proposal.path), None)
            if target is None:
                raise ValueError("unknown subtree path")
            replacement = proposal.ast
            if proposal.action == "replace_summary":
                if target.get("op") != "motif" or not replacement or replacement.get("op") != "motif":
                    raise ValueError("replace_summary must replace one summary-reference leaf")
            if proposal.action == "change_cap_type":
                from inventory.policy.ast import cap, CAP_MODES
                if proposal.mode not in CAP_MODES:
                    raise ValueError("invalid cap type")
                if target["op"] == "cap":
                    replacement = cap(target["arg"], target["limit"], proposal.mode)
                elif target["op"] == "minimum":
                    cap_slots = [(target[a], target[b]) for a, b in
                              (("left", "right"), ("right", "left"))
                              if target[a].get("op") == "parameter"
                              and str(target[a].get("name", "")).startswith("C")]
                    if len(cap_slots) != 1:
                        raise ValueError("cap-type change requires exactly one cap parameter")
                    limit, arg = cap_slots[0]
                    replacement = cap(arg, limit, proposal.mode)
                else:
                    raise ValueError("cap-type change requires a cap slot")
            if replacement is None:
                raise ValueError("replacement requires ast")
            candidate = replace_subtree(left, proposal.path, replacement)
        else:
            if proposal.other_parent_hash not in parents:
                raise ValueError("unknown other_parent_hash")
            if proposal.other_parent_hash == proposal.parent_hash:
                raise ValueError("crossover requires two distinct parents")
            right = parents[str(proposal.other_parent_hash)]
            candidate = minimum(left, right) if proposal.action == "minimum_crossover" else maximum(left, right)
    return ast_check(
        candidate,
        motifs=motifs,
        limits=limits or ASTLimits(max_depth=10, max_nodes=64, max_parameters=8),
        allowed_state_names=allowed_state_names,
    ).ast
