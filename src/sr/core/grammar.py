"""Model-neutral motif expansion and AST complexity accounting."""

from __future__ import annotations

from typing import Mapping, Sequence

from inventory.policy.ast import AST, ASTLimits, ast_check


def expanded_complexity(
    ast: AST,
    motifs: Mapping[str, AST],
    *,
    allowed_state_names: Sequence[str] | None = None,
) -> float:
    """Return complexity after recursively replacing motif leaves."""

    expanded = expand_motifs(ast, motifs)
    return float(
        ast_check(
            expanded,
            motifs={},
            limits=ASTLimits(max_depth=128, max_nodes=10000, max_parameters=128),
            allowed_state_names=allowed_state_names,
        ).stats.complexity
    )


def expand_motifs(
    ast: AST, motifs: Mapping[str, AST], stack: tuple[str, ...] = ()
) -> AST:
    if ast.get("op") == "motif":
        name = str(ast["name"])
        if name in stack:
            raise ValueError(f"recursive motif expansion: {name}")
        if name not in motifs:
            raise ValueError(f"unknown motif: {name}")
        return expand_motifs(motifs[name], motifs, stack + (name,))
    output: AST = {}
    for key, value in ast.items():
        if isinstance(value, dict):
            output[key] = expand_motifs(value, motifs, stack)
        elif isinstance(value, list):
            output[key] = [
                expand_motifs(item, motifs, stack) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            output[key] = value
    return output


__all__ = ["expand_motifs", "expanded_complexity"]
