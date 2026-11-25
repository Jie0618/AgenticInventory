"""Small current-grammar tree helpers for the policy AST."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Sequence

from inventory.policy.ast import AST, ASTValidationError


Path = tuple[Any, ...]


@dataclass(frozen=True)
class SubtreeRef:
    path: Path
    subtree: AST


def iter_subtrees(ast: AST) -> tuple[SubtreeRef, ...]:
    refs: list[SubtreeRef] = []

    def visit(node: Any, path: Path) -> None:
        if not _is_node(node):
            return
        refs.append(SubtreeRef(path, node))
        for key, value in node.items():
            if key == "op":
                continue
            if _is_node(value):
                visit(value, path + (key,))
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    if _is_node(child):
                        visit(child, path + (key, index))

    visit(ast, ())
    return tuple(refs)


def walk_nodes(value: Any):
    """Yield AST-shaped mappings recursively, including list members."""

    if isinstance(value, dict):
        if isinstance(value.get("op"), str):
            yield value
        for child in value.values():
            yield from walk_nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_nodes(child)


def contains_subtree(value: Any, target: Any) -> bool:
    """Return whether ``target`` occurs as an exact structural subtree."""

    return value == target or any(node == target for node in walk_nodes(value))


def matching_paths(value: Any, target: Any, prefix: Path = ()) -> tuple[Path, ...]:
    """Find field/index paths whose value equals ``target``."""

    matches: list[Path] = []
    if value == target:
        matches.append(prefix)
    if isinstance(value, dict):
        for key, child in value.items():
            if key != "op":
                matches.extend(matching_paths(child, target, prefix + (key,)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            matches.extend(matching_paths(child, target, prefix + (index,)))
    return tuple(matches)


def get_subtree(ast: AST, path: Sequence[Any]) -> AST:
    node = _navigate(ast, tuple(path))
    if not _is_node(node):
        raise ASTValidationError(f"path does not point to an AST node: {tuple(path)!r}")
    return deepcopy(node)


def replace_subtree(ast: AST, path: Sequence[Any], replacement: AST) -> AST:
    path = tuple(path)
    if not path:
        return deepcopy(replacement)
    root = deepcopy(ast)
    parent = _navigate(root, path[:-1])
    last = path[-1]
    if isinstance(parent, dict) and isinstance(last, str) and last in parent:
        parent[last] = deepcopy(replacement)
        return root
    if isinstance(parent, list) and isinstance(last, int) and 0 <= last < len(parent):
        parent[last] = deepcopy(replacement)
        return root
    raise ASTValidationError(f"invalid replacement path: {path!r}")


def _navigate(root: Any, path: Path) -> Any:
    node = root
    for part in path:
        if isinstance(node, dict) and isinstance(part, str) and part in node:
            node = node[part]
        elif isinstance(node, list) and isinstance(part, int) and 0 <= part < len(node):
            node = node[part]
        else:
            raise ASTValidationError(f"invalid AST path: {path!r}")
    return node


def _is_node(value: Any) -> bool:
    return isinstance(value, dict) and isinstance(value.get("op"), str)


__all__ = [
    "Path",
    "SubtreeRef",
    "contains_subtree",
    "get_subtree",
    "iter_subtrees",
    "matching_paths",
    "replace_subtree",
    "walk_nodes",
]
