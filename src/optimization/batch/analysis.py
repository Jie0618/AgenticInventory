from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping

from inventory.policy.ast import AST

from optimization.batch.samples import _BatchSamples


@dataclass(frozen=True)
class _NodeDependencies:
    parameters: frozenset[str] = frozenset()
    states: frozenset[str] = frozenset()

    def union(self, other: "_NodeDependencies") -> "_NodeDependencies":
        return _NodeDependencies(
            self.parameters | other.parameters,
            self.states | other.states,
        )


def _node_dependencies(node: AST, motifs: Mapping[str, AST]) -> _NodeDependencies:
    op = str(node.get("op", ""))
    if op == "parameter":
        return _NodeDependencies(parameters=frozenset({str(node["name"])}))
    if op == "state":
        return _NodeDependencies(states=frozenset({str(node["name"])}))
    if op == "motif":
        return _node_dependencies(motifs[str(node["name"])], motifs)
    result = _NodeDependencies(states=frozenset({"time"})) if op == "cap" and node["mode"] == "cyclic_integer" else _NodeDependencies()
    for value in node.values():
        if isinstance(value, dict) and "op" in value:
            result = result.union(_node_dependencies(value, motifs))
        elif isinstance(value, list):
            for child in value:
                if isinstance(child, dict) and "op" in child:
                    result = result.union(_node_dependencies(child, motifs))
    return result


class _BatchEvaluationContext:
    """Shared context for one vectorized AST evaluation workload.

    Most callers change parameters between evaluations and therefore keep the
    cache disabled.  Proposal MC fixes one parameter batch while advancing the
    same trajectories over many periods; it enables ``cache_state_independent``
    so constants, parameter-only expressions, and distribution/cost features
    are materialized once per candidate chunk instead of once per period.
    """

    def __init__(
        self,
        batch: _BatchSamples,
        motifs: Mapping[str, AST],
        *,
        cache_state_independent: bool = False,
        cache_within_evaluation: bool = False,
        dependency_cache: MutableMapping[int, _NodeDependencies] | None = None,
    ) -> None:
        self.batch = batch
        self.motifs = motifs
        self.cache_state_independent = bool(cache_state_independent)
        self.cache_within_evaluation = bool(cache_within_evaluation)
        self.cache_hits = 0
        self.cache_misses = 0
        self._dependencies: MutableMapping[int, _NodeDependencies] = (
            {} if dependency_cache is None else dependency_cache
        )
        self._persistent_values: dict[int, object] = {}
        self._transient_values: dict[int, object] = {}

    def begin_evaluation(self) -> None:
        """Invalidate values that depend on the current mutable state.

        State-independent values remain valid across horizon periods. Shared
        AST nodes that do depend on state may still be reused inside one
        evaluation, which gives generic DAG/common-subexpression elimination
        without assuming anything about a particular formula family.
        """

        self._transient_values.clear()

    def dependencies(self, node: AST) -> _NodeDependencies:
        key = id(node)
        result = self._dependencies.get(key)
        if result is None:
            result = _node_dependencies(node, self.motifs)
            self._dependencies[key] = result
        return result

    def cache_key(
        self,
        node: AST,
        local_vars: Mapping[str, object] | None = None,
    ) -> tuple[bool, int] | None:
        del local_vars
        state_dependent = bool(self.dependencies(node).states)
        if state_dependent:
            return (False, id(node)) if self.cache_within_evaluation else None
        if self.cache_state_independent:
            return (True, id(node))
        return (False, id(node)) if self.cache_within_evaluation else None

    def get(self, key: tuple[bool, int]):
        persistent, node_id = key
        values = self._persistent_values if persistent else self._transient_values
        value = values.get(node_id)
        if value is None:
            self.cache_misses += 1
        else:
            self.cache_hits += 1
        return value

    def put(self, key: tuple[bool, int], value) -> None:
        persistent, node_id = key
        values = self._persistent_values if persistent else self._transient_values
        values[node_id] = value

    def compact_batch(self, node: AST):
        return self.batch, range(self.batch.size)
