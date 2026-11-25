"""Case-derived domain contract for policy-family search.

The family-search layer consumes this immutable description instead of
embedding inventory state, motif, or feature names in prompts.  Every context
is backed by a concrete :class:`InventoryCase`, so combinations such as lead
time plus backlog or perishability plus backlog use the same contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

from inventory.core import InventoryCase
from inventory.policy.ast import AST
from inventory.policy.domain import policy_domain_for_case
from inventory.policy.features import FeatureSpec, feature_specs_for_config
from inventory.policy.motifs import motifs_for_config


@dataclass(frozen=True)
class InventoryFamilyContext:
    context_id: str
    case: InventoryCase

    def __post_init__(self) -> None:
        if not str(self.context_id).strip():
            raise ValueError("family context_id must be nonempty")
        if not isinstance(self.case, InventoryCase):
            raise TypeError("family context case must be an InventoryCase")


@dataclass(frozen=True)
class InventoryFamilyDomain:
    """Vocabulary shared by one policy family across concrete contexts."""

    contexts: tuple[InventoryFamilyContext, ...]
    training_case_count: int
    cyclic_cap: bool = False

    def __post_init__(self) -> None:
        if not self.contexts:
            raise ValueError("family domain requires at least one context")
        ids = tuple(row.context_id for row in self.contexts)
        if len(ids) != len(set(ids)):
            raise ValueError("family context ids must be unique")
        if int(self.training_case_count) <= 0:
            raise ValueError("training_case_count must be positive")

    @classmethod
    def from_cases(
        cls,
        contexts: Mapping[str, InventoryCase],
        *,
        training_case_count: int,
        cyclic_cap: bool = False,
    ) -> "InventoryFamilyDomain":
        return cls(
            tuple(
                InventoryFamilyContext(str(context_id), case)
                for context_id, case in contexts.items()
            ),
            int(training_case_count),
            cyclic_cap=cyclic_cap,
        )

    @property
    def context_ids(self) -> tuple[str, ...]:
        return tuple(row.context_id for row in self.contexts)

    @property
    def common_motif_names(self) -> tuple[str, ...]:
        names = [set(motifs_for_config(row.case.config)) for row in self.contexts]
        return tuple(sorted(set.intersection(*names)))

    @property
    def common_features(self) -> Mapping[str, FeatureSpec]:
        catalogs = [feature_specs_for_config(row.case.config) for row in self.contexts]
        common_names = set(catalogs[0])
        for catalog in catalogs[1:]:
            common_names.intersection_update(catalog)
        output: dict[str, FeatureSpec] = {}
        for name in sorted(common_names):
            definitions = {catalog[name] for catalog in catalogs}
            if len(definitions) == 1:
                output[name] = definitions.pop()
        return output

    @property
    def common_motifs(self) -> Mapping[str, Mapping[str, AST]]:
        names = self.common_motif_names
        return {
            row.context_id: {
                name: motifs_for_config(row.case.config)[name]
                for name in names
            }
            for row in self.contexts
        }

    def prompt_payload(self) -> dict[str, object]:
        """Return the complete machine-readable vocabulary for an SR prompt."""

        context_rows = []
        rules: set[str] = set()
        for row in self.contexts:
            domain = policy_domain_for_case(row.case)
            rules.update(domain.rules)
            context_rows.append(
                {
                    "context_id": row.context_id,
                    "model": domain.name,
                    "components": list(domain.components),
                    "state_names": list(domain.state_names),
                    "distribution_names": list(domain.distribution_names),
                    "cost_names": list(domain.cost_names),
                }
            )
        return {
            "contexts": context_rows,
            "training_case_count": int(self.training_case_count),
            "cyclic_cap_enabled": self.cyclic_cap,
            "rules": sorted(rules),
            "common_motifs": {
                context_id: dict(motifs)
                for context_id, motifs in self.common_motifs.items()
            },
            "common_features": {
                name: asdict(spec)
                for name, spec in self.common_features.items()
            },
        }

    def validate_family(self, family: object) -> None:
        """Validate through the inventory-owned family materializer."""

        from inventory.policy.family import PolicyFamily, validate_family_for_cases

        if not isinstance(family, PolicyFamily):
            raise TypeError("family must be a PolicyFamily")
        def contains_cyclic(node):
            if isinstance(node, Mapping):
                return (node.get("op") == "cap" and node.get("mode") == "cyclic_integer") or any(contains_cyclic(v) for v in node.values())
            return isinstance(node, (list, tuple)) and any(contains_cyclic(v) for v in node)
        if not self.cyclic_cap and (contains_cyclic(family.ast) or contains_cyclic(family.motifs)):
            raise ValueError("cyclic cap is not enabled for this family domain")
        validate_family_for_cases(family, [row.case for row in self.contexts])

    def semantic_hash(self, family: object) -> str:
        from inventory.policy.family import (
            PolicyFamily,
            semantic_family_hash_for_cases,
        )

        if not isinstance(family, PolicyFamily):
            raise TypeError("family must be a PolicyFamily")
        return semantic_family_hash_for_cases(
            family,
            [(row.context_id, row.case) for row in self.contexts],
        )

    def complexity_stats(self, family: object) -> dict[str, object]:
        from inventory.policy.family import PolicyFamily, family_complexity_for_cases

        if not isinstance(family, PolicyFamily):
            raise TypeError("family must be a PolicyFamily")
        return family_complexity_for_cases(
            family,
            [(row.context_id, row.case) for row in self.contexts],
        )


__all__ = ["InventoryFamilyContext", "InventoryFamilyDomain"]
