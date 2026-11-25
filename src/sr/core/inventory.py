"""Reusable SR bindings for any canonical inventory composition."""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import partial
from typing import Mapping, Sequence

from inventory.policy.ast import AST
from inventory.policy.domain import policy_domain_for_case
from optimization.models.types import FitResult, OptimizationConfig
from optimization.search_context import InventorySearchContext
from optimization.policies.parameters import parameter_specs_for_case, target_roles_for_motifs
from inventory.policy.motifs import motifs_for_config
from inventory.core import InventoryCase
from sr.core.gp import generate_gp_descendants, make_candidate
from sr.core.models import SearchCandidate, SearchConfig, SearchIndividual
from sr.llm.search import SearchDomain


@dataclass(frozen=True)
class InventorySearchBindings:
    """Case-derived operations consumed by a generic agentic controller.

    The context owns automatic evaluator/optimizer routing while the binding
    exposes one structural interface for every component composition.
    """

    context: InventorySearchContext | None
    config: SearchConfig
    domain: SearchDomain
    target_roles: Mapping[str, str]
    case: InventoryCase
    motif_registry: Mapping[str, AST]

    @classmethod
    def build(
        cls,
        context: InventorySearchContext,
        config: SearchConfig = SearchConfig(),
    ) -> "InventorySearchBindings":
        if not isinstance(context, InventorySearchContext):
            raise TypeError("context must be an InventorySearchContext")
        return cls(
            context=context,
            config=config,
            domain=SearchDomain.from_spec(policy_domain_for_case(context.case)),
            target_roles=target_roles_for_motifs(context.motifs),
            case=context.case,
            motif_registry=dict(context.motifs),
        )

    @classmethod
    def build_context_free(
        cls,
        case: InventoryCase,
        config: SearchConfig = SearchConfig(),
    ) -> "InventorySearchBindings":
        """Build structural bindings when numerical references are not needed."""

        motifs = motifs_for_config(case.config)
        return cls(
            context=None,
            config=config,
            domain=SearchDomain.from_spec(policy_domain_for_case(case)),
            target_roles=target_roles_for_motifs(motifs),
            case=case,
            motif_registry=motifs,
        )

    @property
    def motifs(self) -> Mapping[str, AST]:
        return self.motif_registry

    def candidate(
        self,
        ast: AST,
        *,
        source: str,
        parents: Sequence[str] = (),
        iteration: int = 0,
        island: int = 0,
        detail: Mapping[str, object] | None = None,
        motifs: Mapping[str, AST] | None = None,
    ) -> SearchCandidate:
        active_motifs = self.motifs if motifs is None else motifs
        return make_candidate(
            ast,
            motifs=active_motifs,
            config=self.config,
            source=source,
            parameter_validator=partial(parameter_specs_for_case, case=self.case),
            allowed_state_names=self.case.schema.allowed_state_names,
            parents=parents,
            iteration=iteration,
            island=island,
            detail=detail,
        )

    def descendants(
        self,
        parents: Sequence[SearchIndividual],
        *,
        count: int,
        rng: random.Random,
        iteration: int,
        island: int,
        operator_weights: Mapping[str, float] | None = None,
        parent_weights: Mapping[str, float] | None = None,
        motifs: Mapping[str, AST] | None = None,
    ) -> list[SearchCandidate]:
        active_motifs = self.motifs if motifs is None else motifs
        return generate_gp_descendants(
            parents,
            count=count,
            motifs=active_motifs,
            config=self.config,
            rng=rng,
            iteration=iteration,
            island=island,
            candidate_builder=partial(self.candidate, motifs=active_motifs),
            target_roles=target_roles_for_motifs(active_motifs),
            operator_weights=operator_weights,
            parent_weights=parent_weights,
        )

    def fit(
        self,
        candidate: SearchCandidate,
        *,
        mode: str = "standard",
        random_seed: int = 0,
        warm_starts: Sequence[Mapping[str, float]] = (),
        motifs: Mapping[str, AST] | None = None,
    ) -> FitResult:
        if mode not in {"fast", "standard", "certify"}:
            raise ValueError("mode must be fast, standard, or certify")
        if self.context is None:
            raise RuntimeError("numerical fitting requires an InventorySearchContext")
        return self.context.fit(
            candidate.ast,
            motifs=self.motifs if motifs is None else motifs,
            config=OptimizationConfig(mode=mode, random_seed=int(random_seed)),
            warm_starts=warm_starts,
        )


__all__ = ["InventorySearchBindings"]
