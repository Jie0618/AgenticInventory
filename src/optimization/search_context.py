"""Build model-dependent optimization inputs for symbolic search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from evaluation.backends.lead_time.dp import DPResult
from evaluation.dp import ComposedDPResult
from evaluation.routing import BackendDecision, select_evaluation_backend, solve_inventory_dp, validate_reference_for_case
from inventory.core import InventoryCase
from inventory.policy.ast import AST
from inventory.policy.motifs import motifs_for_config
from optimization.models.types import FitResult, OptimizationConfig, ParameterSpec, SupervisedSample
from optimization.policies.parameters import parameter_specs_for_case
from optimization.samples.reference import reference_action_samples
from optimization.solvers.api import fit_inventory_policy


@dataclass(frozen=True)
class InventorySearchContext:
    """All model-dependent inputs needed to evaluate generated structures."""

    case: InventoryCase
    horizon: int
    reference: DPResult | ComposedDPResult
    motifs: Mapping[str, AST]
    samples: tuple[SupervisedSample, ...]
    backend: BackendDecision

    @classmethod
    def build(
        cls,
        case: InventoryCase,
        horizon: int,
        *,
        max_states: int = 2_000_000,
        max_table_cells: int = 50_000_000,
    ) -> "InventorySearchContext":
        decision = select_evaluation_backend(case)
        reference = solve_inventory_dp(
            case,
            horizon,
            max_states=max_states,
            max_table_cells=max_table_cells,
        )
        samples = reference_action_samples(reference)
        return cls(
            case=case,
            horizon=int(horizon),
            reference=reference,
            motifs=motifs_for_config(case.config),
            samples=tuple(samples),
            backend=decision,
        )

    @classmethod
    def from_reference(
        cls,
        case: InventoryCase,
        horizon: int,
        reference: DPResult | ComposedDPResult,
        *,
        samples: Sequence[SupervisedSample] | None = None,
    ) -> "InventorySearchContext":
        """Bind an already loaded reference without solving the DP again."""

        decision = select_evaluation_backend(case)
        validate_reference_for_case(case, reference, horizon)
        resolved_samples = (
            tuple(samples)
            if samples is not None
            else tuple(reference_action_samples(reference))
        )
        if not resolved_samples:
            raise ValueError("search context requires at least one sample")
        return cls(
            case=case,
            horizon=int(horizon),
            reference=reference,
            motifs=motifs_for_config(case.config),
            samples=resolved_samples,
            backend=decision,
        )

    def parameter_specs(self, names: Sequence[str]) -> tuple[ParameterSpec, ...]:
        return parameter_specs_for_case(names, self.case)

    def fit(
        self,
        ast: AST,
        *,
        motifs: Mapping[str, AST] | None = None,
        parameter_specs: Sequence[ParameterSpec] | None = None,
        config: OptimizationConfig = OptimizationConfig(),
        warm_starts: Sequence[Mapping[str, float]] = (),
    ) -> FitResult:
        from inventory.policy.ast import ast_check

        active_motifs = self.motifs if motifs is None else motifs
        checked = ast_check(
            ast,
            motifs=active_motifs,
            allowed_state_names=self.case.schema.allowed_state_names,
        )
        specs = (
            self.parameter_specs(checked.stats.parameters)
            if parameter_specs is None
            else tuple(parameter_specs)
        )
        return fit_inventory_policy(
            checked.ast,
            self.case,
            self.horizon,
            specs,
            samples=self.samples,
            metric_samples=self.samples,
            motifs=active_motifs,
            config=config,
            warm_starts=warm_starts,
            reference=self.reference,
        )

__all__ = ["InventorySearchContext"]
