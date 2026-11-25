"""Stage-III MC worker for lead-time + backlog policy families."""

from __future__ import annotations

from runtime.artifacts import stable_key
import time
from pathlib import Path

from inventory.policy.family import materialize_family_for_case
from inventory.policy.family_domain import InventoryFamilyDomain
from optimization import OptimizationConfig, ParameterSpec, fit_inventory_policy_mc
from optimization.evaluators.composed import ComposedMonteCarloPolicyCostEvaluator
from sr.family.composed_reference import catalog_signature
from sr.family.identity import case_suite_signature
from sr.family.models import CaseMetric

from experiments.lead_time_backlog.reference import ReferenceWorker, _REFERENCE_PROTOCOL, _catalog, _seed
from ..cases import CASE_SUFFIXES, CONTEXTS, cases


def _specs(family):
    output = []
    for item in family.parameters:
        integer = item.kind in {"target", "cap", "lag"} and float(item.lower).is_integer() and float(item.upper).is_integer()
        output.append(ParameterSpec(
            item.name, kind="integer" if integer else "continuous",
            lower=item.lower, upper=item.upper,
            default=round((item.lower + item.upper) / 2) if integer else (item.lower + item.upper) / 2,
        ))
    return tuple(output)


class FamilyWorker(ReferenceWorker):
    subprocess_module = "experiments.lead_time_backlog.stage3_multi_context_sr.runtime.family_worker"

    def __init__(self, *, config_path: Path, reference_cache_dir: Path | None = None,
                 reference_mode: str = "best_baseline") -> None:
        super().__init__(config_path=config_path, reference_cache_dir=reference_cache_dir,
                         reference_mode=reference_mode)
        self.case_map = {row.name: row for row in cases(int(self.config.get("state_action_cap", 20)))}
        self.case_suite_signature = case_suite_signature(self.case_map.values())
        signatures = []
        for lead_time, patience in CONTEXTS:
            case = next(row for row in self.case_map.values() if row.name.startswith(f"L{lead_time}_K{patience}__"))
            signatures.append(catalog_signature(_REFERENCE_PROTOCOL, _catalog(case, self.config.get("reference_baselines"))))
        identity = stable_key("|".join(signatures).encode())[:16]
        self.evaluation_protocol_signature = (
            f"{_REFERENCE_PROTOCOL}:{self.reference_mode}:{identity}:"
            f"cases-{self.case_suite_signature}:typed-cap-{bool(self.config.get('cyclic_cap_enabled', False))}"
        )

    @property
    def context_ids(self):
        return tuple(f"L{lead}_K{patience}" for lead, patience in CONTEXTS)

    def family_domain(self):
        representatives = {
            context: next(row for row in self.case_map.values() if row.name.startswith(context + "__"))
            for context in self.context_ids
        }
        return InventoryFamilyDomain.from_cases(representatives, training_case_count=len(CASE_SUFFIXES),
            cyclic_cap=bool(self.config.get("cyclic_cap_enabled", False)))

    def __call__(self, family, task):
        started = time.perf_counter()
        case = self.case_map[f"{task.context_id}__{task.case_id}"]
        ast, motifs = materialize_family_for_case(family, case)
        mode = task.fidelity
        horizon = int(self.config.get("horizon", 1000))
        proposal = int(self.config.get(f"{mode}_proposal_paths", self.config["proposal_paths"]))
        strict = int(self.config.get(f"{mode}_strict_paths", self.config["strict_paths"]))
        final_paths = int(self.config.get(f"{mode}_evaluation_paths", self.config["evaluation_paths"]))
        fit = fit_inventory_policy_mc(
            ast, case, horizon, _specs(family), motifs=motifs,
            config=OptimizationConfig(
                mode=mode, objective="policy_cost", random_seed=task.seed,
                total_evaluations=self.config.get(f"{mode}_proposal_budget"),
            ),
            proposal_paths=proposal, strict_paths=strict,
            seed=_seed("ltb-tuning", case.name, mode),
        )
        final_seed = _seed("ltb-final", case.name, mode)
        final = ComposedMonteCarloPolicyCostEvaluator.from_case(
            ast, case, horizon, motifs=motifs, paths=final_paths, seed=final_seed,
        ).evaluate_many((fit.params,))[0]
        reference, hit, reference_type = self._reference(case, mode)
        cost = float(final.cost)
        return CaseMetric(
            task.context_id, task.case_id,
            100.0 * (cost / max(reference, 1.0e-12) - 1.0),
            time.perf_counter() - started,
            metadata={
                "cost": cost, "gap_reference_cost": reference,
                "reference_type": reference_type, "reference_cache_hit": hit,
                "constants": dict(fit.params), "schema_names": list(case.schema.names),
                "proposal_paths": proposal, "strict_paths": strict,
                "evaluation_paths": final_paths, "evaluation_seed": final_seed,
            },
        )

    def prepare_reference(self, context_id: str, case_id: str, fidelity: str = "standard") -> None:
        self._reference(self.case_map[f"{context_id}__{case_id}"], fidelity)


__all__ = ["FamilyWorker"]


def main() -> None:
    from sr.family.stage3_runner import run_family_worker_cli

    run_family_worker_cli(
        FamilyWorker,
        default_reference_mode="best_baseline",
        reference_modes=("best_baseline",),
    )


if __name__ == "__main__":
    main()
