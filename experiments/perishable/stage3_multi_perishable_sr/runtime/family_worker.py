"""Stage-III MC worker for pure-perishable policy families."""

from __future__ import annotations

from runtime.artifacts import seed_from_text, stable_key
import fcntl
import json
import time
from pathlib import Path

from inventory.policy.family import materialize_family_for_case
from inventory.policy.family_domain import InventoryFamilyDomain
from optimization import OptimizationConfig, ParameterSpec, fit_inventory_policy_mc
from optimization.evaluators.composed import (
    ComposedMonteCarloPolicyCostEvaluator,
    ComputationalMonteCarloPolicyCostEvaluator,
)
from benchmarks.baselines.perishable import computational_baselines
from sr.family.models import CaseMetric
from sr.family.identity import case_suite_signature
from sr.family.composed_reference import select_reference_baselines

from ..cases import CASE_SUFFIXES, CONTEXTS, cases
from optimization.policies.baselines.perishable import baselines_for_perishable


_REFERENCE_PROTOCOL = "perishable-best-baseline-shared-bank"


def _seed(*tokens: object) -> int:
    return seed_from_text("|".join(map(str, tokens)).encode())


def _specs(family):
    output = []
    for item in family.parameters:
        integer = item.kind in {"target", "cap", "lag"} and float(item.lower).is_integer() and float(item.upper).is_integer()
        output.append(ParameterSpec(
            item.name,
            kind="integer" if integer else "continuous",
            lower=item.lower,
            upper=item.upper,
            default=round((item.lower + item.upper) / 2) if integer else (item.lower + item.upper) / 2,
        ))
    return tuple(output)


def _catalogs(case, names):
    formula = baselines_for_perishable(int(case.shelf_life), int(case.n), include_extended=True)
    computational = computational_baselines()
    selected = {row.name for row in select_reference_baselines(
        {row.name: row for row in (*formula, *computational)}, names)}
    return (tuple(row for row in formula if row.name in selected),
            tuple(row for row in computational if row.name in selected))


def _catalog_signature(case, names) -> str:
    formula, computational = _catalogs(case, names)
    payload = {
        "protocol": _REFERENCE_PROTOCOL,
        "formula": [
            {
                "name": spec.name,
                "ast": spec.ast,
                "motifs": spec.motifs,
                "parameters": [vars(parameter) for parameter in spec.parameter_specs],
            }
            for spec in formula
        ],
        "computational": [vars(spec) for spec in computational],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return stable_key(encoded.encode())[:16]


class FamilyWorker:
    subprocess_module = "experiments.perishable.stage3_multi_perishable_sr.runtime.family_worker"

    def __init__(self, *, config_path: Path, reference_cache_dir: Path | None = None, reference_mode: str = "best_baseline"):
        self.config_path = str(config_path)
        self.config = json.loads(Path(config_path).read_text())
        self.case_map = {row.name: row for row in cases(int(self.config.get("state_action_cap", 22)))}
        self.case_suite_signature = case_suite_signature(self.case_map.values())
        self.reference_cache_dir = Path(reference_cache_dir or ".stage3_reference_cache")
        self.reference_mode = str(reference_mode).lower()
        if self.reference_mode != "best_baseline":
            raise ValueError("pure-perishable reference_mode must be best_baseline")
        catalog_signatures = [
            _catalog_signature(next(
                row for row in self.case_map.values()
                if row.name.startswith(f"M{shelf_life}__")
            ), self.config.get("reference_baselines"))
            for shelf_life in CONTEXTS
        ]
        catalog_identity = stable_key("|".join(catalog_signatures).encode())[:16]
        self.evaluation_protocol_signature = (
            f"{_REFERENCE_PROTOCOL}:{self.reference_mode}:{catalog_identity}:"
            f"cases-{self.case_suite_signature}:typed-cap-{bool(self.config.get('cyclic_cap_enabled', False))}"
        )

    @property
    def context_ids(self):
        return tuple(f"M{value}" for value in CONTEXTS)

    def family_domain(self):
        representatives = {
            f"M{shelf_life}": next(
                row for row in self.case_map.values() if row.name.startswith(f"M{shelf_life}__")
            )
            for shelf_life in CONTEXTS
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
                mode=mode,
                objective="policy_cost",
                random_seed=task.seed,
                total_evaluations=self.config.get(f"{mode}_proposal_budget"),
            ),
            proposal_paths=proposal,
            strict_paths=strict,
            seed=_seed("perishable-tuning", case.name, mode),
        )
        final_seed = _seed("perishable-final", case.name, mode)
        final = ComposedMonteCarloPolicyCostEvaluator.from_case(
            ast, case, horizon, motifs=motifs, paths=final_paths,
            seed=final_seed,
        ).evaluate_many((fit.params,))[0]
        reference, hit, reference_type = self._reference(case, mode)
        cost = float(final.cost)
        return CaseMetric(
            task.context_id,
            task.case_id,
            100.0 * (cost / max(reference, 1e-12) - 1.0),
            time.perf_counter() - started,
            metadata={
                "cost": cost,
                "gap_reference_cost": reference,
                "reference_type": reference_type,
                "reference_cache_hit": hit,
                "constants": dict(fit.params),
                "schema_names": list(case.schema.names),
                "proposal_paths": proposal,
                "strict_paths": strict,
                "evaluation_paths": final_paths,
                "evaluation_seed": final_seed,
            },
        )

    def prepare_reference(self, context_id: str, case_id: str, fidelity: str = "standard") -> None:
        """Populate the reference cache without fitting a candidate family."""
        case = self.case_map[f"{context_id}__{case_id}"]
        self._reference(case, fidelity)

    def _reference(self, case, mode):
        self.reference_cache_dir.mkdir(parents=True, exist_ok=True)
        catalog_signature = _catalog_signature(case, self.config.get("reference_baselines"))
        key = (
            f"{_REFERENCE_PROTOCOL}|{catalog_signature}|{self.reference_mode}|"
            f"{case.case_signature}|{mode}|{json.dumps(self.config, sort_keys=True)}"
        )
        path = self.reference_cache_dir / (stable_key(key.encode()) + ".json")
        with path.with_suffix(".lock").open("w") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if path.exists():
                payload = json.loads(path.read_text())
                return float(payload["cost"]), True, str(payload.get("reference_type", "best_baseline_empirical_mc"))
            return self._best_baseline_reference(
                case, mode, path, catalog_signature=catalog_signature
            )

    def _best_baseline_reference(self, case, mode, path, *, catalog_signature):
        """Fit the registered perishable baselines and freeze the best cost."""
        horizon = int(self.config.get("horizon", 1000))
        rows = []
        tuning_seed = _seed("perishable-tuning", case.name, mode)
        final_seed = _seed("perishable-final", case.name, mode)
        formula_specs, computational_specs = _catalogs(case, self.config.get("reference_baselines"))
        def fit_formula(spec):
            fit = fit_inventory_policy_mc(
                spec.ast, case, horizon, spec.parameter_specs, motifs=spec.motifs,
                config=OptimizationConfig(
                    mode=mode, objective="policy_cost",
                    random_seed=_seed("perishable-baseline-fit", spec.name, case.name, mode),
                    total_evaluations=self.config.get(f"{mode}_proposal_budget"),
                ),
                proposal_paths=int(self.config.get(f"{mode}_proposal_paths", self.config["proposal_paths"])),
                strict_paths=int(self.config.get(f"{mode}_strict_paths", self.config["strict_paths"])),
                seed=tuning_seed,
            )
            value = float(ComposedMonteCarloPolicyCostEvaluator.from_case(
                spec.ast, case, horizon, motifs=spec.motifs,
                paths=int(self.config.get(f"{mode}_evaluation_paths", self.config["evaluation_paths"])),
                seed=final_seed,
            ).evaluate_many((fit.params,))[0].cost)
            return value, spec.name, fit.params
        rows.extend(fit_formula(spec) for spec in formula_specs)
        def eval_computational(spec):
            provider = spec.make_provider(case)
            value = float(ComputationalMonteCarloPolicyCostEvaluator(
                case,
                horizon,
                provider,
                require_batch=True,
                paths=int(self.config.get(f"{mode}_evaluation_paths", self.config["evaluation_paths"])),
                seed=final_seed,
            ).evaluate({}).cost)
            return value, spec.name, {}
        rows.extend(eval_computational(spec) for spec in computational_specs)
        value, name, params = min(rows, key=lambda row: (row[0], row[1]))
        payload = {
            "cost": value,
            "policy": name,
            "params": params,
            "reference_type": "best_baseline_empirical_mc",
            "reference_protocol": _REFERENCE_PROTOCOL,
            "catalog_signature": catalog_signature,
            "tuning_seed": tuning_seed,
            "evaluation_seed": final_seed,
            "candidates": [
                {"cost": cost, "policy": policy, "params": candidate_params}
                for cost, policy, candidate_params in rows
            ],
        }
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
        temporary.replace(path)
        return value, False, "best_baseline_empirical_mc"


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
