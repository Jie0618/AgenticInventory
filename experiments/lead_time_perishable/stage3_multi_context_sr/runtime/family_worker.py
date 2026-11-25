"""Stage-III MC worker for lead-time + perishable policy families."""

from __future__ import annotations

from runtime.artifacts import seed_from_text, stable_key
import fcntl
import json
import time
from pathlib import Path

from inventory.policy.family import materialize_family_for_case
from inventory.policy.family_domain import InventoryFamilyDomain
from optimization import OptimizationConfig, ParameterSpec, fit_inventory_policy_mc
from optimization.evaluators.composed import ComposedMonteCarloPolicyCostEvaluator
from optimization.policies.baselines.lead_time_perishable import baselines_for_lead_time_perishable
from sr.family.composed_reference import catalog_signature, fit_best_baseline_reference, select_reference_baselines
from sr.family.identity import case_suite_signature
from sr.family.models import CaseMetric

from ..cases import CASE_SUFFIXES, CONTEXTS, cases


_REFERENCE_PROTOCOL = "lead-time-perishable-best-formula-baseline-shared-bank"


def _seed(*tokens: object) -> int:
    return seed_from_text("|".join(map(str, tokens)).encode())


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


def _catalogs(case, names):
    formula = tuple(baselines_for_lead_time_perishable(
        int(case.lead_time), int(case.shelf_life), int(case.n), include_extended=True,
    ))
    return select_reference_baselines({row.name: row for row in formula}, names)


class FamilyWorker:
    subprocess_module = "experiments.lead_time_perishable.stage3_multi_context_sr.runtime.family_worker"

    def __init__(self, *, config_path: Path, reference_cache_dir: Path | None = None,
                 reference_mode: str = "best_baseline") -> None:
        self.config_path = str(config_path)
        self.config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        self.case_map = {row.name: row for row in cases(int(self.config.get("state_action_cap", 20)))}
        self.case_suite_signature = case_suite_signature(self.case_map.values())
        self.reference_cache_dir = Path(reference_cache_dir or ".stage3_reference_cache")
        self.reference_mode = str(reference_mode).lower()
        if self.reference_mode != "best_baseline":
            raise ValueError("lead-time + perishable reference_mode must be best_baseline")
        signatures = []
        for lead_time, shelf_life in CONTEXTS:
            case = next(row for row in self.case_map.values() if row.name.startswith(f"L{lead_time}_M{shelf_life}__"))
            formula = _catalogs(case, self.config.get("reference_baselines"))
            signatures.append(catalog_signature(_REFERENCE_PROTOCOL, formula))
        identity = stable_key("|".join(signatures).encode())[:16]
        self.evaluation_protocol_signature = (
            f"{_REFERENCE_PROTOCOL}:{self.reference_mode}:{identity}:"
            f"cases-{self.case_suite_signature}:typed-cap-{bool(self.config.get('cyclic_cap_enabled', False))}"
        )

    @property
    def context_ids(self):
        return tuple(f"L{lead}_M{shelf}" for lead, shelf in CONTEXTS)

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
            seed=_seed("ltp-tuning", case.name, mode),
        )
        final_seed = _seed("ltp-final", case.name, mode)
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

    def _reference(self, case, mode):
        self.reference_cache_dir.mkdir(parents=True, exist_ok=True)
        formula = _catalogs(case, self.config.get("reference_baselines"))
        signature = catalog_signature(_REFERENCE_PROTOCOL, formula)
        key = (
            f"{_REFERENCE_PROTOCOL}|{signature}|{self.reference_mode}|"
            f"{case.case_signature}|{mode}|{json.dumps(self.config, sort_keys=True)}"
        )
        path = self.reference_cache_dir / (stable_key(key.encode()) + ".json")
        with path.with_suffix(".lock").open("w") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                return float(payload["cost"]), True, str(payload["reference_type"])
            value, name, params, candidates, tuning_seed, final_seed = fit_best_baseline_reference(
                case=case, mode=mode, config=self.config,
                formula_specs=formula,
                seed=_seed, seed_namespace="ltp",
            )
            payload = {
                "cost": value, "policy": name, "params": params,
                "reference_type": "best_formula_baseline_empirical_mc",
                "reference_protocol": _REFERENCE_PROTOCOL,
                "catalog_signature": signature, "tuning_seed": tuning_seed,
                "evaluation_seed": final_seed, "candidates": candidates,
            }
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
            temporary.replace(path)
            return value, False, "best_formula_baseline_empirical_mc"


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
