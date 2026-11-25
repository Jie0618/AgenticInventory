"""Best-baseline references for MC-only composed Stage-III experiments."""

from __future__ import annotations

from runtime.artifacts import stable_key
import json
from typing import Callable, Mapping, Sequence

from optimization import OptimizationConfig, fit_inventory_policy_mc
from optimization.evaluators.composed import ComposedMonteCarloPolicyCostEvaluator


def select_reference_baselines(catalog, names):
    """Resolve an explicit, nonempty reference roster without silent omissions."""
    if (not isinstance(names, (list, tuple)) or not names
            or any(not isinstance(name, str) or not name for name in names)):
        raise ValueError("reference_baselines must be a nonempty list of policy IDs")
    if len(set(names)) != len(names):
        raise ValueError("Duplicate reference_baselines policy IDs")
    unknown = set(names) - catalog.keys()
    if unknown:
        raise ValueError(f"Unknown reference_baselines policy IDs: {sorted(unknown)}")
    return tuple(catalog[name] for name in names)


def catalog_signature(
    protocol: str,
    formula_specs: Sequence[object],
) -> str:
    """Return a stable identity for every policy eligible as a reference."""

    payload = {
        "protocol": str(protocol),
        "formula": [
            {
                "name": getattr(spec, "name", None) or spec.id,
                "ast": spec.ast,
                "motifs": spec.motifs,
                "parameters": [vars(parameter) for parameter in spec.parameter_specs],
            }
            for spec in formula_specs
        ],
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    )
    return stable_key(encoded.encode())[:16]


def fit_best_baseline_reference(
    *,
    case,
    mode: str,
    config: Mapping[str, object],
    formula_specs: Sequence[object],
    seed: Callable[..., int],
    seed_namespace: str,
) -> tuple[float, str, Mapping[str, float], list[dict[str, object]], int, int]:
    """Tune every registered baseline and evaluate it on one shared final bank."""

    horizon = int(config.get("horizon", 1000))
    proposal_paths = int(config.get(f"{mode}_proposal_paths", config["proposal_paths"]))
    strict_paths = int(config.get(f"{mode}_strict_paths", config["strict_paths"]))
    evaluation_paths = int(
        config.get(f"{mode}_evaluation_paths", config["evaluation_paths"])
    )
    proposal_budget = config.get(f"{mode}_proposal_budget")
    tuning_seed = seed(f"{seed_namespace}-tuning", case.name, mode)
    final_seed = seed(f"{seed_namespace}-final", case.name, mode)
    rows: list[tuple[float, str, Mapping[str, float]]] = []

    for spec in formula_specs:
        fit = fit_inventory_policy_mc(
            spec.ast,
            case,
            horizon,
            spec.parameter_specs,
            motifs=spec.motifs,
            config=OptimizationConfig(
                mode=mode,
                objective="policy_cost",
                random_seed=seed(
                    f"{seed_namespace}-baseline-fit", spec.name, case.name, mode
                ),
                total_evaluations=proposal_budget,
            ),
            proposal_paths=proposal_paths,
            strict_paths=strict_paths,
            seed=tuning_seed,
        )
        value = float(
            ComposedMonteCarloPolicyCostEvaluator.from_case(
                spec.ast,
                case,
                horizon,
                motifs=spec.motifs,
                paths=evaluation_paths,
                seed=final_seed,
            ).evaluate_many((fit.params,))[0].cost
        )
        rows.append((value, spec.name, dict(fit.params)))

    value, name, params = min(rows, key=lambda row: (row[0], row[1]))
    candidates = [
        {"cost": cost, "policy": policy, "params": candidate_params}
        for cost, policy, candidate_params in rows
    ]
    return value, name, params, candidates, tuning_seed, final_seed


__all__ = ["catalog_signature", "fit_best_baseline_reference", "select_reference_baselines"]
