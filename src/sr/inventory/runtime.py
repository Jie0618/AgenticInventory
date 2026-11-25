"""Shared configuration and input loading for multi-case SR CLIs."""

from __future__ import annotations

from runtime.artifacts import seed_from_text, stable_key
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from inventory.core import InventoryCase
from evaluation.cache import load_or_create_inventory_dp
from optimization import InventorySearchContext
from sr.core.models import SearchConfig
from sr.inventory.multi_evaluation import CaseEvaluationInput
from sr.inventory.frontier import import_frontier_seeds
from sr.analysis.frontier import FrontierSeedBundle
from sr.analysis.frontier import initialization_signature
from sr.multi.models import MultiObjective
from sr.inventory.single import diagnostic_samples


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HORIZON = 1000


@dataclass(frozen=True)
class MultiRuntime:
    case_names: tuple[str, ...]
    objective: MultiObjective
    search_config: SearchConfig
    max_pair_retries: int
    bundle: FrontierSeedBundle
    case_inputs: Mapping[str, CaseEvaluationInput]
    diagnostic_paths: Mapping[str, Path]
    initialization_signature: str


@dataclass(frozen=True)
class MultiCliDefaults:
    experiment_config: Path
    search_config: Path
    checkpoint_root: Path
    diagnostic_root: Path
    cache_root: Path
    reference_suite: str = "gap200"
    cyclic_evaluation_config: Path | None = None


def load_runtime(
    *,
    available_cases: Sequence[InventoryCase],
    experiment_config: Path,
    search_config: Path,
    checkpoint_root: Path,
    diagnostic_root: Path | None,
    dp_cache_dir: Path | None,
    reference_suite: str = "gap200",
    horizon: int = HORIZON,
    workers: int | None = None,
) -> MultiRuntime:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not str(reference_suite).strip():
        raise ValueError("reference_suite must be nonempty")
    experiment = json.loads(Path(experiment_config).read_text(encoding="utf-8"))
    if "training_environments" in experiment:
        source = Path(experiment_config).parent / experiment["training_environments"]
        environments = json.loads(source.read_text(encoding="utf-8"))["environments"]
        values = [row["case_name"] for row in environments]
    else:
        values = experiment["training_cases"]
    case_names = tuple(str(value) for value in values)
    if len(case_names) != len(set(case_names)) or len(case_names) < 2:
        raise ValueError("training_cases must contain at least two unique cases")
    if experiment.get("constant_fitting") != "independent_per_case":
        raise ValueError("only independent_per_case constant fitting is supported")
    if experiment.get("optimizer_mode") != "standard":
        raise ValueError("multi-distribution search requires Standard optimization")
    objective = MultiObjective(**dict(experiment["score"]))
    config_values = asdict(SearchConfig())
    incoming = json.loads(Path(search_config).read_text(encoding="utf-8"))
    unknown = sorted(set(incoming) - set(config_values))
    if unknown:
        raise ValueError(f"unknown SearchConfig fields: {unknown}")
    config_values.update(incoming)
    if workers is not None:
        config_values["optimizer_workers"] = int(workers)
    config = SearchConfig(**config_values)

    registry = {row.name: row for row in available_cases}
    missing = [name for name in case_names if name not in registry]
    if missing:
        raise ValueError(f"unknown training cases: {missing}")
    selected = tuple(registry[name] for name in case_names)
    composition_hashes = {row.config.composition_hash for row in selected}
    if len(composition_hashes) != 1:
        raise ValueError("all training cases must share one inventory composition")
    checkpoint_paths = {
        name: Path(checkpoint_root) / name / "checkpoint.json" for name in case_names
    }
    for path in checkpoint_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    bundle = import_frontier_seeds(
        checkpoint_paths,
        case=selected[0],
        config=config,
    )
    prepared = Path(checkpoint_root) / ".inputs"
    if reference_suite == "stage1-reference":
        diagnostic_root = prepared if diagnostic_root is None else Path(diagnostic_root)
        dp_cache_dir = Path(checkpoint_root) / ".dp_cache" if dp_cache_dir is None else Path(dp_cache_dir)
    if diagnostic_root is None or dp_cache_dir is None:
        raise ValueError("diagnostic_root and dp_cache_dir are required for this reference suite")
    diagnostic_paths = {
        case.name: Path(diagnostic_root) / f"{case.name}.json" for case in selected
    }
    missing_diagnostics = [
        str(diagnostic_paths[case.name]) for case in selected
        if not diagnostic_paths[case.name].is_file()
    ]
    if missing_diagnostics:
        joined = "\n  - ".join(missing_diagnostics)
        raise FileNotFoundError(f"missing multi-distribution DP diagnostics:\n  - {joined}")
    case_inputs: dict[str, CaseEvaluationInput] = {}
    runtime_cases: list[dict[str, object]] = []
    for case in selected:
        diagnostic = diagnostic_paths[case.name]
        if reference_suite == "stage1-reference":
            identity = json.loads(diagnostic.read_text()).get("identity", {})
            if identity.get("case_signature") != case.case_signature or identity.get("horizon") != horizon:
                raise ValueError(f"Stage1 preparation identity differs: {diagnostic}")
        dp_result = load_or_create_inventory_dp(
            case,
            horizon,
            suite=str(reference_suite),
            cache_dir=Path(dp_cache_dir),
        )
        seed = _seed(config.random_seed, case.name)
        samples = diagnostic_samples(diagnostic, horizon)
        context = InventorySearchContext.from_reference(
            case, horizon, dp_result, samples=samples
        )
        case_inputs[case.name] = CaseEvaluationInput(case.name, context, samples, seed)
        runtime_cases.append(_case_signature_payload(case, diagnostic))
    base_signature = initialization_signature(
        bundle,
        case_names=case_names,
        horizon=horizon,
        objective_payload=objective.to_payload(),
        config_payload={"random_seed": config.random_seed, "optimizer_mode": "standard"},
    )
    encoded = json.dumps(
        {"base": base_signature, "cases": runtime_cases},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    signature = stable_key(encoded)
    return MultiRuntime(
        case_names=case_names,
        objective=objective,
        search_config=config,
        max_pair_retries=int(experiment.get("max_pair_retries", 5)),
        bundle=bundle,
        case_inputs=case_inputs,
        diagnostic_paths=diagnostic_paths,
        initialization_signature=signature,
    )


def _case_signature_payload(
    case: object, diagnostic: Path
) -> dict[str, object]:
    """Bind the full canonical case identity to its diagnostic artifact."""

    identity_builder = getattr(case, "identity_payload", None)
    if callable(identity_builder):
        identity = identity_builder()
    else:
        # Keep compatibility with the small case-like objects used by older
        # runtime tests and migration tools, while still signing every known
        # component/cost field rather than only lead-time fields.
        demand = getattr(case, "demand")
        identity = {
            "case_name": str(getattr(case, "name")),
            "model_id": str(getattr(case, "model_id", "legacy")),
            "model_version": str(getattr(case, "model_version", "legacy")),
            "n": int(getattr(case, "n")),
            "lead_time": int(getattr(case, "lead_time", 0)),
            "shelf_life": getattr(case, "shelf_life", None),
            "costs": {
                "holding": float(getattr(case, "h", 0.0)),
                "shortage": float(getattr(case, "b", 0.0)),
                "expiry": float(getattr(case, "w", 0.0)),
                "backlog": float(getattr(case, "g", 0.0)),
                "abandonment": float(getattr(case, "a", 0.0)),
                "ordering": float(getattr(case, "k", 0.0)),
            },
            "demand": {
                "values": [int(value) for value in demand.values],
                "probs": [float(value) for value in demand.probs],
            },
        }
    return {
        "case": identity,
        "case_signature": str(getattr(case, "case_signature", "legacy")),
        "diagnostic": json.loads(diagnostic.read_text()),
    }


def _seed(base: int, *tokens: object) -> int:
    encoded = ":".join((str(base), *(str(value) for value in tokens))).encode("utf-8")
    return seed_from_text(encoded)
