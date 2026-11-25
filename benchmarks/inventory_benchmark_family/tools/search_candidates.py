"""Select evaluated Stage3 structures for an independent benchmark campaign."""

from copy import deepcopy
from hashlib import sha256
from importlib import import_module
import json
import math
from pathlib import Path

from benchmarks.inventory_benchmark_family.mc_protocol import (
    BASELINE_TYPES, load_campaign_config,
)
from benchmarks.inventory_benchmark_family.policy_assets import materialize_policy_set
from sr.family.artifacts import artifact_files


ROOT = Path(__file__).resolve().parents[3]
SETTINGS = ("lead_time", "perishable", "lead_time_perishable", "lead_time_backlog")


def completed_candidates(checkpoint):
    """Read completed evaluations and their checkpoint-embedded definitions."""
    checkpoint = Path(checkpoint)
    payload = json.loads(checkpoint.read_text())
    if not isinstance(payload.get("archive"), list):
        raise ValueError("--checkpoint requires a Stage3 CHECKPOINT.json with an archive")
    candidates = {}
    for row in payload["archive"]:
        if row.get("status") != "complete":
            continue
        score = float(row["cross_context_score"])
        if not math.isfinite(score):
            continue
        key = row.get("semantic_hash") or row["family_hash"]
        candidates[key] = {**row, "semantic_hash": key, "family": row.get("family")}
    return payload, sorted(candidates.values(),
                           key=lambda row: (float(row["cross_context_score"]), row["semantic_hash"]))


def _recover_definitions(checkpoint, iteration, selected):
    """Read committed proposals only for selected candidates missing definitions."""
    missing = {row["semantic_hash"]: row for row in selected if not row.get("family")}
    if not missing:
        return
    committed = []
    for path in artifact_files(checkpoint.parent, "PROPOSALS_ITER_*.json"):
        number = path.stem.removeprefix("PROPOSALS_ITER_")
        if number.isdecimal() and 0 <= int(number) <= iteration:
            committed.append((int(number), path))
    # Prefer the latest committed copy; stop once all selected definitions exist.
    for number, path in sorted(committed, reverse=True):
        try:
            proposal = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Cannot recover definitions from invalid committed proposal: {path}") from exc
        if int(proposal["iteration"]) != number:
            raise ValueError(f"Proposal iteration differs from its filename: {path}")
        for family in proposal.get("families", ()):
            key = family.get("semantic_hash") or family.get("family_hash")
            if key in missing:
                missing.pop(key)["family"] = family
        if not missing:
            break


def export_search_candidates(checkpoint, setting, *, top_k=1, family_ids=None, config=None):
    """Freeze training-selected families together with the setting's baseline roster."""
    if setting not in SETTINGS:
        raise ValueError("a supported --setting is required for checkpoint export")
    if type(top_k) is not int or top_k < 1:
        raise ValueError("--top-k must be positive")
    checkpoint = Path(checkpoint)
    payload, candidates = completed_candidates(checkpoint)
    if family_ids is not None:
        if not family_ids or len(set(family_ids)) != len(family_ids):
            raise ValueError("--family-ids must contain unique family IDs or full hashes")
        selected = []
        for identifier in family_ids:
            matches = [row for row in candidates
                       if identifier in (row["family_id"], row["semantic_hash"])]
            if len(matches) != 1:
                raise ValueError(f"Expected one completed candidate for {identifier!r}; use its full hash if ambiguous")
            selected.append(matches[0])
        if len({row["semantic_hash"] for row in selected}) != len(selected):
            raise ValueError("--family-ids selects the same candidate more than once")
    else:
        if len(candidates) < top_k:
            raise ValueError(f"Only {len(candidates)} completed candidates available; requested {top_k}")
        selected = candidates[:top_k]
    _recover_definitions(checkpoint, int(payload["iteration"]), selected)
    campaign = ROOT / "experiments" / setting / "benchmark_campaigns/inventory_benchmark_family"
    baselines = [row for row in materialize_policy_set(campaign / "policies/sets/mc_final.json")["policies"]
                 if row["type"] in BASELINE_TYPES]
    checkpoint_hash = sha256(checkpoint.read_bytes()).hexdigest()
    policies = deepcopy(baselines)
    for row in selected:
        family = row["family"]
        if not isinstance(family, dict) or not isinstance(family.get("ast"), dict):
            raise ValueError(f"Missing definition for {row['family_id']}; keep the original proposals/ beside older checkpoints")
        if family.get("family_id") != row["family_id"]:
            raise ValueError("Checkpoint family ID differs from its saved definition")
        identifier = "search_" + row["semantic_hash"]
        selection = {
            "family_id": row["family_id"], "family_hash": row["semantic_hash"],
            "training_score": float(row["cross_context_score"]),
            "checkpoint_sha256": checkpoint_hash,
        }
        policies.append({
            "id": identifier, "type": "formula_family",
            "family": deepcopy(family), "display_name": row["family_id"],
            "selection": selection,
        })
    frozen = {
        "policy_set": f"{setting}_search_{checkpoint_hash[:12]}",
        "setting": setting, "selection_kind": "search_candidates",
        "selection_iteration": int(payload["iteration"]),
        "selection_rule": "explicit_completed_candidates" if family_ids is not None else "lowest_training_cross_context_score",
        "policies": policies,
    }
    validate_export(frozen, campaign, setting, config=config)
    return frozen


def validate_export(payload, campaign, setting, *, config=None):
    """Check every evaluation context before creating an immutable output set."""
    from benchmarks.inventory_benchmark_family.mc_protocol import MCPolicyDefinition
    from optimization.models.types import ParameterSpec

    settings = load_campaign_config(config or campaign / "configs/standard.json")
    model = import_module(
        f"experiments.{setting}.benchmark_campaigns.inventory_benchmark_family.mc_model")
    if setting in ("lead_time", "perishable"):
        cases = model.MODEL.cases(settings)
        context_field = model.MODEL.context_field
        representatives = {getattr(case, context_field): case for case in cases}
        for row in payload["policies"]:
            if row["type"] != "formula_family":
                continue
            definition = MCPolicyDefinition(
                row["id"], row["type"], family=row["family"],
                parameter_specs=None if "parameter_specs" not in row else tuple(
                    ParameterSpec(**spec) for spec in row["parameter_specs"]))
            for context, case in representatives.items():
                model.resolve_policy(definition, context, case.n, case=case)
    else:
        # Use the exact same in-memory normalization as the public loader.
        frozen = {key: value for key, value in payload.items() if key != "policy_set"}
        frozen["policies"] = [
            {key: value for key, value in row.items() if key != "type"}
            if row["type"] == "formula_family" else row
            for row in payload["policies"]
            if row["type"] == "formula_family" or row.get("external")]
        ids = tuple(row["id"] for row in frozen["policies"] if row.get("family"))
        model.validate_campaign(settings, frozen, ids, baseline_only=False)
        axis = "shelf_life" if setting == "lead_time_perishable" else "backlog_patience"
        representatives = {}
        for case in model.cases(settings):
            second = case.shelf_life if axis == "shelf_life" else case.config.backlog.patience
            representatives.setdefault((case.lead_time, second), case)
        for case in representatives.values():
            model.resolve(case, frozen)
