"""Small model boundary for the shared MC campaign, worker, and report."""

from __future__ import annotations

from copy import deepcopy
from runtime.artifacts import seed_from_text
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Callable
import json

from benchmarks.inventory_benchmark_family.suites import StructuralContext, cases
from optimization.policies.policy_sets_json import PolicyDefinition
from optimization.models.types import ParameterSpec


BASELINE_TYPES = frozenset({"registered_baseline", "computational_baseline"})
MODEL_SETTINGS = ("lead_time", "perishable")


@dataclass(frozen=True)
class MCModel:
    setting: str
    context_field: str
    context_prefix: str
    seed_namespace: str
    default_config: str
    default_policies: str
    resolve_policy: Callable
    make_numerics: Callable
    fixed_computational: bool = False
    extra_case_fields: tuple = ()
    archived_suites: tuple = ()
    randomized_policies: bool = False
    result_is_current: Callable | None = None

    @property
    def suite(self):
        return f"inventory_benchmark_family__{self.setting}_mc_best_baseline"

    @property
    def contexts_key(self):
        return "lead_times" if self.context_field == "lead_time" else "shelf_lives"

    @property
    def data_dir(self):
        # The adapter owns its campaign; the shared runner knows no workspace.
        return Path(import_module(self.resolve_policy.__module__).__file__).resolve().parent

    def seed(self, *tokens):
        return seed_from_text(self.seed_namespace + "|" + "|".join(tokens))

    def cases(self, config, contexts=None):
        caps = {int(key): int(value) for key, value in config["state_caps"].items()}
        requested = tuple(sorted(caps if contexts is None else contexts))
        missing = set(requested) - set(caps)
        if missing:
            raise ValueError(f"requested contexts absent from state_caps: {sorted(missing)}")
        return sorted((row for value in requested for row in cases(
            self.setting, n=caps[value],
            contexts=(StructuralContext(f"{self.context_prefix}{value}", **{self.context_field: value}),),
        )), key=lambda row: (getattr(row, self.context_field), row.name))


def load_model(setting, *, model_module):
    if setting not in MODEL_SETTINGS:
        raise ValueError(f"unsupported MC setting: {setting}")
    model = import_module(model_module).MODEL
    if model.setting != setting:
        raise ValueError(f"model module does not implement setting {setting!r}")
    return model


def report_model(suite, setting=None, *, model_module):
    model = import_module(model_module).MODEL
    if (setting is None or model.setting == setting) and suite in (model.suite, *model.archived_suites):
        return model
    raise ValueError(f"no MC report model for suite {suite!r} and setting {setting!r}")


@dataclass(frozen=True)
class MCPolicyDefinition:
    id: str
    type: str
    registered_name: str | None = None
    ast: dict | None = None
    motifs: dict | None = None
    parameter_specs: tuple | None = None
    family: dict | None = None


def _policy_payload(path):
    path = Path(path)
    payload = json.loads(path.read_text())
    if payload.get("selection_kind") == "baselines_only" and "baseline_source" in payload:
        source = json.loads((path.parent / payload.pop("baseline_source")).read_text())
        payload["policies"] = [p for p in source["policies"] if p["type"] in BASELINE_TYPES]
    return payload


def load_policy_set(path):
    """Read the common frozen policy format; model catalogs resolve names."""
    path = Path(path)
    payload = _policy_payload(path)
    name = str(payload.get("policy_set", "")).strip()
    policies = []
    if not name or not payload.get("policies"):
        raise ValueError("policy_set and a nonempty policies list are required")
    for row in payload["policies"]:
        identifier = str(row.get("id", "")).strip()
        kind = row["type"]
        if not identifier or kind not in (*BASELINE_TYPES, "formula_ast", "formula_family"):
            raise ValueError(f"invalid policy definition: {row}")
        if kind == "formula_family" and not isinstance(row.get("family"), dict):
            raise ValueError("formula_family requires a frozen family")
        ast, motifs = None, None
        if kind == "formula_ast":
            if (row.get("ast") is None) == (row.get("ast_file") is None):
                raise ValueError("provide exactly one of ast or ast_file")
            ast = row["ast"] if row.get("ast") is not None else json.loads((path.parent / row["ast_file"]).read_text())
            if not isinstance(ast, dict):
                raise ValueError("AST must be a JSON object")
            if "ast" in ast and "op" not in ast:
                ast = ast["ast"]
            if "motifs" in row and "motifs_file" in row:
                raise ValueError("provide at most one of motifs or motifs_file")
            motifs = row.get("motifs", {}) if "motifs_file" not in row else json.loads((path.parent / row["motifs_file"]).read_text())
            if not isinstance(motifs, dict) or any(not isinstance(value, dict) for value in motifs.values()):
                raise ValueError("motifs must map names to AST objects")
        specs = None if "parameter_specs" not in row else tuple(ParameterSpec(**item) for item in row["parameter_specs"])
        policies.append(MCPolicyDefinition(identifier, kind, row.get("name", identifier), ast, motifs, specs, row.get("family")))
    if len({row.id for row in policies}) != len(policies):
        raise ValueError("policy ids must be unique")
    return name, tuple(policies)


def load_campaign_selection(path, baseline_ids):
    """Validate explicit membership and adapt frozen families to existing numerics.

    The returned selection retains the scientific identity. Public
    names/types describe the file format and do not alter the fitted policy.
    """
    payload = _policy_payload(path)
    if "policy_set" not in payload:  # Frozen selections without a policy_set envelope.
        return payload
    _, definitions = load_policy_set(path)
    if any(p.type == "formula_ast" for p in definitions):
        raise ValueError("Joint campaigns require formula_family for symbolic policies")
    baselines = tuple(p.id for p in definitions if p.type in BASELINE_TYPES)
    if set(baselines) != set(baseline_ids) or len(baselines) != len(baseline_ids):
        raise ValueError("Explicit baseline membership differs from the setting catalog")
    canonical = {k: v for k, v in payload.items() if k != "policy_set"}
    canonical["policies"] = []
    for row in payload["policies"]:
        if row["type"] == "formula_family":
            canonical["policies"].append({k: v for k, v in row.items() if k != "type"})
        elif row.get("external"):
            canonical["policies"].append(row)
    # Keep frozen families before external baseline definitions.
    canonical["policies"].sort(key=lambda row: row.get("type") == "registered_baseline")
    return canonical


def validate_joint_selection(settings, frozen, sr_ids, *, baseline_ids,
                             reviewed_rosters, baseline_only):
    """Allow exported search candidates while keeping reviewed campaigns strict."""
    geometry = settings.get("contexts", ())
    if (settings.get("schema_version") != 1 or not geometry
            or any(not isinstance(row, (list, tuple)) or len(row) != 2
                   or any(type(value) is not int or value < 1 for value in row)
                   for row in geometry)
            or len({tuple(row) for row in geometry}) != len(geometry)):
        raise ValueError("contexts must contain unique pairs of positive integers")
    if (any(not isinstance(name, str) or not name.strip() for name in sr_ids)
            or len(set(sr_ids)) != len(sr_ids)
            or set(sr_ids).intersection(baseline_ids)):
        raise ValueError("SR policy IDs must be unique and distinct from baseline IDs")
    if frozen.get("selection_kind") == "search_candidates":
        if not sr_ids:
            raise ValueError("search_candidates requires at least one symbolic policy")
        return
    expected = {(lead, second) for lead in (3, 5, 10) for second in (3, 5, 10)}
    if {tuple(row) for row in geometry} != expected:
        raise ValueError("Production geometry must be {3,5,10} x {3,5,10}")
    if not baseline_only and (
            frozen.get("selection_iteration") != 100 or sr_ids not in reviewed_rosters):
        raise ValueError("Expected reviewed frozen 100-round policy selection")


def resolve_family_policy(definition, setting, context, n, *, case=None):
    """Materialize an exported family without changing its declared parameter bounds."""
    from inventory.policy.family import materialize_family_for_case
    from inventory.policy.family_domain import InventoryFamilyDomain
    from inventory.policy.ast import ast_check, ast_to_formula
    from optimization.policies.parameters import apply_cap_modes
    from sr.family.proposals import family_from_json
    from benchmarks.inventory_benchmark_family.mc_runtime import MCPolicy

    if case is None:
        field = "lead_time" if setting == "lead_time" else "shelf_life"
        case = cases(setting, n=n, contexts=(
            StructuralContext(str(context), **{field: context}),))[0]
    domain = InventoryFamilyDomain.from_cases(
        {"evaluation": case}, training_case_count=1, cyclic_cap=True)
    family = family_from_json(definition.family, domain=domain)
    ast, motifs = materialize_family_for_case(family, case)
    checked = ast_check(ast, motifs=motifs, allowed_state_names=case.schema.names)
    specs = []
    for parameter in family.parameters:
        integer = (parameter.kind in {"target", "cap", "lag"}
                   and float(parameter.lower).is_integer()
                   and float(parameter.upper).is_integer())
        default = (parameter.lower + parameter.upper) / 2
        specs.append(ParameterSpec(
            parameter.name, kind="integer" if integer else "continuous",
            lower=parameter.lower, upper=parameter.upper,
            default=round(default) if integer else default))
    if definition.parameter_specs is not None:
        specs = list(definition.parameter_specs)
    if {spec.name for spec in specs} != set(checked.stats.parameters):
        raise ValueError("frozen parameter domains do not match the AST")
    return MCPolicy(definition.id, definition.type, checked.ast,
                    apply_cap_modes(tuple(specs), checked.ast, motifs),
                    motifs, ast_to_formula(checked.ast))


def compatible_resume_protocol(source, destination, allow_search_budget_change=False):
    old, new = deepcopy(source), deepcopy(destination)
    for protocol in (old, new):
        if allow_search_budget_change:
            for key in ('fidelity', 'proposal_budget', 'proposal_paths', 'strict_paths', 'computational_search_overrides'):
                protocol['settings'].pop(key, None)
    return old == new


def load_campaign_config(path):
    """Accept the common tuning_paths name and strict_paths alias."""
    config = json.loads(Path(path).read_text())
    if "tuning_paths" in config:
        paths = config.pop("tuning_paths")
        if "strict_paths" in config and config["strict_paths"] != paths:
            raise ValueError("Conflicting tuning_paths and strict_paths")
        config["strict_paths"] = paths
    for override in config.get("computational_search_overrides", {}).values():
        if "tuning_paths" in override:
            paths = override.pop("tuning_paths")
            if "strict_paths" in override and override["strict_paths"] != paths:
                raise ValueError("Conflicting override tuning_paths and strict_paths")
            override["strict_paths"] = paths
    return config


def formula_definition(definition):
    return PolicyDefinition(definition.id, definition.type, registered_name=definition.registered_name, ast=definition.ast, motifs=definition.motifs)


def resolve_formula_policy(definition, setting, context, n):
    """Resolve a frozen family with dynamic features and per-context domains."""
    from inventory.core import InventoryModelConfig, FixedShelfLife, LostSales
    from inventory.policy.ast import ast_check, ast_to_formula
    from inventory.policy.features import FeatureSpec, materialize_feature
    from inventory.policy.motifs import motifs_for_config
    from inventory.policy.motifs.lead_time import lead_time_motifs
    from optimization.policies.parameters import parameter_specs, apply_cap_modes
    from benchmarks.inventory_benchmark_family.mc_runtime import MCPolicy

    family = definition.family or {}
    motifs = dict(definition.motifs or {})
    for feature in family.get("features", ()):
        motifs[feature["name"]] = materialize_feature(feature["name"], context, FeatureSpec(**feature))
    if setting == "lead_time":
        motifs.update(lead_time_motifs(context))
        motifs["total_inventory"] = {"op": "state", "name": "x"}
        allowed = frozenset({"x", *(f"p{i}" for i in range(1, context+1))})
    else:
        motifs.update(motifs_for_config(InventoryModelConfig(
            perishability=FixedShelfLife(context), backlog=LostSales(), capacity=n)))
        allowed = frozenset(f"a{i}" for i in range(1, context+1))
    def expand(node):
        if isinstance(node, list):
            return [expand(value) for value in node]
        if not isinstance(node, dict):
            return node
        if node.get("op") == "feature":
            return {"op": "motif", "name": node["name"]}
        return {key: expand(value) for key, value in node.items()}
    motifs = {key: expand(value) for key, value in motifs.items()}
    checked = ast_check(expand(definition.ast), motifs=motifs, allowed_state_names=allowed)
    if definition.parameter_specs is not None:
        specs = definition.parameter_specs
    else:
        declarations = {p["name"]: p for p in family.get("parameters", ())}
        specs = []
        for name in sorted(checked.stats.parameters):
            declaration = declarations.get(name)
            if declaration and declaration["kind"] in {"scale", "weight"}:
                specs.append(ParameterSpec(name, kind="continuous", lower=declaration["lower"],
                                           upper=declaration["upper"], default=declaration["lower"]))
            else:
                specs.append(parameter_specs([name], context if setting == "lead_time" else context-1, n)[0])
    if {p.name for p in specs} != set(checked.stats.parameters):
        raise ValueError("frozen parameter domains do not match the AST")
    return MCPolicy(definition.id, definition.type, checked.ast,
                    apply_cap_modes(specs, checked.ast, motifs), motifs, ast_to_formula(checked.ast))
