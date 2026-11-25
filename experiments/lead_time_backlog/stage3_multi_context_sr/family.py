"""Transfer completed Stage-II frontiers into backlog Stage-III families."""

from __future__ import annotations

import json
from pathlib import Path

from inventory.policy.family import FamilyParameter, PolicyFamily
from inventory.policy.ast import ast_check
from inventory.policy.motifs import motifs_for_config
from inventory.policy.family_domain import InventoryFamilyDomain
from optimization.policies.parameters import parameter_specs_for_case


def _motif(name: str):
    return {"op": "motif", "name": name}


def _neg(node):
    return {"op": "neg", "arg": node}


def _add(*args):
    return {"op": "add", "args": list(args)}


def _linear_signature(node, motifs):
    """Exact state-linear signature; never approximate a nonlinear subtree."""
    op = node.get("op")
    if op == "state":
        return {node["name"]: 1.0}
    if op == "const":
        return {"constant": float(node["value"])}
    if op == "motif":
        return _linear_signature(motifs[node["name"]], motifs)
    if op in {"add", "sub", "neg"}:
        children = (node["args"] if op == "add" else
                    [node["left"], {"op": "neg", "arg": node["right"]}] if op == "sub" else [node["arg"]])
        result = {}
        for child in children:
            terms = _linear_signature(child, motifs)
            if terms is None:
                return None
            for name, value in terms.items():
                result[name] = result.get(name, 0.0) + (-value if op == "neg" else value)
        return {name: value for name, value in result.items() if value != 0}
    if op == "mul":
        factor = 1.0
        body = None
        for child in node["args"]:
            if child.get("op") == "const":
                factor *= float(child["value"])
            elif body is None:
                body = child
            else:
                return None
        terms = _linear_signature(body, motifs) if body is not None else {"constant": 1.0}
        return None if terms is None else {name: factor * value for name, value in terms.items() if factor * value != 0}
    return None


def _promote_stage2_ast(ast, source_case, stored_motifs, domain):
    """Lift source summaries, preserving their exact L3K3 algebra.

    Fixed full-source sums become full-context motifs, not truncated p1..p3
    sums. Unrecognised nonportable coordinates are rejected by domain validation.
    These two extra linear identities are the actual winning frontier's
    lag/backlog surplus expressions, not invented replacement policies.
    """
    base = dict(motifs_for_config(source_case.config))
    source = {**stored_motifs, **base}
    replacements = [_motif(name) for name in domain.common_motif_names]
    replacements += [
        _add(_motif("inventory_position"), _neg(_motif("pipeline_lag_moment")),
             _neg(_motif("total_backlog"))),
        _add(_motif("inventory_position"), _neg(_motif("retained_backlog_if_unserved"))),
    ]
    patterns = [(signature, replacement) for replacement in replacements
                if (signature := _linear_signature(replacement, source)) is not None]

    def walk(node, stack=()):
        if isinstance(node, list):
            return [walk(child, stack) for child in node]
        if not isinstance(node, dict):
            return node
        if node.get("op") == "motif":
            name = node["name"]
            if name in domain.common_motif_names:
                return _motif(name)
            if name in stack:
                raise ValueError(f"cyclic source motif: {name}")
            return walk(source[name], (*stack, name))
        signature = _linear_signature(node, source)
        if signature is not None:
            for expected, replacement in patterns:
                if signature == expected:
                    return replacement
        if node.get("op") == "add":
            # Reflected motifs can interleave nonlinear terms with a full
            # inventory sum. Regroup exact linear siblings before recursing;
            # otherwise source p2/p3 coordinates leak into the L=1 context.
            linear_children, nonlinear_children = [], []
            for child in node["args"]:
                (linear_children if _linear_signature(child, source) is not None
                 else nonlinear_children).append(child)
            if nonlinear_children and len(linear_children) > 1:
                combined_signature = _linear_signature(_add(*linear_children), source)
                replacement = next((value for expected, value in patterns
                                    if combined_signature == expected), None)
                if replacement is not None:
                    return _add(replacement, *[walk(child, stack) for child in nonlinear_children])
        return {key: walk(value, stack) for key, value in node.items()}

    return walk(ast)


def families_from_stage2_checkpoint(path: Path, domain: InventoryFamilyDomain,
                                    *, audit: list | None = None,
                                    selection: str | None = None) -> tuple[PolicyFamily, ...]:
    """Import scored Stage-2 AST frontiers as Stage-3 family seeds.

    Stage 2 stores ordinary ASTs and per-case constants.  Stage 3 needs a
    context-independent family, so constants are deliberately discarded and
    role-based bounds are rebuilt from the canonical backlog case schema.
    Fixed source summaries are generalized before validation. Every rejected
    row and selected source is available to the launcher's persisted audit.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    from sr.analysis.frontier import validate_completed_cap_search
    validate_completed_cap_search(Path(path), payload)
    if int(payload["iteration"]) < int(payload["config"]["iterations"]):
        raise ValueError(f"Stage2 checkpoint is incomplete: {path}")
    from sr.family.seed_selection import PROTOCOL, select_frontier_seeds
    if selection not in (None, PROTOCOL):
        raise ValueError(f"unsupported Stage3 seed selection: {selection}; expected {PROTOCOL}")
    frontier = list(payload["parameter_frontier"]) + list(payload["complexity_frontier"])
    rows, selection_audit = select_frontier_seeds(frontier, frontier)
    if audit is not None:
        audit.append({"status": "selection", "protocol": PROTOCOL,
                      "candidate_count": len(rows), "selection": selection_audit})
    representatives = [row.case for row in domain.contexts]
    source_case = next(case for case in representatives
                       if case.lead_time == 3 and case.config.backlog.patience == 3)
    base_motifs = dict(motifs_for_config(source_case.config))
    stored_motifs = dict(payload.get("motifs", {}))
    stored_motifs.update({row["name"]: row["ast"] for row in payload.get("reflection_motifs", ())})
    output = []
    seen = set()
    portable = []
    processed = set()
    for row in rows:
        ast = row.get("ast")
        source_hash = str(row.get("ast_hash", ""))
        if source_hash in processed:
            continue
        processed.add(source_hash)
        try:
            checked = ast_check(ast, motifs={**stored_motifs, **base_motifs},
                                allowed_state_names=source_case.schema.names)
            specs = parameter_specs_for_case(checked.stats.parameters,
                                            max(representatives, key=lambda case: case.schema.dimension))
            kind = {"integer": "target", "grid": "target", "continuous": "scale"}
            from inventory.policy.ast import cap_parameter_modes
            cap_modes = cap_parameter_modes(checked.ast, {**stored_motifs, **base_motifs})
            parameters = tuple(FamilyParameter(spec.name, "cap" if spec.name in cap_modes else kind.get(spec.kind, "scale"),
                                                float(spec.lower), float(spec.upper)) for spec in specs)
            family = PolicyFamily("stage2_import_" + str(row.get("ast_hash", checked.ast_hash))[:16],
                                  _promote_stage2_ast(checked.ast, source_case, stored_motifs, domain),
                                  {}, parameters,
                                  f"L3K3-algebra-preserving generalization of Stage2 {source_hash}; constants refitted.")
            domain.validate_family(family)
            key = domain.semantic_hash(family)
            if key not in seen:
                portable.append((row, family)); seen.add(key)
            elif audit is not None:
                audit.append({"source_hash": source_hash, "status": "semantic_duplicate"})
        except (KeyError, TypeError, ValueError) as exc:
            if audit is not None:
                audit.append({"source_hash": source_hash, "status": "rejected", "reason": str(exc)})
    selected_hashes = set()
    for row, family in portable:
        if family.family_id not in selected_hashes:
            output.append(family); selected_hashes.add(family.family_id)
    if audit is not None:
        for row, family in portable:
            audit.append({"source_hash": row.get("ast_hash"), "source_score": row.get("score"),
                          "status": "selected" if family.family_id in selected_hashes else "not_selected",
                          "family_id": family.family_id})
    if not output:
        raise ValueError(f"Stage-2 checkpoint has no portable family seeds: {path}")
    return tuple(output)


__all__ = ["families_from_stage2_checkpoint"]
