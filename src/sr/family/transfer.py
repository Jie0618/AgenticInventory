"""Import actual completed Stage2 frontiers without supplying cyclic anchors."""
import json
from pathlib import Path

from inventory.policy.family import PolicyFamily, FamilyParameter
from inventory.policy.motifs import motifs_for_config
from optimization.policies.parameters import parameter_specs_for_case
from inventory.policy.ast import ast_check, cap_parameter_modes


def linear(node, motifs, stack=()):
    op = node.get("op")
    if op == "state":
        return {node["name"]: 1.0}
    if op in {"distribution", "cost"}:
        return {op + ":" + node["name"]: 1.0}
    if op == "const":
        return {"constant": float(node["value"])}
    if op == "motif":
        name = node["name"]
        if name in stack:
            raise ValueError("recursive source motif")
        return linear(motifs[name], motifs, (*stack, name))
    if op in {"add", "sub", "neg"}:
        children = node["args"] if op == "add" else [node["arg"]] if op == "neg" else [node["left"], {"op": "neg", "arg": node["right"]}]
        out = {}
        for child in children:
            terms = linear(child, motifs, stack)
            if terms is None:
                return None
            for name, value in terms.items():
                out[name] = out.get(name, 0.0) + (-value if op == "neg" else value)
        return {k: v for k, v in out.items() if v != 0}
    if op == "mul":
        factor, bodies = 1.0, []
        for child in node["args"]:
            if child.get("op") == "const":
                factor *= float(child["value"])
            else:
                bodies.append(child)
        terms = linear(bodies[0], motifs, stack) if len(bodies) == 1 else {"constant": 1.0} if not bodies else None
        return None if terms is None else {k: factor * v for k, v in terms.items()}
    return None


def import_stage2(path, domain, *, source_case, selection=None):
    payload = json.loads(Path(path).read_text())
    from sr.analysis.frontier import validate_completed_cap_search
    validate_completed_cap_search(Path(path), payload)
    if int(payload["iteration"]) != int(payload["config"]["iterations"]):
        raise ValueError("Stage2 must complete all iterations before Stage3 transfer")
    source = {**payload.get("motifs", {}), **motifs_for_config(source_case.config)}
    source.update({row["name"]: row["ast"] for row in payload.get("reflection_motifs", ())})
    # At L3 cumulative_inventory_3 equals inventory_position, but only the
    # latter denotes the full position at L5/L8/L10. Prefer aggregate semantics
    # when recognizing inlined expressions; explicit motif names stay intact.
    preferred = ("inventory_position", "total_pipeline", "available_inventory",
                 "total_inventory", "far_pipeline", "near_pipeline",
                 "pipeline_lag_moment", "pipeline_imbalance")
    names = sorted((name for name in domain.common_motif_names if name in source),
                   key=lambda name: (preferred.index(name) if name in preferred else len(preferred), name))
    patterns = [(name, linear(source[name], source)) for name in names]
    def aggregate_terms(signature):
        """Recover sums of full inventory aggregates without inventing windows.

        E.g. IP+p3-mu remains IP+far_pipeline-mu at larger L. Unmatched
        interior coordinates still fail rather than silently changing meaning.
        """
        remaining = dict(signature)
        terms = []
        aggregates = ("inventory_position", "total_inventory", "total_pipeline")
        for name in aggregates:
            pattern = dict(patterns).get(name)
            if not pattern or any(v != 1 for v in pattern.values()):
                continue
            values = [remaining.get(k, 0) for k in pattern]
            factor = min(values) if all(v > 0 for v in values) else max(values) if all(v < 0 for v in values) else 0
            if factor:
                motif = {"op": "motif", "name": name}
                terms.append(motif if factor == 1 else {"op": "neg", "arg": motif} if factor == -1 else {"op": "mul", "args": [{"op": "const", "value": factor}, motif]})
                for k in pattern:
                    remaining[k] -= factor
        for key, value in remaining.items():
            if not value:
                continue
            if key == "constant":
                terms.append({"op": "const", "value": value})
                continue
            if ":" in key:
                op, name = key.split(":", 1)
                term = {"op": op, "name": name}
            elif key in {"x", "remaining_horizon"}:
                term = {"op": "state", "name": key}
            else:
                name = next((n for n, p in patterns if p == {key: 1.0}), None)
                if name is None:
                    return None
                term = {"op": "motif", "name": name}
            terms.append(term if value == 1 else {"op": "neg", "arg": term} if value == -1 else {"op": "mul", "args": [{"op": "const", "value": value}, term]})
        return terms[0] if len(terms) == 1 else {"op": "add", "args": terms} if terms else {"op": "const", "value": 0}

    def promote(node, stack=()):
        if isinstance(node, list):
            return [promote(v, stack) for v in node]
        if not isinstance(node, dict):
            return node
        if node.get("op") == "motif" and node["name"] in domain.common_motif_names:
            return node
        signature = linear(node, source) if "op" in node else None
        if signature is not None:
            match = next((name for name, value in patterns if value == signature), None)
            if match is not None:
                return {"op": "motif", "name": match}
            # Do not rewrite standalone distribution/constant leaves or change
            # already portable motif semantics merely because L3 aliases them.
            if node.get("op") in {"add", "sub", "neg", "mul"}:
                aggregate = aggregate_terms(signature)
                if aggregate is not None:
                    return aggregate
        if node.get("op") == "motif":
            name = node["name"]
            if name in stack:
                raise ValueError("recursive transfer motif")
            return promote(source[name], (*stack, name))
        if node.get("op") == "state" and node["name"] not in {"x", "remaining_horizon"}:
            raise ValueError("unrecognised fixed coordinate; use a portable aggregate motif")
        return {k: promote(v, stack) for k, v in node.items()}
    def compact(node):
        if isinstance(node, list):
            return [compact(v) for v in node]
        if not isinstance(node, dict):
            return node
        node = {k: compact(v) for k, v in node.items()}
        if node.get("op") in {"minimum", "maximum"}:
            op = node["op"]
            def leaves(v):
                return leaves(v["left"]) + leaves(v["right"]) if v.get("op") == op else [v]
            unique = list({json.dumps(v, sort_keys=True): v for v in leaves(node)}.values())
            def balance(items):
                if len(items) == 1:
                    return items[0]
                mid = len(items) // 2
                return {"op": op, "left": balance(items[:mid]), "right": balance(items[mid:])}
            return balance(unique)
        return node
    from .seed_selection import PROTOCOL, select_frontier_seeds
    if selection not in (None, PROTOCOL):
        raise ValueError(f"unsupported Stage3 seed selection: {selection}; expected {PROTOCOL}")
    frontier = list(payload["parameter_frontier"]) + list(payload["complexity_frontier"])
    rows, selected_audit = select_frontier_seeds(frontier, frontier)
    selection_audit = [{"status": "selection", "protocol": PROTOCOL,
                        "candidate_count": len(rows), "selection": selected_audit}]
    output, audit, seen = [], list(selection_audit), set()
    for row in rows:
        if row["ast_hash"] in seen:
            continue
        seen.add(row["ast_hash"])
        try:
            checked = ast_check(row["ast"], motifs=source, allowed_state_names=source_case.schema.names)
            specs = parameter_specs_for_case(checked.stats.parameters, max((r.case for r in domain.contexts), key=lambda c: c.schema.dimension))
            modes = cap_parameter_modes(checked.ast, source)
            params = tuple(FamilyParameter(p.name, "cap" if p.name in modes or p.name.startswith(("C", "cap")) else "target" if p.kind in {"integer", "grid"} else "scale", float(p.lower), float(p.upper)) for p in specs)
            family = PolicyFamily("stage2_frontier_" + row["ast_hash"][:16], compact(promote(row["ast"])), {}, params,
                                  "Transferred from completed Stage2 results using audited seed selection; all constants refitted.")
            domain.validate_family(family)
            if domain.semantic_hash(family) in {domain.semantic_hash(f) for f in output}:
                audit.append({"hash": row["ast_hash"], "status": "duplicate",
                              "source_score": row["score"]})
                continue
            output.append(family)
            audit.append({"hash": row["ast_hash"], "status": "selected", "source_score": row["score"], "ast": family.ast})
        except (ValueError, KeyError, TypeError) as exc:
            audit.append({"hash": row["ast_hash"], "status": "rejected", "reason": str(exc),
                          "source_score": row["score"]})
    if not output:
        raise ValueError("no portable Stage2 frontier families: " + json.dumps(audit))
    return tuple(output), audit
