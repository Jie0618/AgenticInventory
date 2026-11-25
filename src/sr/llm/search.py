"""Scenario-neutral prompts for structural proposal and motif reflection."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from inventory.policy.ast import AST, ASTLimits, ast_check, ast_to_formula

from sr.core.models import ReflectionMotif, SearchConfig, SearchIndividual
from sr.llm.client import JSONChatClient
from sr.llm.proposals import execute_structural_proposal, parse_structural_proposals


class DomainSpec(Protocol):
    name: str
    objective: str
    state_names: tuple[str, ...]
    components: tuple[str, ...]
    distribution_names: tuple[str, ...]
    cost_names: tuple[str, ...]
    rules: tuple[str, ...]
    parameter_roles: Mapping[str, str]
    features: Mapping[str, Mapping[str, object]]


@dataclass(frozen=True)
class SearchDomain:
    """The model vocabulary injected into otherwise generic SR prompts."""

    name: str
    objective: str
    state_names: tuple[str, ...]
    components: tuple[str, ...] = ()
    distribution_names: tuple[str, ...] = ()
    cost_names: tuple[str, ...] = ()
    rules: tuple[str, ...] = ()
    parameter_roles: Mapping[str, str] = field(default_factory=dict)
    features: Mapping[str, Mapping[str, object]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.objective.strip():
            raise ValueError("search domain name and objective must be nonempty")
        if not self.state_names or any(not str(name) for name in self.state_names):
            raise ValueError("search domain requires nonempty state names")

    @property
    def allowed_state_names(self) -> frozenset[str]:
        return frozenset((*self.state_names, "remaining_horizon"))

    def prompt_payload(self, motifs: Mapping[str, AST]) -> dict[str, object]:
        return {
            "name": self.name,
            "objective": self.objective,
            "components": list(self.components),
            "state_names": list(self.state_names),
            "distribution_names": list(self.distribution_names),
            "cost_names": list(self.cost_names),
            "motifs": {
                name: ast_to_formula(ast) for name, ast in sorted(motifs.items())
            },
            "features": {
                name: dict(spec) for name, spec in sorted(self.features.items())
            },
            "parameter_roles": dict(sorted(self.parameter_roles.items())),
            "rules": list(self.rules),
        }

    @classmethod
    def from_spec(cls, spec: DomainSpec) -> "SearchDomain":
        """Copy any lower-layer domain description into the prompt contract."""

        return cls(
            name=spec.name,
            objective=spec.objective,
            state_names=tuple(spec.state_names),
            components=tuple(spec.components),
            distribution_names=tuple(spec.distribution_names),
            cost_names=tuple(spec.cost_names),
            rules=tuple(spec.rules),
            parameter_roles=dict(spec.parameter_roles),
            features={name: dict(value) for name, value in spec.features.items()},
        )


def propose_asts(
    client: JSONChatClient,
    parents: Sequence[SearchIndividual],
    motifs: Mapping[str, AST],
    config: SearchConfig,
    domain: SearchDomain,
    evidence: Mapping[str, object] | None = None,
) -> tuple[tuple[AST, str, tuple[str, ...]], ...]:
    """Ask for validated local mutations using a supplied model vocabulary."""

    if not parents:
        return ()
    parent_rows = [
        {
            "hash": row.ast_hash,
            "formula": row.candidate.formula,
            "ast": row.candidate.ast,
            "fitted_parameters": dict(row.params),
            "parameter_names": list(row.candidate.parameter_names),
            "policy_cost": row.policy_cost,
            "reference_cost": row.dp_cost,
            "gap_pct": row.gap_pct,
            "num_params": row.num_parameters,
            "expanded_complexity": row.expanded_complexity,
            **(
                {
                    "multi_mean_gap_pct": row.mean_gap_pct,
                    "multi_mean_top_k_gap_pct": row.mean_top_k_gap_pct,
                    "multi_worst_gap_pct": row.worst_gap_pct,
                    "multi_worst_case": row.worst_case,
                    "multi_case_gaps_pct": {
                        name: result.gap_pct for name, result in row.case_results.items()
                    },
                    "case_evaluations": {
                        name: {"fitted_parameters": dict(result.params), "cost": result.policy_cost,
                               "reference_cost": result.dp_cost, "gap_pct": result.gap_pct}
                        for name, result in row.case_results.items()
                    },
                }
                if hasattr(row, "case_results")
                else {}
            ),
        }
        for row in parents
    ]
    schema = {
        "proposals": [
            {
                "action": "minimum_crossover | maximum_crossover | cap | replace_subtree | change_cap_type",
                "parent_hash": "required",
                "other_parent_hash": "required for crossover",
                "path": ["AST field tokens for replacement or cap-type change; empty for root"],
                "ast": {"op": "replacement subtree, required for replace_subtree"},
                "mode": "integer | continuous | cyclic_integer, for change_cap_type",
                "bound": "upper | lower, for cap",
                "rationale": "brief operational reason",
            }
        ]
    }
    prompt = (
        "Propose simple expectation-free inventory-policy AST mutations. Output JSON only.\n"
        f"Return exactly {config.llm_proposals_per_iteration} proposals. Optimize the domain objective while preferring few free constants and low expanded complexity.\n"
        "Use only current parents and the supplied domain vocabulary. Do not return code, expectations, probability recursion, or invented operators, states, distributions, costs, or motifs.\n"
        "Leaves: const(value), parameter(name), state(name), distribution(name), cost(name), motif(name).\n"
        "Operators: add(args), mul(args), neg(arg), sub(left,right), absolute(arg), minimum(left,right), maximum(left,right), indicator_gt(left,right), safe_ratio(left,right), clip(arg,lower,upper).\n"
        + ("Typed cap: {\"op\":\"cap\",\"arg\":expression,\"limit\":{\"op\":\"parameter\",\"name\":\"C\"},\"mode\":\"cyclic_integer\"}. Modes: integer (integer domain), continuous (float domain), cyclic_integer (float C, integer round(C*(t+1))-round(C*t), t starts at 0, ties-to-even). It replaces only a cap constant; final action rounding is unchanged. Caps may be internal. Each typed cap parameter may occur only as a direct cap limit. Never put cyclic logic in a motif or use the removed cyclic root operator. Use change_cap_type for an existing cap slot, or replace_subtree with its path to supply a locally modified expression.\n" if config.cyclic_cap else "")
        +
        "Replenishment actions are made nonnegative after evaluation; do not wrap the outermost policy in maximum(policy, 0) or maximum(0, policy).\n"
        f"Domain vocabulary: {json.dumps(domain.prompt_payload(motifs), sort_keys=True)}\n"
        f"Parents: {json.dumps(parent_rows, sort_keys=True)}\n"
        f"Decision diagnostics: {json.dumps(dict(evidence or {}), sort_keys=True)}\n"
        f"Structural limits: parameters={config.max_parameters}, depth={config.max_depth}, nodes={config.max_nodes}, expanded_complexity={config.max_expanded_complexity}.\n"
        f"Response schema: {json.dumps(schema, sort_keys=True)}"
    )
    proposals = parse_structural_proposals(client.complete(prompt))
    parent_map = {row.ast_hash: row.candidate.ast for row in parents}
    output = []
    for proposal in proposals[: config.llm_proposals_per_iteration]:
        try:
            ast = execute_structural_proposal(
                proposal,
                parent_map,
                motifs,
                limits=ASTLimits(
                    max_depth=config.max_depth,
                    max_nodes=config.max_nodes,
                    max_parameters=config.max_parameters,
                ),
                allowed_state_names=domain.allowed_state_names,
            )
            parent_hashes = tuple(
                value
                for value in (proposal.parent_hash, proposal.other_parent_hash)
                if value is not None
            )
            output.append((ast, proposal.rationale, parent_hashes))
        except (ValueError, KeyError, TypeError):
            continue
    return tuple(output)


def reflect_motifs(
    client: JSONChatClient,
    individuals: Sequence[SearchIndividual],
    motifs: Mapping[str, AST],
    *,
    version: int,
    domain: SearchDomain,
    max_trials: int = 2,
    context: Mapping[str, object] | None = None,
) -> tuple[ReflectionMotif, ...]:
    """Ask for reusable parameter-free summaries within the same vocabulary."""

    rows = [
        {
            "formula": row.candidate.formula,
            "ast": row.candidate.ast,
            "parameter_names": list(row.candidate.parameter_names),
            "fitted_parameters": dict(row.params),
            "policy_cost": row.policy_cost,
            "reference_cost": row.dp_cost,
            "gap_pct": row.gap_pct,
            "num_params": row.num_parameters,
            "expanded_complexity": row.expanded_complexity,
            "source": row.candidate.source,
            "optimizer_status": row.status,
            "optimizer_route": row.route,
            **(
                {
                    "multi_mean_gap_pct": row.mean_gap_pct,
                    "multi_mean_top_k_gap_pct": row.mean_top_k_gap_pct,
                    "multi_worst_gap_pct": row.worst_gap_pct,
                    "multi_worst_case": row.worst_case,
                    "multi_case_gaps_pct": {
                        name: result.gap_pct for name, result in row.case_results.items()
                    },
                    "case_evaluations": {
                        name: {"fitted_parameters": dict(result.params), "cost": result.policy_cost,
                               "reference_cost": result.dp_cost, "gap_pct": result.gap_pct}
                        for name, result in row.case_results.items()
                    },
                }
                if hasattr(row, "case_results")
                else {}
            ),
        }
        for row in individuals
    ]
    prompt = (
        "Reflect on these symbolic inventory policies and propose at most two reusable parameter-free state motifs. Output JSON only.\n"
        "A motif must be a parameter-free subtree of a selected parent, after expanding existing motifs, and use only the supplied vocabulary.\n"
        "Do not rename an existing motif. Prefer compact operational summaries whose work is linear or better in the state dimension.\n"
        "Use exact node encodings: motif leaves are {\"op\":\"motif\",\"name\":\"name\"}; constants are {\"op\":\"const\",\"value\":0}; neg/absolute use arg; minimum/maximum/indicator_gt/sub/safe_ratio use left and right.\n"
        f"Domain vocabulary: {json.dumps(domain.prompt_payload(motifs), sort_keys=True)}\n"
        f"Policies: {json.dumps(rows, sort_keys=True)}\n"
        f"Search diagnostics: {json.dumps(dict(context or {}), sort_keys=True)}\n"
        "Schema: {\"motifs\":[{\"name\":\"short_name\",\"ast\":{\"op\":\"...\"},\"rationale\":\"...\"}]}"
    )
    text = client.complete(prompt).strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    payload = json.loads(text)
    raw = payload.get("motifs", []) if isinstance(payload, Mapping) else []
    output = []
    for index, row in enumerate(raw[:max_trials]):
        if not isinstance(row, Mapping) or not isinstance(row.get("ast"), Mapping):
            continue
        try:
            base = re.sub(
                r"[^a-z0-9_]+", "_", str(row.get("name", "motif")).lower()
            ).strip("_") or "motif"
            name = f"reflection_v{version}_{index}_{base}"
            ast = ast_check(
                _normalize_reflection_ast(dict(row["ast"]), motifs),
                motifs=motifs,
                limits=ASTLimits(max_depth=10, max_nodes=64, max_parameters=0),
                allowed_state_names=domain.allowed_state_names,
            ).ast
            from inventory.policy.ast import has_cyclic_cap
            if has_cyclic_cap(ast, motifs):
                raise ValueError("reflection motifs cannot contain cyclic caps")
            from sr.core.grammar import expand_motifs
            from sr.core.tree import contains_subtree
            expanded = expand_motifs(ast, motifs)
            if not any(contains_subtree(expand_motifs(parent.candidate.ast, motifs), expanded)
                       for parent in individuals):
                raise ValueError("reflection motif is not a subtree of a selected parent")
            output.append(
                ReflectionMotif(
                    name=name,
                    ast=ast,
                    rationale=str(row.get("rationale", "")),
                    version=version,
                )
            )
        except (ValueError, KeyError, TypeError, OverflowError):
            continue
    return tuple(output)


def _normalize_reflection_ast(ast: AST, motifs: Mapping[str, AST]) -> AST:
    op = str(ast.get("op", ""))
    if op in motifs and set(ast) == {"op"}:
        return {"op": "motif", "name": op}
    if op == "constant":
        return {"op": "const", "value": ast.get("value")}
    output: AST = {key: value for key, value in ast.items() if key != "args"}
    raw_args = ast.get("args")
    if isinstance(raw_args, list):
        args = [
            _normalize_reflection_ast(dict(item), motifs)
            if isinstance(item, Mapping)
            else item
            for item in raw_args
        ]
        if op in {"neg", "absolute"} and len(args) == 1:
            output["arg"] = args[0]
        elif op in {"minimum", "maximum", "indicator_gt", "sub", "safe_ratio"} and len(args) == 2:
            output["left"], output["right"] = args
        else:
            output["args"] = args
    for key, value in tuple(output.items()):
        if isinstance(value, Mapping):
            output[key] = _normalize_reflection_ast(dict(value), motifs)
        elif isinstance(value, list):
            output[key] = [
                _normalize_reflection_ast(dict(item), motifs)
                if isinstance(item, Mapping)
                else item
                for item in value
            ]
    return output


__all__ = ["DomainSpec", "SearchDomain", "propose_asts", "reflect_motifs"]
