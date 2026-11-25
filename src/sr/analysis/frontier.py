"""Generic AST and signature helpers for imported SR frontiers."""

from __future__ import annotations

from runtime.artifacts import stable_key
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from inventory.policy.ast import AST, ASTLimits, ast_check
from sr.core.grammar import expand_motifs
from sr.core.gp import structural_signature
from sr.core.models import SearchCandidate


class FrontierBundle(Protocol):
    candidates: tuple[SearchCandidate, ...]
    motifs: Mapping[str, AST]
    provenance: Mapping[str, tuple[str, ...]]
    source_checkpoints: Mapping[str, str]


def validate_completed_cap_search(path: Path, payload: Mapping) -> None:
    """frontiers may transfer only after final drain/report and checkpoint."""
    if not payload.get("config", {}).get("cyclic_cap", False):
        return
    try:
        marker = json.loads(path.with_name("SEARCH_COMPLETED.json").read_text())
        valid = (marker["protocol"] == "cyclic-cap"
                 and int(marker["iteration"]) == int(payload["config"]["iterations"])
                 and path.with_name("REPORT.md").is_file())
    except (OSError, ValueError, KeyError, TypeError):
        valid = False
    if not valid:
        raise ValueError(f"cyclic source search has not completed final drain/report: {path}")


@dataclass(frozen=True)
class FrontierSeedBundle:
    candidates: tuple[SearchCandidate, ...]
    motifs: Mapping[str, AST]
    provenance: Mapping[str, tuple[str, ...]]
    initial_islands: Mapping[str, tuple[int, ...]]
    source_checkpoints: Mapping[str, str]


class FrontierCandidateFactory(Protocol):
    def __call__(
        self,
        ast: AST,
        *,
        motifs: Mapping[str, AST],
        source: str,
        island: int,
        detail: Mapping[str, object],
    ) -> SearchCandidate: ...


def rewrite_motifs(ast: AST, names: Mapping[str, str]) -> AST:
    if ast.get("op") == "motif":
        old_name = str(ast["name"])
        return {"op": "motif", "name": names.get(old_name, old_name)}
    output: AST = {}
    for key, value in ast.items():
        if isinstance(value, dict):
            output[key] = rewrite_motifs(value, names)
        elif isinstance(value, list):
            output[key] = [
                rewrite_motifs(item, names) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            output[key] = value
    return output


def motif_names(ast: AST) -> set[str]:
    if ast.get("op") == "motif":
        return {str(ast["name"])}
    output: set[str] = set()
    for value in ast.values():
        if isinstance(value, dict):
            output.update(motif_names(value))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    output.update(motif_names(item))
    return output


def import_frontier_seeds(
    checkpoint_paths: Mapping[str, Path],
    *,
    base_motifs: Mapping[str, AST],
    candidate_factory: FrontierCandidateFactory,
    allowed_state_names: Sequence[str] | None = None,
) -> FrontierSeedBundle:
    """Merge single-case frontiers with namespaced motifs and structural deduplication."""

    ordered_cases = tuple(str(name) for name in checkpoint_paths)
    if not ordered_cases:
        raise ValueError("at least one source checkpoint is required")
    payloads: dict[str, dict] = {}
    checkpoint_hashes: dict[str, str] = {}
    global_motifs: dict[str, AST] = dict(base_motifs)
    custom_name_maps: dict[str, dict[str, str]] = {}

    for case_name in ordered_cases:
        path = Path(checkpoint_paths[case_name])
        content = path.read_bytes()
        payload = json.loads(content)
        validate_completed_cap_search(path, payload)
        if int(payload.get("schema_version", -1)) != 2:
            raise ValueError(f"unsupported checkpoint schema: {path}")
        if str(payload.get("case")) != case_name:
            raise ValueError(
                f"checkpoint case mismatch: expected {case_name}, found {payload.get('case')}"
            )
        payloads[case_name] = payload
        checkpoint_hashes[case_name] = str(path.resolve())
        local_motifs = dict(base_motifs)
        for row in payload.get("reflection_motifs", ()):
            local_motifs[str(row["name"])] = dict(row["ast"])
        used_custom: set[str] = set()
        for frontier_name in ("parameter_frontier", "complexity_frontier"):
            for frontier_row in payload.get(frontier_name, ()):
                used_custom.update(motif_names(dict(frontier_row["ast"])))
        used_custom.difference_update(base_motifs)
        names: dict[str, str] = {}
        for old_name in sorted(used_custom):
            if old_name not in local_motifs:
                raise ValueError(
                    f"frontier references missing motif {old_name!r} in {path}"
                )
            expanded = expand_motifs(local_motifs[old_name], local_motifs)
            checked = ast_check(
                expanded,
                motifs={},
                limits=ASTLimits(max_depth=256, max_nodes=10000, max_parameters=0),
                allowed_state_names=allowed_state_names,
            )
            new_name = f"seed_motif_{checked.ast_hash[:16]}"
            previous = global_motifs.get(new_name)
            if previous is not None and previous != checked.ast:
                raise RuntimeError(f"seed motif hash collision: {new_name}")
            global_motifs[new_name] = checked.ast
            names[old_name] = new_name
        custom_name_maps[case_name] = names

    selected: dict[str, SearchCandidate] = {}
    provenance: dict[str, set[str]] = {}
    islands: dict[str, set[int]] = {}
    signatures: dict[str, str] = {}
    for island, case_name in enumerate(ordered_cases):
        rows_by_hash: dict[str, Mapping[str, object]] = {}
        for key in ("parameter_frontier", "complexity_frontier"):
            for row in payloads[case_name].get(key, ()):
                rows_by_hash.setdefault(str(row["ast_hash"]), row)
        for source_hash, row in sorted(rows_by_hash.items()):
            rewritten = rewrite_motifs(
                dict(row["ast"]), custom_name_maps[case_name]
            )
            candidate = candidate_factory(
                rewritten,
                motifs=global_motifs,
                source=f"single_frontier:{case_name}",
                island=island,
                detail={"source_ast_hash": source_hash},
            )
            signature = structural_signature(
                expand_motifs(candidate.ast, global_motifs)
            )
            canonical_key = stable_key(signature.encode("utf-8"))
            if canonical_key not in selected:
                selected[canonical_key] = candidate
                signatures[canonical_key] = signature
                provenance[canonical_key] = set()
                islands[canonical_key] = set()
            elif signatures[canonical_key] != signature:
                raise RuntimeError("expanded structural signature hash collision")
            provenance[canonical_key].add(case_name)
            islands[canonical_key].add(island)

    candidates: list[SearchCandidate] = []
    provenance_by_hash: dict[str, tuple[str, ...]] = {}
    islands_by_hash: dict[str, tuple[int, ...]] = {}
    for canonical_key in sorted(selected):
        row = selected[canonical_key]
        sources = tuple(sorted(provenance[canonical_key]))
        row = replace(
            row,
            detail={
                **dict(row.detail),
                "frontier_provenance": list(sources),
                "expanded_structural_hash": canonical_key,
            },
        )
        candidates.append(row)
        provenance_by_hash[row.ast_hash] = sources
        islands_by_hash[row.ast_hash] = tuple(sorted(islands[canonical_key]))
    if len({row.ast_hash for row in candidates}) != len(candidates):
        raise RuntimeError("macro AST collision after expanded-structure deduplication")
    return FrontierSeedBundle(
        candidates=tuple(candidates),
        motifs=global_motifs,
        provenance=provenance_by_hash,
        initial_islands=islands_by_hash,
        source_checkpoints=checkpoint_hashes,
    )


def initialization_signature(
    bundle: FrontierBundle,
    *,
    case_names: Sequence[str],
    horizon: int,
    objective_payload: Mapping[str, object],
    config_payload: Mapping[str, object],
) -> str:
    payload = {
        "protocol": "multi-frontier-standard-initialization",
        "case_names": list(case_names),
        "horizon": int(horizon),
        "objective": dict(objective_payload),
        "search_config": dict(config_payload),
        "source_checkpoints": dict(sorted(bundle.source_checkpoints.items())),
        "candidates": [
            {
                "ast_hash": row.ast_hash,
                "ast": row.ast,
                "provenance": list(bundle.provenance[row.ast_hash]),
            }
            for row in bundle.candidates
        ],
        "motifs": dict(sorted(bundle.motifs.items())),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return stable_key(encoded)


__all__ = [
    "FrontierBundle",
    "FrontierCandidateFactory",
    "FrontierSeedBundle",
    "import_frontier_seeds",
    "initialization_signature",
    "motif_names",
    "rewrite_motifs",
]
