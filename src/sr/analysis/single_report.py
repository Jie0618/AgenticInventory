"""Durable report models and rendering for single-case SR."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from inventory.policy.ast import AST, ast_to_formula
from sr.core.grammar import expand_motifs
from sr.core.models import ReflectionMotif, SearchIndividual


@dataclass(frozen=True)
class SearchRunResult:
    case_name: str
    output_dir: Path
    archive: tuple[SearchIndividual, ...]
    parameter_frontier: tuple[SearchIndividual, ...]
    complexity_frontier: tuple[SearchIndividual, ...]
    parameter_profile: tuple[SearchIndividual, ...]
    complexity_profile: tuple[SearchIndividual, ...]
    scheduler_counts: Mapping[str, int]
    iterations_completed: int
    elapsed_seconds: float
    reflection_motifs: tuple[ReflectionMotif, ...]
    motifs: Mapping[str, AST]


def reflection_summary(row: ReflectionMotif) -> dict[str, object]:
    return {
        "name": row.name,
        "ast": row.ast,
        "rationale": row.rationale,
        "version": row.version,
        "status": row.status,
    }


def write_search_report(result: SearchRunResult) -> Path:
    lines = [
        f"# Symbolic regression report: `{result.case_name}`",
        "",
        f"- Iterations completed: {result.iterations_completed}",
        f"- Completed Standard candidates: {len(result.archive)}",
        f"- Wall time: {result.elapsed_seconds:.2f} s",
        f"- Scheduler states: `{json.dumps(dict(result.scheduler_counts), sort_keys=True)}`",
        f"- Reflection motif trials: {len(result.reflection_motifs)}",
        "",
        "Only independently completed Standard fits appear below. Bellman-screen parameters were discarded.",
        "",
        "## Best formula at each parameter count",
        "",
        "| Gap | Cost | Params | Complexity | Formula | Expanded formula | Constants | Status | Time |",
        "|---:|---:|---:|---:|---|---|---|---|---:|",
    ]
    lines.extend(_report_row(row, result.motifs) for row in result.parameter_profile)
    lines.extend(
        [
            "",
            "## Best formula at each expanded complexity",
            "",
            "| Gap | Cost | Params | Complexity | Formula | Expanded formula | Constants | Status | Time |",
            "|---:|---:|---:|---:|---|---|---|---|---:|",
        ]
    )
    lines.extend(_report_row(row, result.motifs) for row in result.complexity_profile)
    lines.extend(
        [
            "",
            "## Strict two-objective Pareto sets (appendix)",
            "",
            "The profiles above answer the per-budget question. The strict nondominated sets below are retained for mathematical completeness.",
            "",
            "### Strict gap–parameter Pareto set",
            "",
            "| Gap | Cost | Params | Complexity | Formula | Expanded formula | Constants | Status | Time |",
            "|---:|---:|---:|---:|---|---|---|---|---:|",
        ]
    )
    lines.extend(_report_row(row, result.motifs) for row in result.parameter_frontier)
    lines.extend(
        [
            "",
            "### Strict gap–complexity Pareto set",
            "",
            "| Gap | Cost | Params | Complexity | Formula | Expanded formula | Constants | Status | Time |",
            "|---:|---:|---:|---:|---|---|---|---|---:|",
        ]
    )
    lines.extend(_report_row(row, result.motifs) for row in result.complexity_frontier)
    lines.extend(["", "## Reflection motifs", ""])
    if result.reflection_motifs:
        for motif in result.reflection_motifs:
            lines.append(
                f"- `{motif.name}` ({motif.status}): {motif.rationale}; "
                f"AST `{json.dumps(motif.ast, sort_keys=True)}`"
            )
    else:
        lines.append("No executable motif trial was registered.")
    path = result.output_dir / "REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _report_row(row: SearchIndividual, motifs: Mapping[str, AST]) -> str:
    params = ", ".join(
        f"{key}={value:.6g}" for key, value in sorted(row.params.items())
    ) or "—"
    formula = row.candidate.formula.replace("|", "\\|")
    expanded = ast_to_formula(expand_motifs(row.candidate.ast, motifs)).replace(
        "|", "\\|"
    )
    return (
        f"| {row.gap_pct:.4f}% | {row.policy_cost:.6f} | {row.num_parameters} | "
        f"{row.expanded_complexity:.1f} | `{formula}` | `{expanded}` | {params} | "
        f"{row.status}/{row.route} | {row.optimizer_seconds:.2f}s |"
    )


__all__ = ["SearchRunResult", "reflection_summary", "write_search_report"]
