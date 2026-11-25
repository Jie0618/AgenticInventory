"""Durable report models and rendering for multi-case SR."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from inventory.policy.ast import AST, ast_to_formula
from sr.core.grammar import expand_motifs
from sr.core.models import ReflectionMotif
from sr.multi.models import MultiSearchIndividual


@dataclass(frozen=True)
class MultiSearchRunResult:
    case_name: str
    case_names: tuple[str, ...]
    output_dir: Path
    archive: tuple[MultiSearchIndividual, ...]
    parameter_frontier: tuple[MultiSearchIndividual, ...]
    complexity_frontier: tuple[MultiSearchIndividual, ...]
    parameter_profile: tuple[MultiSearchIndividual, ...]
    complexity_profile: tuple[MultiSearchIndividual, ...]
    scheduler_counts: Mapping[str, int]
    iterations_completed: int
    elapsed_seconds: float
    reflection_motifs: tuple[ReflectionMotif, ...]
    motifs: Mapping[str, AST]


def search_evaluation_dir(output_dir: Path) -> Path:
    return Path(output_dir) / "standard_evaluations"


def write_multi_search_report(result: MultiSearchRunResult) -> Path:
    lines = [
        "# Multi-distribution symbolic regression report",
        "",
        f"- Training cases: {len(result.case_names)}",
        f"- Completed Standard structures: {len(result.archive)}",
        f"- Iterations: {result.iterations_completed}",
        f"- Elapsed: {result.elapsed_seconds:.2f}s",
        "- Score: `0.6 × mean gap + 0.4 × CVaR80 gap`",
        "",
        "## Training cases",
        "",
        *[f"- `{name}`" for name in result.case_names],
        "",
        "## Score–parameter frontier",
        "",
        table_header(result.case_names),
        *[multi_report_row(row, result.case_names) for row in result.parameter_frontier],
        "",
        "## Score–complexity frontier",
        "",
        table_header(result.case_names),
        *[multi_report_row(row, result.case_names) for row in result.complexity_frontier],
        "",
        "## Per-case gap matrix for frontier union",
        "",
    ]
    union = {
        row.ast_hash: row
        for row in (*result.parameter_frontier, *result.complexity_frontier)
    }
    header = "| AST | " + " | ".join(result.case_names) + " |"
    lines.extend((header, "|---|" + "---:|" * len(result.case_names)))
    for row in sorted(union.values(), key=quality_key):
        values = " | ".join(
            f"{row.case_results[name].gap_pct:.4f}%" for name in result.case_names
        )
        lines.append(f"| `{row.ast_hash[:12]}` | {values} |")
    path = result.output_dir / "REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def table_header(case_names: Sequence[str]) -> str:
    columns = (
        "Score",
        "Mean gap",
        "CVaR80",
        "Worst gap",
        "Params",
        "Complexity",
        *(f"{name} gap" for name in case_names),
        "AST",
        "Formula",
    )
    alignments = ("---:",) * (6 + len(case_names)) + ("---", "---")
    return "| " + " | ".join(columns) + " |\n|" + "|".join(alignments) + "|"


def multi_report_row(
    row: MultiSearchIndividual, case_names: Sequence[str]
) -> str:
    formula = row.candidate.formula.replace("|", "\\|")
    gaps = " | ".join(
        f"{row.case_results[name].gap_pct:.4f}%" for name in case_names
    )
    return (
        f"| {row.score:.4f}% | {row.mean_gap_pct:.4f}% | "
        f"{row.mean_top_k_gap_pct:.4f}% | {row.worst_gap_pct:.4f}% "
        f"(`{row.worst_case}`) | {row.num_parameters} | "
        f"{row.expanded_complexity:.1f} | {gaps} | `{row.ast_hash[:12]}` | "
        f"`{formula}` |"
    )


def multi_summary(
    row: MultiSearchIndividual, motifs: Mapping[str, AST]
) -> dict[str, object]:
    expanded = expand_motifs(row.candidate.ast, motifs)
    return {
        "ast_hash": row.ast_hash,
        "ast": row.candidate.ast,
        "formula": row.candidate.formula,
        "expanded_ast": expanded,
        "expanded_formula": ast_to_formula(expanded),
        "source": row.candidate.source,
        "parents": list(row.candidate.parents),
        "num_parameters": row.num_parameters,
        "expanded_complexity": row.expanded_complexity,
        "score": row.score,
        "gap_pct": row.score,
        "mean_gap_pct": row.mean_gap_pct,
        "mean_top_k_gap_pct": row.mean_top_k_gap_pct,
        "worst_gap_pct": row.worst_gap_pct,
        "worst_case": row.worst_case,
        "top_k_cases": list(row.top_k_cases),
        "provenance": list(row.provenance),
        "case_results": {
            name: {
                "policy_cost": result.policy_cost,
                "dp_cost": result.dp_cost,
                "gap_pct": result.gap_pct,
                "params": dict(result.params),
                "status": result.status,
                "route": result.route,
                "optimizer_seconds": result.optimizer_seconds,
                "result_path": result.result_path,
            }
            for name, result in row.case_results.items()
        },
    }


def quality_key(row: MultiSearchIndividual) -> tuple[float, int, float, str]:
    return (row.score, row.num_parameters, row.expanded_complexity, row.ast_hash)


__all__ = [
    "MultiSearchRunResult",
    "multi_report_row",
    "multi_summary",
    "quality_key",
    "search_evaluation_dir",
    "table_header",
    "write_multi_search_report",
]
