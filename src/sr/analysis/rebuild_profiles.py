#!/usr/bin/env python3
"""Rebuild SR profiles and reports from completed Standard result files."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from inventory.core import InventoryCase
from inventory.policy.ast import ast_to_formula
from inventory.policy.motifs import motifs_for_config
from sr.analysis.artifacts import load_standard_result, rebuild_checkpoint_profiles
from sr.core.models import ReflectionMotif


def main(available_cases: Sequence[InventoryCase]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-dir", type=Path, required=True)
    args = parser.parse_args()
    checkpoint_path = args.search_dir / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    case = next(
        (case for case in available_cases if case.name == checkpoint["case"]),
        None,
    )
    if case is None:
        raise ValueError(f"unknown checkpoint case: {checkpoint['case']}")
    motifs = motifs_for_config(case.config)
    reflections = tuple(ReflectionMotif(**row) for row in checkpoint.get("reflection_motifs", []))
    for row in reflections:
        motifs[row.name] = row.ast
    archive = tuple(
        load_standard_result(path, motifs)
        for path in sorted((args.search_dir / "standard_results").glob("*.json"))
    )
    checkpoint = rebuild_checkpoint_profiles(checkpoint, archive, motifs)
    _atomic_json(checkpoint, checkpoint_path)
    _write_report(args.search_dir / "REPORT.md", checkpoint, reflections)
    print(json.dumps({"case": checkpoint["case"], "archive": len(archive), "parameter_profile": len(checkpoint["parameter_profile"]), "complexity_profile": len(checkpoint["complexity_profile"])}))


def _write_report(path: Path, checkpoint: dict, reflections: tuple[ReflectionMotif, ...]) -> None:
    lines = [
        f"# Symbolic regression report: `{checkpoint['case']}`",
        "",
        f"- Iterations completed: {checkpoint['iteration']}",
        f"- Completed Standard candidates: {checkpoint['archive_size']}",
        f"- Wall time: {checkpoint['elapsed_seconds']:.2f} s",
        f"- Scheduler states: `{json.dumps(checkpoint['scheduler_counts'], sort_keys=True)}`",
        f"- Reflection motif trials: {len(reflections)}",
        "",
        "Only independently completed Standard fits appear below. Bellman-screen parameters were discarded.",
        "",
        "## Best formula at each parameter count",
        "",
        _header(),
    ]
    lines.extend(_row(row) for row in checkpoint["parameter_profile"])
    lines.extend(["", "## Best formula at each expanded complexity", "", _header()])
    lines.extend(_row(row) for row in checkpoint["complexity_profile"])
    lines.extend([
        "",
        "## Strict two-objective Pareto sets (appendix)",
        "",
        "The profiles above answer the per-budget question. The strict nondominated sets below are retained for mathematical completeness.",
        "",
        "### Strict gap–parameter Pareto set",
        "",
        _header(),
    ])
    lines.extend(_row(row) for row in checkpoint["parameter_frontier"])
    lines.extend(["", "### Strict gap–complexity Pareto set", "", _header()])
    lines.extend(_row(row) for row in checkpoint["complexity_frontier"])
    lines.extend(["", "## Reflection motifs", ""])
    lines.extend(f"- `{row.name}` ({row.status}): {row.rationale}; AST `{json.dumps(row.ast, sort_keys=True)}`" for row in reflections)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _header() -> str:
    return "| Gap | Cost | Params | Complexity | Formula | Expanded formula | Constants | Status | Time |\n|---:|---:|---:|---:|---|---|---|---|---:|"


def _row(row: dict) -> str:
    params = ", ".join(f"{key}={value:.6g}" for key, value in sorted(row["params"].items())) or "—"
    formula = str(row["formula"]).replace("|", "\\|")
    expanded = str(row["expanded_formula"]).replace("|", "\\|")
    return f"| {row['gap_pct']:.4f}% | {row['policy_cost']:.6f} | {row['num_parameters']} | {row['expanded_complexity']:.1f} | `{formula}` | `{expanded}` | {params} | {row['status']}/{row['route']} | {row['optimizer_seconds']:.2f}s |"


def _atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
