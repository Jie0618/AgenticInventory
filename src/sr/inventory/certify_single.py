#!/usr/bin/env python3
"""Certify a frozen Standard SR frontier and compare baselines."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

from inventory.policy.ast import ast_check, ast_to_formula
from inventory.core import InventoryCase
from evaluation.cache import load_or_create_inventory_dp
from optimization import MonteCarloProposalConfig, OptimizationConfig, InventorySearchContext
from inventory.policy.motifs import motifs_for_config
from sr.analysis.certification import certification_seed, select_profile_candidates
from sr.inventory.single import diagnostic_samples
from runtime.artifacts import load_result_records


HORIZON = 1000


def certify_seed(case_name: str, ast_hash: str) -> int:
    """Derive an order-independent seed for one frozen Certify fit."""
    return certification_seed(20260812, "certify", case_name, ast_hash)


def main(
    available_cases: Sequence[InventoryCase],
    *,
    default_diagnostic_root: Path,
    default_baseline_results: Path,
    default_cyclic_evaluation_config: Path | None = None,
) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--search-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dp-cache-dir", type=Path, required=True)
    parser.add_argument("--reference-suite", default="gap200")
    parser.add_argument(
        "--diagnostic-root",
        type=Path,
        default=default_diagnostic_root,
    )
    parser.add_argument(
        "--baseline-results",
        type=Path,
        default=default_baseline_results,
    )
    parser.add_argument("--max-per-frontier", type=int, default=8)
    parser.add_argument("--cyclic-evaluation-config", type=Path, default=default_cyclic_evaluation_config)
    args = parser.parse_args()

    case = next(row for row in available_cases if row.name == args.case)
    checkpoint = json.loads((args.search_dir / "checkpoint.json").read_text(encoding="utf-8"))
    selected = select_profile_candidates(
        checkpoint, max_per_profile=args.max_per_frontier
    )
    dp_result = load_or_create_inventory_dp(
        case, HORIZON, suite=args.reference_suite, cache_dir=args.dp_cache_dir
    )
    motifs = motifs_for_config(case.config)
    for row in checkpoint.get("reflection_motifs", []):
        motifs[str(row["name"])] = dict(row["ast"])
    samples = diagnostic_samples(args.diagnostic_root / f"{case.name}.json", HORIZON)
    context = InventorySearchContext.from_reference(case, HORIZON, dp_result, samples=samples)
    if checkpoint.get("config", {}).get("cyclic_cap", False):
        if args.cyclic_evaluation_config is None:
            parser.error("cyclic certification requires --cyclic-evaluation-config")
        from sr.inventory.cyclic import CyclicLeadTimeSearchContext
        context = CyclicLeadTimeSearchContext.from_inventory(
            context, json.loads(args.cyclic_evaluation_config.read_text()))
    dp_cost = float(dp_result.value(0, case.initial_state))
    results = []
    for row in selected:
        checked = ast_check(
            row["ast"], motifs=motifs, allowed_state_names=case.schema.allowed_state_names
        )
        started = time.perf_counter()
        fit = context.fit(
            checked.ast,
            motifs=motifs,
            config=OptimizationConfig(
                mode="certify",
                objective="policy_cost",
                random_seed=certify_seed(case.name, checked.ast_hash),
                mc_proposal=MonteCarloProposalConfig(enabled=True, routing_mode="auto"),
            ),
            warm_starts=(),
        )
        results.append(
            {
                "ast_hash": checked.ast_hash,
                "ast": checked.ast,
                "formula": ast_to_formula(checked.ast),
                "standard_gap_pct": row["gap_pct"],
                "certify_cost": float(fit.policy_cost),
                "certify_gap_pct": 100.0 * (float(fit.policy_cost) / dp_cost - 1.0),
                "params": dict(fit.params),
                "status": fit.status,
                "route": fit.route,
                "optimizer_seconds": float(fit.elapsed_seconds),
                "wall_seconds": time.perf_counter() - started,
                "independent_warm_starts": 0,
            }
        )
    baselines = []
    for row in load_result_records(args.baseline_results):
        if str(row.get("name")) != case.name:
            continue
        cost = float(row["cost"])
        baseline_dp = float(row.get("dp_cost", dp_cost))
        baselines.append(
            {
                "policy": row.get("policy"),
                "cost": cost,
                "gap_pct": 100.0 * (cost / baseline_dp - 1.0),
                "params": row.get("params"),
                "status": row.get("status"),
            }
        )
    payload = {
        "schema_version": 1,
        "case": case.name,
        "dp_cost": dp_cost,
        "certify_results": sorted(results, key=lambda row: (row["certify_gap_pct"], row["ast_hash"])),
        "seven_baselines": sorted(baselines, key=lambda row: float(row["gap_pct"])),
    }
    if len(baselines) != 7:
        raise RuntimeError(f"expected seven baseline records for {case.name}, found {len(baselines)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    _write_report(payload, args.output.with_suffix(".md"))
    print(json.dumps({"case": case.name, "finalists": len(results), "best_gap_pct": min(row["certify_gap_pct"] for row in results)}, sort_keys=True))


def _write_report(payload: dict, path: Path) -> None:
    lines = [
        f"# Final Certify audit: `{payload['case']}`",
        "",
        f"DP cost: {payload['dp_cost']:.6f}",
        "",
        "## Independently refitted SR finalists",
        "",
        "| Certify gap | Certify cost | Standard gap | Formula | Constants | Status | Time |",
        "|---:|---:|---:|---|---|---|---:|",
    ]
    for row in payload["certify_results"]:
        params = ", ".join(f"{key}={value:.6g}" for key, value in sorted(row["params"].items())) or "—"
        formula = row["formula"].replace("|", "\\|")
        lines.append(f"| {row['certify_gap_pct']:.4f}% | {row['certify_cost']:.6f} | {row['standard_gap_pct']:.4f}% | `{formula}` | {params} | {row['status']}/{row['route']} | {row['optimizer_seconds']:.2f}s |")
    lines.extend(["", "## Existing seven-family Certify benchmark", "", "| Policy | Gap | Cost |", "|---|---:|---:|"])
    for row in payload["seven_baselines"]:
        lines.append(f"| {row['policy']} | {float(row['gap_pct']):.4f}% | {float(row['cost']):.6f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
