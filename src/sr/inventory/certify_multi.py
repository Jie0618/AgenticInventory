#!/usr/bin/env python3
"""Certify every case for a frozen multi-case frontier."""

from __future__ import annotations

from runtime.artifacts import seed_from_text, stable_key
import argparse
import json
import os
import time
from pathlib import Path
from typing import Sequence

os.environ.update(
    {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
    }
)

from inventory.core import InventoryCase
from inventory.policy.ast import ast_check, ast_to_formula
from optimization import MonteCarloProposalConfig, OptimizationConfig
from sr.multi.models import MultiObjective

from sr.inventory.runtime import MultiCliDefaults, load_runtime
from runtime.artifacts import load_result_records


def main(
    available_cases: Sequence[InventoryCase],
    *,
    defaults: MultiCliDefaults,
) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=defaults.experiment_config,
    )
    parser.add_argument(
        "--search-config",
        type=Path,
        default=defaults.search_config,
    )
    parser.add_argument("--checkpoint-root", type=Path, default=defaults.checkpoint_root)
    parser.add_argument("--cyclic-evaluation-config", type=Path, default=defaults.cyclic_evaluation_config)
    parser.add_argument(
        "--diagnostic-root",
        type=Path,
        default=defaults.diagnostic_root,
    )
    parser.add_argument("--dp-cache-dir", type=Path, required=defaults.reference_suite != "stage1-reference", default=defaults.checkpoint_root / ".dp_cache")
    parser.add_argument("--reference-suite", default=defaults.reference_suite)
    parser.add_argument("--baseline-results", type=Path, default=None)
    args = parser.parse_args()

    checkpoint_path = args.search_dir / "checkpoint.json"
    checkpoint_bytes = checkpoint_path.read_bytes()
    checkpoint = json.loads(checkpoint_bytes)
    if checkpoint.get("mode") != "multi_distribution_standard_search":
        raise ValueError(f"not a multi-distribution search checkpoint: {checkpoint_path}")
    runtime = load_runtime(
        available_cases=available_cases,
        experiment_config=args.experiment_config,
        search_config=args.search_config,
        checkpoint_root=args.checkpoint_root,
        diagnostic_root=args.diagnostic_root,
        dp_cache_dir=args.dp_cache_dir,
        reference_suite=args.reference_suite,
    )
    if runtime.search_config.cyclic_cap:
        if args.cyclic_evaluation_config is None:
            parser.error("cyclic certification requires --cyclic-evaluation-config")
        from sr.inventory.cyclic import bind_cyclic_runtime
        runtime = bind_cyclic_runtime(runtime, json.loads(args.cyclic_evaluation_config.read_text()))
    if tuple(checkpoint.get("case_names", ())) != runtime.case_names:
        raise ValueError("checkpoint training cases do not match experiment config")
    objective = MultiObjective(**dict(checkpoint.get("objective", {})))
    if objective != runtime.objective:
        raise ValueError("checkpoint objective does not match experiment config")
    motifs = {str(name): dict(ast) for name, ast in dict(checkpoint["motifs"]).items()}
    finalists = _frontier_union(checkpoint)
    if not finalists:
        raise RuntimeError("checkpoint contains no frontier finalists")

    checkpoint_source = str(args.checkpoint.resolve())
    pair_dir = args.output_dir / "pair_results"
    pair_dir.mkdir(parents=True, exist_ok=True)
    aggregate_rows: list[dict[str, object]] = []
    total_pairs = len(finalists) * len(runtime.case_names)
    completed_pairs = 0
    started_all = time.perf_counter()
    for finalist in finalists:
        checked = ast_check(dict(finalist["ast"]), motifs=motifs)
        case_rows: dict[str, dict[str, object]] = {}
        for case_name in runtime.case_names:
            pair_path = pair_dir / checked.ast_hash / f"{case_name}.json"
            signature = _certify_signature(
                checkpoint_source=checkpoint_source,
                runtime_signature=runtime.initialization_signature,
                ast_hash=checked.ast_hash,
                case_name=case_name,
            )
            cached = _load_pair(pair_path, signature)
            if cached is None:
                case = runtime.case_inputs[case_name]
                context = case.context
                if context is None:
                    raise ValueError(f"missing search context for {case_name}")
                seed = _certify_seed(checked.ast_hash, case_name)
                fit_started = time.perf_counter()
                fit = context.fit(
                    checked.ast,
                    motifs=motifs,
                    config=OptimizationConfig(
                        mode="certify",
                        objective="policy_cost",
                        random_seed=seed,
                        mc_proposal=MonteCarloProposalConfig(enabled=True, routing_mode="auto"),
                    ),
                    warm_starts=(),
                )
                dp_cost = (float(context.reference_cost) if hasattr(context, "reference_cost")
                           else float(context.reference.value(0, context.case.initial_state)))
                cost = float(fit.policy_cost)
                cached = {
                    "schema_version": 1,
                    "mode": "multi_independent_certify_pair",
                    "signature": signature,
                    "ast_hash": checked.ast_hash,
                    "case": case_name,
                    "policy_cost": cost,
                    "dp_cost": dp_cost,
                    "gap_pct": 100.0 * (cost / dp_cost - 1.0),
                    "params": dict(fit.params),
                    "status": fit.status,
                    "route": fit.route,
                    "optimizer_seconds": float(fit.elapsed_seconds),
                    "wall_seconds": time.perf_counter() - fit_started,
                    "random_seed": seed,
                    "independent_warm_starts": 0,
                }
                _atomic_json(cached, pair_path)
            case_rows[case_name] = cached
            completed_pairs += 1
            print(
                json.dumps(
                    {
                        "progress": "multi_certify",
                        "completed_pairs": completed_pairs,
                        "total_pairs": total_pairs,
                        "ast_hash": checked.ast_hash,
                        "case": case_name,
                        "gap_pct": cached["gap_pct"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        gaps = tuple(float(case_rows[name]["gap_pct"]) for name in runtime.case_names)
        mean, mean_top2, score, top_indices = objective.aggregate(gaps)
        aggregate_rows.append(
            {
                "ast_hash": checked.ast_hash,
                "ast": checked.ast,
                "formula": ast_to_formula(checked.ast),
                "num_parameters": len(checked.stats.parameters),
                "expanded_complexity": float(finalist["expanded_complexity"]),
                "standard_score": float(finalist["score"]),
                "certify_score": score,
                "certify_mean_gap_pct": mean,
                "certify_mean_top2_gap_pct": mean_top2,
                "certify_worst_gap_pct": gaps[top_indices[0]],
                "certify_worst_case": runtime.case_names[top_indices[0]],
                "certify_top2_cases": [runtime.case_names[index] for index in top_indices],
                "case_results": case_rows,
            }
        )

    baseline_rows = _load_baselines(args.baseline_results, runtime.case_names, objective)
    payload = {
        "schema_version": 1,
        "mode": "multi_distribution_frontier_certify",
        "case_names": list(runtime.case_names),
        "objective": objective.to_payload(),
        "source_checkpoint": checkpoint_source,
        "elapsed_seconds": time.perf_counter() - started_all,
        "certify_results": sorted(aggregate_rows, key=lambda row: (row["certify_score"], row["ast_hash"])),
        "seven_baselines": baseline_rows,
    }
    _atomic_json(payload, args.output_dir / "RESULTS.json")
    _write_report(payload, args.output_dir / "REPORT.md")


def _frontier_union(checkpoint: dict) -> list[dict]:
    selected: dict[str, dict] = {}
    for name in ("parameter_frontier", "complexity_frontier"):
        for row in checkpoint.get(name, ()):
            selected.setdefault(str(row["ast_hash"]), dict(row))
    return sorted(
        selected.values(),
        key=lambda row: (float(row["score"]), int(row["num_parameters"]), float(row["expanded_complexity"]), str(row["ast_hash"])),
    )


def _certify_seed(ast_hash: str, case_name: str) -> int:
    encoded = f"20260818|multi-certify|{ast_hash}|{case_name}".encode("utf-8")
    return seed_from_text(encoded)


def _certify_signature(*, checkpoint_source: str, runtime_signature: str, ast_hash: str, case_name: str) -> str:
    payload = {
        "protocol": "multi-independent-certify",
        "source_checkpoint": checkpoint_source,
        "runtime_signature": runtime_signature,
        "ast_hash": ast_hash,
        "case": case_name,
        "seed": _certify_seed(ast_hash, case_name),
    }
    return stable_key(json.dumps(payload, sort_keys=True).encode("utf-8"))


def _load_pair(path: Path, signature: str) -> dict[str, object] | None:
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("mode") != "multi_independent_certify_pair" or row.get("signature") != signature:
            return None
        if int(row.get("independent_warm_starts", -1)) != 0:
            return None
        return row
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _load_baselines(path: Path | None, case_names: tuple[str, ...], objective: MultiObjective) -> list[dict[str, object]]:
    if path is None:
        return []
    by_policy: dict[str, dict[str, float]] = {}
    available = load_result_records(path)
    for case_name in case_names:
        rows = [row for row in available if str(row.get("name")) == case_name]
        if len(rows) != 7:
            raise RuntimeError(f"expected seven baseline records for {case_name}, found {len(rows)}")
        for row in rows:
            cost = float(row["cost"])
            dp_cost = float(row["dp_cost"])
            by_policy.setdefault(str(row["policy"]), {})[case_name] = 100.0 * (cost / dp_cost - 1.0)
    output = []
    for policy, case_gaps in sorted(by_policy.items()):
        if set(case_gaps) != set(case_names):
            raise RuntimeError(f"incomplete baseline {policy}")
        gaps = tuple(case_gaps[name] for name in case_names)
        mean, top2, score, top_indices = objective.aggregate(gaps)
        output.append(
            {
                "policy": policy,
                "score": score,
                "mean_gap_pct": mean,
                "mean_top2_gap_pct": top2,
                "worst_gap_pct": gaps[top_indices[0]],
                "worst_case": case_names[top_indices[0]],
                "case_gaps_pct": case_gaps,
            }
        )
    return sorted(output, key=lambda row: (row["score"], row["policy"]))


def _write_report(payload: dict, path: Path) -> None:
    lines = [
        "# Multi-distribution final Certify audit",
        "",
        "每个结构在八个环境上分别从零拟合常数；不继承 Standard 常数，也不跨环境共享常数。",
        "",
        "| Certify score | Mean gap | CVaR80 | Worst gap | Standard score | Params | Complexity | Formula |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["certify_results"]:
        formula = str(row["formula"]).replace("|", "\\|")
        lines.append(
            f"| {row['certify_score']:.4f}% | {row['certify_mean_gap_pct']:.4f}% | "
            f"{row['certify_mean_top2_gap_pct']:.4f}% | {row['certify_worst_gap_pct']:.4f}% "
            f"(`{row['certify_worst_case']}`) | {row['standard_score']:.4f}% | "
            f"{row['num_parameters']} | {row['expanded_complexity']:.1f} | `{formula}` |"
        )
    if payload["seven_baselines"]:
        lines.extend(("", "## Seven-baseline aggregate comparison", "", "| Baseline | Score | Mean gap | CVaR80 | Worst gap |", "|---|---:|---:|---:|---:|"))
        for row in payload["seven_baselines"]:
            lines.append(
                f"| {row['policy']} | {row['score']:.4f}% | {row['mean_gap_pct']:.4f}% | "
                f"{row['mean_top2_gap_pct']:.4f}% | {row['worst_gap_pct']:.4f}% (`{row['worst_case']}`) |"
            )
    lines.extend(("", "## Per-case Certify gaps", ""))
    cases = tuple(payload["case_names"])
    lines.append("| AST | " + " | ".join(cases) + " |")
    lines.append("|---|" + "---:|" * len(cases))
    for row in payload["certify_results"]:
        values = " | ".join(f"{row['case_results'][name]['gap_pct']:.4f}%" for name in cases)
        lines.append(f"| `{row['ast_hash'][:12]}` | {values} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    main()
