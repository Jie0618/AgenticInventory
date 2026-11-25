#!/usr/bin/env python3
"""Build hierarchical mean–CVaR80 reports for the empirical-MC benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from importlib import import_module
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Mapping, Sequence
from benchmarks.inventory_benchmark_family.suites import profiles_for
from inventory.common.fitness import SCORE_PROTOCOL

try:
    from .policy_display import display_name
    from .mean_cvar_fitness import configuration_scores
    from .result_tables import write_csv
except ImportError:  # Direct script execution.
    from policy_display import display_name
    from mean_cvar_fitness import configuration_scores
    from result_tables import write_csv


def write_partial_results(path, rows):
    """Keep nested worker metadata as JSON in partial comparison tables."""
    temporary = path.with_suffix(".csv.tmp")
    write_csv(temporary, [{k: json.dumps(v, sort_keys=True) if isinstance(v, dict) else v
                           for k, v in row.items()} for row in rows])
    temporary.replace(path)


def write_joint_report(output, rows, selected, *, model, method_ids):
    """Require the full case-policy roster before scoring joint settings."""
    if not method_ids or len(set(method_ids)) != len(method_ids):
        raise ValueError("Unsupported comparison membership")
    by_case = defaultdict(dict)
    for row in rows:
        if row["policy"] in by_case[row["case"]]:
            raise ValueError("Duplicate case-policy result")
        by_case[row["case"]][row["policy"]] = row
    if set(by_case) != {case.name for case in selected} or any(
            set(methods) != set(method_ids) for methods in by_case.values()):
        raise ValueError("Refusing to report an incomplete comparison")
    backlog = model.SETTING == "lead_time_backlog"
    for methods in by_case.values():
        # Every method in a case must use the same final bank and path count.
        banks = {(row["bank_seeds"]["final"],
                  row["evaluation_paths"]) for row in methods.values()}
        if len(banks) != 1:
            raise ValueError("Final demand banks differ between methods")
    from reporting.scores import write_scores
    records = [methods[pid] for _, methods in sorted(by_case.items()) for pid in method_ids]
    return write_scores(output, records, "K" if backlog else "M", report_note=(
        "SQ, QR and GBS are published rules transferred from related models; "
        "not native fixed-patience methods.") if backlog else "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", choices=("lead_time", "perishable", "lead_time_backlog", "lead_time_perishable"))
    parser.add_argument("--model-module")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.setting in ("lead_time_backlog", "lead_time_perishable"):
        protocol = json.loads((args.run_dir / "MC_PROTOCOL.json").read_text())
        module = args.model_module or protocol.get("model_module")
        if not module:
            parser.error("--model-module is required when the protocol has no adapter")
        model = import_module(module)
        if model.SETTING != args.setting:
            raise ValueError("MC model does not match the requested setting")
        cases = [case for case in model.cases(protocol["settings"])
                 if case.name in protocol["case_signatures"]]
        if {case.name: case.case_signature for case in cases} != protocol["case_signatures"]:
            raise ValueError("Protocol cases differ from the current suite")
        method_ids = model.BASELINES + tuple(
            p["id"] for p in protocol["frozen_selection"]["policies"] if p.get("family"))
        rows = [json.loads(p.read_text()) for p in (args.run_dir / "results").glob("*.json")]
        by_case = {case.name: case for case in cases}
        axis = "K" if args.setting == "lead_time_backlog" else "M"
        for row in rows:
            case = by_case[row["case"]]
            value = case.config.backlog.patience if axis == "K" else case.shelf_life
            policy = next(p for p in protocol["resolved_policies"][f"L{case.lead_time}_{axis}{value}"]
                          if p["id"] == row["policy"])
            policy = model.resolve_case_policy(case, policy)
            model.validate_cached(row, case, policy, protocol["settings"])
        write_joint_report(args.run_dir, rows, cases, model=model, method_ids=method_ids)
        return
    if args.model_module is None:
        parser.error("--model-module is required for L/P reports")
    from benchmarks.inventory_benchmark_family.mc_protocol import report_model
    run_dir = args.run_dir.resolve()
    manifest = json.loads((run_dir / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    model = report_model(str(manifest["suite"]), args.setting, model_module=args.model_module)
    context = model.context_field
    instance_count = int(manifest.get("instance_count", manifest["task_count"]))
    existing_summary = (
        json.loads((run_dir / "SUMMARY.json").read_text(encoding="utf-8"))
        if (run_dir / "SUMMARY.json").is_file()
        else {}
    )
    instance_payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((run_dir / "results").glob("*.json"))
    ]
    if len(instance_payloads) == instance_count:
        if {row["instance_id"] for row in instance_payloads} != set(manifest["instances"]):
            raise RuntimeError("result instances do not match the campaign manifest")
        if any(row["suite"] != manifest["suite"] or [policy["policy_id"] for policy in row["policies"]] != manifest["policies"] for row in instance_payloads):
            raise RuntimeError("result suite or policy identities do not match the campaign manifest")
        rows = _flatten(instance_payloads, context, model.extra_case_fields)
    elif not instance_payloads and (run_dir / "RESULTS.csv").is_file():
        rows = _read_results(run_dir / "RESULTS.csv")
    else:
        raise RuntimeError(
            f"expected {instance_count} instance results, found {len(instance_payloads)}"
        )
    expected_rows = instance_count * len(manifest["policies"])
    if len(rows) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} policy rows, found {len(rows)}")
    valid_rows = _valid_measurement_rows(rows)
    reference_path = run_dir / "REFERENCE_COSTS.json"
    if reference_path.exists():
        _apply_frozen_reference(valid_rows, json.loads(reference_path.read_text())["costs"])
    context_scores, overall_scores = configuration_scores(valid_rows, context)
    _write_results(run_dir / "RESULTS.csv", rows, context, model.extra_case_fields)
    write_csv(run_dir / "SUMMARY.csv", context_scores)
    write_csv(run_dir / "OVERALL.csv", overall_scores)
    skipped_rows = [row for row in rows if row.get("status") == "skipped_timeout"]
    tle_rows = [row for row in rows if row.get("status") == "TLE"]
    instance_best = _instance_best(valid_rows, context)
    _write_instance_best(run_dir / "INSTANCE_BEST.csv", instance_best, context)
    summary = {
        "schema_version": 2,
        "suite": manifest["suite"],
        "reference_policy": ("all_formal_policies" if reference_path.exists()
                             else manifest.get("reference_policy", "best_baseline_per_instance")),
        "instances": instance_count,
        "policy_rows": len(rows),
        "by_policy": _group(valid_rows, lambda row: str(row["policy_id"])),
        "fitness_protocol": {
            "protocol": SCORE_PROTOCOL,
            "within_configuration": "0.60*mean_gap + 0.40*CVaR80_gap",
            "across_configurations": "0.60*mean_fitness + 0.40*CVaR80_fitness",
            "tail_boundary": "fractional_probability_mass",
            "configuration_weight": "equal; shared measured configurations",
            "negative_gaps": "preserved",
        },
        "robust_fitness_by_policy": {
            row["policy_id"]: {key: value for key, value in row.items() if key != "policy_id"}
            for row in overall_scores
        },
        "by_configuration": context_scores,
        "overall": overall_scores,
        f"by_policy_and_{context}": _group(
            valid_rows, lambda row: f"{row['policy_id']}|{model.context_prefix}{row[context]}"
        ),
        f"task_runtime_by_{context}": _task_runtime_by_context(
            instance_payloads,
            rows,
            existing_summary.get(f"task_runtime_by_{context}", {}),
            context,
        ),
        "instance_best": instance_best,
    }
    if skipped_rows:
        summary["skipped_timeout_rows"] = len(skipped_rows)
        summary["skipped_timeout_by_policy"] = dict(Counter(str(row["policy_id"]) for row in skipped_rows))
    if tle_rows:
        summary["tle_rows"] = len(tle_rows)
        summary["tle_by_policy"] = dict(Counter(str(row["policy_id"]) for row in tle_rows))
    (run_dir / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run_dir / "REPORT.md").write_text(
        _render_report(manifest, summary, rows, context, model.context_prefix, model.contexts_key, len(profiles_for(model.setting))), encoding="utf-8"
    )
    print(
        json.dumps(
            {"event": "mc_benchmark_report", "rows": len(rows), "run_dir": str(run_dir)},
            sort_keys=True,
        ),
        flush=True,
    )


def _valid_measurement_rows(rows):
    return [row for row in rows if row.get("evaluation_cost") not in (None, "")
            and math.isfinite(float(row["evaluation_cost"]))
            and row.get("status") not in {"TLE", "skipped_timeout", "failed", "error", "pending"}]


def _apply_frozen_reference(rows, references):
    """Retain the adopted formal comparison when rebuilding worker reports."""
    for row in rows:
        reference = float(references[row["instance_id"]])
        cost = float(row["evaluation_cost"])
        if not math.isfinite(reference) or reference <= 0 or cost < reference:
            raise ValueError("Frozen formal reference must be positive and no greater than measured cost")
        if float(row["reference_cost"]) != reference:
            # Pathwise uncertainty against a different policy cannot be
            # reconstructed from scalar archived measurements.
            for key in ("relative_gap_se_pct", "paired_cost_difference", "paired_cost_difference_se"):
                row[key] = None
        row.update(reference_cost=reference, reference_policy_id="all_formal_policies",
                   relative_gap_pct=100 * (cost / reference - 1))


def _flatten(payloads: Sequence[Mapping[str, object]], context="lead_time", extra_fields=()) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    shared_fields = (
        "instance_id",
        context,
        "distribution_id",
        "distribution_family",
        "h",
        "b",
        "b_over_h",
        "n",
        "horizon",
        "tuning_paths",
        "evaluation_paths",
        "tuning_seed",
        "evaluation_seed",
        "total_task_seconds",
    )
    shared_fields += tuple(extra_fields)
    for payload in payloads:
        for policy in payload["policies"]:  # type: ignore[index]
            row = {field: payload[field] for field in shared_fields}
            row.update({key: value for key, value in payload.items() if key != "policies"})
            row.update(dict(policy))
            output.append(row)
    return output


def _read_results(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, float | int]:
    gaps = sorted(float(row["relative_gap_pct"]) for row in rows)
    costs = [float(row["evaluation_cost"]) for row in rows]
    standard_errors = [
        float(row["relative_gap_se_pct"])
        for row in rows
        if row.get("relative_gap_se_pct") not in (None, "")
    ]
    times = [float(row["policy_wall_seconds"]) for row in rows]
    return {
        "count": len(rows),
        "mean_relative_gap_pct": statistics.fmean(gaps),
        "median_relative_gap_pct": statistics.median(gaps),
        "p90_relative_gap_pct": _quantile(gaps, 0.90),
        "p95_relative_gap_pct": _quantile(gaps, 0.95),
        "best_relative_gap_pct": min(gaps),
        "worst_relative_gap_pct": max(gaps),
        "prob_cost_below_reference": sum(value < -1.0e-12 for value in gaps) / len(gaps),
        "mean_evaluation_cost": statistics.fmean(costs),
        "mean_gap_se_pct": (
            statistics.fmean(standard_errors)
            if len(standard_errors) == len(rows)
            else None
        ),
        "gap_se_count": len(standard_errors),
        "mean_policy_seconds": statistics.fmean(times),
        "max_policy_seconds": max(times),
    }


def _group(
    rows: Sequence[Mapping[str, object]], key: Callable[[Mapping[str, object]], str]
) -> dict[str, object]:
    groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    return {name: _metrics(group) for name, group in sorted(groups.items())}


def _instance_best(rows: Sequence[Mapping[str, object]], context="lead_time") -> list[dict[str, object]]:
    groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["instance_id"])].append(row)
    output = []
    for instance_id, group in sorted(groups.items()):
        best = min(
            group,
            key=lambda row: (float(row["evaluation_cost"]), str(row["policy_id"])),
        )
        output.append(
            {
                "instance_id": instance_id,
                context: int(best[context]),
                "distribution_family": best["distribution_family"],
                "b_over_h": float(best["b_over_h"]),
                "best_policy": best["policy_id"],
                "best_evaluation_cost": float(best["evaluation_cost"]),
                "best_relative_gap_pct": float(best["relative_gap_pct"]),
                "reference_policy_id": best["reference_policy_id"],
                "reference_cost": float(best["reference_cost"]),
            }
        )
    return output


def _task_runtime_by_context(
    payloads: Sequence[Mapping[str, object]],
    rows: Sequence[Mapping[str, object]],
    existing: object,
    context="lead_time",
) -> dict[str, dict[str, float | int]]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for payload in payloads:
        grouped[int(payload[context])].append(float(payload["total_task_seconds"]))
    if not payloads:
        rows_with_runtime = [row for row in rows if row.get("total_task_seconds") not in (None, "")]
        if rows_with_runtime:
            per_instance = {
                str(row["instance_id"]): (
                    int(row[context]),
                    float(row["total_task_seconds"]),
                )
                for row in rows_with_runtime
            }
            for lead_time, seconds in per_instance.values():
                grouped[lead_time].append(seconds)
        elif isinstance(existing, dict):
            return dict(existing)
    return {
        str(lead_time): {
            "count": len(values),
            "mean_seconds": statistics.fmean(values),
            "median_seconds": statistics.median(values),
            "p90_seconds": _quantile(sorted(values), 0.90),
            "max_seconds": max(values),
        }
        for lead_time, values in sorted(grouped.items())
    }


def _write_results(path: Path, rows: Sequence[Mapping[str, object]], context="lead_time", extra_fields=()) -> None:
    fields = (
        "instance_id",
        "policy_id",
        "policy_type",
        context,
        "distribution_id",
        "distribution_family",
        "h",
        "b",
        "b_over_h",
        "n",
        "horizon",
        "tuning_paths",
        "evaluation_paths",
        "evaluation_cost",
        "evaluation_se",
        "reference_policy_id",
        "reference_cost",
        "relative_gap_pct",
        "relative_gap_se_pct",
        "paired_cost_difference",
        "paired_cost_difference_se",
        "tuning_cost",
        "tuning_se",
        "status",
        "route",
        "candidate_count",
        "optimizer_seconds",
        "policy_wall_seconds",
        "constants",
    )
    fields += tuple(extra_fields)
    fields = tuple(dict.fromkeys((*fields, *(key for row in rows for key in row))))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in sorted(
            rows,
            key=lambda value: (
                str(value["policy_id"]),
                int(value[context]),
                str(value["instance_id"]),
            ),
        ):
            output = {field: row.get(field) for field in fields}
            constants = row.get("constants", {})
            if isinstance(constants, str):
                try:
                    constants = json.loads(constants)
                except json.JSONDecodeError:
                    pass
            output["constants"] = json.dumps(constants, sort_keys=True, separators=(",", ":"))
            for key, value in output.items():
                if isinstance(value, (dict, list, tuple)):
                    output[key] = json.dumps(value, sort_keys=True, separators=(",", ":"))
            writer.writerow(output)


def _write_instance_best(path: Path, rows: Sequence[Mapping[str, object]], context="lead_time") -> None:
    fields = (
        "instance_id",
        context,
        "distribution_family",
        "b_over_h",
        "best_policy",
        "best_evaluation_cost",
        "best_relative_gap_pct",
        "reference_policy_id",
        "reference_cost",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _render_report(
    manifest: Mapping[str, object],
    summary: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    context="lead_time", prefix="L", contexts_key="lead_times", profile_count=4,
) -> str:
    by_policy = summary["by_policy"]
    formulas = manifest.get("policy_formulas", {})
    all_policies = summary["reference_policy"] in {
        "all_formal_policies", "best_all_policies_per_instance"}
    reference = ("the frozen adopted formal-policy reference in REFERENCE_COSTS.json"
                 if summary["reference_policy"] == "all_formal_policies"
                 else "the lowest independent evaluation cost across all policies in this run"
                 if all_policies else "the lowest independent evaluation cost across eligible baselines")
    lines = [
        f"# Inventory Benchmark 50×{profile_count} — Empirical MC",
        "",
        f"- {contexts_key.replace('_', ' ').capitalize()}: {', '.join(prefix+'='+str(value) for value in manifest[contexts_key])}",
        f"- Instances: {len(manifest['instances']):,}",
        f"- Policies: {len(manifest['policies']):,}",
        f"- Tuning trajectories per instance: {manifest['tuning_paths']:,}",
        f"- Independent evaluation trajectories per instance: {manifest['evaluation_paths']:,}",
        "- Common random numbers: shared across all policies within each instance",
        f"- Reference: {reference}",
        "- Reference selection bank: the same evaluation bank used for the reported gap",
        "",
        "> Relative gaps use an empirical comparison reference, not a certified optimum.",
        "",
        "Fitness = `0.60 Mean + 0.40 CVaR80` with fractional tail-boundary mass. "
        "Compute it over instance gaps within each configuration, then over configuration "
        "fitness with equal weights on configurations measured by every reported policy. "
        "Negative gaps are preserved. Missing/TLE measurements are excluded, not imputed.",
        "",
        "## Overall summary",
        "",
        "| Policy | Formula | Fitness | Mean context fitness | CVaR80 context fitness | Mean instance gap | Median | P90 | P95 | Worst | Mean time | Max time |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in manifest["policies"]:
        if policy not in by_policy:
            lines.append(f"| `{policy}` | No completed evaluations (timeout/TLE) |")
            continue
        metric = by_policy[policy]
        fitness = summary["robust_fitness_by_policy"][policy]
        formula_map = dict(formulas.get(policy, {})) if isinstance(formulas, dict) else {}
        unique_formulas = list(dict.fromkeys(str(value) for value in formula_map.values()))
        formula = unique_formulas[0] if len(unique_formulas) == 1 else f"{context.replace('_', '-')} dependent"
        lines.append(
            "| {label} (`{policy}`) | `{formula}` | {fitness:.4f} | {context_mean:.4f} | {context_cvar:.4f} | "
                "{mean:.4f}% | {median:.4f}% | {p90:.4f}% | {p95:.4f}% | {worst:.4f}% | {mean_time:.2f}s | {max_time:.2f}s |".format(
                policy=policy,
                label=display_name(str(policy), manifest),
                formula=formula.replace("|", "\\|"),
                fitness=fitness["fitness"],
                mean=metric["mean_relative_gap_pct"],
                median=metric["median_relative_gap_pct"],
                p90=metric["p90_relative_gap_pct"],
                p95=metric["p95_relative_gap_pct"],
                context_mean=fitness["mean_gap_pct"],
                context_cvar=fitness["cvar80_gap_pct"],
                worst=metric["worst_relative_gap_pct"],
                mean_time=metric["mean_policy_seconds"],
                max_time=metric["max_policy_seconds"],
            )
        )
    lines.extend(
        [
            "",
            f"## By {context.replace('_', ' ')}",
            "",
            f"| Policy | {prefix} | N | Fitness | Mean gap | CVaR80 gap | P90 | Worst |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    by_policy_l = summary[f"by_policy_and_{context}"]
    context_scores = {(row["policy_id"], row[context]): row for row in summary["by_configuration"]}
    n_by_l = {int(row[context]): int(row["n"]) for row in rows}
    for policy in manifest["policies"]:
        for lead_time in manifest[contexts_key]:
            if f"{policy}|{prefix}{lead_time}" not in by_policy_l:
                lines.append(f"| `{policy}` | {lead_time} | {n_by_l[int(lead_time)]} | N/A (timeout/TLE) | N/A | N/A | N/A | N/A |")
                continue
            metric = by_policy_l[f"{policy}|{prefix}{lead_time}"]
            fitness = context_scores[policy, int(lead_time)]
            lines.append(
                f"| {display_name(str(policy), manifest)} (`{policy}`) | {lead_time} | {n_by_l[int(lead_time)]} | "
                f"{fitness['fitness']:.4f} | "
                f"{metric['mean_relative_gap_pct']:.4f}% | "
                f"{fitness['cvar80_gap_pct']:.4f}% | "
                f"{metric['p90_relative_gap_pct']:.4f}% | "
                f"{metric['worst_relative_gap_pct']:.4f}% |"
            )
    if summary.get("skipped_timeout_rows"):
        lines.extend(["", f"Timeout skips: {summary['skipped_timeout_rows']} policy rows. Cost/gap statistics exclude these rows; policy sample counts may differ.", ""])
    if summary.get("tle_rows"):
        lines.extend(["", f"TLE: {summary['tle_rows']} policy rows skipped by user request without running. Cost/gap statistics exclude these rows; policy sample counts may differ.", ""])
    lines.extend(
        [
            "",
            f"## Worker-task runtime by {context.replace('_', ' ')}",
            "",
            f"| {prefix} | Tasks | Mean | Median | P90 | Max |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    task_runtime = summary[f"task_runtime_by_{context}"]
    for lead_time in manifest[contexts_key]:
        metric = task_runtime.get(str(lead_time))
        if metric is None:
            lines.append(f"| {lead_time} | N/A | N/A | N/A | N/A | N/A |")
            continue
        lines.append(
            f"| {lead_time} | {metric['count']} | {metric['mean_seconds']:.2f}s | "
            f"{metric['median_seconds']:.2f}s | {metric['p90_seconds']:.2f}s | "
            f"{metric['max_seconds']:.2f}s |"
        )
    lines.extend(
        [
            "",
            "## Methodological interpretation",
            "",
            "- Constants are selected on the tuning trajectory bank; reported costs use a separate evaluation bank.",
            "- Demand paths and randomized-policy uniforms are deterministic and shared across policies, enabling paired comparisons.",
            "- Selecting a lower envelope on the reporting bank introduces a best-of-noise advantage; it is an empirical comparison.",
            "- No DP, Exact evaluator, stationary extrapolation, CSR construction, or certify timeout is used.",
            "- The reference includes all formal policies, including SR and registered/computational baselines." if all_policies else "- This run explicitly uses an eligible-baseline reference.",
            "",
        ]
    )
    return "\n".join(lines)


def _quantile(values: Sequence[float], probability: float) -> float:
    if len(values) == 1:
        return float(values[0])
    position = probability * (len(values) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    fraction = position - low
    return float(values[low] * (1.0 - fraction) + values[high] * fraction)


if __name__ == "__main__":
    main()
