"""Fit and rank inventory policies using shared demand paths."""

import argparse
from dataclasses import replace
from importlib import import_module
import json
import sys

from _common import ROOT, positive_int
import numpy as np

# The policy catalog lives in the repository's experiment and benchmark packages.
sys.path.insert(0, str(ROOT))

from benchmarks.inventory_benchmark_family.mc_protocol import load_policy_set
from benchmarks.inventory_benchmark_family.mc_runtime import MCDemandBank
from inventory.common.distributions import DiscreteDemand
from inventory.core import (
    DeterministicLeadTime,
    FixedShelfLife,
    LostSales,
    inventory_case,
)
from optimization.evaluators.composed import (
    ComposedMonteCarloPolicyCostEvaluator,
    TwoFidelityComposedMCEvaluator,
)
from optimization.models.types import (
    MonteCarloProposalConfig,
    OptimizationConfig,
)
from optimization.solvers.policy_cost import optimize_policy_constants


POLICY_GROUPS = {
    "lead_time": ("co", "base_stock", "cbs", "weighted_tilted_pic", "SCDS", "IA-SCDS"),
    "perishable": ("base_stock", "bsp_low", "weighted_base_stock", "EABS", "RLCS"),
}
POLICY_IDS = tuple(dict.fromkeys(
    policy for group in POLICY_GROUPS.values() for policy in group
))
DEMAND_ID = "random_pmf_4_m12_alpha1"


def comparison_case(setting):
    """Use one frozen demand PMF in two explicitly selected demonstration cases."""
    catalog = ROOT / "benchmarks/inventory_benchmark_family/demand_catalogs/inventory_50/distributions.json"
    distribution = next(
        row for row in json.loads(catalog.read_text(encoding="utf-8"))
        if row["id"] == DEMAND_ID
    )
    return inventory_case(
        name=f"example_comparison_{setting}",
        demand=DiscreteDemand(
            values=distribution["support"], probs=distribution["probabilities"],
            name=DEMAND_ID,
        ),
        n=20,
        capacity=None,
        costs={
            "holding": 1.,
            "shortage": 20. if setting == "lead_time" else 9.,
            "expiry": 0. if setting == "lead_time" else 16.,
        },
        lead_time=DeterministicLeadTime(6) if setting == "lead_time" else None,
        perishability=FixedShelfLife(2) if setting == "perishable" else None,
        backlog=LostSales(),
    )


def resolved_policies(setting, case, identifiers):
    """Reuse the campaign resolvers, including their per-case parameter domains."""
    manifest = (
        ROOT / "experiments" / setting / "benchmark_campaigns"
        / "inventory_benchmark_family/policies/sets/mc_final.json"
    )
    model = import_module(
        f"experiments.{setting}.benchmark_campaigns.inventory_benchmark_family.mc_model"
    )
    _, definitions = load_policy_set(manifest)
    by_id = {row.id: row for row in definitions}
    policies = [
        model.resolve_policy(
            by_id[name],
            case.lead_time if setting == "lead_time" else case.shelf_life,
            case.n,
            case=case,
        )
        for name in identifiers
    ]
    # At pre-order epochs, P carries at most (M-1)*n units, and its mean
    # remaining life is at most M-1. Above M*(n+1), S always saturates the
    # action cap; a cyclic C>=n is also indistinguishable after projection.
    # Remove only these redundant regions of the frozen RLCS search domain.
    policies = [
        replace(
            policy,
            parameter_specs=tuple(
                replace(spec, upper=min(spec.upper, case.shelf_life * (case.n + 1)))
                if spec.name == "S" else
                replace(spec, upper=min(spec.upper, case.n))
                if spec.name == "C" else spec
                for spec in policy.parameter_specs
            ),
        ) if policy.id == "RLCS" else policy
        for policy in policies
    ]
    return [
        {
            "id": policy.id,
            "ast": policy.ast,
            "motifs": policy.motifs,
            "parameter_specs": policy.parameter_specs,
        }
        for policy in policies
    ]


def fit_policy(case, policy, args, tuning_bank, evaluation_bank):
    def evaluator(bank):
        result = ComposedMonteCarloPolicyCostEvaluator.from_case(
            policy["ast"],
            case,
            args.horizon,
            motifs=policy["motifs"],
            paths=bank.num_paths,
            seed=bank.seed,
            batch_size=16,
        )
        result.bind_demand_paths(bank.demand_values)
        return result

    tuning = evaluator(tuning_bank)
    proposals = evaluator(tuning_bank.prefix(args.proposal_paths, args.horizon))
    fitted = optimize_policy_constants(
        policy["parameter_specs"],
        TwoFidelityComposedMCEvaluator(tuning, proposals),
        config=OptimizationConfig(
            mode="fast",
            total_evaluations=args.proposal_budget,
            random_seed=args.seed + 2,
            action_cap=case.n,
            mc_proposal=MonteCarloProposalConfig(enabled=False, routing_mode="never"),
        ),
    )
    # Final evaluation uses one shared bank that was never used for fitting.
    evaluation = evaluator(evaluation_bank)
    path_costs = evaluation.simulate_path_costs(fitted.best_params) / args.horizon
    return {
        "policy": policy["id"],
        "parameter_count": len(policy["parameter_specs"]),
        "params": dict(fitted.best_params),
        "tuning_cost_per_period": fitted.best_cost / args.horizon,
        "evaluation_cost_per_period": float(path_costs.mean()),
        "standard_error_per_period": float(path_costs.std(ddof=1) / np.sqrt(len(path_costs))),
        "status": fitted.status,
    }


def markdown_table(headers, rows):
    text = [[str(cell).replace("|", "\\|") for cell in row] for row in rows]
    widths = [
        max(len(header), *(len(row[i]) for row in text))
        for i, header in enumerate(headers)
    ]

    def line(cells):
        return "| " + " | ".join(
            cell.ljust(width) for cell, width in zip(cells, widths)
        ) + " |"

    return "\n".join([
        line(headers),
        "| " + " | ".join("-" * width for width in widths) + " |",
        *(line(row) for row in text),
    ])


def format_report(rows):
    """Rank policies separately by independent MC cost; displayed ties share rank."""
    sections = []
    for setting in POLICY_GROUPS:
        group = sorted(
            (row for row in rows if row["setting"] == setting),
            key=lambda row: row["evaluation_cost_per_period"],
        )
        if not group:
            continue
        ranked = []
        previous_cost = None
        rank = 0
        for position, row in enumerate(group, 1):
            displayed_cost = f'{row["evaluation_cost_per_period"]:.6f}'
            if displayed_cost != previous_cost:
                rank = position
            previous_cost = displayed_cost
            ranked.append([
                rank, row["policy"], row["parameter_count"],
                f'{row["tuning_cost_per_period"]:.6f}', displayed_cost,
                f'{row["standard_error_per_period"]:.6f}', row["status"],
            ])
        summary = markdown_table(
            ["Rank", "Policy", "#Params", "Tuning/period", "MC/period", "MC SE/period", "Status"],
            ranked,
        )
        parameters = markdown_table(
            ["Policy", "Fitted parameters"],
            [
                [row["policy"], ", ".join(
                    f"{name}={value:.6g}" for name, value in sorted(row["params"].items())
                )]
                for row in group
            ],
        )
        sections.append(f"{setting} — ranked by MC/period (lower is better)\n\n"
                        + summary + "\n\n" + parameters)
    return "\n\n".join(sections)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", "--policies", dest="policies", nargs="+",
                        choices=("all", *POLICY_IDS), default=["all"])
    parser.add_argument("--setting", choices=("all", *POLICY_GROUPS), default="all")
    parser.add_argument("--horizon", type=positive_int, default=100)
    parser.add_argument("--tuning-paths", type=positive_int, default=4096,
                        help="shared fitting path count for both settings")
    parser.add_argument("--proposal-paths", type=positive_int, default=512)
    parser.add_argument("--paths", type=positive_int, default=50000)
    parser.add_argument("--proposal-budget", type=positive_int, default=4096,
                        help="shared fast proposal budget for both settings")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--evaluation-seed", type=int, default=20261125)
    args = parser.parse_args()
    if args.proposal_paths > args.tuning_paths:
        parser.error("--proposal-paths must not exceed --tuning-paths")
    if args.seed < 0 or args.evaluation_seed < 0:
        parser.error("seeds must be nonnegative")
    if args.evaluation_seed == args.seed:
        parser.error("--evaluation-seed must differ from the fitting seed")
    if args.paths < 2:
        parser.error("--paths must be at least 2 to estimate the standard error")
    if len(args.policies) != len(set(args.policies)):
        parser.error("duplicate policy names")
    if "all" in args.policies and len(args.policies) != 1:
        parser.error("'all' must be used alone")
    allowed = POLICY_IDS if args.setting == "all" else POLICY_GROUPS[args.setting]
    if args.policies != ["all"] and not set(args.policies) <= set(allowed):
        parser.error("selected policies do not belong to --setting")
    # Catalog order and seeds are independent of subset size and input order.
    selected = tuple(
        name for name in allowed if args.policies == ["all"] or name in args.policies
    )
    groups = {
        setting: names for setting, names in POLICY_GROUPS.items()
        if args.setting in ("all", setting)
    }
    total = sum(name in selected for names in groups.values() for name in names)
    print(
        "Optimizer: fast\n"
        f"Demand: {DEMAND_ID}\n"
        f"Fixed seeds: tuning={args.seed}, evaluation={args.evaluation_seed}, "
        f"optimizer={args.seed + 2}",
        flush=True,
    )
    completed = 0
    for setting, names in groups.items():
        identifiers = tuple(name for name in names if name in selected)
        if not identifiers:
            continue
        case = comparison_case(setting)
        print(
            f"{setting}: L={case.lead_time}, M={case.shelf_life}, n={case.n}, "
            f"h={case.costs.holding:g}, b={case.costs.shortage:g}, "
            f"w={case.costs.expiry:g}; horizon={args.horizon}; "
            f"tuning paths={args.tuning_paths}; "
            f"proposal budget={args.proposal_budget}",
            flush=True,
        )
        policies = resolved_policies(setting, case, identifiers)
        tuning_bank = MCDemandBank.create(
            case.demand, args.horizon, args.tuning_paths, args.seed
        )
        evaluation_bank = MCDemandBank.create(
            case.demand, args.horizon, args.paths, args.evaluation_seed
        )
        rows = []
        for policy in policies:
            print(f'[{completed + 1}/{total}] Fitting {setting}/{policy["id"]}',
                  flush=True)
            rows.append({
                "setting": setting,
                **fit_policy(case, policy, args, tuning_bank, evaluation_bank),
            })
            completed += 1
        print("\n" + format_report(rows) + "\n", flush=True)


if __name__ == "__main__":
    main()
