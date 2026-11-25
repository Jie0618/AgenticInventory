"""Compare Exact and Monte Carlo evaluation on four tiny inventory settings."""

import argparse

from _common import SETTINGS, make_case, positive_int
from evaluation import evaluate_composed_policy_exact, evaluate_composed_policy_mc


def constant_order(state, remaining):
    return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", choices=("all", *SETTINGS), default="all")
    parser.add_argument("--horizon", type=positive_int, default=8)
    parser.add_argument("--paths", type=positive_int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    settings = SETTINGS if args.setting == "all" else (args.setting,)
    print("Policy: order 1 each period; empty initial state")
    print(f"{'setting':<24} {'Exact/period':>14} {'MC/period':>14} {'MC SE/period':>14}")
    for setting in settings:
        case = make_case(setting)
        exact = evaluate_composed_policy_exact(case, constant_order, args.horizon)
        mc = evaluate_composed_policy_mc(
            case, constant_order, args.horizon, paths=args.paths, seed=args.seed
        )
        print(
            f"{setting:<24} {exact.cost / args.horizon:14.6f}"
            f" {mc.mean_cost / args.horizon:14.6f}"
            f" {mc.standard_error / args.horizon:14.6f}"
        )


if __name__ == "__main__":
    main()
