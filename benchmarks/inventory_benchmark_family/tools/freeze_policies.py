"""Freeze benchmark candidates into the common sets/ + assets/ layout."""

import argparse
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path

from benchmarks.inventory_benchmark_family.policy_assets import materialize_policy_set, write_policy_set


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="freeze an existing materialized or external policy set")
    source.add_argument("--selection-module", help="Importable policy-selection adapter")
    source.add_argument("--checkpoint", type=Path, help="export completed candidates from a Stage3 CHECKPOINT.json")
    parser.add_argument("--setting", choices=("lead_time", "perishable", "lead_time_perishable", "lead_time_backlog"))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--top-k", type=int, help="export the lowest training-score candidates (default: 1)")
    selection.add_argument("--family-ids", nargs="+", help="select completed candidates by family ID or full semantic hash")
    parser.add_argument("--config", type=Path, help="benchmark configuration used to validate exported families")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.checkpoint is not None:
        from benchmarks.inventory_benchmark_family.tools.search_candidates import export_search_candidates
        if args.setting is None:
            parser.error("--checkpoint requires --setting")
        try:
            payload = export_search_candidates(
                args.checkpoint, args.setting, top_k=1 if args.top_k is None else args.top_k,
                family_ids=args.family_ids, config=args.config)
        except (ValueError, KeyError, OSError) as exc:
            parser.error(str(exc))
    elif any(value is not None for value in (args.setting, args.top_k, args.family_ids, args.config)):
        parser.error("--setting, --top-k, --family-ids and --config require --checkpoint")
    elif args.input is not None:
        payload = materialize_policy_set(args.input)
    else:
        module_name = args.selection_module
        if find_spec(module_name) is None:
            parser.error("this experiment already has frozen candidates; provide --input to export a set")
        selection = import_module(module_name)
        payload = selection.build_mc()
    write_policy_set(payload, args.output, externalize=True)
    print(f"Frozen {len(payload['policies'])} policies: {args.output}")
    for row in payload["policies"]:
        if row.get("family"):
            print(f"  {row['id']}: {row.get('display_name', row['id'])}")


if __name__ == "__main__":
    main()
