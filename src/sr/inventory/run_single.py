#!/usr/bin/env python3
"""Run a production symbolic search for one inventory case."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

# Numerical libraries must see these values before NumPy/SciPy are imported.
# The asynchronous optimizer uses one process per worker, so allowing an
# additional BLAS/OpenMP pool inside every process causes severe CPU
# oversubscription.
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

from evaluation.cache import load_or_create_inventory_dp
from inventory.core import InventoryCase
from sr.inventory.single import SymbolicSearch, diagnostic_samples
from optimization import InventorySearchContext
from sr import SearchConfig
from sr.core.inventory import InventorySearchBindings
from sr.llm import JSONChatClient, LLMConfig


HORIZON = 1000


def _config(path: Path | None, workers: int | None) -> SearchConfig:
    values = asdict(SearchConfig())
    if path is not None:
        incoming = json.loads(path.read_text(encoding="utf-8"))
        unknown = sorted(set(incoming) - set(values))
        if unknown:
            raise ValueError(f"unknown SearchConfig fields: {unknown}")
        values.update(incoming)
    if workers is not None:
        values["optimizer_workers"] = int(workers)
    return SearchConfig(**values)


def main(
    available_cases: Sequence[InventoryCase],
    *,
    default_diagnostic_root: Path,
    prepare_inputs: bool = False,
    default_search_config: Path | None = None,
    default_cyclic_evaluation_config: Path | None = None,
) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--diagnostic-root",
        type=Path,
        default=None if prepare_inputs else default_diagnostic_root,
    )
    parser.add_argument("--dp-cache-dir", type=Path, required=not prepare_inputs)
    parser.add_argument("--reference-suite", default="gap200")
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--search-config", type=Path, default=default_search_config)
    parser.add_argument("--cyclic-evaluation-config", type=Path, default=default_cyclic_evaluation_config)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--mutation-config", type=Path, default=None)
    parser.add_argument("--reflection-config", type=Path, default=None)
    parser.add_argument(
        "--write-run-logs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write events.jsonl and llm_calls.jsonl (default: enabled)",
    )
    args = parser.parse_args()
    if args.horizon <= 0:
        parser.error("--horizon must be positive")
    if (args.output_dir / "checkpoint.json").exists():
        raise ValueError(f"Existing Stage1 checkpoint; use a new output directory: {args.output_dir}")

    case = next((row for row in available_cases if row.name == args.case), None)
    if case is None:
        raise ValueError(f"unknown inventory case: {args.case}")
    if prepare_inputs:
        args.diagnostic_root = args.diagnostic_root or args.output_dir.parent / ".inputs"
        args.dp_cache_dir = args.dp_cache_dir or args.output_dir.parent / ".dp_cache"
        args.reference_suite = "stage1-reference"
        from sr.inventory.preparation import prepare_case
        dp_result, diagnostic = prepare_case(case, args.horizon, args.diagnostic_root, args.dp_cache_dir)
    diagnostic = args.diagnostic_root / f"{case.name}.json"
    if not diagnostic.is_file():
        raise FileNotFoundError(diagnostic)
    config = _config(args.search_config, args.workers)
    if not prepare_inputs:
        dp_result = load_or_create_inventory_dp(
            case, args.horizon, suite=args.reference_suite, cache_dir=args.dp_cache_dir
        )
    samples = diagnostic_samples(diagnostic, args.horizon)
    context = InventorySearchContext.from_reference(
        case,
        args.horizon,
        dp_result,
        samples=samples,
    )
    bindings = InventorySearchBindings.build(context, config)
    screen = None
    if config.cyclic_cap:
        if args.cyclic_evaluation_config is None:
            parser.error("cyclic_cap requires --cyclic-evaluation-config")
        from sr.inventory.cyclic import CyclicLeadTimeSearchContext
        from sr.inventory.mc import MonteCarloScreen
        context = CyclicLeadTimeSearchContext.from_inventory(
            context, json.loads(args.cyclic_evaluation_config.read_text()))
        bindings = InventorySearchBindings.build_context_free(case, config)
        screen = MonteCarloScreen(context, context.motifs, diagnostic_json=diagnostic)
    llm_audit_log = (
        args.output_dir / "llm_calls.jsonl" if args.write_run_logs else None
    )
    mutation = None if args.mutation_config is None else JSONChatClient(
        LLMConfig.from_json(args.mutation_config),
        audit_log=llm_audit_log,
        call_kind="mutation",
    )
    reflection_path = args.reflection_config or args.mutation_config
    reflection = None if reflection_path is None else JSONChatClient(
        LLMConfig.from_json(reflection_path),
        audit_log=llm_audit_log,
        call_kind="reflection",
    )
    result = SymbolicSearch(
        case_name=case.name,
        context=context,
        diagnostic_json=diagnostic,
        output_dir=args.output_dir,
        config=config,
        bindings=bindings,
        samples=samples,
        screen=screen,
        mutation_client=mutation,
        reflection_client=reflection,
        write_run_logs=args.write_run_logs,
    ).run()
    print(
        json.dumps(
            {
                "case": result.case_name,
                "archive": len(result.archive),
                "best_gap_pct": min(row.gap_pct for row in result.archive),
                "iterations": result.iterations_completed,
                "elapsed_seconds": result.elapsed_seconds,
                "report": str(result.output_dir / "REPORT.md"),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
