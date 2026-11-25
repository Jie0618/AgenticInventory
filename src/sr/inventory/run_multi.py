#!/usr/bin/env python3
"""Run production multi-case symbolic regression."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from inventory.core import InventoryCase

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

from sr.inventory.runtime import MultiCliDefaults, load_runtime
from sr.llm import JSONChatClient, LLMConfig
from sr.inventory.multi import MultiDistributionSearch


def main(
    available_cases: Sequence[InventoryCase],
    *,
    defaults: MultiCliDefaults,
) -> None:
    parser = argparse.ArgumentParser()
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
        default=None if defaults.reference_suite == "stage1-reference" else defaults.diagnostic_root,
    )
    parser.add_argument("--dp-cache-dir", type=Path, required=defaults.reference_suite != "stage1-reference")
    parser.add_argument("--reference-suite", default=defaults.reference_suite)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--cache-root", type=Path, default=defaults.cache_root)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Restore existing evaluated structures and continue checkpoint iteration")
    parser.add_argument("--mutation-config", type=Path, default=None)
    parser.add_argument("--reflection-config", type=Path, default=None)
    parser.add_argument(
        "--write-run-logs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write events.jsonl and llm_calls.jsonl (default: enabled)",
    )
    args = parser.parse_args()

    runtime = load_runtime(
        available_cases=available_cases,
        experiment_config=args.experiment_config,
        search_config=args.search_config,
        checkpoint_root=args.checkpoint_root,
        diagnostic_root=args.diagnostic_root,
        dp_cache_dir=args.dp_cache_dir,
        reference_suite=args.reference_suite,
        horizon=args.horizon,
        workers=args.workers,
    )
    audit_log = args.output_dir / "llm_calls.jsonl" if args.write_run_logs else None
    screens = None
    if runtime.search_config.cyclic_cap:
        if args.cyclic_evaluation_config is None:
            parser.error("cyclic_cap requires --cyclic-evaluation-config")
        from sr.inventory.cyclic import bind_cyclic_runtime
        from sr.inventory.mc import MonteCarloScreen
        settings = json.loads(args.cyclic_evaluation_config.read_text())
        runtime = bind_cyclic_runtime(runtime, settings)
        screens = {
            name: MonteCarloScreen(
                row.context, runtime.bundle.motifs,
                diagnostic_json=runtime.diagnostic_paths[name],
            )
            for name, row in runtime.case_inputs.items()
        }
    mutation = None if args.mutation_config is None else JSONChatClient(
        LLMConfig.from_json(args.mutation_config),
        audit_log=audit_log,
        call_kind="multi_mutation",
    )
    reflection_path = args.reflection_config or args.mutation_config
    reflection = None if reflection_path is None else JSONChatClient(
        LLMConfig.from_json(reflection_path),
        audit_log=audit_log,
        call_kind="multi_reflection",
    )
    initialization_cache_dir = args.cache_root / "initialization" / runtime.initialization_signature
    result = MultiDistributionSearch(
        case_inputs=runtime.case_inputs,
        diagnostic_paths=runtime.diagnostic_paths,
        seed_bundle=runtime.bundle,
        initialization_cache_dir=initialization_cache_dir,
        initialization_signature=runtime.initialization_signature,
        output_dir=args.output_dir,
        config=runtime.search_config,
        objective=runtime.objective,
        max_pair_retries=runtime.max_pair_retries,
        mutation_client=mutation,
        reflection_client=reflection,
        write_run_logs=args.write_run_logs,
        screens=screens,
        resume=args.resume,
    ).run()
    print(
        json.dumps(
            {
                "status": "complete",
                "cases": list(result.case_names),
                "archive": len(result.archive),
                "best_score": min(row.score for row in result.archive),
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
