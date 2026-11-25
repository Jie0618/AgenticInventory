#!/usr/bin/env python3
"""Run or resume no-LLM multi-case frontier initialization."""

from __future__ import annotations

import argparse
import json
import os
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
from sr.inventory.initialization import run_frontier_initialization


def main(
    available_cases: Sequence[InventoryCase],
    *,
    defaults: MultiCliDefaults,
) -> None:
    parser = argparse.ArgumentParser()
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
    if runtime.search_config.cyclic_cap:
        if args.cyclic_evaluation_config is None:
            parser.error("cyclic_cap requires --cyclic-evaluation-config")
        from sr.inventory.cyclic import bind_cyclic_runtime
        runtime = bind_cyclic_runtime(runtime, json.loads(args.cyclic_evaluation_config.read_text()))
    cache_dir = args.cache_root / "initialization" / runtime.initialization_signature
    result = run_frontier_initialization(
        runtime.bundle,
        runtime.case_inputs,
        config=runtime.search_config,
        objective=runtime.objective,
        cache_dir=cache_dir,
        initialization_signature=runtime.initialization_signature,
        max_pair_retries=runtime.max_pair_retries,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "cache_dir": str(result.cache_dir),
                "structures": len(result.individuals),
                "pairs": len(result.individuals) * len(runtime.case_names),
                "elapsed_seconds": result.elapsed_seconds,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
