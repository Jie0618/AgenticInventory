#!/usr/bin/env python3
"""Run a multi-phase SR workflow with streamed logs.

The controller deliberately runs Phase 0 and Phase 1 in separate Python
processes.  This releases all DP/optimizer worker memory after initialization,
while preserving the pair cache that Phase 1 validates and reuses.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "PYTHONUNBUFFERED": "1",
}


@dataclass(frozen=True)
class WorkflowCliDefaults:
    repository_root: Path
    initialization_entrypoint: Path
    search_entrypoint: Path
    experiment_config: Path
    search_config: Path
    checkpoint_root: Path
    diagnostic_root: Path
    cache_root: Path
    reference_suite: str = "gap200"
    cyclic_evaluation_config: Path | None = None
    initialization_arguments: tuple[str, ...] = ()
    search_arguments: tuple[str, ...] = ()


def _shared_args(args: argparse.Namespace) -> list[str]:
    return [
        "--experiment-config",
        str(args.experiment_config),
        "--search-config",
        str(args.search_config),
        "--checkpoint-root",
        str(args.checkpoint_root),
        "--diagnostic-root",
        str(args.diagnostic_root),
        "--dp-cache-dir",
        str(args.dp_cache_dir),
        "--horizon",
        str(getattr(args, "horizon", 1000)),
        "--cache-root",
        str(args.cache_root),
        "--workers",
        str(args.workers),
        "--reference-suite",
        str(args.reference_suite),
    ]


def build_commands(
    args: argparse.Namespace,
    *,
    defaults: WorkflowCliDefaults,
) -> list[tuple[str, list[str]]]:
    """Return the ordered child commands for the requested workflow phase."""

    shared = _shared_args(args)
    if getattr(args, "cyclic_evaluation_config", None) is not None:
        shared.extend(("--cyclic-evaluation-config", str(args.cyclic_evaluation_config)))
    commands: list[tuple[str, list[str]]] = []
    if args.phase in {"all", "initialize"}:
        commands.append(
            (
                "initialization",
                [
                    sys.executable,
                    "-u",
                    str(defaults.initialization_entrypoint),
                    *defaults.initialization_arguments,
                    *shared,
                ],
            )
        )
    if args.phase in {"all", "search"}:
        search = [
            sys.executable,
            "-u",
            str(defaults.search_entrypoint),
            *defaults.search_arguments,
            "--output-dir",
            str(args.output_dir),
            *shared,
        ]
        if args.mutation_config is not None:
            search.extend(["--mutation-config", str(args.mutation_config)])
        if args.reflection_config is not None:
            search.extend(["--reflection-config", str(args.reflection_config)])
        if getattr(args, "resume", False):
            search.append("--resume")
        commands.append(("search", search))
    return commands


def _run_streamed(
    label: str,
    command: Sequence[str],
    *,
    log_path: Path,
    environment: dict[str, str],
    repository_root: Path,
) -> float:
    """Run one phase and mirror its merged output to terminal and a log file."""

    started = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as handle:
        header = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] phase={label} command={shlex.join(command)}"
        print(header, flush=True)
        handle.write(header + "\n")
        process = subprocess.Popen(
            list(command),
            cwd=repository_root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                handle.write(line)
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            raise
        return_code = process.wait()
        elapsed = time.monotonic() - started
        footer = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] phase={label} returncode={return_code} elapsed_seconds={elapsed:.3f}"
        print(footer, flush=True)
        handle.write(footer + "\n")
    if return_code != 0:
        raise SystemExit(f"phase {label!r} failed with return code {return_code}; see {log_path}")
    return elapsed


def parse_args(
    argv: Sequence[str] | None = None,
    *,
    defaults: WorkflowCliDefaults,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cached frontier initialization and multi-distribution SR sequentially."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=("all", "initialize", "search"),
        default="all",
        help="'all' initializes first and starts search only after successful completion.",
    )
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
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--cache-root", type=Path, default=defaults.cache_root)
    parser.add_argument("--workers", type=int, default=60)
    parser.add_argument("--reference-suite", default=defaults.reference_suite)
    parser.add_argument("--mutation-config", type=Path, default=None)
    parser.add_argument("--reflection-config", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume Stage2 search from its checkpoint")
    args = parser.parse_args(argv)
    if args.reference_suite == "stage1-reference":
        args.diagnostic_root = args.diagnostic_root or args.checkpoint_root / ".inputs"
        args.dp_cache_dir = args.dp_cache_dir or args.checkpoint_root / ".dp_cache"
    elif args.diagnostic_root is None or args.dp_cache_dir is None:
        parser.error("--diagnostic-root and --dp-cache-dir are required for this reference suite")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.horizon <= 0:
        parser.error("--horizon must be positive")
    return args


def main(
    argv: Sequence[str] | None = None,
    *,
    defaults: WorkflowCliDefaults,
) -> None:
    args = parse_args(argv, defaults=defaults)
    commands = build_commands(args, defaults=defaults)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "phase": args.phase,
                    "workers": args.workers,
                    "commands": [
                        {"phase": label, "command": shlex.join(command)}
                        for label, command in commands
                    ],
                },
                indent=2,
            )
        )
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(_THREAD_ENV)
    existing_pythonpath = environment.get("PYTHONPATH", "")
    root = defaults.repository_root
    required_pythonpath = f"{root / 'src'}:{root}"
    environment["PYTHONPATH"] = (
        f"{required_pythonpath}:{existing_pythonpath}" if existing_pythonpath else required_pythonpath
    )

    total_started = time.monotonic()
    timings: dict[str, float] = {}
    for label, command in commands:
        timings[label] = _run_streamed(
            label,
            command,
            log_path=args.output_dir / f"{label}.log",
            environment=environment,
            repository_root=root,
        )
    summary = {
        "status": "complete",
        "phase": args.phase,
        "workers": args.workers,
        "phase_seconds": timings,
        "elapsed_seconds": time.monotonic() - total_started,
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "controller_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
