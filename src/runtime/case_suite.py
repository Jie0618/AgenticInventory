#!/usr/bin/env python3
"""Run a configured suite of independent symbolic-search jobs serially."""

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
from typing import Collection, Sequence


@dataclass(frozen=True)
class SuiteConfig:
    name: str
    case_names: tuple[str, ...]
    workers: int


@dataclass(frozen=True)
class CaseSuiteCliDefaults:
    repository_root: Path
    search_entrypoint: Path
    diagnostic_entrypoint: Path
    diagnostic_root: Path
    dp_cache_dir: Path
    search_config: Path
    cyclic_evaluation_config: Path | None = None
    prepare_inputs_in_case: bool = False
    search_arguments: tuple[str, ...] = ()


def load_suite(path: Path, *, known_case_names: Collection[str]) -> SuiteConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    name = str(payload["name"])
    if "training_environments" in payload:
        source = Path(path).parent / payload["training_environments"]
        environments = json.loads(source.read_text(encoding="utf-8"))["environments"]
        values = [row["case_name"] for row in environments]
    else:
        values = payload["cases"]
    case_names = tuple(str(value) for value in values)
    workers = int(payload.get("workers", 60))
    if not name or not case_names or len(case_names) != len(set(case_names)):
        raise ValueError("suite name and unique nonempty cases are required")
    if workers <= 0:
        raise ValueError("suite workers must be positive")
    known = {str(name) for name in known_case_names}
    unknown = [case_name for case_name in case_names if case_name not in known]
    if unknown:
        raise ValueError(f"unknown suite cases: {unknown}")
    return SuiteConfig(name, case_names, workers)


def build_search_command(
    *,
    python: Path,
    case_name: str,
    output_dir: Path,
    diagnostic_root: Path,
    dp_cache_dir: Path,
    search_config: Path,
    workers: int,
    mutation_config: Path | None,
    reflection_config: Path | None,
    search_entrypoint: Path,
    cyclic_evaluation_config: Path | None = None,
    search_arguments: Sequence[str] = (),
) -> tuple[str, ...]:
    command = [
        str(python),
        str(search_entrypoint),
        *search_arguments,
        "--case",
        case_name,
        "--output-dir",
        str(output_dir),
        "--diagnostic-root",
        str(diagnostic_root),
        "--dp-cache-dir",
        str(dp_cache_dir),
        "--search-config",
        str(search_config),
        "--workers",
        str(int(workers)),
    ]
    if mutation_config is not None:
        command.extend(("--mutation-config", str(mutation_config)))
    if reflection_config is not None:
        command.extend(("--reflection-config", str(reflection_config)))
    if cyclic_evaluation_config is not None:
        command.extend(("--cyclic-evaluation-config", str(cyclic_evaluation_config)))
    return tuple(command)


def main(
    *,
    known_case_names: Collection[str],
    defaults: CaseSuiteCliDefaults,
) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--diagnostic-root", type=Path, default=None if defaults.prepare_inputs_in_case else defaults.diagnostic_root)
    parser.add_argument("--dp-cache-dir", type=Path, default=None if defaults.prepare_inputs_in_case else defaults.dp_cache_dir)
    parser.add_argument("--search-config", type=Path, default=defaults.search_config)
    parser.add_argument("--cyclic-evaluation-config", type=Path, default=defaults.cyclic_evaluation_config)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--mutation-config", type=Path, default=None)
    parser.add_argument("--reflection-config", type=Path, default=None)
    parser.add_argument("--prepare-diagnostics", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if defaults.prepare_inputs_in_case:
        args.diagnostic_root = args.diagnostic_root or args.output_root / ".inputs"
        args.dp_cache_dir = args.dp_cache_dir or args.output_root / ".dp_cache"
    suite = load_suite(args.suite_config, known_case_names=known_case_names)
    workers = suite.workers if args.workers is None else int(args.workers)
    if workers <= 0:
        raise ValueError("workers must be positive")
    missing = [
        case_name for case_name in suite.case_names
        if not (args.diagnostic_root / f"{case_name}.json").is_file()
    ]
    if defaults.prepare_inputs_in_case:
        missing = []
    if missing and args.prepare_diagnostics:
        diagnostic_command = (
            str(args.python),
            str(defaults.diagnostic_entrypoint),
            "--cases",
            *missing,
            "--horizon",
            "1000",
            "--sample-count",
            "16",
            "--prune-probability",
            "1e-13",
            "--output-dir",
            str(args.diagnostic_root),
            "--dp-cache-dir",
            str(args.dp_cache_dir),
            "--resume",
        )
        _run_streaming(
            diagnostic_command,
            args.output_root / "diagnostics.log",
            args.dry_run,
            repository_root=defaults.repository_root,
        )
        if not args.dry_run:
            still_missing = [
                case_name for case_name in missing
                if not (args.diagnostic_root / f"{case_name}.json").is_file()
            ]
            if still_missing:
                raise RuntimeError(f"diagnostic command completed without producing: {still_missing}")
    elif missing:
        joined = "\n  - ".join(missing)
        raise FileNotFoundError(
            f"missing diagnostics for suite {suite.name}:\n  - {joined}\n"
            "rerun with --prepare-diagnostics or generate them separately"
        )

    if not args.dry_run:
        args.output_root.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "event": "suite_started",
                "suite": suite.name,
                "cases": list(suite.case_names),
                "workers": workers,
                "dry_run": bool(args.dry_run),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    for case_name in suite.case_names:
        output_dir = args.output_root / case_name
        if output_dir.exists() and any(output_dir.iterdir()):
            if args.skip_existing:
                print(json.dumps({"event": "case_skipped", "case": case_name}, sort_keys=True), flush=True)
                continue
            raise FileExistsError(
                f"refusing to reuse nonempty output directory: {output_dir}; "
                "choose a new --output-root or pass --skip-existing"
            )
        command = build_search_command(
            search_arguments=defaults.search_arguments,
            python=args.python,
            case_name=case_name,
            output_dir=output_dir,
            diagnostic_root=args.diagnostic_root,
            dp_cache_dir=args.dp_cache_dir,
            search_config=args.search_config,
            workers=workers,
            mutation_config=args.mutation_config,
            reflection_config=args.reflection_config or args.mutation_config,
            search_entrypoint=defaults.search_entrypoint,
            cyclic_evaluation_config=args.cyclic_evaluation_config,
        )
        print(json.dumps({"event": "case_started", "case": case_name, "time": time.time()}, sort_keys=True), flush=True)
        _run_streaming(
            command,
            output_dir / "run.log",
            args.dry_run,
            repository_root=defaults.repository_root,
        )
        print(json.dumps({"event": "case_finished", "case": case_name, "time": time.time()}, sort_keys=True), flush=True)
    print(json.dumps({"event": "suite_finished", "suite": suite.name, "time": time.time()}, sort_keys=True), flush=True)


def _run_streaming(
    command: Sequence[str],
    log_path: Path,
    dry_run: bool,
    *,
    repository_root: Path,
) -> None:
    print("COMMAND " + shlex.join(tuple(str(value) for value in command)), flush=True)
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    current_python_path = environment.get("PYTHONPATH", "")
    root = Path(repository_root)
    required = f"{root / 'src'}:{root}"
    environment["PYTHONPATH"] = required if not current_python_path else f"{required}:{current_python_path}"
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            tuple(str(value) for value in command),
            cwd=root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
