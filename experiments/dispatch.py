"""Bind shared training and maintenance commands to a setting's inputs."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from importlib import import_module
import sys

from experiments.launch import ROOT, SETTINGS, output_roots, stage_directories

COMMANDS = (
    "suite", "single", "workflow", "initialize", "multi", "mc1", "mc2",
    "rebuild-profiles", "certify-frontier",
)
ENTRY = ROOT / "experiments" / "dispatch.py"


def multi_defaults(setting):
    from sr.inventory.runtime import MultiCliDefaults

    first, second, _ = stage_directories(setting)
    return MultiCliDefaults(
        experiment_config=second / "configs/training.json",
        cyclic_evaluation_config=second / "configs/evaluation.json",
        search_config=second / "configs/search.json",
        checkpoint_root=output_roots(setting, {})[0],
        diagnostic_root=first / "results/.inputs",
        cache_root=second / ".cache",
        reference_suite="stage1-reference",
    )


def workflow_defaults(setting):
    from runtime.sr_workflow import WorkflowCliDefaults

    return WorkflowCliDefaults(
        repository_root=ROOT,
        initialization_entrypoint=ENTRY, initialization_arguments=(setting, "initialize"),
        search_entrypoint=ENTRY, search_arguments=(setting, "multi"),
        **asdict(multi_defaults(setting)),
    )


def suite_defaults(setting):
    from runtime.case_suite import CaseSuiteCliDefaults

    first = stage_directories(setting)[0]
    return CaseSuiteCliDefaults(
        repository_root=ROOT,
        search_entrypoint=ENTRY, search_arguments=(setting, "single"),
        diagnostic_entrypoint=ENTRY,
        diagnostic_root=first / "results/.inputs",
        dp_cache_dir=first / "results/.dp_cache",
        search_config=first / "configs/search.json",
        cyclic_evaluation_config=first / "configs/evaluation.json",
        prepare_inputs_in_case=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("setting", choices=SETTINGS)
    parser.add_argument("command", choices=COMMANDS)
    # Forward all backend flags (including --help) to that backend's parser.
    args = parser.parse_args(sys.argv[1:3])
    sys.argv = [sys.argv[0], *sys.argv[3:]]
    simple = args.setting in ("lead_time", "perishable")
    if not simple and args.command not in ("mc1", "mc2"):
        parser.error("this command requires lead_time or perishable")
    if simple and args.command in ("mc1", "mc2"):
        parser.error("MC-only commands require a combined setting")
    cases = import_module(f"experiments.{args.setting}.configs").cases
    if args.command == "suite":
        from runtime.case_suite import main as run
        run(known_case_names={case.name for case in cases()}, defaults=suite_defaults(args.setting))
    elif args.command == "single":
        from sr.inventory.run_single import main as run
        first = stage_directories(args.setting)[0]
        run(cases(), prepare_inputs=True,
            default_diagnostic_root=first / "results/.inputs",
            default_search_config=first / "configs/search.json",
            default_cyclic_evaluation_config=first / "configs/evaluation.json")
    elif args.command == "workflow":
        from runtime.sr_workflow import main as run
        run(defaults=workflow_defaults(args.setting))
    elif args.command in ("initialize", "multi", "certify-frontier"):
        module = {
            "initialize": "initialize_multi", "multi": "run_multi",
            "certify-frontier": "certify_multi",
        }[args.command]
        run = import_module(f"sr.inventory.{module}").main
        run(cases(), defaults=multi_defaults(args.setting))
    elif args.command == "rebuild-profiles":
        from sr.analysis.rebuild_profiles import main as run
        run(cases())
    else:
        from sr.inventory.run_mc import main as run
        worker = import_module(f"experiments.{args.setting}.reference").ReferenceWorker
        run(int(args.command[-1]), root=ROOT / "experiments" / args.setting,
            case_provider=cases, worker_type=worker, prepare_per_case=True)


if __name__ == "__main__":
    main()
