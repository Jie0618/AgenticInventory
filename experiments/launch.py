"""Shared command construction for the four settings' stage launchers."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import sys
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ("lead_time", "perishable", "lead_time_backlog", "lead_time_perishable")
STAGE3_DIRS = {
    "lead_time": "stage3_multi_leadtime_sr",
    "perishable": "stage3_multi_perishable_sr",
    "lead_time_backlog": "stage3_multi_context_sr",
    "lead_time_perishable": "stage3_multi_context_sr",
}
# Keep the complete pipeline's established output roots, including capitalization.
OUTPUT_NAMES = {
    "lead_time": ("l3_ten_case_production", "l3_ten_case_standard"),
    "perishable": ("m3_n22_stationary_headroom_ten_case_production", "M3_ten_case_standard"),
    "lead_time_backlog": ("l3k3_balanced_10_standard", "l3k3_balanced_10_standard"),
    "lead_time_perishable": ("l3m3_balanced_10", "l3m3_balanced_10"),
}


def stage_directories(setting: str) -> tuple[Path, Path, Path]:
    base = ROOT / "experiments" / setting
    return (
        base / "stage1_single_distribution_sr",
        base / "stage2_multi_distribution_sr",
        base / STAGE3_DIRS[setting],
    )


def output_roots(setting: str, env: Mapping[str, str]) -> tuple[Path, Path, Path]:
    names = (*OUTPUT_NAMES[setting], "stage3_ml_md_run")
    return tuple(
        Path(env.get(f"STAGE{i}", "") or directory / "results" / name)
        for i, (directory, name) in enumerate(zip(stage_directories(setting), names), 1)
    )


def _option(arguments: Sequence[str], name: str) -> str | None:
    for i, arg in reversed(list(enumerate(arguments))):
        if arg.startswith(name + "="):
            return arg.split("=", 1)[1]
        if arg == name and i + 1 < len(arguments):
            return arguments[i + 1]
    return None


def stage_command(
    setting: str, stage: int, arguments: Sequence[str] = (),
    *, environ: Mapping[str, str] | None = None,
) -> list[str]:
    """Resolve defaults without preparing inputs, touching results or starting work."""
    if setting not in SETTINGS or stage not in (1, 2, 3):
        raise ValueError("Unknown setting or stage")
    env = os.environ if environ is None else environ
    local = {} if env.get("EXPERIMENT_PIPELINE") == "1" else env
    directories = stage_directories(setting)
    outputs = output_roots(setting, env)
    directory = directories[stage - 1]
    simple = setting in ("lead_time", "perishable")
    selected_case = _option(arguments, "--case")
    single_case = stage == 1 and simple and any(
        arg.split("=", 1)[0] == "--case" for arg in arguments
    )
    output = local.get("OUTPUT_DIR") or (
        local.get("OUTPUT_ROOT") if stage == 1 else None
    ) or str(outputs[stage - 1])
    if stage == 1 and simple and not single_case:
        operation = "suite"
        defaults = [
            "--suite-config", str(directory / "configs/suite.json"),
            "--output-root", output,
        ]
    else:
        operation = "single" if stage == 1 and simple else (
            "workflow" if stage == 2 and simple else f"mc{stage}"
        )
        if single_case:
            output = str(Path(output) / (selected_case or "case"))
        defaults = ["--output-dir", output]
    if stage == 2:
        defaults += ["--checkpoint-root", local.get("CHECKPOINT_ROOT") or str(outputs[0])]
    if stage == 3:
        checkpoint = local.get("STAGE2_CHECKPOINT") or local.get("SEED_ARTIFACTS")
        defaults += ["--stage2-checkpoint", checkpoint or str(outputs[1] / "checkpoint.json")]
        if local.get("ITERATIONS"):
            defaults += ["--iterations", local["ITERATIONS"]]
    evaluation = local.get("EVALUATION_CONFIG") or local.get("EVAL_CONFIG")
    if evaluation:
        flag = "--config" if stage == 3 else (
            "--cyclic-evaluation-config" if simple else "--evaluation-config"
        )
        defaults += [flag, evaluation]
    if local.get("FIDELITY") and (stage == 3 or not simple):
        defaults += ["--fidelity", local["FIDELITY"]]
    defaults += [
        "--workers", env.get("WORKERS") or "60",
        "--mutation-config", env.get("MUTATION_CONFIG") or str(ROOT / "configs/mutation_api_key.json"),
        "--reflection-config", env.get("REFLECTION_CONFIG") or str(ROOT / "configs/motif_reflection_api_key.json"),
    ]
    python = env.get("PYTHON_BIN") or sys.executable
    backend = (
        [str(directory / "runtime/run_search.py")] if stage == 3 else
        [str(ROOT / "experiments/dispatch.py"), setting, operation]
    )
    command = [python, "-u", *backend, *defaults, *arguments]
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("setting", choices=SETTINGS)
    parser.add_argument("stage", type=int, choices=(1, 2, 3))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = stage_command(args.setting, args.stage, args.arguments)
    if os.environ.get("DRY_RUN", "0") == "1":
        print(shlex.join(command), flush=True)
        return
    os.chdir(ROOT)
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT}" + (
        ":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    # These are the existing complete-pipeline thread settings.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    if args.setting == "lead_time_backlog" and args.stage == 3:
        env.update(VECLIB_MAXIMUM_THREADS="1", BLIS_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    # Logging is explicit; help and planning must never create result directories.
    log_file = env.get("LOG_FILE")
    if log_file and not any(arg in ("--help", "-h", "--dry-run") for arg in args.arguments):
        log = Path(log_file)
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as stream:
            os.dup2(stream.fileno(), 1)
            os.dup2(stream.fileno(), 2)
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
