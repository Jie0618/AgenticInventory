"""Stage3 result layout, including reads of flat run directories."""

from pathlib import Path


def artifact_path(root: Path, name: str) -> Path:
    if name.startswith("PROPOSALS_ITER_"):
        directory = "proposals"
    elif name.startswith("MOTIF_REFLECTION_ITER_"):
        directory = "reflections"
    elif name.startswith("ITERATION_AUDIT_"):
        directory = "audit/iterations"
    elif name.startswith("PARENT_EVIDENCE_ITER_"):
        directory = "audit/parents"
    elif name.startswith("PROPOSAL_FAILURE_ITER_"):
        directory = "audit"
    elif name in {"LLM_TEMPLATE_INIT.json", "SEED_SETUP.json"}:
        directory = "initialization"
    elif name in {"STAGE2_TRANSFER_AUDIT.json", "SCREENING_AUDIT.json"}:
        directory = "audit"
    else:
        directory = ""
    return Path(root) / directory / name


def write_path(root: Path, name: str) -> Path:
    path = artifact_path(root, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_path(root: Path, name: str) -> Path:
    path = artifact_path(root, name)
    return path if path.exists() else Path(root) / name


def artifact_files(root: Path, pattern: str) -> tuple[Path, ...]:
    """Prefer the organized copy when both flat and organized locations exist."""
    paths = {path.name: path for path in Path(root).glob(pattern)}
    directory = artifact_path(root, pattern).parent
    paths.update({path.name: path for path in directory.glob(pattern)})
    return tuple(paths[name] for name in sorted(paths))
