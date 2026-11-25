"""Immutable policy manifests and external AST/motif assets."""

import json
from pathlib import Path


def materialize_policy_set(path):
    path = Path(path)
    payload = json.loads(path.read_text())
    for row in payload["policies"]:
        for key in ("ast", "motifs"):
            reference = row.get(key + "_file")
            if reference is not None:
                if key in row:
                    raise ValueError(f"both {key} and {key}_file supplied")
                row[key] = json.loads((path.parent / reference).read_text())
                del row[key + "_file"]
    return payload


def write_policy_set(payload, path, *, externalize=False):
    path = Path(path)
    # Work on a copy: selection and semantic identities use materialized ASTs.
    manifest = json.loads(json.dumps(payload))
    files = {}
    if externalize:
        assets = path.parent.parent / "assets" / path.stem
        for row in manifest["policies"]:
            identifier = row["id"]
            if Path(identifier).name != identifier or identifier in (".", ".."):
                raise ValueError("policy ids must be safe asset filenames")
            for key in ("ast", "motifs"):
                if key in row:
                    destination = assets / f"{identifier}.{key}.json"
                    files[destination] = row.pop(key)
                    row[key + "_file"] = str(Path("..") / "assets" / path.stem / destination.name)
    files[path] = manifest
    encoded = {target: json.dumps(value, indent=2, sort_keys=True) + "\n" for target, value in files.items()}
    # Validate every file before writing any: a frozen version cannot mutate.
    for target, value in encoded.items():
        if target.exists() and target.read_text() != value:
            raise ValueError(f"frozen policy set differs; use a new version: {target}")
    for target, value in encoded.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_text(value)
