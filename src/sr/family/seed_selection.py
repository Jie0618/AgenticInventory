"""Merge rule-defined Stage2 boundary candidates for Stage3 initialization."""

import json
from pathlib import Path


PROTOCOL = "dual-frontier-union"


def select_frontier_seeds(frontier, archive):
    """Use the complete union of the parameter and complexity frontiers."""
    frontier_hashes = {str(row["ast_hash"]) for row in frontier}
    union = {str(row["ast_hash"]): row for row in archive
             if str(row["ast_hash"]) in frontier_hashes}
    missing = frontier_hashes - union.keys()
    if missing:
        raise ValueError(f"frontier candidates missing from supplied inventory: {sorted(missing)}")
    if not union:
        raise ValueError("frontier union contains no portable candidates")
    audit = [{"protocol": PROTOCOL, "selected": sorted(union)}]
    selected = sorted(union.values(), key=lambda r: (
        float(r["expanded_complexity"]), float(r["score"]), r["ast_hash"]))
    return selected, audit


def load_archive(checkpoint, payload):
    """Load exported AST inventory and scores without depending on original paths.

    A portable score index covers initialization-only candidates as well as
    event-log evaluations. Missing scores require the original case files;
    silently omitting potentially competitive policies is not allowed.
    """
    root = Path(checkpoint).parent
    inventory = root / "ARCHIVE_INVENTORY.jsonl"
    if not inventory.exists():
        raise ValueError(f"archive selection requires {inventory}")
    rows = {r["ast_hash"]: r for line in inventory.open() if (r := json.loads(line))}
    scores = {}
    index = root / "STAGE3_SCORE_INDEX.json"
    if index.exists():
        scores.update(json.loads(index.read_text()))
    else:
        events = root / "events.jsonl"
        if events.exists():
            with events.open() as stream:
                for line in stream:
                    event = json.loads(line)
                    if event["event"] == "group_completed":
                        scores[event["ast_hash"]] = event["score"]
    for key in ("parameter_frontier", "complexity_frontier", "parameter_profile", "complexity_profile"):
        for row in payload.get(key, ()):
            scores[row["ast_hash"]] = row["score"]
            rows.setdefault(row["ast_hash"], row)
    complete_count = sum(row.get("complete_cases", len(payload["case_names"])) == len(payload["case_names"])
                         for row in rows.values())
    if "archive_size" in payload and complete_count != int(payload["archive_size"]):
        raise ValueError(
            f"incomplete archive inventory: {complete_count}/{payload['archive_size']} complete policies; "
            "export both initialization and standard evaluation directories")
    output, missing = [], []
    for key, row in rows.items():
        if row.get("complete_cases", len(payload["case_names"])) != len(payload["case_names"]):
            continue
        if key not in scores:
            files = [root / "standard_evaluations" / key / f"{name}.json"
                     for name in payload["case_names"]]
            if all(p.exists() for p in files):
                gaps = [float(json.loads(p.read_text())["gap_pct"]) for p in files]
                from inventory.common.fitness import metrics
                scores[key] = metrics(gaps)["fitness"]
            else:
                missing.append(key)
                continue
        output.append({**row, "score": float(scores[key]),
                       "num_parameters": row.get("num_parameters", len(row.get("parameter_names", ())))})
    if missing:
        raise ValueError(f"missing archive scores for {len(missing)} complete policies; export STAGE3_SCORE_INDEX.json from original results")
    return output


def export_score_index(checkpoint):
    """Generate a portable derived score artifact alongside original results."""
    root = Path(checkpoint).parent
    payload = json.loads(Path(checkpoint).read_text())
    # The loader resolves non-event initialization scores from original files.
    rows = load_archive(checkpoint, payload)
    target = root / "STAGE3_SCORE_INDEX.json"
    target.write_text(json.dumps({r["ast_hash"]: r["score"] for r in rows}, sort_keys=True) + "\n")
    return target
