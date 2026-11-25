"""Durable, seed-by-seed LLM template initialization."""
from runtime.artifacts import stable_key
from dataclasses import asdict, replace
import json
from pathlib import Path

from runtime.artifacts import atomic_json
from sr.core.search_control import SEARCH_CONTROL_PROTOCOL
from .proposals import family_from_json, llm_template_initialization
from .validation import validate_search_family

PROTOCOL = "per-seed-llm-template"


def initialize_templates(client, parents, *, domain, output_dir, count=6,
                         minimum=5, max_attempts=3):
    if minimum != 5 or count not in (5, 6) or max_attempts < 1:
        raise ValueError("paper initialization requires five or six templates per seed")
    root = Path(output_dir) / "template_init_seeds"
    seed_keys = {domain.semantic_hash(seed) for seed in parents}
    seen = set(seed_keys)
    families, reports = [], []
    for seed in parents:
        seed_key = domain.semantic_hash(seed)
        identity = dict(protocol=PROTOCOL, seed_hash=seed_key, count=count,
                        minimum=minimum, domain=domain.prompt_payload(),
                        search_control_protocol=SEARCH_CONTROL_PROTOCOL)
        signature = stable_key(json.dumps(identity, sort_keys=True).encode())
        path = root / (seed_key + ".json")
        if path.exists():
            saved = json.loads(path.read_text())
            if saved.get("signature") != signature:
                raise ValueError("per-seed template initialization identity mismatch")
            children = [family_from_json(row, domain=domain) for row in saved["families"]]
            attempts = list(saved["attempts"])
        else:
            children, attempts = [], []
        local_keys = set()
        for child in children:
            validate_search_family(child, domain)
            key = domain.semantic_hash(child)
            if child.parent_family_hash != seed_key or key in seed_keys or key in local_keys:
                raise ValueError("invalid persisted per-seed template lineage or duplicate")
            local_keys.add(key)
        # A resumed invocation gets a fresh bounded refill budget; prior
        # failures remain audited instead of permanently poisoning the seed.
        invocation_attempts = 0
        while len(children) < minimum and invocation_attempts < max_attempts:
            proposed, audit = llm_template_initialization(
                client, (seed,), domain=domain, count=count, existing_templates=children,
                previous_rejections=[row for attempt in attempts[-3:]
                                     for row in attempt.get('rejected', ())])
            invocation_attempts += 1
            attempts.append(audit)
            for child in proposed:
                key = domain.semantic_hash(child)
                if key in seed_keys or key in local_keys or len(children) >= count:
                    continue
                local_keys.add(key)
                children.append(replace(child, family_id=f"template_{seed_key[:16]}_{len(children)+1}"))
            atomic_json(dict(signature=signature, identity=identity, attempts=attempts,
                             families=[asdict(f) for f in children]), path)
            if len(children) >= minimum:
                break
        # Persist valid partial replies for retry, but do not start numerical
        # search with a different population after exhausted initialization.
        retained = []
        for child in children:
            key = domain.semantic_hash(child)
            if key not in seen:
                seen.add(key)
                families.append(child)
                retained.append(key)
        reports.append(dict(seed_hash=seed_key, seed_id=seed.family_id,
                            accepted_count=len(children), retained_hashes=retained,
                            status="complete" if len(children) >= minimum else "partial",
                            cross_seed_duplicates=len(children)-len(retained), attempts=attempts))
    partial = [report["seed_id"] for report in reports if report["status"] == "partial"]
    if partial:
        raise RuntimeError(f"template initialization requires at least five valid templates per seed: {partial}; partial replies saved for retry")
    return tuple(families), dict(protocol=PROTOCOL, status="complete", seed_count=len(parents),
                                 requested_per_seed=count, minimum_per_seed=minimum,
                                 accepted_count=len(families), seeds=reports,
                                 partial_seed_count=sum(r['status'] == 'partial' for r in reports))
