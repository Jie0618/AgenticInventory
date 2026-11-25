"""Shared LTP baseline references for Stage1/2/3."""
from __future__ import annotations

from runtime.artifacts import seed_from_text, stable_key
import fcntl
import json
from pathlib import Path

from optimization.policies.baselines.lead_time_perishable import baselines_for_lead_time_perishable
from runtime.artifacts import atomic_json
from sr.family.composed_reference import catalog_signature, fit_best_baseline_reference, select_reference_baselines

PROTOCOL = "lead-time-perishable-best-formula-baseline-shared-bank"


def seed(*tokens):
    return seed_from_text("|".join(map(str, tokens)).encode())


class ReferenceWorker:
    seed_namespace = "ltp"

    def __init__(self, *, config_path, reference_cache_dir=None, require_cache=False):
        self.config = json.loads(Path(config_path).read_text())
        self.reference_cache_dir = Path(reference_cache_dir or ".stage3_reference_cache")
        self.require_cache = require_cache

    def catalog(self, case):
        rows = baselines_for_lead_time_perishable(
            int(case.lead_time), int(case.shelf_life), int(case.n), include_extended=True)
        return select_reference_baselines({row.name: row for row in rows},
                                          self.config.get("reference_baselines"))

    def identity(self, case, mode):
        signature = catalog_signature(PROTOCOL, self.catalog(case))
        key = f"{PROTOCOL}|{signature}|best_baseline|{case.case_signature}|{mode}|{json.dumps(self.config, sort_keys=True)}"
        return signature, self.reference_cache_dir/(stable_key(key.encode())+".json")

    def reference_details(self, case, mode):
        signature, path = self.identity(case, mode)
        if self.require_cache:
            if not path.is_file():
                raise FileNotFoundError(f"missing reference for {case.name}; complete the corresponding Stage1 first: {path}")
            return json.loads(path.read_text()), True
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_suffix(".lock").open("w") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if path.exists():
                return json.loads(path.read_text()), True
            value, name, params, candidates, tuning, final = fit_best_baseline_reference(
                case=case, mode=mode, config=self.config, formula_specs=self.catalog(case),
                seed=seed, seed_namespace=self.seed_namespace)
            payload = {"cost": value, "policy": name, "params": params,
                       "reference_type": "best_formula_baseline_empirical_mc",
                       "reference_protocol": PROTOCOL, "catalog_signature": signature,
                       "tuning_seed": tuning, "evaluation_seed": final, "candidates": candidates}
            atomic_json(payload, path)
            return payload, False

    def _reference(self, case, mode):
        payload, hit = self.reference_details(case, mode)
        return float(payload["cost"]), hit, str(payload["reference_type"])
