"""Shared L+B baseline references for all three training stages."""
from __future__ import annotations

from runtime.artifacts import seed_from_text, stable_key
import fcntl
import json
from pathlib import Path

from optimization.policies.baselines.lead_time_backlog import baselines_for_lead_time_backlog
from sr.family.composed_reference import catalog_signature, fit_best_baseline_reference, select_reference_baselines


_REFERENCE_PROTOCOL = "lead-time-backlog-best-formula-baseline-shared-bank"


def _seed(*tokens: object) -> int:
    return seed_from_text("|".join(map(str, tokens)).encode())


def _catalog(case, names=None):
    rows = tuple(baselines_for_lead_time_backlog(
        int(case.lead_time), int(case.config.backlog.patience), int(case.n)
    ))
    return select_reference_baselines({row.name: row for row in rows}, names)


class ReferenceWorker:
    def __init__(self, *, config_path: Path, reference_cache_dir: Path | None = None,
                 reference_mode: str = "best_baseline") -> None:
        self.config_path = str(config_path)
        self.config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        self.reference_cache_dir = Path(reference_cache_dir or ".stage3_reference_cache")
        self.reference_mode = str(reference_mode).lower()
        if self.reference_mode != "best_baseline":
            raise ValueError("lead-time + backlog reference_mode must be best_baseline")

    def _reference(self, case, mode):
        payload, hit = self.reference_details(case, mode)
        return float(payload["cost"]), hit, str(payload["reference_type"])

    def reference_details(self, case, mode):
        """Return the signed baseline results used by every experiment stage."""
        self.reference_cache_dir.mkdir(parents=True, exist_ok=True)
        formula = _catalog(case, self.config.get("reference_baselines"))
        signature = catalog_signature(_REFERENCE_PROTOCOL, formula)
        key = (
            f"{_REFERENCE_PROTOCOL}|{signature}|{self.reference_mode}|"
            f"{case.case_signature}|{mode}|{json.dumps(self.config, sort_keys=True)}"
        )
        path = self.reference_cache_dir / (stable_key(key.encode()) + ".json")
        with path.with_suffix(".lock").open("w") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                return payload, True
            value, name, params, candidates, tuning_seed, final_seed = fit_best_baseline_reference(
                case=case, mode=mode, config=self.config, formula_specs=formula,
                seed=_seed, seed_namespace="ltb",
            )
            payload = {
                "cost": value, "policy": name, "params": params,
                "reference_type": "best_formula_baseline_empirical_mc",
                "reference_protocol": _REFERENCE_PROTOCOL,
                "catalog_signature": signature, "tuning_seed": tuning_seed,
                "evaluation_seed": final_seed, "candidates": candidates,
            }
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
            temporary.replace(path)
            return payload, False
