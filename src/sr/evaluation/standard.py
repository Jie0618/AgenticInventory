"""Canonical Standard-mode SR result and configuration contracts."""

from __future__ import annotations

import json
import math
from pathlib import Path

from optimization.models.types import MonteCarloProposalConfig, OptimizationConfig
from sr.core.models import SearchCandidate, SearchIndividual


def standard_optimization_config(random_seed: int) -> OptimizationConfig:
    return OptimizationConfig(
        mode="standard",
        objective="policy_cost",
        random_seed=int(random_seed),
        mc_proposal=MonteCarloProposalConfig(enabled=True, routing_mode="auto"),
    )


def load_standard_result(
    path: Path,
    candidate: SearchCandidate,
    case_name: str,
    random_seed: int,
    evaluation_signature: str,
) -> SearchIndividual | None:
    try:
        row = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(row, dict):
            return None
        if (
            row.get("schema_version") != 2
            or row.get("case") != case_name
            or row.get("mode") != "standard"
            or row.get("ast_hash") != candidate.ast_hash
            or row.get("ast") != candidate.ast
            or int(row.get("independent_warm_starts", -1)) != 0
            or row.get("optimizer_config_version") != "standard-independent"
            or int(row.get("random_seed", -1)) != int(random_seed)
            or row.get("evaluation_signature") != evaluation_signature
        ):
            return None
        params = {str(name): float(value) for name, value in dict(row["params"]).items()}
        if set(params) != set(candidate.parameter_names) or not all(math.isfinite(value) for value in params.values()):
            return None
        policy_cost = float(row["policy_cost"])
        dp_cost = float(row["dp_cost"])
        gap_pct = float(row["gap_pct"])
        optimizer_seconds = float(row["optimizer_seconds"])
        completed_at = float(row["completed_at"])
        if not all(math.isfinite(value) for value in (policy_cost, dp_cost, gap_pct, optimizer_seconds, completed_at)) or dp_cost <= 0 or optimizer_seconds < 0:
            return None
        if not math.isclose(gap_pct, 100.0 * (policy_cost / dp_cost - 1.0), rel_tol=1.0e-10, abs_tol=1.0e-10):
            return None
        return SearchIndividual(
            candidate=candidate,
            params=params,
            policy_cost=policy_cost,
            dp_cost=dp_cost,
            gap_pct=gap_pct,
            status=str(row["status"]),
            route=str(row["route"]),
            optimizer_seconds=optimizer_seconds,
            completed_at=completed_at,
            result_path=str(path),
            optimizer_metadata=dict(row.get("optimizer_metadata", {})),
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


__all__ = ["load_standard_result", "standard_optimization_config"]
