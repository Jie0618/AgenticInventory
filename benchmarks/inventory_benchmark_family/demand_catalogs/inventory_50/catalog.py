"""Load and validate the frozen inventory demand catalog."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

from inventory.common.distributions import DiscreteDemand


SOURCE = Path(__file__).resolve().with_name("distributions.json")


@dataclass(frozen=True)
class DemandRecord:
    demand_id: str
    family: str
    demand: DiscreteDemand
    construction: Mapping[str, object]


@lru_cache(maxsize=1)
def load_demands() -> tuple[DemandRecord, ...]:
    raw = SOURCE.read_bytes()
    payload = json.loads(raw)
    records = tuple(
        DemandRecord(
            demand_id=str(row["id"]),
            family=str(row["family"]),
            demand=DiscreteDemand(
                row["support"],
                row["probabilities"],
                name=str(row["id"]),
                params=dict(row["construction"]),
            ),
            construction=dict(row["construction"]),
        )
        for row in payload
    )
    if len(records) != 50 or len({row.demand_id for row in records}) != 50:
        raise RuntimeError("catalog requires exactly 50 unique distributions")
    if any(
        not math.isclose(float(row.demand.probs.sum()), 1.0, abs_tol=1.0e-12)
        for row in records
    ):
        raise RuntimeError("frozen demand probabilities must sum to one")
    return records


__all__ = ["DemandRecord", "SOURCE", "load_demands"]
