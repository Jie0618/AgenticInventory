"""Static metadata and constructors for perishable computational baselines."""

from __future__ import annotations

from dataclasses import dataclass

from inventory.core import InventoryCase
from optimization.models.types import ParameterSpec

from .policies import NAMES, PerishableBalancingPolicy


@dataclass(frozen=True)
class ComputationalPerishableSpec:
    name: str
    display_name: str
    paper: str
    doi: str
    parameter_specs: tuple[ParameterSpec, ...] = ()
    implementation_version: str = "perishable_balancing_randomized_integer"
    implementation_note: str = (
        "Adjacent feasible integer actions randomized to preserve expected balance residual; "
        "Finite-horizon holding/expiry accounting; batched MC; DB critical level unchanged. "
        "No source-paper approximation guarantee claimed for this discrete/capacitated variant."
    )

    def make_provider(self, case: InventoryCase) -> PerishableBalancingPolicy:
        return PerishableBalancingPolicy(case, self.name)


def computational_baselines() -> tuple[ComputationalPerishableSpec, ...]:
    names = {
        "pb_chao": "Proportional Balancing (Chao)",
        "db_chao": "Dual Balancing (Chao)",
        "b_zaw": "Balancing (Zhang–Ayer–White)",
        "tb_zaw": "Truncated Balancing (Zhang–Ayer–White)",
    }
    return tuple(
        ComputationalPerishableSpec(
            name=name,
            display_name=names[name],
            paper=(
                "Chao et al. (2015), Approximation Algorithms for Perishable Inventory Systems"
                if name.endswith("chao")
                else "Zhang, Ayer, and White (2023), Truncated Balancing Policy for Perishable Inventory Management"
            ),
            doi=("10.1287/opre.2015.1386" if name.endswith("chao") else "10.1287/msom.2022.0644"),
        )
        for name in NAMES
    )


__all__ = ["ComputationalPerishableSpec", "computational_baselines"]
