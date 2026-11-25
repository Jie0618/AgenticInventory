"""Case-derived binding for importing prior SR frontiers."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Mapping

from inventory.core import InventoryCase
from inventory.policy.motifs import motifs_for_config
from sr.analysis.frontier import FrontierSeedBundle
from sr.analysis.frontier import import_frontier_seeds as import_generic_frontier_seeds
from sr.core.inventory import InventorySearchBindings
from sr.core.models import SearchConfig


def import_frontier_seeds(
    checkpoint_paths: Mapping[str, Path],
    *,
    case: InventoryCase,
    config: SearchConfig,
) -> FrontierSeedBundle:
    bindings = InventorySearchBindings.build_context_free(case, config)
    return import_generic_frontier_seeds(
        checkpoint_paths,
        base_motifs=motifs_for_config(case.config),
        candidate_factory=partial(bindings.candidate),
        allowed_state_names=case.schema.allowed_state_names,
    )


__all__ = ["FrontierSeedBundle", "import_frontier_seeds"]
