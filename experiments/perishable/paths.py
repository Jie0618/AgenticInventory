"""Stable repository paths for pure-perishable workflow adapters."""

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = REPOSITORY_ROOT / "experiments" / "perishable"
STAGE1_DATA = EXPERIMENT_ROOT / "stage1_single_distribution_sr"
STAGE1_M3_RESULTS = STAGE1_DATA / "results" / "m3_n22_stationary_headroom_ten_case_production"
STAGE2_DATA = EXPERIMENT_ROOT / "stage2_multi_distribution_sr"
STAGE1_INPUTS = STAGE1_DATA / "results" / ".inputs"

__all__ = [
    "EXPERIMENT_ROOT",
    "REPOSITORY_ROOT",
    "STAGE1_INPUTS",
    "STAGE1_DATA",
    "STAGE1_M3_RESULTS",
    "STAGE2_DATA",
]
