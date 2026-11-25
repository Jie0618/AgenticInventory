"""Stable repository paths for lead-time workflow adapters."""

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = REPOSITORY_ROOT / "experiments" / "lead_time"
STAGE1_DATA = EXPERIMENT_ROOT / "stage1_single_distribution_sr"
STAGE1_L3_RESULTS = STAGE1_DATA / "results" / "l3_ten_case_production"
STAGE2_DATA = EXPERIMENT_ROOT / "stage2_multi_distribution_sr"
STAGE3_DATA = EXPERIMENT_ROOT / "stage3_multi_leadtime_sr"
STAGE1_INPUTS = STAGE1_DATA / "results" / ".inputs"


__all__ = [
    "EXPERIMENT_ROOT",
    "REPOSITORY_ROOT",
    "STAGE1_INPUTS",
    "STAGE1_DATA",
    "STAGE1_L3_RESULTS",
    "STAGE2_DATA",
    "STAGE3_DATA",
]
