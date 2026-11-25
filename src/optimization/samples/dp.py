"""Supervised action samples from lead-time DP tables."""

from __future__ import annotations

from evaluation.backends.lead_time.dp import DPResult
from evaluation.dp import ComposedDPResult

from optimization.models.types import SupervisedSample
from optimization.samples.reference import reference_action_samples


def dp_action_samples(dp_result: DPResult) -> list[SupervisedSample]:
    """Flatten a DP policy table into supervised samples."""
    return reference_action_samples(dp_result)


def composed_dp_action_samples(
    dp_result: ComposedDPResult,
) -> list[SupervisedSample]:
    """Flatten a canonical composed DP policy into named-state samples."""
    return reference_action_samples(dp_result)


__all__ = ["composed_dp_action_samples", "dp_action_samples"]
