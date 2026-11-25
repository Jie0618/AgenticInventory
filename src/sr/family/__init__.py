"""Scenario-neutral policy-family search primitives.

The family layer knows only about a policy object, an evaluation context, and
case metrics. Domain packages provide materialization, workers, and initial
seeds; domain vocabulary does not belong here.
"""

from sr.family.models import CaseMetric, FamilyEvaluation, FamilyTask, ScenarioScore
from sr.family.planner import plan_family_tasks
from sr.family.scoring import cross_context_score, scenario_score

__all__ = [
    "CaseMetric",
    "FamilyEvaluation",
    "FamilyTask",
    "ScenarioScore",
    "plan_family_tasks",
    "scenario_score",
    "cross_context_score",
]
