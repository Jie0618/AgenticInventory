"""Compatible benchmark imports for shared result scoring."""

from inventory.common.fitness import metrics
from reporting.scores import configuration_scores, cross_configuration_metrics

__all__ = ["metrics", "configuration_scores", "cross_configuration_metrics"]
