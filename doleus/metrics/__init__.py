"""Metric computation utilities for model evaluation and analysis."""

from doleus.metrics.base import METRIC_FUNCTIONS, calculate_metric
from doleus.metrics.classification import calculate_classification_metric
from doleus.metrics.detection import calculate_detection_metric

__all__ = [
    "calculate_metric",
    "calculate_classification_metric",
    "calculate_detection_metric",
    "METRIC_FUNCTIONS",
]
