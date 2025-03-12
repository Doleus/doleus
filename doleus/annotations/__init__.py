"""Annotation classes for storing model predictions and ground truths."""

from doleus.annotations.base import Annotation, AnnotationStore, GroundTruths, Predictions
from doleus.annotations.classification import Labels
from doleus.annotations.detection import BoundingBoxes

__all__ = [
    "Annotation",
    "AnnotationStore",
    "GroundTruths",
    "Predictions",
    "Labels",
    "BoundingBoxes"
] 