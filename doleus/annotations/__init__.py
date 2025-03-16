"""Annotation classes for storing model predictions and ground truths."""

from doleus.annotations.base import Annotation, Annotations
from doleus.annotations.classification import Labels
from doleus.annotations.detection import BoundingBoxes

__all__ = [
    "Annotation",
    "Annotations",
    "Labels",
    "BoundingBoxes",
]
