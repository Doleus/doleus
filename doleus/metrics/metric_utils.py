"""Utility functions and constants for metric calculations."""

from typing import Any, Dict, Optional, Union

import torch
import torchmetrics

from doleus.datasets import Doleus

# -----------------------------------------------------------------------------
# Torchmetrics Registry
# -----------------------------------------------------------------------------

METRIC_FUNCTIONS = {
    "Accuracy": torchmetrics.functional.accuracy,
    "Precision": torchmetrics.functional.precision,
    "Recall": torchmetrics.functional.recall,
    "F1_Score": torchmetrics.functional.f1_score,
    "HammingDistance": torchmetrics.functional.hamming_distance,
    "mAP": torchmetrics.detection.MeanAveragePrecision,
    "mAP_small": torchmetrics.detection.MeanAveragePrecision,
    "mAP_medium": torchmetrics.detection.MeanAveragePrecision,
    "mAP_large": torchmetrics.detection.MeanAveragePrecision,
    "CompleteIntersectionOverUnion": torchmetrics.detection.CompleteIntersectionOverUnion,
    "DistanceIntersectionOverUnion": torchmetrics.detection.DistanceIntersectionOverUnion,
    "GeneralizedIntersectionOverUnion": torchmetrics.detection.GeneralizedIntersectionOverUnion,
    "IntersectionOverUnion": torchmetrics.detection.IntersectionOverUnion,
}

METRIC_KEYS = {
    "mAP": "map",
    "mAP_small": "map_small",
    "mAP_medium": "map_medium",
    "mAP_large": "map_large",
    "CompleteIntersectionOverUnion": "ciou",
    "DistanceIntersectionOverUnion": "diou",
    "GeneralizedIntersectionOverUnion": "giou",
    "IntersectionOverUnion": "iou",
}

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def parse_metric_class(
    metric_class: Optional[Union[int, str]], dataset: Doleus
) -> Optional[int]:
    """Parse and validate the metric class parameter.

    Parameters
    ----------
    metric_class : Optional[Union[int, str]]
        The metric class to parse. Can be an integer (class ID) or string (class name).
    dataset : Doleus
        The dataset containing the label_to_name mapping.

    Returns
    -------
    Optional[int]
        The parsed class ID, or None if metric_class was None.

    Raises
    ------
    ValueError
        If label_to_name mapping is not provided or class name not found.
    TypeError
        If metric_class is of unsupported type.
    """
    if metric_class is None:
        return None

    if dataset.label_to_name is None:
        raise ValueError(
            "label_to_name mapping not provided; cannot parse metric_class."
        )

    if isinstance(metric_class, int):
        return metric_class

    if isinstance(metric_class, str):
        if metric_class not in dataset.label_to_name.values():
            raise ValueError(
                f"Class name '{metric_class}' not found in dataset.label_to_name."
            )
        # Invert the mapping: {id: name} becomes {name: id}
        for k, v in dataset.label_to_name.items():
            if v == metric_class:
                return int(k)

    raise TypeError(f"Unsupported type for metric_class: {type(metric_class)}")


def convert_detection_dicts(annotations_list: list[Dict[str, Any]]) -> None:
    """Convert bounding box dictionary values to torch.Tensor format.

    Parameters
    ----------
    annotations_list : list[Dict[str, Any]]
        List of annotation dictionaries containing bounding box information.
    """
    for ann in annotations_list:
        ann["boxes"] = torch.tensor(ann["boxes"], dtype=torch.float32)
        ann["labels"] = torch.tensor(ann["labels"], dtype=torch.int64)
        if "scores" in ann:
            ann["scores"] = torch.tensor(ann["scores"], dtype=torch.float32)
