"""Base metric functionality for classification and detection tasks."""

from typing import Any, Dict, List, Optional, Union

import torch
import torchmetrics

from doleus.datasets import Doleus, Slice
from doleus.utils.data import TaskType

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


def convert_detection_dicts(annotations_list: List[Dict[str, Any]]) -> None:
    """Convert bounding box dictionary values to torch.Tensor format.

    Parameters
    ----------
    annotations_list : List[Dict[str, Any]]
        List of annotation dictionaries containing bounding box information.
    """
    for ann in annotations_list:
        ann["boxes"] = torch.tensor(ann["boxes"], dtype=torch.float32)
        ann["labels"] = torch.tensor(ann["labels"], dtype=torch.int64)
        if "scores" in ann:
            ann["scores"] = torch.tensor(ann["scores"], dtype=torch.float32)


# -----------------------------------------------------------------------------
# High-Level Metric Calculation
# -----------------------------------------------------------------------------

def calculate_metric(
    dataset: Union[Doleus, Slice],
    indices: List[int],
    metric: str,
    metric_parameters: Optional[Dict[str, Any]] = None,
    metric_class: Optional[Union[int, str]] = None,
) -> float:
    """Compute a metric (classification or detection) on the provided dataset.

    Parameters
    ----------
    dataset : Union[Doleus, Slice]
        The dataset or slice to compute metrics on.
    indices : List[int]
        List of indices to compute the metric for.
    metric : str
        Name of the metric to compute.
    metric_parameters : Optional[Dict[str, Any]], optional
        Additional parameters for the metric computation, by default None.
    metric_class : Optional[Union[int, str]], optional
        Optional class ID or name to compute class-specific metrics, by default None.

    Returns
    -------
    float
        The computed metric value.

    Raises
    ------
    ValueError
        If the task type is not supported.
    """
    metric_parameters = metric_parameters or {}

    groundtruths_loaded = [dataset.groundtruths.get(i) for i in indices]
    predictions_loaded = [dataset.predictions.get(i) for i in indices]

    if dataset.task_type == TaskType.CLASSIFICATION.value:
        # Import here to avoid circular import
        from doleus.metrics.classification import calculate_classification_metric
        return calculate_classification_metric(
            groundtruths_loaded,
            predictions_loaded,
            dataset,
            metric,
            metric_parameters,
            metric_class,
        )
    elif dataset.task_type == TaskType.DETECTION.value:
        # Import here to avoid circular import
        from doleus.metrics.detection import calculate_detection_metric
        return calculate_detection_metric(
            groundtruths_loaded,
            predictions_loaded,
            dataset,
            metric,
            metric_parameters,
            metric_class,
        )
    else:
        raise ValueError(f"Unsupported task type: {dataset.task_type}") 