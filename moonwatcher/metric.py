"""Metric computation utilities for classification and detection tasks."""

from typing import Union, Dict, Any, List, Optional

import torch
import torchmetrics

from moonwatcher.utils.data import TaskType
from moonwatcher.dataset.dataset import Moonwatcher, Slice
from moonwatcher.annotations import Labels, BoundingBoxes

# -----------------------------------------------------------------------------
# Torchmetrics Registry
# -----------------------------------------------------------------------------

_METRIC_FUNCTIONS = {
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

_METRIC_KEYS = {
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


def _parse_metric_class(metric_class: Optional[Union[int, str]], dataset: Moonwatcher) -> Optional[int]:
    """Parse and validate the metric class parameter.

    Parameters
    ----------
    metric_class : Optional[Union[int, str]]
        The metric class to parse. Can be an integer (class ID) or string (class name).
    dataset : Moonwatcher
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
            "label_to_name mapping not provided; cannot parse metric_class.")

    if isinstance(metric_class, int):
        return metric_class

    if isinstance(metric_class, str):
        if metric_class not in dataset.label_to_name.values():
            raise ValueError(
                f"Class name '{metric_class}' not found in dataset.label_to_name.")
        # Invert the mapping: {id: name} becomes {name: id}
        for k, v in dataset.label_to_name.items():
            if v == metric_class:
                return int(k)

    raise TypeError(f"Unsupported type for metric_class: {type(metric_class)}")


def _convert_detection_dicts(annotations_list: List[Dict[str, Any]]) -> None:
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
# Classification Metric
# -----------------------------------------------------------------------------


def _calculate_classification_metric(
    groundtruths_loaded: List[Labels],
    predictions_loaded: List[Labels],
    dataset: Moonwatcher,
    metric: str,
    metric_parameters: Dict[str, Any],
    metric_class: Optional[Union[int, str]],
) -> float:
    """Compute a classification metric using the provided Labels annotations.

    Parameters
    ----------
    groundtruths_loaded : List[Labels]
        List of ground truth label annotations.
    predictions_loaded : List[Labels]
        List of predicted label annotations.
    dataset : Moonwatcher
        The dataset containing task and class information.
    metric : str
        Name of the metric to compute.
    metric_parameters : Dict[str, Any]
        Additional parameters for the metric computation.
    metric_class : Optional[Union[int, str]]
        Optional class ID or name to compute class-specific metrics.

    Returns
    -------
    float
        The computed metric value.

    Raises
    ------
    RuntimeError
        If there is an error during metric computation.
    ValueError
        If no predictions are provided.
    """
    try:
        gt_tensor = torch.stack([ann.labels.squeeze()
                                for ann in groundtruths_loaded])

        pred_list = [
            ann.scores if ann.scores is not None else ann.labels.squeeze()
            for ann in predictions_loaded
        ]
        if not pred_list:
            raise ValueError("No predictions provided to compute the metric.")
        pred_tensor = torch.stack(pred_list)

        # Set default averaging if not specified.
        if "average" not in metric_parameters:
            metric_parameters["average"] = "macro"

        # If a specific class is requested, override averaging.
        metric_class_id = _parse_metric_class(metric_class, dataset)
        if metric_class_id is not None:
            metric_parameters["average"] = "none"

        metric_fn = _METRIC_FUNCTIONS[metric]
        metric_value = metric_fn(
            pred_tensor,
            gt_tensor,
            task=dataset.task,
            num_classes=dataset.num_classes,
            **metric_parameters,
        )

        if metric_class_id is not None:
            metric_value = metric_value[metric_class_id]

        return float(metric_value.item()) if hasattr(metric_value, "item") else float(metric_value)
    except Exception as e:
        raise RuntimeError(f"Error in classification metric computation: {str(e)}") from e

# -----------------------------------------------------------------------------
# Detection Metric
# -----------------------------------------------------------------------------


def _calculate_detection_metric(
    groundtruths_loaded: List[BoundingBoxes],
    predictions_loaded: List[BoundingBoxes],
    dataset: Moonwatcher,
    metric: str,
    metric_parameters: Dict[str, Any],
    metric_class: Optional[Union[int, str]],
) -> float:
    """Compute a detection metric using BoundingBoxes annotations.

    Parameters
    ----------
    groundtruths_loaded : List[BoundingBoxes]
        List of ground truth bounding box annotations.
    predictions_loaded : List[BoundingBoxes]
        List of predicted bounding box annotations.
    dataset : Moonwatcher
        The dataset containing task and class information.
    metric : str
        Name of the metric to compute.
    metric_parameters : Dict[str, Any]
        Additional parameters for the metric computation.
    metric_class : Optional[Union[int, str]]
        Optional class ID or name to compute class-specific metrics.

    Returns
    -------
    float
        The computed metric value.

    Raises
    ------
    RuntimeError
        If there is an error during metric computation.
    """
    try:
        gt_list = [ann.to_dict() for ann in groundtruths_loaded]
        pred_list = [ann.to_dict() for ann in predictions_loaded]

        _convert_detection_dicts(gt_list)
        _convert_detection_dicts(pred_list)

        metric_class_id = _parse_metric_class(metric_class, dataset)
        if metric_class_id is not None:
            metric_parameters["class_metrics"] = True

        if metric in ["mAP", "mAP_small", "mAP_medium", "mAP_large"]:
            metric_parameters["iou_type"] = "bbox"

        metric_fn = _METRIC_FUNCTIONS[metric](**metric_parameters)
        metric_fn.update(pred_list, gt_list)
        metric_value_dict = metric_fn.compute()

        if metric_class_id is not None:
            if metric in ["mAP", "mAP_small", "mAP_medium", "mAP_large"]:
                classes = metric_value_dict.get("classes", None)
                if classes is not None:
                    index = torch.where(classes == metric_class_id)[0]
                    result = metric_value_dict["map_per_class"][index].item(
                    ) if index.numel() > 0 else 0.0
                else:
                    result = 0.0
            else:
                key = f"{_METRIC_KEYS[metric]}/cl_{metric_class_id}"
                result = metric_value_dict.get(key, 0.0)
        else:
            result = metric_value_dict[_METRIC_KEYS[metric]]

        return float(result.item()) if hasattr(result, "item") else float(result)
    except Exception as e:
        raise RuntimeError(f"Error in detection metric computation: {str(e)}") from e

# -----------------------------------------------------------------------------
# High-Level Metric Calculation
# -----------------------------------------------------------------------------


def calculate_metric(
    dataset: Union[Moonwatcher, Slice],
    indices: List[int],
    metric: str,
    metric_parameters: Optional[Dict[str, Any]] = None,
    metric_class: Optional[Union[int, str]] = None,
) -> float:
    """Compute a metric (classification or detection) on the provided dataset.

    Parameters
    ----------
    dataset : Union[Moonwatcher, Slice]
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

    groundtruths_loaded = [
        dataset.groundtruths.get(i) for i in indices]
    predictions_loaded = [
        dataset.predictions.get(i) for i in indices]

    if dataset.task_type == TaskType.CLASSIFICATION.value:
        return _calculate_classification_metric(
            groundtruths_loaded,
            predictions_loaded,
            dataset,
            metric,
            metric_parameters,
            metric_class,
        )
    elif dataset.task_type == TaskType.DETECTION.value:
        return _calculate_detection_metric(
            groundtruths_loaded,
            predictions_loaded,
            dataset,
            metric,
            metric_parameters,
            metric_class,
        )
    else:
        raise ValueError(f"Unsupported task type: {dataset.task_type}")
