"""Classification-specific metric functionality."""

from typing import Any, Dict, List, Optional, Union

import torch

from doleus.annotations import Labels
from doleus.datasets import Doleus
from doleus.metrics.base import METRIC_FUNCTIONS, parse_metric_class


def calculate_classification_metric(
    groundtruths_loaded: List[Labels],
    predictions_loaded: List[Labels],
    dataset: Doleus,
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
    dataset : Doleus
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
        # Build ground truth tensor: squeeze each annotation to get a scalar.
        gt_tensor = torch.stack([ann.labels.squeeze() for ann in groundtruths_loaded])

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
        metric_class_id = parse_metric_class(metric_class, dataset)
        if metric_class_id is not None:
            metric_parameters["average"] = "none"

        metric_fn = METRIC_FUNCTIONS[metric]
        metric_value = metric_fn(
            pred_tensor,
            gt_tensor,
            task=dataset.task,
            num_classes=dataset.num_classes,
            **metric_parameters,
        )

        if metric_class_id is not None:
            metric_value = metric_value[metric_class_id]

        return (
            float(metric_value.item())
            if hasattr(metric_value, "item")
            else float(metric_value)
        )
    except Exception as e:
        raise RuntimeError(
            f"Error in classification metric computation: {str(e)}"
        ) from e
