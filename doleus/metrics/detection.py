"""Detection-specific metric functionality."""

from typing import Any, Dict, List, Optional, Union

import torch

from doleus.annotations import BoundingBoxes
from doleus.datasets import Doleus
from doleus.metrics.base import (METRIC_FUNCTIONS, METRIC_KEYS,
                                 convert_detection_dicts, parse_metric_class)


def calculate_detection_metric(
    groundtruths_loaded: List[BoundingBoxes],
    predictions_loaded: List[BoundingBoxes],
    dataset: Doleus,
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
    """
    try:
        gt_list = [ann.to_dict() for ann in groundtruths_loaded]
        pred_list = [ann.to_dict() for ann in predictions_loaded]

        convert_detection_dicts(gt_list)
        convert_detection_dicts(pred_list)

        metric_class_id = parse_metric_class(metric_class, dataset)
        if metric_class_id is not None:
            metric_parameters["class_metrics"] = True

        if metric in ["mAP", "mAP_small", "mAP_medium", "mAP_large"]:
            metric_parameters["iou_type"] = "bbox"

        metric_fn = METRIC_FUNCTIONS[metric](**metric_parameters)
        metric_fn.update(pred_list, gt_list)
        metric_value_dict = metric_fn.compute()

        if metric_class_id is not None:
            if metric in ["mAP", "mAP_small", "mAP_medium", "mAP_large"]:
                classes = metric_value_dict.get("classes", None)
                if classes is not None:
                    index = torch.where(classes == metric_class_id)[0]
                    result = (
                        metric_value_dict["map_per_class"][index].item()
                        if index.numel() > 0
                        else 0.0
                    )
                else:
                    result = 0.0
            else:
                key = f"{METRIC_KEYS[metric]}/cl_{metric_class_id}"
                result = metric_value_dict.get(key, 0.0)
        else:
            result = metric_value_dict[METRIC_KEYS[metric]]

        return float(result.item()) if hasattr(result, "item") else float(result)
    except Exception as e:
        raise RuntimeError(f"Error in detection metric computation: {str(e)}") from e
