# metric.py

from typing import Union, Dict, Any, List

import torch
import torchmetrics

from moonwatcher.utils.data import TaskType
from moonwatcher.dataset.dataset import Moonwatcher, Slice
from moonwatcher.annotations import Labels, BoundingBoxes


def get_original_indices(dataset_or_slice: Union[Moonwatcher, Slice]) -> List[int]:
    """
    Recursively retrieve the 'original' indices from a dataset or slice.
    """
    if isinstance(dataset_or_slice, Slice):
        parent_indices = get_original_indices(
            dataset_or_slice.original_dataset)
        return [parent_indices[i] for i in dataset_or_slice.indices]
    elif isinstance(dataset_or_slice, Moonwatcher):
        return list(range(len(dataset_or_slice.dataset)))
    else:
        raise TypeError("Unsupported dataset type for get_original_indices.")


def _parse_metric_class(metric_class: Any, dataset) -> int:
    """
    If metric_class is an int, return it directly.
    If it's a string, convert to label_id using dataset.label_to_name.
    """
    if metric_class is None:
        return None

    if dataset.label_to_name is None:
        raise ValueError(
            "label_to_name mapping not provided; cannot parse metric_class.")

    if isinstance(metric_class, int):
        return metric_class

    if isinstance(metric_class, str):
        # invert {id: "class_name"} to {"class_name": id}
        if metric_class not in dataset.label_to_name.values():
            raise ValueError(
                f"Class name '{metric_class}' not found in dataset.label_to_name.")
        for k, v in dataset.label_to_name.items():
            if v == metric_class:
                return int(k)

    raise TypeError(f"Unsupported type for metric_class: {type(metric_class)}")


# -----------------------------------------------------------------------------
# CLASSIFICATION METRIC
# -----------------------------------------------------------------------------

def _calculate_classification_metric(
    groundtruths_loaded: List[Labels],
    predictions_loaded: List[Labels],
    dataset: Moonwatcher,
    metric: str,
    metric_parameters: Dict[str, Any],
    metric_class: Any,
) -> float:
    """
    Compute classification metrics using the annotation objects:
      - groundtruths_loaded[i] is a Labels object for ground truth
      - predictions_loaded[i] is a Labels object for predictions (with .scores possibly)
    """
    try:
        # Build ground truth tensor
        # Each groundtruth.labels might be shape [1] if single-label
        # We flatten them into shape [N]
        gt_list = []
        for ann in groundtruths_loaded:
            # e.g., ann.labels shape [1] for single-label classification
            # or shape [k] for multi-label
            # We assume single-label currently:
            gt_list.append(ann.labels.squeeze())

        gt_tensor = torch.stack(gt_list, dim=0)  # shape [N]

        # Build predictions as EITHER:
        #   - The argmax of ann.scores (if scores exist)
        #   - The ann.labels (if user stored predicted class IDs directly)
        pred_list = []
        for ann in predictions_loaded:
            if ann.scores is not None:
                # We store the raw distribution in pred_list if shape [num_classes].
                pred_list.append(ann.scores)
            else:
                # If user gave a single class label as "prediction"
                pred_list.append(ann.labels.squeeze())

        # If any item in pred_list is a scalar int, we want a final shape [N] of ints.
        # If it's a 1D distribution [num_classes], we want shape [N, num_classes].
        # We check the type of the first element to see if it's a distribution or a single scalar.
        if len(pred_list) == 0:
            raise ValueError("No predictions to compute metric on.")

        # Check shape of the first element
        # TODO: This is a bit hacky, we should find a better way to handle this.
        first_pred = pred_list[0]
        if first_pred.dim() == 0:
            # e.g. shape [] => single scalar int
            # We cat them into shape [N]
            pred_tensor = torch.stack([p.unsqueeze(0)
                                      for p in pred_list]).squeeze(-1)
        elif first_pred.dim() == 1:
            # e.g. shape [num_classes] => distribution
            pred_tensor = torch.stack(pred_list, dim=0)  # shape [N, C]
        else:
            raise ValueError(
                "Unsupported shape for prediction. Expect either scalar or 1D distribution."
            )

        if "average" not in metric_parameters:
            metric_parameters["average"] = "macro"

        if metric_class is not None:
            metric_class_id = _parse_metric_class(metric_class, dataset)
            metric_parameters["average"] = "none"
        else:
            metric_class_id = None

        metric_function = _METRIC_FUNCTIONS[metric]
        # If pred_tensor is shape [N, C], we pass it as distribution
        # If shape [N], we pass it as integer labels
        metric_value = metric_function(
            pred_tensor,
            gt_tensor,
            task=dataset.task,  # e.g., "multiclass"
            num_classes=dataset.num_classes,
            **metric_parameters,
        )

        # If user requested a specific class
        if metric_class_id is not None:
            metric_value = metric_value[metric_class_id]

        return float(metric_value.item()) if hasattr(metric_value, "item") else float(metric_value)

    except Exception as e:
        raise RuntimeError(f"Error in classification metric computation: {e}")


# -----------------------------------------------------------------------------
# DETECTION METRIC
# -----------------------------------------------------------------------------

def _convert_detection_dicts(annotations_list: List[Dict[str, Any]]) -> None:
    """
    In-place convert the bounding box dicts to proper torch.Tensor format.
    """
    for ann in annotations_list:
        ann["boxes"] = torch.tensor(ann["boxes"], dtype=torch.float32)
        ann["labels"] = torch.tensor(ann["labels"], dtype=torch.int64)
        if "scores" in ann:
            ann["scores"] = torch.tensor(ann["scores"], dtype=torch.float32)


def _calculate_detection_metric(
    groundtruths_loaded: List[BoundingBoxes],
    predictions_loaded: List[BoundingBoxes],
    dataset: Moonwatcher,
    metric: str,
    metric_parameters: Dict[str, Any],
    metric_class: Any,
) -> float:
    """
    For detection, each annotation is a BoundingBoxes object.
    We'll convert them to the dict format expected by torchmetrics, then run e.g. MeanAveragePrecision.
    """
    try:
        gt_list = [ann.to_dict() for ann in groundtruths_loaded]
        pred_list = [ann.to_dict() for ann in predictions_loaded]

        _convert_detection_dicts(gt_list)
        _convert_detection_dicts(pred_list)

        if metric_class is not None:
            metric_parameters["class_metrics"] = True
            metric_class_id = _parse_metric_class(metric_class, dataset)
        else:
            metric_class_id = None

        # For mAP and variations, specify iou_type="bbox"
        if metric in ["mAP", "mAP_small", "mAP_medium", "mAP_large"]:
            metric_parameters["iou_type"] = "bbox"

        metric_function = _METRIC_FUNCTIONS[metric](**metric_parameters)
        metric_function.update(pred_list, gt_list)
        metric_value_dict = metric_function.compute()

        if metric_class_id is not None:
            # e.g. user wants mAP for class X
            if metric in ["mAP", "mAP_small", "mAP_medium", "mAP_large"]:
                # find the index of the requested class
                index = torch.where(
                    metric_value_dict["classes"] == metric_class_id)[0]
                if index.numel() > 0:
                    result = metric_value_dict["map_per_class"][index].item()
                else:
                    result = 0.0
            else:
                # For IoU-based metrics
                key = f"{_METRIC_KEYS[metric]}/cl_{metric_class_id}"
                result = metric_value_dict.get(key, 0.0)
        else:
            # just get the top-level metric
            result = metric_value_dict[_METRIC_KEYS[metric]]

        return float(result.item()) if hasattr(result, "item") else float(result)

    except Exception as e:
        raise RuntimeError(f"Error in detection metric computation: {e}")


# -----------------------------------------------------------------------------
# HIGH-LEVEL ENTRY POINT
# -----------------------------------------------------------------------------
def calculate_metric(
    dataset_or_slice: Union[Moonwatcher, Slice],
    predictions: Union[torch.Tensor, List[Dict[str, Any]]],
    metric: str,
    metric_parameters: Dict[str, Any] = None,
    metric_class: Any = None,
) -> float:
    """
    High-level function to compute a metric for classification or detection tasks.
    Now includes debugging prints to check datapoint alignment.
    """
    if metric_parameters is None:
        metric_parameters = {}

    # Identify underlying dataset if 'dataset_or_slice' is a Slice
    if isinstance(dataset_or_slice, Slice):
        parent_dataset = dataset_or_slice.original_dataset
    else:
        parent_dataset = dataset_or_slice

    # Convert raw 'predictions' into annotation objects on the entire parent dataset
    parent_dataset.add_predictions_from_model_outputs(predictions)

    # Gather relevant indices (if slice, this is a subset; if full dataset, it's [0..N-1])
    relevant_ids = get_original_indices(dataset_or_slice)

    # Load ground truths & predictions for these indices
    groundtruths_loaded = [
        parent_dataset.groundtruths.get(i) for i in relevant_ids]
    predictions_loaded = [
        parent_dataset.predictions.get(i) for i in relevant_ids]

    # Now proceed to classification or detection logic
    if parent_dataset.task_type == TaskType.CLASSIFICATION.value:
        return _calculate_classification_metric(
            groundtruths_loaded,
            predictions_loaded,
            parent_dataset,
            metric,
            metric_parameters,
            metric_class,
        )
    elif parent_dataset.task_type == TaskType.DETECTION.value:
        return _calculate_detection_metric(
            groundtruths_loaded,
            predictions_loaded,
            parent_dataset,
            metric,
            metric_parameters,
            metric_class,
        )
    else:
        raise ValueError(f"Unsupported task type: {parent_dataset.task_type}")


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
