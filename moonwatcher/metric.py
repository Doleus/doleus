# metric.py

from typing import Union, Dict, Any, List

import torch
import torchmetrics

from moonwatcher.utils.data import TaskType
from moonwatcher.dataset.dataset import Moonwatcher, Slice


def get_original_indices(dataset_or_slice: Union[Moonwatcher, Slice]) -> List[int]:
    """
    Recursively retrieve the 'original' indices from a dataset or slice.

    :param dataset_or_slice: either a Moonwatcher object or a Slice
    :return: list of integer indices corresponding to the underlying data items
    """
    if isinstance(dataset_or_slice, Slice):
        parent_indices = get_original_indices(
            dataset_or_slice.original_dataset)
        # The slice's indices refer to the parent's indices, so map them
        return [parent_indices[i] for i in dataset_or_slice.indices]
    elif isinstance(dataset_or_slice, Moonwatcher):
        return list(range(len(dataset_or_slice.dataset)))
    else:
        raise TypeError("Unsupported dataset type for get_original_indices.")


def load_data(
    dataset_or_slice: Union[Moonwatcher, Slice],
    predictions: Union[List[Any], torch.Tensor],
):
    """
    Given a dataset (or slice) plus a predictions array (or list),
    extract the relevant ground truths and predictions for the portion of the dataset.
    """
    relevant_ids = get_original_indices(dataset_or_slice=dataset_or_slice)

    # If it's a Slice, the 'original_dataset' is the reference to the entire dataset
    dataset = (
        dataset_or_slice.original_dataset
        if isinstance(dataset_or_slice, Slice)
        else dataset_or_slice
    )

    groundtruths_loaded = [dataset.groundtruths.get(i) for i in relevant_ids]
    predictions_loaded = [predictions[i] for i in relevant_ids]

    return relevant_ids, dataset, groundtruths_loaded, predictions_loaded


def _parse_metric_class(
    metric_class: Any, dataset
) -> int:
    """
    If metric_class is a string, convert it to the corresponding label ID.
    If metric_class is already an int, just return it.
    Raise an error if the class name is not found in dataset.label_to_name.
    """
    if metric_class is None:
        return None

    # If dataset.label_to_name is not provided, we cannot parse string class names
    if dataset.label_to_name is None:
        raise ValueError(
            "label_to_name mapping is not provided, cannot parse metric_class.")

    # If it's already an int, return it directly
    if isinstance(metric_class, int):
        return metric_class

    # If it's a string, find the corresponding label ID
    if isinstance(metric_class, str):
        if metric_class not in dataset.label_to_name.values():
            raise ValueError(
                f"Class name '{metric_class}' not found in label_to_name mapping.")
        # Convert from class name -> label id
        for k, v in dataset.label_to_name.items():
            if v == metric_class:
                return int(k)

    # If none of the above, user passed an unsupported type
    raise TypeError(f"Unsupported type for metric_class: {type(metric_class)}")


def _calculate_classification_metric(
    groundtruths_loaded,
    predictions_loaded,
    dataset,
    metric: str,
    metric_parameters: Dict[str, Any],
    metric_class: Any,
) -> float:
    """
    Compute classification metrics using torchmetrics.functional, handling
    optional 'metric_class' for per-class values (e.g., per-class Precision).
    """
    try:
        # groundtruths_loaded: list of annotation objects (Labels). E.g. each has .labels
        # predictions_loaded: list of torch.Tensor or same shape
        gt_tensor = torch.stack(
            [gt.labels for gt in groundtruths_loaded]).squeeze()
        pred_tensor = torch.stack([pred for pred in predictions_loaded])

        # Ensure a default 'average' if not provided
        if "average" not in metric_parameters:
            metric_parameters["average"] = "macro"

        # If a class is specified, adjust parameters accordingly
        if metric_class is not None:
            metric_class_id = _parse_metric_class(metric_class, dataset)
            metric_parameters["average"] = "none"
        else:
            metric_class_id = None

        # call the torchmetrics functional metric
        metric_function = _METRIC_FUNCTIONS[metric]
        metric_value = metric_function(
            pred_tensor,
            gt_tensor,
            task=dataset.task,
            num_classes=dataset.num_classes,
            **metric_parameters,
        )

        # If user requested a specific class, pick that index out
        if metric_class_id is not None:
            metric_value = metric_value[metric_class_id]

        if hasattr(metric_value, "item"):
            metric_value = metric_value.item()

        return float(metric_value)

    except Exception as e:
        raise RuntimeError(f"Error in classification metric computation: {e}")


def _convert_detection_dicts(annotations_list: List[Dict[str, Any]]) -> None:
    """
    In-place convert the bounding box dicts to proper torch.Tensor format.
    Each dictionary is expected to have 'boxes' and 'labels'.
    If it has 'scores', convert that as well.
    """
    for ann in annotations_list:
        ann["boxes"] = torch.tensor(ann["boxes"], dtype=torch.float32)
        ann["labels"] = torch.tensor(ann["labels"], dtype=torch.int64)
        if "scores" in ann:
            ann["scores"] = torch.tensor(ann["scores"], dtype=torch.float32)


def _calculate_detection_metric(
    groundtruths_loaded,
    predictions_loaded,
    dataset,
    metric: str,
    metric_parameters: Dict[str, Any],
    metric_class: Any,
) -> float:
    """
    Compute detection metrics using the MeanAveragePrecision or IoU-based
    metrics from torchmetrics.detection.
    """
    try:
        # Convert each groundtruth/prediction to a dict with Tensors
        gt_list = [gt.to_dict() for gt in groundtruths_loaded]
        pred_list = [pred.to_dict() for pred in predictions_loaded]

        _convert_detection_dicts(pred_list)
        _convert_detection_dicts(gt_list)

        # If user requests a specific class, set class_metrics=True
        if metric_class is not None:
            metric_parameters["class_metrics"] = True
            metric_class_id = _parse_metric_class(metric_class, dataset)
        else:
            metric_class_id = None

        # For mAP and variations, we need to specify iou_type="bbox"
        if metric in ["mAP", "mAP_small", "mAP_medium", "mAP_large"]:
            metric_parameters["iou_type"] = "bbox"

        metric_function = _METRIC_FUNCTIONS[metric](**metric_parameters)
        metric_function.update(pred_list, gt_list)
        metric_value_dict = metric_function.compute()

        # If the user requested a particular class, extract that result
        if metric_class_id is not None:
            # For mAP-based metrics
            if metric in ["mAP", "mAP_small", "mAP_medium", "mAP_large"]:
                # We look for where metric_value_dict["classes"] == metric_class_id
                index = torch.where(
                    metric_value_dict["classes"] == metric_class_id)[0]
                if index.numel() > 0:
                    result = metric_value_dict["map_per_class"][index].item()
                    result = 0.0 if result < 0.0 else result  # clamp negative
                else:
                    result = 0.0
            else:
                # e.g., IntersectionOverUnion => look up "iou/cl_{id}" or something
                key = f"{_METRIC_KEYS[metric]}/cl_{metric_class_id}"
                result = metric_value_dict.get(key, 0.0)
        else:
            # No class specified, just get the top-level metric
            result = metric_value_dict[_METRIC_KEYS[metric]]

        if hasattr(result, "item"):
            result = result.item()
        return round(float(result), 5)

    except Exception as e:
        raise RuntimeError(f"Error in detection metric computation: {e}")


def calculate_metric(
    dataset_or_slice: Union[Moonwatcher, Slice],
    predictions: torch.Tensor,
    metric: str,
    metric_parameters: Dict[str, Any] = None,
    metric_class: Any = None,
) -> float:
    """
    High-level function to compute a metric for classification or detection tasks.

    :param dataset_or_slice: either a full Moonwatcher dataset or a slice
    :param predictions: a torch.Tensor or list of predictions (shape depends on classification/detection)
    :param metric: a string key, e.g., "Accuracy", "mAP", "IntersectionOverUnion", etc.
    :param metric_parameters: optional dictionary of parameters to be passed to torchmetrics
    :param metric_class: optional class (as an int label or string name) for per-class metrics
    :return: A float value representing the computed metric
    """
    if metric_parameters is None:
        metric_parameters = {}

    # Load relevant subset of ground truths and predictions
    relevant_ids, dataset, groundtruths_loaded, predictions_loaded = load_data(
        dataset_or_slice, predictions
    )

    # Branch based on classification vs. detection
    if dataset.task_type == TaskType.CLASSIFICATION.value:
        return round(
            _calculate_classification_metric(
                groundtruths_loaded,
                predictions_loaded,
                dataset,
                metric,
                metric_parameters,
                metric_class,
            ),
            5,
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


# ---------------------------------------------------------------------
# Define or extend your metric registry and the related keys
# ---------------------------------------------------------------------
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
