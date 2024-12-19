from typing import Union

import torch
import torchmetrics

from moonwatcher.utils.data import TaskType, Task
from moonwatcher.dataset.dataset import Moonwatcher, Slice


def get_original_indices(dataset_or_slice):
    if isinstance(dataset_or_slice, Slice):
        parent_indices = get_original_indices(
            dataset_or_slice.original_dataset)
        return [parent_indices[i] for i in dataset_or_slice.indices]
    elif isinstance(dataset_or_slice, Moonwatcher):
        return list(range(len(dataset_or_slice.dataset)))
    else:
        raise TypeError("Unsupported dataset type")


def load_data(dataset_or_slice: Union[Moonwatcher, Slice], predictions):
    relevant_ids = get_original_indices(dataset_or_slice=dataset_or_slice)
    dataset = (
        dataset_or_slice.original_dataset
        if isinstance(dataset_or_slice, Slice)
        else dataset_or_slice
    )

    groundtruths_loaded = [dataset.groundtruths.get(i) for i in relevant_ids]
    predictions_loaded = [predictions[i] for i in relevant_ids]

    return relevant_ids, dataset, groundtruths_loaded, predictions_loaded


def calculate_metric_internal(
    relevant_ids,
    dataset,
    groundtruths_loaded,
    predictions_loaded,
    metric: str,
    metric_parameters=None,
    metric_class=None,
):
    if metric_parameters is None:
        metric_parameters = {}

    metric_function = _METRIC_FUNCTIONS[metric]

    if dataset.task_type == TaskType.CLASSIFICATION.value:
        try:
            groundtruths = torch.stack(
                [gt.labels for gt in groundtruths_loaded]
            )

            predictions = torch.stack(
                [pred for pred in predictions_loaded]
            )

            if "average" not in metric_parameters:
                metric_parameters["average"] = "macro"

            if metric_class is not None:
                metric_parameters["average"] = "none"
                if isinstance(metric_class, str):
                    if dataset.label_to_name is None:
                        raise ValueError(
                            "label_to_name mapping is not provided.")
                    if metric_class not in dataset.label_to_name.values():
                        raise ValueError(
                            f"Class name '{metric_class}' not found in label_to_name mapping."
                        )
                    metric_class = list(dataset.label_to_name.keys())[
                        list(dataset.label_to_name.values()).index(metric_class)
                    ]
                metric_class = int(metric_class)
            metric_value = metric_function(
                predictions,
                groundtruths,
                task=dataset.task,
                # TODO: Naming, num_classes or num_labels?
                num_classes=dataset.num_classes,
                **metric_parameters,
            )

            if metric_class is not None:
                metric_value = metric_value[metric_class]

        except Exception as e:
            raise Exception(f"Error occurred during metric computation: {e}")

    elif dataset.task_type == TaskType.DETECTION.value:
        try:
            groundtruths = [gt.to_dict() for gt in groundtruths_loaded]
            predictions = [pred.to_dict() for pred in predictions_loaded]

            for pred in predictions:
                pred['boxes'] = torch.tensor(
                    pred['boxes'], dtype=torch.float32)
                pred['labels'] = torch.tensor(
                    pred['labels'], dtype=torch.int64)
                if 'scores' in pred:
                    pred['scores'] = torch.tensor(
                        pred['scores'], dtype=torch.float32)

            for gt in groundtruths:
                gt['boxes'] = torch.tensor(gt['boxes'], dtype=torch.float32)
                gt['labels'] = torch.tensor(gt['labels'], dtype=torch.int64)

            if metric_class is not None:
                metric_parameters["class_metrics"] = True

            if metric in ["mAP", "mAP_small", "mAP_medium", "mAP_large"]:
                metric_parameters["iou_type"] = "bbox"

            metric_function_instance = metric_function(**metric_parameters)
            metric_function_instance.update(predictions, groundtruths)
            metric_value = metric_function_instance.compute()

            if metric_class is not None:
                if isinstance(metric_class, str):
                    if dataset.label_to_name is None:
                        raise ValueError(
                            "label_to_name mapping is not provided.")
                    if metric_class not in dataset.label_to_name.values():
                        raise ValueError(
                            f"Class name '{metric_class}' not found in label_to_name mapping."
                        )
                    metric_class = list(dataset.label_to_name.keys())[
                        list(dataset.label_to_name.values()).index(metric_class)
                    ]
                metric_class = int(metric_class)

                if metric in ["mAP", "mAP_small", "mAP_medium", "mAP_large"]:
                    index = torch.where(
                        metric_value["classes"] == metric_class)[0]
                    metric_value = (
                        metric_value["map_per_class"][index].item()
                        if index.numel() > 0
                        else 0.0
                    )
                    metric_value = 0.0 if metric_value < 0.0 else metric_value
                else:
                    metric_class_key = f"{_METRIC_KEYS[metric]}/cl_{metric_class}"
                    metric_value = metric_value.get(metric_class_key, 0.0)
            else:
                metric_value = metric_value[_METRIC_KEYS[metric]]

        except Exception as e:
            raise Exception(f"Error occurred during metric computation: {e}")
    else:
        raise ValueError(f"Unsupported task type: {dataset.task_type}")

    if hasattr(metric_value, "item"):
        metric_value = metric_value.item()

    return round(metric_value, 5)


def calculate_metric(
    dataset_or_slice: Union[Moonwatcher, Slice],
    predictions: torch.Tensor,
    metric: str,
    metric_parameters=None,
    metric_class=None,
):
    relevant_ids, dataset, groundtruths_loaded, predictions_loaded = load_data(
        dataset_or_slice, predictions
    )

    return calculate_metric_internal(
        relevant_ids,
        dataset,
        groundtruths_loaded,
        predictions_loaded,
        metric,
        metric_parameters,
        metric_class,
    )


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
