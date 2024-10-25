from typing import Union

import torch
import numpy as np
import torchmetrics
from sklearn.preprocessing import LabelEncoder

from moonwatcher.utils.data import TaskType, Task
from moonwatcher.inference.inference import inference
from moonwatcher.dataset.dataset import Slice, MoonwatcherDataset
from moonwatcher.model.model import MoonwatcherModel
from moonwatcher.utils.data_storage import (
    load_groundtruths,
    load_predictions,
    do_predictions_exist,
)


def run_inference_if_necessary(model, dataset):
    if not do_predictions_exist(dataset_name=dataset.name, model_name=model.name):
        inference(model=model, dataset=dataset, device=model.device)


def get_original_indices(dataset_or_slice):
    if isinstance(dataset_or_slice, Slice):
        parent_indices = get_original_indices(
            dataset_or_slice.moonwatcher_dataset)
        return [parent_indices[i] for i in dataset_or_slice.indices]
    elif isinstance(dataset_or_slice, MoonwatcherDataset):
        return list(range(len(dataset_or_slice.dataset)))
    else:
        raise TypeError("Unsupported dataset type")

# CHANGE: Added model to the parameters and assign model.predictions to prediction_loaded


def load_data(dataset_or_slice: Union[MoonwatcherDataset, Slice], model: MoonwatcherModel):
    relevant_ids = get_original_indices(dataset_or_slice=dataset_or_slice)
    dataset = (
        dataset_or_slice.original_dataset
        if isinstance(dataset_or_slice, Slice)
        else dataset_or_slice
    )

    groundtruths_loaded = load_groundtruths(dataset_name=dataset.name)
    predictions_loaded = load_predictions(model)

    return relevant_ids, dataset, groundtruths_loaded, predictions_loaded

# TODO Adapt calculate_metric_internal function slightly and use the task parameter in the model to determine the correct use case


def calculate_metric_internal(
    relevant_ids,
    dataset,
    model,
    # name_to_label,
    groundtruths_loaded,
    predictions_loaded,
    metric: str,
    metric_parameters=None,
    metric_class=None,
):
    if metric_parameters is None:
        metric_parameters = {}

    metric_function = _METRIC_FUNCTIONS[metric]

    # Differentiate between different use cases

    # CHANGE: Added multilabel as a potential task type
    if model.task_type == TaskType.CLASSIFICATION.value:
        # Code to handle classification tasks
        try:

            groundtruths = torch.stack(
                [groundtruths_loaded.annotations[i].labels for i in relevant_ids])

            # This is a temporary solution to handle the case where the predictions are not in the correct format.
            # We need to change create a transform prediction function in the model class so that predictions are already in the correct format.
            predictions = [torch.tensor(pred) if isinstance(
                pred, list) else pred for pred in predictions_loaded]
            predictions = torch.stack(predictions)

            if "average" not in metric_parameters:
                metric_parameters["average"] = "macro"
            if metric_class is not None:
                metric_parameters["average"] = "none"
                if isinstance(metric_class, str):
                    if metric_class not in name_to_label:
                        raise ValueError(
                            f"Class name '{
                                metric_class}' not found in label_to_name dictionary."
                        )
                    # metric_class = name_to_label[metric_class]
                metric_class = label_encoder.transform([metric_class])[0]

            metric_value = metric_function(
                predictions,
                groundtruths,
                task=dataset.task,
                # TODO: Calculate num_classes based on the dataset
                # TODO: Check if num_labels is a valid parameter. Previously it was num_classes
                num_labels=4,
                **metric_parameters,
            )

            if metric_class is not None:
                metric_value = metric_value[metric_class]

        except Exception as e:
            raise Exception(f"Error occurred during metric computation: {e}")

    elif model.task_type == TaskType.DETECTION.value:
        try:
            groundtruths = [groundtruths_loaded[i].to_dict()
                            for i in relevant_ids]

            # TODO: Not a nice solution, maybe we can instead convert predictions_loaded to our PredictedBoundingBoxes class?
            predictions = [predictions_loaded[i] for i in relevant_ids]
            for pred in predictions:
                pred['boxes'] = torch.tensor(
                    pred['boxes'], dtype=torch.float32)
                pred['labels'] = torch.tensor(
                    pred['labels'], dtype=torch.int64)
                pred['scores'] = torch.tensor(
                    pred['scores'], dtype=torch.float32)

            groundtruths, predictions = zip(
                *[
                    (gt, pred)
                    for gt, pred in zip(groundtruths, predictions)
                    if len(gt["boxes"]) > 0 and len(pred["boxes"]) > 0
                ]
            )

            if not groundtruths or not predictions:
                return -1.0

            if metric_class is not None:
                metric_parameters["class_metrics"] = True

            if metric in ["mAP", "mAP_small", "mAP_medium", "mAP_large"]:
                metric_parameters["iou_type"] = "bbox"

            metric_function = metric_function(**metric_parameters)
            metric_function.update(predictions, groundtruths)
            metric_value = metric_function.compute()

            if metric_class is not None:
                if isinstance(metric_class, str):
                    if metric_class not in name_to_label:
                        raise ValueError(
                            f"Class name '{
                                metric_class}' not found in label_to_name dictionary."
                        )
                    metric_class = name_to_label[metric_class]

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
                    metric_class = "iou/cl_" + str(metric_class)
                    metric_value = metric_value.get(metric_class, 0.0)
            else:
                metric_value = metric_value[_METRIC_KEYS[metric]]

        except Exception as e:
            raise Exception(f"Error occurred during metric computation: {e}")

    if hasattr(metric_value, "item"):
        metric_value = metric_value.item()

    return round(metric_value, 5)


def calculate_metric(
    dataset_or_slice: Union[MoonwatcherDataset, Slice],
    model: MoonwatcherModel,
    metric: str,
    metric_parameters=None,
    metric_class=None,
):
    relevant_ids, dataset, groundtruths_loaded, predictions_loaded = load_data(
        dataset_or_slice,
        model
    )

    # label_to_name = (
    #     dataset_or_slice.moonwatcher_dataset.label_to_name
    #     if isinstance(dataset_or_slice, Slice)
    #     else dataset_or_slice.label_to_name
    # )
    # name_to_label = {v: k for k, v in label_to_name.items()}

    return calculate_metric_internal(
        relevant_ids,
        dataset,
        model,
        # name_to_label, #TODO: Delete parameter
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
