from typing import List, Dict, Any, Union, Optional
from abc import ABC, abstractmethod

from torch.nn import Module
from torch import Tensor

from moonwatcher.utils.data import DataType, TaskType, Task
from moonwatcher.annotations import Predictions, PredictedLabels, Labels, PredictedBoundingBoxes, BoundingBoxes
from moonwatcher.base.base import MoonwatcherObject
from moonwatcher.utils.helpers import get_current_timestamp
from moonwatcher.utils.api_connector import upload_if_possible

# CHANGE: Added a function to transform the predictions of classification and detection models into the required format
# CHANGE: Deleted ModelOutputInputTransformation class


def transform_classification_predictions(self, predictions: Tensor, scores: Tensor = None) -> Union[PredictedLabels, Labels]:
    """
        Transform classification predictions into standardized Labels or PredictedLabels objects.

        Args:
            predictions (Tensor): Model predictions tensor.
                - For binary/multiclass: shape (dataset_size,) containing predicted numeric labels
                - For multilabel: shape (dataset_size, x) where x ≤ num_classes, containing the predicted numeric labels
            scores (Optional[Tensor], optional): Prediction scores tensor.
                - For binary: shape (dataset_size, 2) containing scores for both classes
                - For multiclass: shape (dataset_size, num_classes) containing scores for all classes
                - For multilabel: shape (dataset_size, num_classes) containing scores for all classes
                - Must be float values between 0 and 1
                - Defaults to None

        Returns:
            List[Union[PredictedLabels, Labels]]: List of prediction objects, one per sample

        Warning:
            This function assumes that the order of predictions matches exactly with the order
            of datapoints in your dataset. The index i in predictions[i] is used as the
            datapoint_number, assuming it corresponds to the i-th sample in your dataset.
            You must provide predictions for ALL datapoints in your dataset.

        Example:
            Binary: 
                predictions = [0, 1, 0]  # shape: (3,)
                scores = [[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]]  # shape: (3, 2)

            Multiclass:
                predictions = [0, 2, 1]  # shape: (3,)
                scores = [[0.8, 0.1, 0.1], [0.1, 0.2, 0.7], [0.2, 0.6, 0.2]]  # shape: (3, 3)

            Multilabel:
                predictions = [[0,1], [2], [0,2]]  # shape: (3, x) where x varies
                scores = [[0.8, 0.7, 0.1], [0.2, 0.3, 0.9], [0.7, 0.2, 0.8]]  # shape: (3, 3)
    """

    # Convert to int64 if needed to ensure compatibility with Labels object
    if predictions.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        predictions = predictions.to(torch.int64)

    if scores is not None:
        # Convert to float32 if needed
        if scores.dtype not in (torch.float16, torch.float32, torch.float64):
            scores = scores.to(torch.float32)

        assert len(scores) == len(
            predictions), "The length of scores must match the length of predictions"

        return [
            PredictedLabels(datapoint_number=i, labels=pred, scores=score)
            for i, (pred, score) in enumerate(zip(predictions, scores))
        ]

    return [
        Labels(datapoint_number=i, labels=pred)
        for i, pred in enumerate(predictions)
    ]


def transform_detection_predictions(self, predictions: Tensor, scores: Optional[Tensor] = None, labels: Optional[Tensor] = None) -> Union[PredictedBoundingBoxes, BoundingBoxes]:
    # TODO: The function still needs significant editing since I, Hendrik, am not sure how we expect the data to be passed.
    """
    Transform detection predictions into standardized BoundingBoxes or PredictedBoundingBoxes objects.

    Args:
        predictions (Tensor): Model predictions tensor containing bounding box coordinates.
            - Shape (dataset_size, num_boxes_i, 4) containing box coordinates in xyxy format
            - num_boxes_i is the number of boxes for the i-th datapoint
        labels (List[Tensor]): List of label tensors, one per datapoint.
            - Each tensor has shape (num_boxes_i,) matching the number of boxes in predictions
            - The tensor contains numeric labels that correspond to the predicted classes
        scores (Optional[List[Tensor]]): List of score tensors, one per datapoint.
            - Each tensor has shape (num_boxes_i,) matching the number of boxes in predictions
            - Contains float values between 0 and 1
            - If provided, returns PredictedBoundingBoxes, otherwise returns BoundingBoxes
            - Defaults to None

    Returns:
        List[Union[PredictedBoundingBoxes, BoundingBoxes]]: List of prediction objects, one per sample

    Warning:
        This function assumes that the order of predictions matches exactly with the order
        of datapoints in your dataset. The index i in predictions[i] is used as the
        datapoint_number, assuming it corresponds to the i-th sample in your dataset.
        You must provide predictions for ALL datapoints in your dataset.

    Example:
        predictions = torch.tensor([[[x1, y1, x2, y2], ...], ...])  # shape: (dataset_size, num_boxes, 4)
        scores = torch.tensor([[0.9, 0.8, ...], ...])  # shape: (dataset_size, num_boxes)
        labels = torch.tensor([[1, 2, ...], ...])  # shape: (dataset_size, num_boxes)
    """
    # Convert boxes to float32 if needed
    if predictions.dtype not in (torch.float16, torch.float32, torch.float64):
        predictions = predictions.to(torch.float32)

    if scores is not None:
        # Convert scores to float32 if needed
        if scores.dtype not in (torch.float16, torch.float32, torch.float64):
            scores = scores.to(torch.float32)

        # Convert labels to int64 if needed
        if labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            labels = labels.to(torch.int64)

        assert len(scores) == len(
            predictions), "The length of scores must match the length of predictions"
        assert len(labels) == len(
            predictions), "The length of labels must match the length of predictions"

        return [
            PredictedBoundingBoxes(
                datapoint_number=i,
                boxes_xyxy=boxes,          # shape: (num_boxes, 4)
                scores=box_scores,         # shape: (num_boxes,)
                labels=box_labels          # shape: (num_boxes,)
            )
            for i, (boxes, box_scores, box_labels) in enumerate(zip(predictions, scores, labels))
        ]

    return [
        BoundingBoxes(
            datapoint_id=i,
            boxes_xyxy=boxes,
            labels=labels[i]
        )
        for i, boxes in enumerate(predictions)
    ]

# CHANGE: Added predictions transform to simplify how users pass their model predictions


def transform_prediction(self, predictions: Tensor, scores):
    # TODO: Adapt
    """
    Transforms user-provided predictions into a Predictions object.
    WARNING: Users have to provide the predictions in the same order as the datapoints in the dataset.

    :param predictions: A list containing user predictions in various formats.
    :return: A Predictions object standardized for the Moonwatcher framework.
    """

    # Case 1: Classification
    if self.task_type == TaskType.CLASSIFICATION:
        formatted_predictions = self.transform_classification_predictions(
            predictions, scores)

    # Case 2: Detection
    elif self.task_type == TaskType.DETECTION:
        return ("The function is not yet implemented")

    return formatted_predictions


class MoonwatcherModel(MoonwatcherObject, Module):
    def __init__(
        self,
        name: str,
        task_type: str,
        task: str,
        predictions: Tensor,
        scores: Optional[Tensor] = None
    ):
        """
        Creates a moonwatcher model wrapper around an existing model that can be used with the moonwatcher framework

        :param name: the name you want to give this model
        :param task_type: either classification or detection
        :param task: either binary, multiclass or multilabel
        :param predictions: a list of predictions provided by the users. Format depends on task_type:
            - For classification: Tensor of shape (dataset_size,) or (dataset_size, x) where x ≤ num_classes
            - For detection: #To be Discussed
        :param scores: Optional Tensor of prediction scores. Format depends on task_type:
            - For classification: Tensor of shape (dataset_size,) or (dataset_size, x) where x ≤ num_classes
            - For detection: #To be Discussed
        """
        MoonwatcherObject.__init__(self, name=name, datatype=DataType.MODEL)

        self.name = name
        self.task_type = task_type
        self.task = task
        self.predictions = predictions

    # CHANGE: Deleted delete and _upload function. Both aren't requried anymore
