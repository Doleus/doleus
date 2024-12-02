from typing import List, Union, Optional

import torch
from torch import Tensor
from moonwatcher.utils.data import DataType
from moonwatcher.base.base import MoonwatcherObject


def validate_datapoint_number(datapoint_number: int):
    if not isinstance(datapoint_number, int):
        raise TypeError(f"Datapoint number must be an integer but got {
                        datapoint_number}")
    if datapoint_number < 0:
        raise ValueError(f"Datapoint number must be a positive integer but got {
                         datapoint_number}")


def validate_boxes_xyxy(boxes_xyxy: Tensor):
    if not isinstance(boxes_xyxy, Tensor):
        raise TypeError(
            "Bounding boxes must be a Tensor of shape (num_boxes, 4)")
    if not (boxes_xyxy.dim() == 2 and boxes_xyxy.shape[1] == 4):
        raise ValueError(
            f"Bounding boxes must be a Tensor of shape (num_boxes, 4) but has shape {
                boxes_xyxy.shape}"
        )
    for box in boxes_xyxy:
        if not (box[0] <= box[2] and box[1] <= box[3]):
            raise ValueError(
                f"Bounding box coordinates are not in an acceptable format. x1 <= x2 and y1 <= y2 must be true. But got {
                    box}"
            )
        if not (box >= 0).all():
            raise ValueError(
                f"Bounding box coordinates must be positive. But got {box}"
            )


def validate_labels(labels: Tensor, expected_length: Optional[int] = None):
    if not isinstance(labels, Tensor):
        raise TypeError("labels must be a Tensor")
    if labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError(
            "labels must be an integer Tensor of type int8, int16, int32, or int64"
        )
    if labels.dim() != 1:
        raise ValueError(
            f"labels must be a 1-dimensional Tensor, i.e. (x,) but {labels} has shape {
                labels.shape}"
        )
    if expected_length is not None and len(labels) != expected_length:
        raise ValueError(
            f"Expected number of labels ({expected_length}) does not match actual ({
                len(labels)})"
        )


def validate_scores(scores: Tensor, expected_length: Optional[int] = None):
    if not isinstance(scores, Tensor):
        raise TypeError("scores must be a float Tensor")
    if scores.dim() != 1:
        raise ValueError("Scores must be a 1-dimensional Tensor")
    if expected_length is not None and len(scores) != expected_length:
        raise ValueError(
            "The number of scores must match the number of corresponding elements."
        )
    if not (scores.min() >= 0.0 and scores.max() <= 1.0):
        raise ValueError("Scores must be between 0.0 and 1.0 (inclusive).")


class Annotation:
    def __init__(self, datapoint_number: int):
        validate_datapoint_number(datapoint_number)
        self.datapoint_number = datapoint_number


class BoundingBoxes(Annotation):
    def __init__(
        self,
        datapoint_number: int,
        boxes_xyxy: Tensor,
        labels: Tensor,
    ):
        """
        Initializes a BoundingBoxes object
        :param datapoint_number: The unique identifier for the data point.
        :param boxes_xyxy: A tensor of shape (num_boxes, 4) representing the bounding box coordinates.
        :param labels: An integer tensor of shape (num_boxes) representing labels for each bounding box.
        """
        Annotation.__init__(self, datapoint_number)
        validate_boxes_xyxy(boxes_xyxy)
        validate_labels(labels, expected_length=len(boxes_xyxy))
        self.boxes_xyxy = boxes_xyxy
        self.labels = labels

    def to_dict(self):
        return {
            "boxes": self.boxes_xyxy,
            "labels": self.labels,
        }


class PredictedBoundingBoxes(BoundingBoxes):
    def __init__(
        self,
        datapoint_number: int,
        boxes_xyxy: Tensor,
        labels: Tensor,
        scores: Tensor,
    ):
        """
        Initializes a PredictedBoundingBoxes object
        :param datapoint_number: The unique identifier for the data point.
        :param boxes_xyxy: A tensor of shape (num_boxes, 4) representing bounding box coordinates.
        :param labels: An integer tensor of shape (num_boxes) representing labels for each bounding box.
        :param scores: A float tensor of shape (num_boxes) representing the confidence score for each bounding box.
        """
        BoundingBoxes.__init__(self, datapoint_number, boxes_xyxy, labels)
        validate_scores(scores, expected_length=len(boxes_xyxy))
        self.scores = scores

    def to_dict(self):
        return {
            "boxes": self.boxes_xyxy,
            "scores": self.scores,
            "labels": self.labels,
        }


class Labels(Annotation):
    def __init__(self, datapoint_number: int, labels: Tensor):
        """
        Initialize a Labels object
        :param datapoint_number: The unique identifier for the data point.
        :param labels: A 1-dimensional integer tensor of shape (x,) representing the label(s).
        """
        Annotation.__init__(self, datapoint_number)
        validate_labels(labels)
        self.labels = labels


class PredictedLabels(Labels):
    def __init__(self, datapoint_number: int, labels: Tensor, scores: Tensor):
        """
        Initialize a PredictedLabels object
        :param datapoint_number: The unique identifier for the data point.
        :param labels: A 1-dimensional integer tensor representing the predicted label(s).
        :param scores: A float tensor representing the confidence scores for each class.
        """
        Labels.__init__(self, datapoint_number, labels)
        validate_scores(scores, expected_length=len(labels))
        self.scores = scores


class Annotations:
    def __init__(self, annotations: List[Annotation] = None):
        """
        Initializes an Annotations collection.

        :param annotations: A list of Annotation objects to initialize the collection with. Defaults to None.
        :raises TypeError: If any of the provided annotations are not instances of Annotation.
        """
        # Check if annotations is a list of Annotation objects
        if annotations is not None:
            if not all(isinstance(annotation, Annotation) for annotation in annotations):
                raise TypeError(
                    "All annotations must be instances of Annotation"
                )

        self.annotations = [] if annotations is None else annotations
        self.datapoint_number_to_annotation_index = {}
        for annotation_index, annotation in enumerate(self.annotations):
            self.datapoint_number_to_annotation_index[
                annotation.datapoint_number
            ] = annotation_index

    def add(self, annotation: Annotation):
        # Check if annotation is an instance of Annotation
        if not isinstance(annotation, Annotation):
            raise TypeError("annotation must be an instance of Annotation")

        # Check if the datapoint number already exists
        if annotation.datapoint_number in self.datapoint_number_to_annotation_index:
            raise KeyError(
                f"An annotation for datapoint number {
                    annotation.datapoint_number} already exists."
            )
        self.annotations.append(annotation)
        self.datapoint_number_to_annotation_index[annotation.datapoint_number] = (
            len(self.annotations) - 1
        )

    def get(self, datapoint_number):
        index = self.datapoint_number_to_annotation_index.get(datapoint_number)
        if index is not None:
            return self.annotations[index]
        else:
            raise KeyError(
                f"No annotation found for datapoint number {datapoint_number}"
            )

    def get_datapoint_ids(self):
        return list(self.datapoint_number_to_annotation_index.keys())

    def __getitem__(self, datapoint_number):
        return self.get(datapoint_number=datapoint_number)

    def __len__(self):
        return len(self.annotations)

    def __iter__(self):
        return iter(self.annotations)


class Predictions(Annotations, MoonwatcherObject):
    def __init__(
        self,
        dataset,
        predictions: List[Union[PredictedBoundingBoxes,
                                PredictedLabels]] = None,
    ):
        """
        Initializes a Predictions object
        :param dataset: The Moonwatcher dataset associated with these predictions.
        :param predictions: A list of predicted annotations.
        """
        if predictions is not None:
            if not (
                all(isinstance(prediction, PredictedBoundingBoxes) for prediction in predictions) or
                all(isinstance(prediction, PredictedLabels)
                    for prediction in predictions)
            ):
                raise TypeError(
                    "All predictions must be instances of PredictedBoundingBoxes or all instances of PredictedLabels."
                )
        Annotations.__init__(self, annotations=predictions)
        MoonwatcherObject.__init__(
            self, name=dataset.name, datatype=DataType.PREDICTIONS
        )


class GroundTruths(Annotations, MoonwatcherObject):
    def __init__(
        self, dataset, groundtruths: List[Union[BoundingBoxes, Labels]] = None
    ):
        if groundtruths is not None:
            if not (
                all(isinstance(groundtruth, BoundingBoxes) and not isinstance(groundtruth, PredictedBoundingBoxes)
                    for groundtruth in groundtruths) or
                all(isinstance(groundtruth, Labels) and not isinstance(groundtruth, PredictedLabels)
                    for groundtruth in groundtruths)
            ):
                raise TypeError(
                    "All ground truths must be instances of BoundingBoxes or Labels (not their Predicted subclasses)."
                )
        Annotations.__init__(self, annotations=groundtruths)
        MoonwatcherObject.__init__(
            self, name=dataset.name, datatype=DataType.GROUNDTRUTHS
        )
