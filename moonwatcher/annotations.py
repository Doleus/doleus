from typing import List, Union

import torch
from torch import Tensor
from moonwatcher.utils.data import DataType
from moonwatcher.base.base import MoonwatcherObject


class Annotation:
    def __init__(self, datapoint_number: int):
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
        if not isinstance(boxes_xyxy, Tensor):
            raise TypeError(
                "bounding boxes must be a Tensor of shape (num_boxes, 4)")
        if not isinstance(labels, Tensor):
            raise TypeError(
                "labels must be an int Tensor of shape (num_boxes)")

        super().__init__(datapoint_number)
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
        if not isinstance(scores, Tensor):
            raise TypeError(
                "scores must be a float Tensor of shape (num_boxes)")

        super().__init__(datapoint_number, boxes_xyxy, labels)
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
        if (
            not isinstance(labels, Tensor)
            or len(labels.shape) != 1
            or labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64)
        ):
            raise TypeError("labels must be a 1-dimensional int Tensor")

        super().__init__(datapoint_number)
        self.labels = labels


class PredictedLabels(Labels):
    def __init__(self, datapoint_number: int, labels: Tensor, scores: Tensor):
        """
        Initialize a PredictedLabels object
        :param datapoint_number: The unique identifier for the data point.
        :param labels: A 1-dimensional integer tensor representing the predicted label(s).
        :param scores: A float tensor representing the confidence scores for each class.
        """
        if not isinstance(scores, Tensor):
            raise TypeError("scores must be a float Tensor")

        super().__init__(datapoint_number, labels)
        self.scores = scores


class Annotations:
    def __init__(self, annotations: List[Annotation] = None):
        self.annotations = [] if annotations is None else annotations
        self.datapoint_number_to_annotation_index = {}
        for annotation_index, annotation in enumerate(self.annotations):
            self.datapoint_number_to_annotation_index[
                annotation.datapoint_number
            ] = annotation_index

    def add(self, annotation: Annotation):
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
                f"No annotation found for datapoint number {datapoint_number}")

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
        predictions: List[
            Union[PredictedBoundingBoxes, PredictedLabels]
        ] = None,
    ):
        """
        Initializes a Predictions object
        :param dataset: The Moonwatcher dataset associated with these predictions.
        :param predictions: A list of predicted annotations.
        """
        Annotations.__init__(self, annotations=predictions)
        MoonwatcherObject.__init__(
            self, name=dataset.name, datatype=DataType.PREDICTIONS
        )


class GroundTruths(Annotations, MoonwatcherObject):
    def __init__(
        self, dataset, groundtruths: List[Union[BoundingBoxes, Labels]] = None
    ):
        """
        Initializes a GroundTruths object
        :param dataset: The Moonwatcher dataset associated with these ground truths.
        :param groundtruths: A list of ground truth annotations.
        """
        Annotations.__init__(self, annotations=groundtruths)
        MoonwatcherObject.__init__(
            self, name=dataset.name, datatype=DataType.GROUNDTRUTHS
        )
