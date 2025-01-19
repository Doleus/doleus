from typing import List, Union, Optional

import torch
from torch import Tensor


class Annotation:
    """
    Base annotation class that stores the datapoint identifier (index).
    Used to unify both classification and detection annotations.
    """

    def __init__(self, datapoint_number: int):
        """
        :param datapoint_number: Integer index or ID corresponding to a sample in the dataset.
        """
        self.datapoint_number = datapoint_number

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(datapoint_number={self.datapoint_number})"


class BoundingBoxes(Annotation):
    """
    Detection annotation storing bounding boxes, labels, and optional scores.
    If 'scores' is None, these boxes are treated as ground truths.
    If 'scores' is present, they are treated as predictions.
    """

    def __init__(
        self,
        datapoint_number: int,
        boxes_xyxy: Tensor,
        labels: Tensor,
        scores: Optional[Tensor] = None,
    ):
        """
        :param datapoint_number: Unique identifier (index) for the data point.
        :param boxes_xyxy: Tensor of shape (num_boxes, 4) with bounding box coordinates
                           in [x1, y1, x2, y2] format.
        :param labels: Integer tensor of shape (num_boxes,) for class labels.
        :param scores: (Optional) Float tensor of shape (num_boxes,) for box confidence scores.
        """
        super().__init__(datapoint_number)
        self.boxes_xyxy = boxes_xyxy
        self.labels = labels
        self.scores = scores  # None => ground truths; not None => predictions

    def to_dict(self) -> dict:
        """
        Convert bounding boxes to a dictionary format that can be consumed
        by detection metrics or post-processing functions.

        :return: Dictionary with keys 'boxes', 'labels', and optionally 'scores'.
        """
        output = {
            "boxes": self.boxes_xyxy,
            "labels": self.labels,
        }
        if self.scores is not None:
            output["scores"] = self.scores
        return output

    def __repr__(self) -> str:
        n_boxes = self.boxes_xyxy.shape[0]
        return (f"{self.__class__.__name__}(datapoint_number={self.datapoint_number}, "
                f"num_boxes={n_boxes}, scores_present={self.scores is not None})")


class Labels(Annotation):
    """
    Classification (single-label or multi-label) annotation,
    with optional scores for predictions.
    """

    def __init__(
        self,
        datapoint_number: int,
        labels: Tensor,
        scores: Optional[Tensor] = None
    ):
        """
        :param datapoint_number: Unique identifier (index) for the data point.
        :param labels: A 1D integer tensor representing the label(s).
                       For single-label classification, shape might be [1].
                       For multi-label, shape might be [k].
        :param scores: (Optional) A float tensor representing predicted confidence or probabilities.
                       For single-label, shape might be [num_classes].
                       For multi-label, shape might be [k] or [num_classes], depending on your format.
        """
        super().__init__(datapoint_number)
        self.labels = labels
        self.scores = scores

    def to_dict(self) -> dict:
        """
        Convert label annotation to a dictionary for potential processing
        in metrics or other downstream tasks.

        :return: Dictionary with keys 'labels' and optionally 'scores'.
        """
        output = {"labels": self.labels}
        if self.scores is not None:
            output["scores"] = self.scores
        return output

    def __repr__(self) -> str:
        labels_str = self.labels.tolist() if self.labels.numel() < 6 else "..."
        scores_str = "scores_present" if self.scores is not None else "no_scores"
        return (f"{self.__class__.__name__}(datapoint_number={self.datapoint_number}, "
                f"labels={labels_str}, {scores_str})")


class Annotations:
    """
    A generic container for annotation objects (Labels or BoundingBoxes).
    Allows indexing by datapoint_number and iteration over all annotations.
    """

    def __init__(self, annotations: List[Annotation] = None):
        """
        :param annotations: Optional initial list of annotation objects to store.
        """
        self.annotations = annotations if annotations is not None else []
        # Map from datapoint_number -> index in self.annotations
        self.datapoint_number_to_annotation_index = {}
        for idx, annotation in enumerate(self.annotations):
            dp_num = annotation.datapoint_number
            self.datapoint_number_to_annotation_index[dp_num] = idx

    def add(self, annotation: Annotation):
        """
        Add a new annotation to the container. The datapoint_number must be unique.

        :param annotation: An annotation of type Labels or BoundingBoxes.
        """
        if not isinstance(annotation, Annotation):
            raise TypeError(
                "annotation must be an instance of the base Annotation class.")

        dp_num = annotation.datapoint_number
        if dp_num in self.datapoint_number_to_annotation_index:
            raise KeyError(
                f"Annotation for datapoint {dp_num} already exists.")

        self.annotations.append(annotation)
        self.datapoint_number_to_annotation_index[dp_num] = len(
            self.annotations) - 1

    def get(self, datapoint_number: int) -> Annotation:
        """
        Retrieve the annotation object for a given datapoint_number.

        :param datapoint_number: The ID of the sample in the dataset.
        :return: Annotation (Labels or BoundingBoxes).
        :raises KeyError: if no annotation found for that datapoint_number.
        """
        index = self.datapoint_number_to_annotation_index.get(datapoint_number)
        if index is None:
            raise KeyError(
                f"No annotation found for datapoint {datapoint_number}.")
        return self.annotations[index]

    def get_datapoint_ids(self) -> List[int]:
        """
        :return: A list of all datapoint IDs (keys) for which annotations exist.
        """
        return list(self.datapoint_number_to_annotation_index.keys())

    def __getitem__(self, datapoint_number: int) -> Annotation:
        return self.get(datapoint_number)

    def __len__(self) -> int:
        return len(self.annotations)

    def __iter__(self):
        return iter(self.annotations)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_annotations={len(self.annotations)})"


class Predictions(Annotations):
    """
    A specialized container for predicted annotations (Labels/BoundingBoxes with scores).
    """

    def __init__(self, dataset, predictions: List[Annotation] = None):
        """
        :param dataset: Reference to the dataset (e.g., Moonwatcher object).
        :param predictions: List of prediction annotations (Labels or BoundingBoxes with scores).
        """
        super().__init__(annotations=predictions)
        self.dataset = dataset

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dataset='{self.dataset.name}', num_annotations={len(self.annotations)})"


class GroundTruths(Annotations):
    """
    A specialized container for ground-truth annotations (Labels/BoundingBoxes without scores).
    """

    def __init__(self, dataset, groundtruths: List[Annotation] = None):
        """
        :param dataset: Reference to the dataset (e.g., Moonwatcher object).
        :param groundtruths: List of ground-truth annotations (Labels/BoundingBoxes).
        """
        super().__init__(annotations=groundtruths)
        self.dataset = dataset

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dataset='{self.dataset.name}', num_annotations={len(self.annotations)})"
