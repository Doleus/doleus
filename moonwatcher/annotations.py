from typing import List, Union, Optional

import torch
from torch import Tensor


def validate_datapoint_number(datapoint_number: int):
    if not isinstance(datapoint_number, int):
        raise TypeError(
            f"Datapoint number must be an integer but got {datapoint_number}")
    if datapoint_number < 0:
        raise ValueError(
            f"Datapoint number must be a positive integer but got {datapoint_number}")


def validate_boxes_xyxy(boxes_xyxy: Tensor):
    if not isinstance(boxes_xyxy, Tensor):
        raise TypeError(
            "Bounding boxes must be a Tensor of shape (num_boxes, 4)")
    if not (boxes_xyxy.dim() == 2 and boxes_xyxy.shape[1] == 4):
        raise ValueError(
            f"Bounding boxes must be a Tensor of shape (num_boxes, 4), but has shape {boxes_xyxy.shape}"
        )
    for box in boxes_xyxy:
        if not (box[0] <= box[2] and box[1] <= box[3]):
            raise ValueError(
                f"Bounding box coordinates are not in an acceptable format. "
                f"x1 <= x2 and y1 <= y2 must be true. But got {box}"
            )
        if not (box >= 0).all():
            raise ValueError(
                f"Bounding box coordinates must be non-negative. But got {box}"
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
            f"labels must be a 1-dimensional Tensor, i.e. (x,), but got shape {labels.shape}"
        )
    if expected_length is not None and len(labels) != expected_length:
        raise ValueError(
            f"Expected number of labels ({expected_length}) does not match actual ({len(labels)})"
        )


def validate_scores(scores: Tensor, expected_length: Optional[int] = None):
    if not isinstance(scores, Tensor):
        raise TypeError("scores must be a torch.Tensor")
    if scores.dtype not in (torch.float16, torch.float32, torch.float64):
        raise TypeError("scores must be a float Tensor")
    if scores.dim() != 1:
        raise ValueError("scores must be a 1-dimensional Tensor")
    if expected_length is not None and len(scores) != expected_length:
        raise ValueError(
            "The number of scores must match the number of corresponding elements."
        )
    if not (scores.min() >= 0.0 and scores.max() <= 1.0):
        raise ValueError("Scores must be between 0.0 and 1.0 (inclusive).")


class Annotation:
    """
    Base annotation class storing the datapoint_number (index).
    """

    def __init__(self, datapoint_number: int):
        validate_datapoint_number(datapoint_number)
        self.datapoint_number = datapoint_number


class BoundingBoxes(Annotation):
    """
    BoundingBoxes can optionally have scores. If scores is provided,
    we treat these bounding boxes as 'predicted'; if not, 'ground truth'.
    """

    def __init__(
        self,
        datapoint_number: int,
        boxes_xyxy: Tensor,
        labels: Tensor,
        scores: Optional[Tensor] = None,
    ):
        """
        :param datapoint_number: The unique identifier for the data point.
        :param boxes_xyxy: A tensor of shape (num_boxes, 4) representing
                           bounding box coordinates [x1, y1, x2, y2].
        :param labels: An integer tensor of shape (num_boxes) representing
                       labels for each bounding box.
        :param scores: (Optional) A float tensor of shape (num_boxes)
                       representing confidence scores for each bounding box.
        """
        super().__init__(datapoint_number)
        validate_boxes_xyxy(boxes_xyxy)
        validate_labels(labels, expected_length=len(boxes_xyxy))

        if scores is not None:
            validate_scores(scores, expected_length=len(boxes_xyxy))

        self.boxes_xyxy = boxes_xyxy
        self.labels = labels
        self.scores = scores  # None if ground-truth, else predicted confidence

    def to_dict(self):
        output = {
            "boxes": self.boxes_xyxy,
            "labels": self.labels,
        }
        if self.scores is not None:
            output["scores"] = self.scores
        return output


class Labels(Annotation):
    """
    Classification or multi-label annotation, with optional scores for predictions.
    """

    def __init__(
        self,
        datapoint_number: int,
        labels: Tensor,
        scores: Optional[Tensor] = None
    ):
        """
        :param datapoint_number: The unique identifier for the data point.
        :param labels: A 1-dimensional integer tensor (e.g., shape [1] or [k]) 
                       representing the label(s).
        :param scores: (Optional) A float tensor representing the confidence
                       or probability for each label class. Must match length of 'labels'
                       if it's multi-label, or it can represent probability for all classes.
        """
        super().__init__(datapoint_number)
        validate_labels(labels)

        if scores is not None:
            # We don't strictly require the length to match 'labels' if it's e.g.
            # a distribution over all classes. But you can enforce it if you want.
            validate_scores(scores)

        self.labels = labels
        self.scores = scores

    def to_dict(self):
        output = {
            "labels": self.labels,
        }
        if self.scores is not None:
            output["scores"] = self.scores
        return output


class Annotations:
    """
    A collection of annotation objects, either bounding boxes or labels. 
    """

    def __init__(self, annotations: List[Annotation] = None):
        if annotations is not None:
            if not all(isinstance(annotation, Annotation) for annotation in annotations):
                raise TypeError(
                    "All annotations must be instances of Annotation")

        self.annotations = [] if annotations is None else annotations
        self.datapoint_number_to_annotation_index = {}
        for idx, annotation in enumerate(self.annotations):
            dp_num = annotation.datapoint_number
            self.datapoint_number_to_annotation_index[dp_num] = idx

    def add(self, annotation: Annotation):
        if not isinstance(annotation, Annotation):
            raise TypeError("annotation must be an instance of Annotation")

        if annotation.datapoint_number in self.datapoint_number_to_annotation_index:
            raise KeyError(
                f"An annotation for datapoint number {annotation.datapoint_number} already exists."
            )
        self.annotations.append(annotation)
        self.datapoint_number_to_annotation_index[annotation.datapoint_number] = len(
            self.annotations) - 1

    def get(self, datapoint_number: int) -> Annotation:
        index = self.datapoint_number_to_annotation_index.get(datapoint_number)
        if index is None:
            raise KeyError(
                f"No annotation found for datapoint number {datapoint_number}")
        return self.annotations[index]

    def get_datapoint_ids(self) -> List[int]:
        return list(self.datapoint_number_to_annotation_index.keys())

    def __getitem__(self, datapoint_number: int):
        return self.get(datapoint_number)

    def __len__(self):
        return len(self.annotations)

    def __iter__(self):
        return iter(self.annotations)


class Predictions(Annotations):
    """
    Predictions container. By using our new unified classes (Labels/BoundingBoxes with optional scores),
    we no longer need separate PredictedLabels or PredictedBoundingBoxes classes.
    """

    def __init__(self, dataset, predictions: List[Annotation] = None):
        """
        :param dataset: The Moonwatcher dataset or relevant dataset object.
        :param predictions: A list of annotation objects (Labels or BoundingBoxes) 
                            with optional scores.
        """
        # Optional: we could enforce that if it’s a detection dataset, predictions
        # should be bounding boxes, etc. But that’s up to your logic.
        super().__init__(annotations=predictions)
        self.dataset = dataset


class GroundTruths(Annotations):
    """
    Ground truths container. Similarly does not need separate bounding box vs. label classes,
    because each annotation can handle optional scores (which presumably won't be set for ground truths).
    """

    def __init__(self, dataset, groundtruths: List[Annotation] = None):
        # You could similarly do checks if you want to enforce no 'scores' in ground truths,
        # or just rely on user discipline.
        super().__init__(annotations=groundtruths)
        self.dataset = dataset
