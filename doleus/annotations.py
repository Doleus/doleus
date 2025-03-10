"""Annotation classes for storing and managing model predictions and ground truths."""

from typing import List, Optional

from torch import Tensor


class Annotation:
    """Base annotation class for storing datapoint identifiers.

    This class serves as a base for both classification and detection annotations,
    providing a unified interface for datapoint identification.
    """

    def __init__(self, datapoint_number: int):
        """Initialize an Annotation instance.

        Parameters
        ----------
        datapoint_number : int
            Integer index or ID corresponding to a sample in the dataset.
        """
        self.datapoint_number = datapoint_number

    def __repr__(self) -> str:
        """Return string representation of the annotation.

        Returns
        -------
        str
            String representation including the class name and datapoint number.
        """
        return f"{self.__class__.__name__}(datapoint_number={self.datapoint_number})"


class BoundingBoxes(Annotation):
    """Detection annotation for storing bounding boxes, labels, and scores.

    This class handles both ground truth boxes (no scores) and predicted boxes
    (with scores). The presence of scores determines whether the boxes are
    treated as predictions or ground truths.
    """

    def __init__(
        self,
        datapoint_number: int,
        boxes_xyxy: Tensor,
        labels: Tensor,
        scores: Optional[Tensor] = None,
    ):
        """Initialize a BoundingBoxes instance.

        Parameters
        ----------
        datapoint_number : int
            Unique identifier (index) for the data point.
        boxes_xyxy : Tensor
            Tensor of shape (num_boxes, 4) with bounding box coordinates
            in [x1, y1, x2, y2] format.
        labels : Tensor
            Integer tensor of shape (num_boxes,) for class labels.
        scores : Optional[Tensor], optional
            Float tensor of shape (num_boxes,) for box confidence scores.
            If None, boxes are treated as ground truths, by default None.
        """
        super().__init__(datapoint_number)
        self.boxes_xyxy = boxes_xyxy
        self.labels = labels
        self.scores = scores  # None => ground truths; not None => predictions

    def to_dict(self) -> dict:
        """Convert bounding boxes to a dictionary format.

        Returns
        -------
        dict
            Dictionary with keys 'boxes', 'labels', and optionally 'scores',
            suitable for detection metrics or post-processing.
        """
        output = {
            "boxes": self.boxes_xyxy,
            "labels": self.labels,
        }
        if self.scores is not None:
            output["scores"] = self.scores
        return output

    def __repr__(self) -> str:
        """Return string representation of the bounding boxes.

        Returns
        -------
        str
            String representation including number of boxes and scores presence.
        """
        n_boxes = self.boxes_xyxy.shape[0]
        return (
            f"{self.__class__.__name__}(datapoint_number={self.datapoint_number}, "
            f"num_boxes={n_boxes}, scores_present={self.scores is not None})"
        )


class Labels(Annotation):
    """Classification annotation for single-label or multi-label tasks.

    This class handles both ground truth labels (no scores) and predicted labels
    (with scores) for classification tasks.
    """

    def __init__(
        self, datapoint_number: int, labels: Tensor, scores: Optional[Tensor] = None
    ):
        """Initialize a Labels instance.

        Parameters
        ----------
        datapoint_number : int
            Unique identifier (index) for the data point.
        labels : Tensor
            A 1D integer tensor representing the label(s).
            For single-label classification, shape might be [1].
            For multi-label, shape might be [k].
        scores : Optional[Tensor], optional
            A float tensor representing predicted confidence or probabilities.
            For single-label, shape might be [num_classes].
            For multi-label, shape might be [k] or [num_classes], by default None.
        """
        super().__init__(datapoint_number)
        self.labels = labels
        self.scores = scores

    def to_dict(self) -> dict:
        """Convert label annotation to a dictionary format.

        Returns
        -------
        dict
            Dictionary with keys 'labels' and optionally 'scores', suitable for
            metrics or downstream tasks.
        """
        output = {"labels": self.labels}
        if self.scores is not None:
            output["scores"] = self.scores
        return output

    def __repr__(self) -> str:
        """Return string representation of the labels.

        Returns
        -------
        str
            String representation including labels and scores presence.
        """
        labels_str = self.labels.tolist() if self.labels.numel() < 6 else "..."
        scores_str = "scores_present" if self.scores is not None else "no_scores"
        return (
            f"{self.__class__.__name__}(datapoint_number={self.datapoint_number}, "
            f"labels={labels_str}, {scores_str})"
        )


class Annotations:
    """Generic container for managing annotation objects.

    This class provides a container for Labels or BoundingBoxes annotations,
    allowing indexing by datapoint number and iteration over all annotations.
    """

    def __init__(self, annotations: List[Annotation] = None):
        """Initialize an Annotations container.

        Parameters
        ----------
        annotations : List[Annotation], optional
            Initial list of annotation objects to store, by default None.
        """
        self.annotations = annotations if annotations is not None else []
        # Map from datapoint_number -> index in self.annotations
        self.datapoint_number_to_annotation_index = {}
        for idx, annotation in enumerate(self.annotations):
            dp_num = annotation.datapoint_number
            self.datapoint_number_to_annotation_index[dp_num] = idx

    def add(self, annotation: Annotation):
        """Add a new annotation to the container.

        Parameters
        ----------
        annotation : Annotation
            An annotation of type Labels or BoundingBoxes.

        Raises
        ------
        TypeError
            If annotation is not an instance of the base Annotation class.
        KeyError
            If an annotation for the datapoint already exists.
        """
        if not isinstance(annotation, Annotation):
            raise TypeError(
                "annotation must be an instance of the base Annotation class."
            )

        dp_num = annotation.datapoint_number
        if dp_num in self.datapoint_number_to_annotation_index:
            raise KeyError(f"Annotation for datapoint {dp_num} already exists.")

        self.annotations.append(annotation)
        self.datapoint_number_to_annotation_index[dp_num] = len(self.annotations) - 1

    def get(self, datapoint_number: int) -> Annotation:
        """Retrieve the annotation object for a given datapoint.

        Parameters
        ----------
        datapoint_number : int
            The ID of the sample in the dataset.

        Returns
        -------
        Annotation
            The annotation (Labels or BoundingBoxes) for the datapoint.

        Raises
        ------
        KeyError
            If no annotation is found for the datapoint number.
        """
        index = self.datapoint_number_to_annotation_index.get(datapoint_number)
        if index is None:
            raise KeyError(f"No annotation found for datapoint {datapoint_number}.")
        return self.annotations[index]

    def get_datapoint_ids(self) -> List[int]:
        """Get all datapoint IDs with annotations.

        Returns
        -------
        List[int]
            A list of all datapoint IDs (keys) for which annotations exist.
        """
        return list(self.datapoint_number_to_annotation_index.keys())

    def __getitem__(self, datapoint_number: int) -> Annotation:
        """Get annotation by datapoint number using indexing syntax.

        Parameters
        ----------
        datapoint_number : int
            The ID of the sample in the dataset.

        Returns
        -------
        Annotation
            The annotation for the datapoint.
        """
        return self.get(datapoint_number)

    def __len__(self) -> int:
        """Get the number of annotations.

        Returns
        -------
        int
            The total number of annotations in the container.
        """
        return len(self.annotations)

    def __iter__(self):
        """Get an iterator over all annotations.

        Returns
        -------
        iterator
            Iterator over all annotations in the container.
        """
        return iter(self.annotations)

    def __repr__(self) -> str:
        """Return string representation of the annotations container.

        Returns
        -------
        str
            String representation including the number of annotations.
        """
        return f"{self.__class__.__name__}(num_annotations={len(self.annotations)})"


class Predictions(Annotations):
    """Specialized container for predicted annotations.

    This container is specifically for storing predictions (Labels or
    BoundingBoxes with scores) and maintains a reference to its dataset.
    """

    def __init__(self, dataset, predictions: List[Annotation] = None):
        """Initialize a Predictions container.

        Parameters
        ----------
        dataset
            Reference to the dataset (e.g., Doleus object).
        predictions : List[Annotation], optional
            List of prediction annotations (Labels or BoundingBoxes with scores),
            by default None.
        """
        super().__init__(annotations=predictions)
        self.dataset = dataset

    def __repr__(self) -> str:
        """Return string representation of the predictions container.

        Returns
        -------
        str
            String representation including dataset name and number of annotations.
        """
        return f"{self.__class__.__name__}(dataset='{self.dataset.name}', num_annotations={len(self.annotations)})"


class GroundTruths(Annotations):
    """Specialized container for ground truth annotations.

    This container is specifically for storing ground truths (Labels or
    BoundingBoxes without scores) and maintains a reference to its dataset.
    """

    def __init__(self, dataset, groundtruths: List[Annotation] = None):
        """Initialize a GroundTruths container.

        Parameters
        ----------
        dataset
            Reference to the dataset (e.g., Doleus object).
        groundtruths : List[Annotation], optional
            List of ground truth annotations (Labels or BoundingBoxes),
            by default None.
        """
        super().__init__(annotations=groundtruths)
        self.dataset = dataset

    def __repr__(self) -> str:
        """Return string representation of the ground truths container.

        Returns
        -------
        str
            String representation including dataset name and number of annotations.
        """
        return f"{self.__class__.__name__}(dataset='{self.dataset.name}', num_annotations={len(self.annotations)})"
