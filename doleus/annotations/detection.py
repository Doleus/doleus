"""Detection annotation classes for storing bounding boxes and object predictions."""

from typing import Optional

from torch import Tensor

from doleus.annotations.base import Annotation


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