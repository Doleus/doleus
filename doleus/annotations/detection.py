from typing import Optional

from torch import Tensor

from doleus.annotations.base import Annotation


class BoundingBoxes(Annotation):
    """Annotation for storing bounding boxes and their corresponding labels and scores.

    This class handles both ground truth boxes (no probability scores) and predicted boxes
    (with probability scores).
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
            Index for the corresponding data point.
        boxes_xyxy : Tensor
            A tensor of shape (num_boxes, 4) with bounding box coordinates
            in [x1, y1, x2, y2] format.
        labels : Tensor
            An integer tensor of shape (num_boxes,) for class labels.
        scores : Optional[Tensor], optional
            A float tensor of shape (num_boxes,) containing predicted probability scores (optional).
        """
        super().__init__(datapoint_number)
        self.boxes_xyxy = boxes_xyxy
        self.labels = labels
        self.scores = scores

    def to_dict(self) -> dict:
        """Convert bounding boxes annotation to a dictionary format.

        Returns
        -------
        dict
            Dictionary with keys 'boxes', 'labels', and optionally 'scores'.
        """
        output = {
            "boxes": self.boxes_xyxy,
            "labels": self.labels,
        }
        if self.scores is not None:
            output["scores"] = self.scores
        return output
