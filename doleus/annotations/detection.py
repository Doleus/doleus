# SPDX-FileCopyrightText: 2025 Doleus contributors
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from torch import Tensor

from doleus.annotations.base import Annotation


class BoundingBoxes(Annotation):
    """Annotation for a sample in an object detection task.

    This class handles both ground truth bounding boxes (no probability scores) and predicted bounding boxes
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
        """Convert annotation to a dictionary format.

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
