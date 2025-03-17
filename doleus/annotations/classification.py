from typing import Optional

from torch import Tensor

from doleus.annotations.base import Annotation


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
            Dictionary with keys 'labels' and optionally 'scores'.
        """
        output = {"labels": self.labels}
        if self.scores is not None:
            output["scores"] = self.scores
        return output
