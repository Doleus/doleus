"""Classification annotation classes for storing labels and predictions."""

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
