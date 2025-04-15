from typing import Optional

from torch import Tensor

from doleus.annotations.base import Annotation


class Labels(Annotation):
    """Annotation for single-label or multi-label classification.

    This class handles both ground truth labels (no probability scores) and predicted labels
    (with probability scores) for classification tasks.
    """

    def __init__(
        self, datapoint_number: int, labels: Tensor, scores: Optional[Tensor] = None
    ):
        """Initialize a Labels instance.

        Parameters
        ----------
        datapoint_number : int
            Index for the corresponding data point.
        labels : Tensor
            A 1D integer tensor representing the label(s).
        scores : Optional[Tensor], optional
            A float tensor containing predicted probability scores (optional).
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
