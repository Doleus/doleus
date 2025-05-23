from typing import Optional

from torch import Tensor

from doleus.annotations.base import Annotation


class Labels(Annotation):
    """Annotation for single-label or multi-label classification.

    This class handles both ground truth labels (no probability scores) and predicted labels
    (with probability scores) for classification tasks.
    """

    def __init__(
        self, datapoint_number: int, labels: Optional[Tensor], scores: Optional[Tensor] = None
    ):
        """Initialize a Labels instance.

        Parameters
        ----------
        datapoint_number : int
            Index for the corresponding data point.
        labels : Optional[Tensor]
            A 1D integer tensor. For single-label tasks, this typically contains one class index
            (e.g., `tensor([2])`). For multilabel tasks, this is typically a multi-hot encoded
            tensor (e.g., `tensor([1, 0, 1])`). Can be `None` if only `scores` are provided.
        scores : Optional[Tensor], optional
            A 1D float tensor. For single-label tasks (e.g. multiclass), this usually contains
            probabilities for each class (e.g., `tensor([0.1, 0.2, 0.7])`). For multilabel
            tasks, this contains independent probabilities for each label (e.g.,
            `tensor([0.8, 0.1, 0.9])`). Optional.
        """
        if labels is None and scores is None:
            raise ValueError("Either 'labels' or 'scores' must be provided but both are None.")
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
        output = {}
        if self.labels is not None:
            output["labels"] = self.labels
        if self.scores is not None:
            output["scores"] = self.scores
        return output
