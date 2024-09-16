from typing import Optional

from torch import Tensor

from doleus.annotations.base import Annotation


class Labels(Annotation):
    """Annotation for a sample in a single-label or multi-label classification task.

    This class handles both ground truth labels (no probability scores) and predicted labels
    (with probability scores) for classification tasks.
    """

    def __init__(
        self,
        datapoint_number: int,
        labels: Optional[Tensor] = None,
        scores: Optional[Tensor] = None,
    ):
        """Initialize a Labels instance.

        Parameters
        ----------
        datapoint_number : int
            Index for the corresponding data point.
        labels : Optional[Tensor], optional
            A 1D integer tensor. For single-label tasks (e.g. multiclass), this contains one class index
            (e.g., `tensor([3])` for the fourth class in a 5-class problem). For multilabel tasks, this is
            a multi-hot encoded tensor (e.g., `tensor([1, 0, 1, 0])` indicating presence of classes 0 and 2
            in a 4-class problem). Can be `None` if `scores` are provided.
        scores : Optional[Tensor], optional
            A 1D float tensor. For single-label tasks (e.g. multiclass), this contains
            probabilities for each class (e.g., `tensor([0.1, 0.05, 0.15, 0.6, 0.1])` summing to 1.0
            for a 5-class problem). For multilabel tasks, this contains independent probabilities
            for each label (e.g., `tensor([0.9, 0.2, 0.8, 0.1])` where each value is between 0 and 1
            for a 4-class problem). Optional.
        """
        if labels is None and scores is None:
            raise ValueError(
                "Either 'labels' or 'scores' must be provided but both are None."
            )
        super().__init__(datapoint_number)
        self.labels = labels
        self.scores = scores

    def to_dict(self) -> dict:
        """Convert annotation to a dictionary format.

        Returns
        -------
        dict
            Dictionary with keys 'labels' and/or 'scores'.
        """
        output = {}
        if self.labels is not None:
            output["labels"] = self.labels
        if self.scores is not None:
            output["scores"] = self.scores
        return output
