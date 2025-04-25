from typing import Optional

from doleus.annotations import Annotations


class GroundTruthStore:
    """Storage for ground truth annotations for a specific dataset instance.

    Each Doleus Dataset has its own GroundTruthStore instance to manage
    ground truth annotations for that specific dataset.
    """

    def __init__(self):
        self.groundtruths: Optional[Annotations] = None

    def add_groundtruths(self, groundtruths: Annotations) -> None:
        """
        Store ground truth annotations.

        Parameters
        ----------
        groundtruths : Annotations
            Ground truth annotations to store.
        """
        self.groundtruths = groundtruths

    def get_groundtruths(self) -> Annotations:
        """Retrieve ground truth annotations.

        Returns
        -------
        Annotations
            The stored ground truth annotations
        """
        if self.groundtruths is None:
            raise ValueError("No ground truth annotations found")
        return self.groundtruths

    def get(self, datapoint_number: int):
        """Get annotation by datapoint number.

        Parameters
        ----------
        datapoint_number : int
            The ID of the sample in the dataset.

        Returns
        -------
        Annotation
            The annotation for the datapoint.
        """
        if self.groundtruths is None:
            raise ValueError("No ground truth annotations found")
        return self.groundtruths[datapoint_number]
