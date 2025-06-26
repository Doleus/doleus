# SPDX-FileCopyrightText: 2025 Doleus contributors
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Optional

from torch.utils.data import Dataset

from doleus.annotations import Annotation, Annotations


class BaseGroundTruthStore(ABC):
    """Base storage for ground truth data for a specific dataset instance."""

    def __init__(self, dataset: Dataset):
        """
        Initialize the ground truth store.

        Parameters
        ----------
        dataset : Dataset
            The PyTorch dataset object.
        """
        self.dataset = dataset
        self.groundtruths: Optional[Annotations] = None
        self.groundtruths = self._process_groundtruths()

    @abstractmethod
    def _process_groundtruths(self) -> Annotations:
        """
        Process raw ground truth data from the dataset into the standard annotation format.
        Actual implementation will depend on the task type (classification, detection).

        Returns
        -------
        Annotations
            Processed ground truths in standard annotation format.
        """
        pass

    def get(self, datapoint_number: int) -> Optional[Annotation]:
        """
        Get a single ground truth annotation object by datapoint number.

        Parameters
        ----------
        datapoint_number : int
            The ID of the sample in the dataset.

        Returns
        -------
        Optional[Annotation]
            The specific Annotation object (e.g., Labels, BoundingBoxes) for the datapoint,
            or None if not found.
        """
        if self.groundtruths is None:
            return None
        try:
            return self.groundtruths[datapoint_number]
        except KeyError:
            return None
