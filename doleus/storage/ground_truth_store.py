from typing import Any, Optional

import torch

from doleus.annotations import Annotations, BoundingBoxes, Labels
from doleus.utils import TaskType


class GroundTruthStore:
    """Storage for ground truth annotations for a specific dataset instance.

    Each Doleus Dataset has its own GroundTruthStore instance to manage
    ground truth annotations for that specific dataset.
    """

    def __init__(self, task_type: str, dataset: Any):
        """Initialize the ground truth store.

        Parameters
        ----------
        task_type : str
            Type of task (e.g., "classification", "detection").
        dataset : Any
            The underlying dataset to process ground truths from.
        """
        self.task_type = task_type
        self.dataset = dataset
        self.groundtruths: Optional[Annotations] = None
        self._process_groundtruths()

    def _process_groundtruths(self):
        """Process and store ground truth annotations from the dataset."""
        groundtruths = Annotations()

        if self.task_type == TaskType.CLASSIFICATION.value:
            for idx in range(len(self.dataset)):
                data = self.dataset[idx]
                if len(data) < 2:
                    raise ValueError(
                        f"Expected (image, label(s)) from dataset at index {idx}, got {len(data)} elements."
                    )
                _, labels = data

                # Convert label(s) to tensor of shape [N] if needed
                if not isinstance(labels, torch.Tensor):
                    labels = torch.tensor(labels)
                if labels.dim() == 0:
                    labels = labels.unsqueeze(0)

                ann = Labels(datapoint_number=idx, labels=labels)
                groundtruths.add(ann)

        elif self.task_type == TaskType.DETECTION.value:
            for idx in range(len(self.dataset)):
                data = self.dataset[idx]
                if len(data) != 3:
                    raise ValueError(
                        f"Expected (image, bounding_boxes, labels) for detection at index {idx}, got {len(data)} elements."
                    )
                _, bounding_boxes, labels = data

                if not isinstance(bounding_boxes, torch.Tensor):
                    bounding_boxes = torch.tensor(bounding_boxes, dtype=torch.float32)
                if not isinstance(labels, torch.Tensor):
                    labels = torch.tensor(labels, dtype=torch.long)

                ann = BoundingBoxes(
                    datapoint_number=idx, boxes_xyxy=bounding_boxes, labels=labels
                )
                groundtruths.add(ann)

        self.groundtruths = groundtruths

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
