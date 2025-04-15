from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from doleus.annotations.base import Annotations
from doleus.annotations.detection import BoundingBoxes
from doleus.datasets.base import Doleus
from doleus.utils.data import TaskType


class DoleusDetection(Doleus):
    """Dataset wrapper for detection tasks."""

    def __init__(
        self,
        dataset: Dataset,
        name: str,
        label_to_name: Optional[Dict[int, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        datapoints_metadata: Optional[List[Dict[str, Any]]] = None,
    ):
        """Initialize a DoleusDetection dataset.

        Parameters
        ----------
        dataset : Dataset
            The PyTorch dataset to wrap.
        name : str
            Name of the dataset.
        label_to_name : Optional[Dict[int, str]], optional
            Mapping from class IDs to class names, by default None.
        metadata : Optional[Dict[str, Any]], optional
            Dataset-level metadata, by default None.
        datapoints_metadata : Optional[List[Dict[str, Any]]], optional
            Per-datapoint metadata, by default None.
        """
        super().__init__(
            dataset=dataset,
            name=name,
            task_type=TaskType.DETECTION.value,
            label_to_name=label_to_name,
            metadata=metadata,
            datapoints_metadata=datapoints_metadata,
        )
        # Call add_groundtruths after super().__init__ has initialized self.groundtruths
        self.add_groundtruths()

    def add_groundtruths(self):
        """Add detection ground truth annotations.

        Extracts bounding boxes and labels from the underlying dataset
        and stores them as BoundingBoxes.
        """
        self.groundtruths = Annotations()  # Re-initialize
        for idx in tqdm(range(len(self.dataset)), desc="Building DETECTION ground truths"):
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
            self.groundtruths.add(ann)

    def _set_predictions(self, predictions: List[Dict[str, Any]]):
        """Add detection model predictions.

        Parameters
        ----------
        predictions : List[Dict[str, Any]]
            List of length N, where each element is a dictionary containing
            'boxes' (M,4), 'labels' (M,), and 'scores' (M,) tensors.
        """
        self.predictions = Annotations()

        if not isinstance(predictions, list):
            raise TypeError("For detection, predictions must be a list of length N.")

        if len(predictions) != len(self.dataset):
            raise ValueError("Mismatch between predictions list and dataset length.")

        # Each element should look like {"boxes": (M,4), "labels": (M,), "scores": (M,)}
        for i, pred_dict in enumerate(
            tqdm(predictions, desc="Building DETECTION predictions")
        ):
            # Ensure tensors are on the correct device and dtype if necessary
            # Basic conversion is done here, assuming inputs might be lists/numpy arrays
            boxes_xyxy = torch.tensor(pred_dict["boxes"], dtype=torch.float32)
            labels = torch.tensor(pred_dict["labels"], dtype=torch.long)
            scores = torch.tensor(pred_dict["scores"], dtype=torch.float32)

            ann = BoundingBoxes(
                datapoint_number=i,
                boxes_xyxy=boxes_xyxy,
                labels=labels,
                scores=scores,
            )
            self.predictions.add(ann)
