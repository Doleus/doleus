from typing import Any, Dict, List, Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from doleus.annotations.base import Annotations
from doleus.annotations.classification import Labels
from doleus.datasets.base import Doleus
from doleus.utils.data import TaskType


class DoleusClassification(Doleus):
    """Dataset wrapper for classification tasks."""

    def __init__(
        self,
        dataset: Dataset,
        name: str,
        task: str,
        num_classes: int,
        label_to_name: Optional[Dict[int, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        datapoints_metadata: Optional[List[Dict[str, Any]]] = None,
    ):
        """Initialize a DoleusClassification dataset.

        Parameters
        ----------
        dataset : Dataset
            The PyTorch dataset to wrap.
        name : str
            Name of the dataset.
        task : str
            Specific classification task description.
        num_classes : int
            Number of classes in the dataset.
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
            task_type=TaskType.CLASSIFICATION.value,
            task=task,
            label_to_name=label_to_name,
            metadata=metadata,
            datapoints_metadata=datapoints_metadata,
        )
        self.num_classes = num_classes

    def _set_predictions(self, predictions: Tensor):
        """Add classification model predictions.

        Parameters
        ----------
        predictions : Tensor
            Tensor of shape [N, num_classes] (logits/probabilities) or
            [N] (label indices).
        """
        self.predictions = Annotations()

        if not isinstance(predictions, torch.Tensor):
            raise TypeError("For classification, predictions must be a torch.Tensor.")

        num_samples = predictions.shape[0]
        if num_samples != len(self.dataset):
            raise ValueError("Mismatch between predictions size and dataset length.")

        # If shape is [N], assume these are predicted labels (class IDs)
        # If shape is [N, C], assume these are logits or probabilities
        # TODO: Add support for multi-label predictions
        if predictions.dim() == 1:
            for i in tqdm(
                range(num_samples), desc="Building CLASSIFICATION predictions"
            ):
                label_val = predictions[i].unsqueeze(0)
                ann = Labels(datapoint_number=i, labels=label_val, scores=None)
                self.predictions.add(ann)

        elif predictions.dim() == 2:
            # logits or probabilities of shape [N, C]
            # currently we always interpret them as logits, with an argmax
            for i in tqdm(
                range(num_samples), desc="Building CLASSIFICATION predictions"
            ):
                logit_row = predictions[i]
                # "labels" is the top-1 predicted label
                pred_label = logit_row.argmax(dim=0).unsqueeze(0)
                scores = torch.softmax(logit_row, dim=0)
                ann = Labels(
                    datapoint_number=i,
                    labels=pred_label,  # shape [1]
                    scores=scores,  # shape [self.num_classes]
                )
                self.predictions.add(ann)

        else:
            raise ValueError("Classification predictions must be 1D or 2D tensor.")
