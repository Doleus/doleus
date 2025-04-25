from typing import Any, Dict, List, Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset
from tqdm import tqdm

from doleus.annotations import Annotations, Labels
from doleus.datasets.base import Doleus
from doleus.utils import TaskType


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
        per_datapoint_metadata: Optional[List[Dict[str, Any]]] = None,
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
        per_datapoint_metadata : Optional[List[Dict[str, Any]]], optional
            Per-datapoint metadata, by default None.
        """
        self.num_classes = num_classes
        super().__init__(
            dataset=dataset,
            name=name,
            task_type=TaskType.CLASSIFICATION.value,
            task=task,
            label_to_name=label_to_name,
            metadata=metadata,
            per_datapoint_metadata=per_datapoint_metadata,
        )

    def process_groundtruths(self):
        """Process and store classification ground truth annotations.

        Extracts labels from the underlying dataset and stores them as Labels.
        """
        groundtruths = Annotations()
        for idx in tqdm(
            range(len(self.dataset)), desc="Building CLASSIFICATION ground truths"
        ):
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

        self.groundtruth_store.add_groundtruths(groundtruths)

    def process_predictions(self, predictions: Tensor):
        """Process and store classification model predictions.

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

    def _create_new_instance(self, dataset, indices):
        subset = Subset(dataset, indices)
        new_metadata = [self.metadata_store.metadata[i] for i in indices]
        new_instance = DoleusClassification(
            dataset=subset,
            name=f"{self.name}_subset",
            task=self.task,
            num_classes=self.num_classes,
            label_to_name=self.label_to_name,
            metadata=self.metadata.copy(),
            per_datapoint_metadata=new_metadata,
        )

        # Copy model predictions if available
        for model_id in self.prediction_store.predictions:
            preds = self.prediction_store.predictions[model_id]
            sliced_preds = preds[indices]
            new_instance.prediction_store.add_predictions(sliced_preds, model_id)

        return new_instance
