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

        for model_id in self.prediction_store.predictions:
            preds = self.prediction_store.predictions[model_id]
            sliced_preds = (
                preds[indices]
                if isinstance(preds, torch.Tensor)
                else [preds[i] for i in indices]
            )
            new_instance.prediction_store.add_predictions(sliced_preds, model_id)

        return new_instance
