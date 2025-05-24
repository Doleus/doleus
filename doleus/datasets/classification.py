from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset, Subset

from doleus.datasets.base import Doleus
from doleus.utils import TaskType
from doleus.storage import ClassificationGroundTruthStore, ClassificationPredictionStore


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
        
        # Instantiate the classification-specific stores
        self.groundtruth_store = ClassificationGroundTruthStore(
            dataset=self.dataset, task=self.task, num_classes=self.num_classes
        )
        self.prediction_store = ClassificationPredictionStore()

    def _create_new_instance(self, dataset, indices, name):
        subset = Subset(dataset, indices)
        metadata_subset = self.metadata_store.get_subset(indices)
        new_instance = DoleusClassification(
            dataset=subset,
            name=name,
            task=self.task,
            num_classes=self.num_classes,
            label_to_name=self.label_to_name,
            metadata=self.metadata.copy(),
            per_datapoint_metadata=metadata_subset,
        )

        # Copy sliced predictions directly to the new instance
        for model_id in self.prediction_store.predictions:
            sliced_preds = self.prediction_store.get_subset(model_id, indices)
            new_instance.prediction_store.predictions[model_id] = sliced_preds

        return new_instance
