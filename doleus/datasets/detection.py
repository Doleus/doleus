# SPDX-FileCopyrightText: 2025 Doleus contributors
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional

from torch.utils.data import Dataset, Subset

from doleus.datasets.base import Doleus
from doleus.storage import DetectionGroundTruthStore, DetectionPredictionStore
from doleus.utils import TaskType


class DoleusDetection(Doleus):
    """Dataset wrapper for detection tasks."""

    def __init__(
        self,
        dataset: Dataset,
        name: str,
        label_to_name: Optional[Dict[int, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        per_datapoint_metadata: Optional[List[Dict[str, Any]]] = None,
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
        per_datapoint_metadata : Optional[List[Dict[str, Any]]], optional
            Per-datapoint metadata, by default None.
        """
        super().__init__(
            dataset=dataset,
            name=name,
            task_type=TaskType.DETECTION.value,
            label_to_name=label_to_name,
            metadata=metadata,
            per_datapoint_metadata=per_datapoint_metadata,
        )
        self.groundtruth_store = DetectionGroundTruthStore(dataset=self.dataset)
        self.prediction_store = DetectionPredictionStore()

    def _create_new_instance(self, dataset, indices, slice_name):
        subset = Subset(dataset, indices)
        new_metadata = self.metadata_store.get_subset(indices)
        new_instance = DoleusDetection(
            dataset=subset,
            name=slice_name,
            label_to_name=self.label_to_name,
            metadata=self.metadata.copy(),
            per_datapoint_metadata=new_metadata,
        )

        if self.prediction_store and self.prediction_store.predictions:
            for model_id in self.prediction_store.predictions:
                sliced_preds_annotations = self.prediction_store.get_subset(
                    model_id, indices
                )
                new_instance.prediction_store.predictions[model_id] = (
                    sliced_preds_annotations
                )

        return new_instance
