from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import torch

from doleus.annotations import Annotation, Annotations


class BasePredictionStore(ABC):
    """Base storage for model predictions for a specific dataset instance."""

    def __init__(self):
        """Initialize the prediction store."""
        self.predictions: Dict[str, Annotations] = {}

    @abstractmethod
    def add_predictions(
        self,
        predictions: Union[torch.Tensor, List[Dict[str, Any]]],
        model_id: str,
        **kwargs,
    ) -> None:
        """
        Store predictions for a model.
        Actual implementation will depend on the task type (classification, detection).

        Parameters
        ----------
        predictions : Union[torch.Tensor, List[Dict[str, Any]]]
            Model predictions to store. For classification, this should be a
            tensor of shape [N, C] where N is the number of samples and C is the
            number of classes. For detection, this should be a list of dictionaries
            with 'boxes', 'labels', and 'scores' keys.
        model_id : str
            Identifier of the specified model.
        kwargs : dict
            Additional arguments specific to the subclass implementation (e.g., 'task' for classification).
        """
        pass

    @abstractmethod
    def _process_predictions(
        self,
        predictions: Union[torch.Tensor, List[Dict[str, Any]], Annotations],
        **kwargs,
    ) -> Annotations:
        """
        Process raw predictions into the standard annotation format.
        Actual implementation will depend on the task type.

        Parameters
        ----------
        predictions : Union[torch.Tensor, List[Dict[str, Any]], Annotations]
            Raw predictions to process.
        kwargs : dict
            Additional arguments specific to the subclass implementation (e.g., 'task' for classification).

        Returns
        -------
        Annotations
            Processed predictions in standard annotation format.
        """
        pass

    def get(self, model_id: str, datapoint_number: int) -> Annotation:
        """Get a single annotation object by datapoint number.

        Parameters
        ----------
        model_id : str
            Identifier of the model to get predictions for.
        datapoint_number : int
            The ID of the sample in the dataset.

        Returns
        -------
        Annotation
            The specific Annotation object (e.g., Labels, BoundingBoxes) for the datapoint.
        """
        if model_id not in self.predictions:
            raise KeyError(f"No predictions found for model: {model_id}")
        return self.predictions[model_id][datapoint_number]

    def get_subset(self, model_id: str, indices: List[int]) -> Annotations:
        """Get a subset of predictions for a specific model based on indices.

        Parameters
        ----------
        model_id : str
            Identifier of the model to get predictions for.
        indices : List[int]
            List of indices to get predictions for.

        Returns
        -------
        Annotations
            An Annotations object containing predictions for the specified indices.
        """
        if model_id not in self.predictions:
            raise KeyError(f"No predictions found for model: {model_id}")
        
        subset_annotations = Annotations()
        for i in indices:
            subset_annotations.add(self.predictions[model_id][i])
        return subset_annotations

    def get_predictions(self, model_id: str) -> Annotations:
        """Get all predictions for a specific model.

        Parameters
        ----------
        model_id : str
            Identifier of the model to get predictions for.

        Returns
        -------
        Annotations
            An Annotations object containing all predictions for the specified model.
        """
        if model_id not in self.predictions:
            raise KeyError(f"No predictions found for model: {model_id}")
        return self.predictions[model_id] 