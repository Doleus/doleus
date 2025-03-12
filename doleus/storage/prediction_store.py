"""Centralized storage for model predictions."""

from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch

from doleus.utils.helpers import get_current_timestamp


@dataclass
class PredictionMetadata:
    """Metadata for a set of predictions.
    
    Contains information about when and by which model predictions were generated.
    """

    model_id: str
    timestamp: str


class PredictionStore:
    """Centralized storage for model predictions across multiple datasets.
    
    This class provides a way to store, retrieve, and manage model predictions
    with associated metadata.
    """

    def __init__(self):
        """Initialize an empty prediction store."""
        self._predictions: Dict[str, Dict[str, torch.Tensor]] = (
            {}
        )  # {dataset_id: {model_id: predictions}}
        self._metadata: Dict[str, PredictionMetadata] = {}  # {model_id: metadata}

    def add_predictions(
        self,
        predictions: Union[torch.Tensor, List[Dict[str, Any]]],
        dataset_id: str,
        model_id: str,
    ) -> None:
        """
        Store predictions with associated metadata.

        Parameters
        ----------
        predictions : Union[torch.Tensor, List[Dict[str, Any]]]
            Model predictions to store. For classification, this is typically a
            tensor of shape [N, C] where N is the number of samples and C is the
            number of classes. For detection, this is typically a list of dictionaries
            with 'boxes', 'labels', and 'scores' keys.
        dataset_id : str
            ID of the dataset these predictions are for
        model_id : str
            Name of the model that generated these predictions
        """
        if dataset_id not in self._predictions:
            self._predictions[dataset_id] = {}

        self._predictions[dataset_id][model_id] = predictions
        self._metadata[model_id] = PredictionMetadata(
            model_id=model_id,
            timestamp=get_current_timestamp(),
        )

    def get_predictions(
        self, dataset_id: str, model_id: str
    ) -> Union[torch.Tensor, List[Dict[str, Any]]]:
        """Retrieve predictions for a specific dataset and model.
        
        Parameters
        ----------
        dataset_id : str
            ID of the dataset to get predictions for
        model_id : str
            ID of the model to get predictions from
            
        Returns
        -------
        Union[torch.Tensor, List[Dict[str, Any]]]
            The stored predictions
            
        Raises
        ------
        KeyError
            If no predictions found for the specified dataset or model
        """
        if dataset_id not in self._predictions:
            raise KeyError(f"No predictions found for dataset: {dataset_id}")
        if model_id not in self._predictions[dataset_id]:
            raise KeyError(f"No predictions found for model: {model_id}")
        return self._predictions[dataset_id][model_id]

    def get_metadata(self, model_id: str) -> PredictionMetadata:
        """Get metadata for a specific model's predictions.
        
        Parameters
        ----------
        model_id : str
            ID of the model to get metadata for
            
        Returns
        -------
        PredictionMetadata
            Metadata for the specified model
            
        Raises
        ------
        KeyError
            If no metadata found for the specified model
        """
        if model_id not in self._metadata:
            raise KeyError(f"No metadata found for model: {model_id}")
        return self._metadata[model_id]

    def list_models(self, dataset_id: str = None) -> List[str]:
        """List all model names, optionally filtered by dataset.
        
        Parameters
        ----------
        dataset_id : str, optional
            ID of the dataset to filter by, by default None
            
        Returns
        -------
        List[str]
            List of model IDs
            
        Raises
        ------
        KeyError
            If the specified dataset_id doesn't exist
        """
        if dataset_id:
            if dataset_id not in self._predictions:
                raise KeyError(f"No predictions found for dataset: {dataset_id}")
            return list(self._predictions[dataset_id].keys())
        return list(self._metadata.keys()) 