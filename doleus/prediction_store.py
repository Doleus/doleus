from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch

from doleus.utils.helpers import get_current_timestamp


@dataclass
class PredictionMetadata:
    """Metadata for a set of predictions"""

    model_id: str
    timestamp: str


class PredictionStore:
    """Centralized storage for model predictions"""

    def __init__(self):
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
            Model predictions to store
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
        """Retrieve predictions for a specific dataset and model"""
        if dataset_id not in self._predictions:
            raise KeyError(f"No predictions found for dataset: {dataset_id}")
        if model_id not in self._predictions[dataset_id]:
            raise KeyError(f"No predictions found for model: {model_id}")
        return self._predictions[dataset_id][model_id]

    def get_metadata(self, model_id: str) -> PredictionMetadata:
        """Get metadata for a specific model's predictions"""
        return self._metadata[model_id]

    def list_models(self, dataset_id: str = None) -> List[str]:
        """List all model names, optionally filtered by dataset"""
        if dataset_id:
            return list(self._predictions[dataset_id].keys())
        return list(self._metadata.keys())
