from typing import Dict, Union, List, Any
import torch
from dataclasses import dataclass
import uuid
from moonwatcher.utils.helpers import get_current_timestamp

@dataclass
class PredictionMetadata:
    """Metadata for a set of predictions"""
    model_name: str
    timestamp: str
    model_metadata: Dict[str, Any] = None

class PredictionStore:
    """Centralized storage for model predictions"""
    def __init__(self):
        self._predictions: Dict[str, Dict[str, torch.Tensor]] = {}  # {dataset_name: {model_id: predictions}}
        self._metadata: Dict[str, PredictionMetadata] = {}  # {model_id: metadata}
    
    def add_predictions(
        self,
        predictions: Union[torch.Tensor, List[Dict[str, Any]]],
        dataset_name: str,
        model_name: str,
        model_metadata: Dict[str, Any] = None
    ) -> str:
        """
        Store predictions with associated metadata.
        
        Returns:
            model_id: Unique identifier for this set of predictions
        """
        model_id = str(uuid.uuid4())
        
        if dataset_name not in self._predictions:
            self._predictions[dataset_name] = {}
            
        self._predictions[dataset_name][model_id] = predictions
        self._metadata[model_id] = PredictionMetadata(
            model_name=model_name,
            timestamp=get_current_timestamp(),
            model_metadata=model_metadata
        )
        
        return model_id
    
    def get_predictions(
        self,
        dataset_name: str,
        model_id: str
    ) -> Union[torch.Tensor, List[Dict[str, Any]]]:
        """Retrieve predictions for a specific dataset and model"""
        if dataset_name not in self._predictions:
            raise KeyError(f"No predictions found for dataset: {dataset_name}")
        if model_id not in self._predictions[dataset_name]:
            raise KeyError(f"No predictions found for model_id: {model_id}")
        return self._predictions[dataset_name][model_id]
    
    def get_metadata(self, model_id: str) -> PredictionMetadata:
        """Get metadata for a specific model's predictions"""
        return self._metadata[model_id]
    
    def list_models(self, dataset_name: str = None) -> List[str]:
        """List all model IDs, optionally filtered by dataset"""
        if dataset_name:
            return list(self._predictions[dataset_name].keys())
        return list(self._metadata.keys()) 