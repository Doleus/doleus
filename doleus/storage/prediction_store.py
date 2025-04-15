from typing import Any, Dict, List, Union

import torch


class PredictionStore:
    """Storage for model predictions for a specific dataset instance.

    Each Doleus Dataset has its own PredictionStore instance to manage
    predictions from different models for that specific dataset.
    """

    def __init__(self):
        self.predictions: Dict[str, Union[torch.Tensor, List[Dict[str, Any]]]] = {}

    def add_predictions(
        self,
        predictions: Union[torch.Tensor, List[Dict[str, Any]]],
        model_id: str,
    ) -> None:
        """
        Store predictions for a model.

        Parameters
        ----------
        predictions : Union[torch.Tensor, List[Dict[str, Any]]]
            Model predictions to store. For classification, this should be a
            tensor of shape [N, C] where N is the number of samples and C is the
            number of classes. For detection, this should be a list of dictionaries
            with 'boxes', 'labels', and 'scores' keys.
        model_id : str
            Identifier of the specified model.
        """
        self.predictions[model_id] = predictions

    def get_predictions(
        self, model_id: str
    ) -> Union[torch.Tensor, List[Dict[str, Any]]]:
        """Retrieve predictions for a specific model.

        Parameters
        ----------
        model_id : str
            Identifier of the specified model.

        Returns
        -------
        Union[torch.Tensor, List[Dict[str, Any]]]
            The stored predictions
        """
        if model_id not in self.predictions:
            raise KeyError(f"No predictions found for model: {model_id}")
        return self.predictions[model_id]
