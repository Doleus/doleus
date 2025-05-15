from typing import Any, Dict, List, Union

import torch

from doleus.annotations import Annotations, BoundingBoxes, Labels
from doleus.utils import TaskType


class PredictionStore:
    """Storage for model predictions for a specific dataset instance.

    Each Doleus Dataset has its own PredictionStore instance to manage
    predictions from different models for that specific dataset.
    """

    def __init__(self, task_type: str):
        """Initialize the prediction store.

        Parameters
        ----------
        task_type : str
            Type of task (e.g., "classification", "detection").
        """
        self.task_type = task_type
        self.predictions: Dict[str, Annotations] = {}

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
        processed_predictions = self._process_predictions(predictions)
        self.predictions[model_id] = processed_predictions

    def _process_predictions(
        self, predictions: Union[torch.Tensor, List[Dict[str, Any]], Annotations]
    ) -> Annotations:
        """Process raw predictions into the standard annotation format.

        Parameters
        ----------
        predictions : Union[torch.Tensor, List[Dict[str, Any]], Annotations]
            Raw predictions to process. Can be:
            - A torch.Tensor for classification tasks
            - A list of dictionaries for detection tasks
            - An already processed Annotations object

        Returns
        -------
        Annotations
            Processed predictions in standard annotation format.
        """
        if isinstance(predictions[0], Labels) or isinstance(
            predictions[0], BoundingBoxes
        ):
            return predictions

        processed = Annotations()

        if self.task_type == TaskType.CLASSIFICATION.value:
            if not isinstance(predictions, torch.Tensor):
                raise TypeError(
                    "For classification, predictions must be a torch.Tensor."
                )

            num_samples = predictions.shape[0]

            # If shape is [N], assume these are predicted labels (class IDs)
            # If shape is [N, C], assume these are logits or probabilities
            if predictions.dim() == 1:
                for i in range(num_samples):
                    label_val = predictions[i].unsqueeze(0)
                    ann = Labels(datapoint_number=i, labels=label_val, scores=None)
                    processed.add(ann)

            elif predictions.dim() == 2:
                # logits or probabilities of shape [N, C]
                # currently we always interpret them as logits, with an argmax
                for i in range(num_samples):
                    logit_row = predictions[i]
                    # "labels" is the top-1 predicted label
                    pred_label = logit_row.argmax(dim=0).unsqueeze(0)
                    scores = torch.softmax(logit_row, dim=0)
                    ann = Labels(
                        datapoint_number=i,
                        labels=pred_label,  # shape [1]
                        scores=scores,  # shape [self.num_classes]
                    )
                    processed.add(ann)

            else:
                raise ValueError("Classification predictions must be 1D or 2D tensor.")

        elif self.task_type == TaskType.DETECTION.value:
            if not isinstance(predictions, list):
                raise TypeError(
                    "For detection, predictions must be a list of length N."
                )

            # Each element should look like {"boxes": (M,4), "labels": (M,), "scores": (M,)}
            for i, pred_dict in enumerate(predictions):
                boxes_xyxy = torch.tensor(pred_dict["boxes"], dtype=torch.float32)
                labels = torch.tensor(pred_dict["labels"], dtype=torch.long)
                scores = torch.tensor(pred_dict["scores"], dtype=torch.float32)

                ann = BoundingBoxes(
                    datapoint_number=i,
                    boxes_xyxy=boxes_xyxy,
                    labels=labels,
                    scores=scores,
                )
                processed.add(ann)

        return processed

    def get(self, model_id: str, datapoint_number: int):
        """Get annotation by datapoint number.

        Parameters
        ----------
        datapoint_number : int
            The ID of the sample in the dataset.

        Returns
        -------
        Annotation
            The annotation for the datapoint.
        """
        if model_id not in self.predictions:
            raise KeyError(f"No predictions found for model: {model_id}")
        return self.predictions[model_id][datapoint_number]

    def get_subset(self, model_id: str, indices: List[int]) -> List[Any]:
        """Get a subset of predictions for a specific model based on indices.

        Parameters
        ----------
        model_id : str
            Identifier of the model to get predictions for.
        indices : List[int]
            List of indices to get predictions for.

        Returns
        -------
        List[Any]
            List of predictions for the specified indices.
        """
        if model_id not in self.predictions:
            raise KeyError(f"No predictions found for model: {model_id}")
        return [self.predictions[model_id][i] for i in indices]

    def get_predictions(self, model_id: str) -> List[Any]:
        """Get all predictions for a specific model.

        Parameters
        ----------
        model_id : str
            Identifier of the model to get predictions for.

        Returns
        -------
        List[Any]
            List of all predictions for the specified model.
        """
        if model_id not in self.predictions:
            raise KeyError(f"No predictions found for model: {model_id}")
        return self.predictions[model_id].annotations
