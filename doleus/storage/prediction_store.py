from typing import Any, Dict, List, Optional, Union

import torch

from doleus.annotations import Annotations, BoundingBoxes, Labels
from doleus.utils import Task, TaskType


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
        task: Optional[str] = None,
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
        task : Optional[str], optional
            The specific task (e.g., "multilabel", "multiclass"), by default None.
        """
        processed_predictions = self._process_predictions(predictions, task)
        self.predictions[model_id] = processed_predictions

    def _process_predictions(
        self,
        predictions: Union[torch.Tensor, List[Dict[str, Any]], Annotations],
        task: Optional[str] = None,
    ) -> Annotations:
        """Process raw predictions into the standard annotation format.

        Parameters
        ----------
        predictions : Union[torch.Tensor, List[Dict[str, Any]], Annotations]
            Raw predictions to process. Can be:
            - A torch.Tensor for classification tasks
            - A list of dictionaries for detection tasks
            - An already processed Annotations object
        task : Optional[str], optional
            The specific task (e.g., "multilabel", "multiclass"), by default None.


        Returns
        -------
        Annotations
            Processed predictions in standard annotation format.
        """
        if isinstance(predictions, Annotations) and (
            isinstance(predictions[0], Labels)
            or isinstance(predictions[0], BoundingBoxes)
        ):
            return predictions

        processed = Annotations()

        if self.task_type == TaskType.CLASSIFICATION.value:
            if not isinstance(predictions, torch.Tensor):
                raise TypeError(
                    "For classification, predictions must be a torch.Tensor."
                )

            num_samples = predictions.shape[0]

            if predictions.dim() == 1:
                # Assume these are predicted labels (class IDs) for single-label tasks
                for i in range(num_samples):
                    label_val = predictions[i].unsqueeze(0) # Ensure [1] shape
                    ann = Labels(datapoint_number=i, labels=label_val, scores=None)
                    processed.add(ann)

            elif predictions.dim() == 2: # Shape [N, C]
                for i in range(num_samples):
                    prediction_row = predictions[i]  # This is the [C] tensor for the i-th sample
                    current_labels: torch.Tensor
                    current_scores: Optional[torch.Tensor]

                    if task == Task.MULTILABEL.value:
                        if prediction_row.dtype in (torch.long, torch.int, torch.bool):
                            # Input is integer multi-hot
                            current_labels = prediction_row
                            current_scores = None
                        else:  # Float input, assumed to be logits or probabilities
                            # Apply sigmoid if not already probabilities in [0,1]
                            if not (prediction_row.min() >= 0 and prediction_row.max() <= 1):
                                processed_scores_for_row = torch.sigmoid(prediction_row)
                            else: # Already probabilities
                                processed_scores_for_row = prediction_row
                            
                            current_labels = (processed_scores_for_row >= 0.5).long() # Default threshold 0.5
                            current_scores = processed_scores_for_row
                        
                        ann = Labels(
                            datapoint_number=i,
                            labels=current_labels,
                            scores=current_scores,
                        )
                    else:  # Binary, Multiclass, or task is None (default to old behavior)
                        current_labels = prediction_row.argmax(dim=0).unsqueeze(0) # [1] tensor
                        
                        if prediction_row.dtype == torch.float:
                            current_scores = torch.softmax(prediction_row, dim=0) # [C] tensor
                        else:  # Integer input
                            current_scores = None
                        
                        ann = Labels(
                            datapoint_number=i,
                            labels=current_labels,
                            scores=current_scores,
                        )
                    processed.add(ann)
            else:
                raise ValueError(
                    "Classification predictions must be a 1D or 2D tensor."
                )

        elif self.task_type == TaskType.DETECTION.value:
            if not isinstance(predictions, list):
                raise TypeError(
                    "For detection, predictions must be a list of length N."
                )
            if not all(isinstance(p, dict) for p in predictions):
                raise TypeError(
                    "Each item in detection predictions list must be a dictionary."
                )


            # Each element should look like {"boxes": (M,4), "labels": (M,), "scores": (M,)}
            for i, pred_dict in enumerate(predictions):
                # Validate keys
                required_keys = {"boxes", "labels", "scores"}
                if not required_keys.issubset(pred_dict.keys()):
                    raise ValueError(f"Detection prediction dict for sample {i} missing keys. Required: {required_keys}")

                boxes_xyxy = torch.as_tensor(pred_dict["boxes"], dtype=torch.float32)
                labels = torch.as_tensor(pred_dict["labels"], dtype=torch.long)
                scores = torch.as_tensor(pred_dict["scores"], dtype=torch.float32)

                # Validate shapes
                num_detections = boxes_xyxy.shape[0]
                if not (boxes_xyxy.ndim == 2 and boxes_xyxy.shape[1] == 4):
                    raise ValueError(f"boxes for sample {i} must have shape (M,4)")
                if not (labels.ndim == 1 and labels.shape[0] == num_detections):
                    raise ValueError(f"labels for sample {i} must have shape (M,)")
                if not (scores.ndim == 1 and scores.shape[0] == num_detections):
                    raise ValueError(f"scores for sample {i} must have shape (M,)")


                ann = BoundingBoxes(
                    datapoint_number=i,
                    boxes_xyxy=boxes_xyxy,
                    labels=labels,
                    scores=scores,
                )
                processed.add(ann)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")


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
