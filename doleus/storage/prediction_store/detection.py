from typing import Any, Dict, List

import torch

from doleus.annotations import Annotations, BoundingBoxes
from doleus.storage.prediction_store.base import BasePredictionStore


class DetectionPredictionStore(BasePredictionStore):
    """Storage for detection model predictions."""

    def add_predictions(
        self,
        predictions: List[Dict[str, Any]],
        model_id: str,
        **kwargs,
    ) -> None:
        """
        Store predictions for a detection model.

        Parameters
        ----------
        predictions : List[Dict[str, Any]]
            Model predictions to store. This should be a list of dictionaries,
            each with 'boxes', 'labels', and 'scores' keys.
        model_id : str
            Identifier of the specified model.
        """
        if not isinstance(predictions, list):
            raise TypeError(
                "For detection, predictions must be a list of dictionaries."
            )
        if not all(isinstance(p, dict) for p in predictions):
            raise TypeError(
                "Each item in detection predictions list must be a dictionary."
            )

        processed_predictions = self._process_predictions(predictions)
        self.predictions[model_id] = processed_predictions

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
            An Annotations object containing predictions for the specified indices,
            with datapoint_number values re-indexed starting from 0.
        """
        if model_id not in self.predictions:
            raise KeyError(f"No predictions found for model: {model_id}")

        subset_annotations = Annotations()
        for new_idx, original_idx in enumerate(indices):
            original_annotation = self.predictions[model_id][original_idx]

            # Create a new BoundingBoxes annotation with re-indexed datapoint_number
            new_annotation = BoundingBoxes(
                datapoint_number=new_idx,
                boxes_xyxy=original_annotation.boxes_xyxy,
                labels=original_annotation.labels,
                scores=original_annotation.scores,
            )
            subset_annotations.add(new_annotation)
        return subset_annotations

    def _process_predictions(
        self,
        predictions: List[Dict[str, Any]],
        **kwargs,
    ) -> Annotations:
        """Process raw detection predictions into the standard annotation format.

        Parameters
        ----------
        predictions : List[Dict[str, Any]]
            Raw predictions to process. Must be a list of dictionaries.

        Returns
        -------
        Annotations
            Processed predictions in standard annotation format.
        """
        processed = Annotations()

        for i, pred_dict in enumerate(predictions):
            # Validate keys
            required_keys = {"boxes", "labels"}
            if not required_keys.issubset(pred_dict.keys()):
                raise ValueError(
                    f"Detection prediction dict for sample {i} missing keys. "
                    f"Required: {required_keys}, Got: {list(pred_dict.keys())}"
                )

            boxes_xyxy = torch.as_tensor(pred_dict["boxes"], dtype=torch.float32)
            labels = torch.as_tensor(pred_dict["labels"], dtype=torch.long)

            scores = None
            if "scores" in pred_dict:
                scores = torch.as_tensor(pred_dict["scores"], dtype=torch.float32)

            # Validate shapes
            num_detections = boxes_xyxy.shape[0]
            if not (boxes_xyxy.ndim == 2 and boxes_xyxy.shape[1] == 4):
                raise ValueError(
                    f"boxes for sample {i} must have shape (M,4), Got: {boxes_xyxy.shape}"
                )
            if not (labels.ndim == 1 and labels.shape[0] == num_detections):
                raise ValueError(
                    f"labels for sample {i} must have shape (M,), Got: {labels.shape}"
                )
            if scores is not None and not (
                scores.ndim == 1 and scores.shape[0] == num_detections
            ):
                raise ValueError(
                    f"scores for sample {i} must have shape (M,), Got: {scores.shape}"
                )

            ann = BoundingBoxes(
                datapoint_number=i,
                boxes_xyxy=boxes_xyxy,
                labels=labels,
                scores=scores,
            )
            processed.add(ann)
        return processed
