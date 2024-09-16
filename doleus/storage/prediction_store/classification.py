from typing import List, Optional

import torch
from torch import Tensor

from doleus.annotations import Annotations, Labels
from doleus.storage.prediction_store.base import BasePredictionStore
from doleus.utils import Task


class ClassificationPredictionStore(BasePredictionStore):
    """Storage for classification model predictions."""

    def add_predictions(
        self,
        predictions: torch.Tensor,
        model_id: str,
        task: str,
    ) -> None:
        """
        Store predictions for a classification model.

        Parameters
        ----------
        predictions : torch.Tensor
            Model predictions to store. This should be a tensor of shape [N, C]
            where N is the number of samples and C is the number of classes.
        model_id : str
            Identifier of the specified model.
        task : str
            The specific classification task (e.g., "multilabel", "multiclass", "binary").
        """
        if not isinstance(predictions, torch.Tensor):
            raise TypeError("For classification, predictions must be a torch.Tensor.")

        if predictions.dtype == torch.bool:
            raise TypeError(
                "Bool tensors are not supported. Please use int or float tensors."
            )

        processed_predictions = self._process_predictions(predictions, task=task)
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

            # Create a new Labels annotation with re-indexed datapoint_number
            new_annotation = Labels(
                datapoint_number=new_idx,
                labels=original_annotation.labels,
                scores=original_annotation.scores,
            )
            subset_annotations.add(new_annotation)
        return subset_annotations

    def _process_predictions(
        self,
        predictions: torch.Tensor,
        task: str,
    ) -> Annotations:
        """Process raw classification predictions into the standard annotation format.

        The behavior depends on the `task` and the shape/dtype of `predictions`:

        - Task.BINARY.value:
            - If `predictions` is 1D and `dtype` is float (scores/logits for the positive class):
                - `Labels.scores` will store the raw float score/logit unchanged.
                - `Labels.labels` will be `None`.
            - If `predictions` is 1D and `dtype` is int (0 or 1):
                - `Labels.labels` will store the integer label.
                - `Labels.scores` will be `None`.
            - 2D predictions currently raise a ValueError.

        - Task.MULTICLASS.value:
            - If `predictions` is 1D and `dtype` is int (class indices):
                - `Labels.labels` will store the integer class index.
                - `Labels.scores` will be `None`.
            - If `predictions` is 1D and `dtype` is float: Raises ValueError (expected class indices).
            - If `predictions` is 2D (shape [N, C]) and `dtype` is float (logits or probabilities per class):
                - `Labels.scores` will store the raw [C] float tensor (logits/probabilities unchanged).
                - `Labels.labels` will store the class index derived from `argmax` of raw scores.
            - If `predictions` is 2D and `dtype` is int: Raises ValueError (expected float scores/logits).

        - Task.MULTILABEL.value:
            - If `predictions` is 2D (shape [N, C]) and `dtype` is float (logits or probabilities per class):
                - `Labels.scores` will store the raw [C] float tensor (logits/probabilities unchanged).
                - `Labels.labels` will be `None`.
            - If `predictions` is 2D (shape [N, C]) and `dtype` is int (multi-hot encoded):
                - `Labels.labels` will store the [C] integer tensor.
                - `Labels.scores` will be `None`.
            - Boolean inputs are not supported for multilabel. 1D predictions or other dtypes for 2D currently raise a ValueError.

        Parameters
        ----------
        predictions : torch.Tensor
            Raw predictions to process. Typically shape [N] or [N, C].
        task : str
            The specific classification task ("binary", "multiclass", "multilabel").

        Returns
        -------
        Annotations
            Processed predictions where each element is a `Labels` object.
        """
        processed = Annotations()
        num_samples = predictions.shape[0]

        for i in range(num_samples):
            current_labels: Optional[Tensor] = None
            current_scores: Optional[Tensor] = None

            if task == Task.BINARY.value:
                if predictions.dim() == 1:
                    if predictions.dtype.is_floating_point:
                        current_scores = predictions[i].unsqueeze(0)
                        current_labels = (
                            None  # Scores are provided, so labels can be None
                        )
                    else:  # Integer type
                        current_labels = predictions[i].unsqueeze(0)
                        label_value = current_labels.item()
                        if not (label_value == 0 or label_value == 1):
                            raise ValueError(
                                f"Binary prediction labels must be 0 or 1. Got: {label_value} at sample {i}"
                            )
                        current_scores = None
                elif predictions.dim() == 2:
                    # TODO: We need to handle samplewise predictions at some point.
                    raise ValueError(
                        f"{task} classification predictions must be 1D tensor. Got {predictions.dim()}D"
                    )
                else:
                    raise ValueError(
                        f"{task} classification predictions must be 1D or 2D tensor. Got {predictions.dim()}D"
                    )

            elif task == Task.MULTICLASS.value:
                if predictions.dim() == 1:
                    if predictions.dtype.is_floating_point:
                        raise ValueError(
                            f"For {task} with 1D predictions, dtype must be integer, got {predictions.dtype}"
                        )
                    else:  # Integer type
                        current_labels = predictions[i].unsqueeze(0)
                        # Validate multiclass labels are non-negative
                        label_value = current_labels.item()
                        if label_value < 0:
                            raise ValueError(
                                f"Multiclass prediction labels must be non-negative. Got: {label_value} at sample {i}"
                            )
                        current_scores = None
                elif predictions.dim() == 2:  # Shape [N, C]
                    prediction_sample = predictions[i]  # Shape [C]
                    if prediction_sample.dtype.is_floating_point:
                        current_labels = prediction_sample.argmax(dim=0).unsqueeze(0)
                        current_scores = prediction_sample
                    else:  # Integer type
                        raise ValueError(
                            f"For {task} with 2D predictions, dtype must be float (scores/logits), got {prediction_sample.dtype}"
                        )
                else:
                    raise ValueError(
                        f"{task} classification predictions must be 1D or 2D tensor. Got {predictions.dim()}D"
                    )

            elif task == Task.MULTILABEL.value:
                if predictions.dim() == 2:  # Expect [N, C]
                    prediction_sample = predictions[i]  # Shape [C]

                    if prediction_sample.dtype.is_floating_point:
                        current_scores = prediction_sample
                        current_labels = None

                    else:  # Integer type
                        current_labels = prediction_sample.int()
                        if not torch.all((current_labels == 0) | (current_labels == 1)):
                            raise ValueError(
                                f"Multilabel prediction labels must be multi-hot encoded (0s and 1s only). Got: {current_labels} at sample {i}"
                            )
                        current_scores = None
                else:
                    raise ValueError(
                        f"{task} classification predictions must be a 2D tensor of shape [N, C]. "
                        f"Got {predictions.dim()}D with shape {predictions.shape}"
                    )

            else:
                raise ValueError(f"Unsupported task: {task}")

            ann = Labels(
                datapoint_number=i, labels=current_labels, scores=current_scores
            )
            processed.add(ann)

        return processed
