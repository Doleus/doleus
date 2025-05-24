import torch
from typing import Any

from doleus.storage.ground_truth_store.base import BaseGroundTruthStore
from doleus.annotations import Annotations
from doleus.annotations.classification import Labels
from doleus.utils.data import Task


class ClassificationGroundTruthStore(BaseGroundTruthStore):
    """Ground truth store for classification tasks."""

    def __init__(self, dataset: Any, task: str, num_classes: int):
        """
        Initialize the classification ground truth store.

        Parameters
        ----------
        dataset : Any
            The raw PyTorch dataset object.
        task : str
            The specific classification task (e.g., Task.BINARY.value, Task.MULTICLASS.value, Task.MULTILABEL.value).
        num_classes : int
            The number of classes for the task.
        """
        self.task = task
        self.num_classes = num_classes
        super().__init__(dataset) 

    def _process_groundtruths(self) -> Annotations:
        """
        Process raw ground truth data from the dataset for classification tasks.

        Returns
        -------
        Annotations
            Processed ground truths in standard annotation format.
        
        Raises
        ------
        ValueError
            If the task is unsupported or if ground truth data is in an invalid format.
        """
        processed_annotations = Annotations()

        for idx, data in enumerate(self.dataset):
            # Assuming standard (image, label) structure for dataset items
            if not (isinstance(data, (list, tuple)) and len(data) > 1):
                raise ValueError(
                    f"Dataset item at index {idx} is not in the expected format (e.g., (image, target)). "
                    f"Got: {type(data)}"
                )
            
            label = data[1]
            processed_label_tensor: torch.Tensor

            if self.task == Task.BINARY.value:
                if not isinstance(label, torch.Tensor):
                    label = torch.tensor(label)
                
                if not (label.ndim == 0 or (label.ndim == 1 and label.numel() == 1)):
                    raise ValueError(
                        f"Binary ground truth for item {idx} must be a scalar or 1-element tensor. Got shape: {label.shape}"
                    )
                if not (label.item() == 0 or label.item() == 1):
                    raise ValueError(
                        f"Binary ground truth for item {idx} must be 0 or 1. Got: {label.item()}"
                    )
                processed_label_tensor = torch.tensor([label.item()], dtype=torch.long)

            elif self.task == Task.MULTICLASS.value:
                if not isinstance(label, torch.Tensor):
                    label = torch.tensor(label)

                if not (label.ndim == 0 or (label.ndim == 1 and label.numel() == 1)):
                    raise ValueError(
                        f"Multiclass ground truth for item {idx} must be a scalar or 1-element tensor. Got shape: {label.shape}"
                    )
                label_value = label.item()
                if not (0 <= label_value < self.num_classes):
                    raise ValueError(
                        f"Multiclass ground truth for item {idx} must be between 0 and {self.num_classes - 1}. Got: {label_value}"
                    )
                processed_label_tensor = torch.tensor([label_value], dtype=torch.long)

            elif self.task == Task.MULTILABEL.value:
                if not isinstance(label, torch.Tensor):
                    try:
                        label = torch.tensor(label) 
                    except Exception as e:
                        raise ValueError(f"Could not convert label for item {idx} to tensor: {label}. Error: {e}")

                if label.dim() != 1:
                    raise ValueError(f"Multilabel ground truth for item {idx} must be a 1D tensor. Got {label.dim()} dimensions.")
                if label.shape[0] != self.num_classes:
                    raise ValueError(
                        f"Multilabel ground truth tensor shape for item {idx} must be ({self.num_classes},). Got {label.shape}."
                    )
                if not (label.dtype == torch.int or label.dtype == torch.long):
                    raise ValueError(
                        f"Multilabel ground truth tensor for item {idx} must be of integer type (torch.int or torch.long). Got {label.dtype}."
                    )
                if not torch.all((label == 0) | (label == 1)):
                    raise ValueError(
                        f"Multilabel ground truth tensor for item {idx} must be multi-hot encoded (contain only 0s and 1s). Got: {label}"
                    )
                processed_label_tensor = label.long()

            else:
                raise ValueError(f"Unsupported task for ClassificationGroundTruthStore: {self.task}")

            ann = Labels(datapoint_number=idx, labels=processed_label_tensor, scores=None) # scores=None for ground truth
            processed_annotations.add(ann)

        return processed_annotations 