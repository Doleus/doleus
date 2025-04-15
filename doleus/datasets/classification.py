from typing import Any, Dict, List

from torch.utils.data import Dataset

from doleus.datasets.base import Doleus
from doleus.utils.data import TaskType


class DoleusClassification(Doleus):
    """Dataset wrapper for classification tasks."""

    def __init__(
        self,
        dataset: Dataset,
        name: str,
        task: str,
        num_classes: int,
        label_to_name: Dict[int, str] = None,
        metadata: Dict[str, Any] = None,
        datapoints_metadata: List[Dict[str, Any]] = None,
    ):
        """Initialize a DoleusClassification dataset.

        Parameters
        ----------
        dataset : Dataset
            The PyTorch dataset to wrap.
        name : str
            Name of the dataset.
        task : str
            Specific classification task description.
        num_classes : int
            Number of classes in the dataset.
        label_to_name : Dict[int, str], optional
            Mapping from class IDs to class names, by default None.
        metadata : Dict[str, Any], optional
            Dataset-level metadata, by default None.
        datapoints_metadata : List[Dict[str, Any]], optional
            Per-datapoint metadata, by default None.
        """
        super().__init__(
            dataset=dataset,
            name=name,
            task_type=TaskType.CLASSIFICATION.value,
            task=task,
            label_to_name=label_to_name,
            metadata=metadata,
            datapoints_metadata=datapoints_metadata,
        )
        self.num_classes = num_classes
