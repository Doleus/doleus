"""Detection dataset classes for model evaluation and analysis."""

from typing import Any, Dict, List

from torch.utils.data import Dataset

from doleus.datasets.base import Doleus
from doleus.utils.data import TaskType


class DoleusDetection(Doleus):
    """Doleus dataset wrapper specialized for detection tasks."""

    def __init__(
        self,
        dataset: Dataset,
        name: str,
        num_classes: int = None,
        label_to_name: Dict[int, str] = None,
        metadata: Dict[str, Any] = None,
        datapoints_metadata: List[Dict[str, Any]] = None,
    ):
        """Initialize a DoleusDetection dataset.

        Parameters
        ----------
        dataset : Dataset
            The PyTorch dataset to wrap.
        name : str
            Name of the dataset.
        num_classes : int, optional
            Number of classes in the dataset, by default None.
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
            task_type=TaskType.DETECTION.value,
            num_classes=num_classes,
            label_to_name=label_to_name,
            metadata=metadata,
            datapoints_metadata=datapoints_metadata,
        )
