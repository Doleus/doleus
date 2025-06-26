# SPDX-FileCopyrightText: 2025 Doleus contributors
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from doleus.annotations import BoundingBoxes, Labels
from doleus.storage import (
    ClassificationGroundTruthStore,
    ClassificationPredictionStore,
    DetectionGroundTruthStore,
    DetectionPredictionStore,
    MetadataStore,
)
from doleus.utils import (
    ATTRIBUTE_FUNCTIONS,
    OPERATOR_DICT,
    TaskType,
    create_filename,
    get_current_timestamp,
    to_numpy_image,
)


class Doleus(Dataset, ABC):
    """Dataset wrapper for Doleus.

    This class provides functionality for:
    - Storing the dataset
    - Storing predictions from multiple models for the same dataset
    - Storing metadata
    - Slicing the dataset based on various criteria
    """

    def __init__(
        self,
        dataset: Dataset,
        name: str,
        task_type: str,
        task: Optional[str] = None,
        label_to_name: Dict[int, str] = None,
        metadata: Dict[str, Any] = None,
        per_datapoint_metadata: List[Dict[str, Any]] = None,
    ):
        """Initialize a dataset wrapper.

        Parameters
        ----------
        dataset : Dataset
            The PyTorch dataset to wrap.
        name : str
            Name of the dataset.
        task_type : str
            Type of task (e.g., "classification", "detection").
        task : str
            Specific task description.
        label_to_name : Dict[int, str], optional
            Mapping from class IDs to class names, by default None.
        metadata : Dict[str, Any], optional
            Dataset-level metadata, by default None.
        per_datapoint_metadata : List[Dict[str, Any]], optional
            Per-datapoint metadata, by default None.
        """
        super().__init__()
        self.name = name
        self.dataset = dataset
        self.task_type = task_type
        self.task = task
        self.label_to_name = label_to_name
        self.name_to_label = {}
        if label_to_name is not None:
            self.name_to_label = {v: k for k, v in label_to_name.items()}

        # Dataset-level metadata
        self.metadata = metadata if metadata is not None else {}
        self.metadata["_timestamp"] = get_current_timestamp()

        # Specific instantiations will be handled by subclasses (DoleusClassification, DoleusDetection).
        self.groundtruth_store: Optional[
            Union[ClassificationGroundTruthStore, DetectionGroundTruthStore]
        ] = None
        self.prediction_store: Optional[
            Union[ClassificationPredictionStore, DetectionPredictionStore]
        ] = None

        self.metadata_store = MetadataStore(
            num_datapoints=len(dataset), metadata=per_datapoint_metadata
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]

    def __getattr__(self, attr):
        return getattr(self.dataset, attr)

    @abstractmethod
    def _create_new_instance(self, dataset, indices, slice_name):
        pass

    def add_model_predictions(
        self,
        predictions: Union[torch.Tensor, List[Dict[str, Any]]],
        model_id: str,
    ) -> None:
        """Add model predictions to the dataset.

        Parameters
        ----------
        predictions : Union[torch.Tensor, List[Dict[str, Any]]]
            Model predictions to store. For classification, this should be a tensor
            of shape [N, C] where N is the number of samples and C is the number
            of classes. For detection, this should be a list of dictionaries with
            'boxes', 'labels', and 'scores' keys.
        model_id : str
            Name of the model that generated these predictions
        """
        kwargs = {}
        if self.task_type == TaskType.CLASSIFICATION.value:
            kwargs["task"] = self.task
            # Ensure predictions is a Tensor for classification
            if not isinstance(predictions, torch.Tensor):
                raise TypeError(
                    "For classification tasks, predictions must be a torch.Tensor."
                )
        elif self.task_type == TaskType.DETECTION.value:
            # Ensure predictions is a List[Dict] for detection
            if not isinstance(predictions, list) or not all(
                isinstance(p, dict) for p in predictions
            ):
                raise TypeError(
                    "For detection tasks, predictions must be a list of dictionaries."
                )

        self.prediction_store.add_predictions(
            predictions=predictions,
            model_id=model_id,
            **kwargs,
        )

    # -------------------------------------------------------------------------
    #                            METADATA FUNCTIONS
    # -------------------------------------------------------------------------
    def add_metadata(
        self, metadata_key: str, value_or_func: Union[Any, Callable[[np.ndarray], Any]]
    ):
        """Add metadata to all datapoints in the dataset.

        Parameters
        ----------
        metadata_key : str
            Key under which to store the metadata.
        value_or_func : Union[Any, Callable[[np.ndarray], Any]]
            Either a static value to store for all datapoints, or a function
            that takes a numpy image array and returns a computed value.
        """
        is_func = callable(value_or_func)

        for i in tqdm(
            range(len(self.dataset)), desc=f"Adding metadata '{metadata_key}'"
        ):
            if is_func:
                image = to_numpy_image(self.dataset, i)
                value = value_or_func(image)
            else:
                value = value_or_func

            self.metadata_store.add_metadata(i, metadata_key, value)

    def add_metadata_from_list(self, metadata_list: List[Dict[str, Any]]):
        """Add metadata from a list of dictionaries.

        Parameters
        ----------
        metadata_list : List[Dict[str, Any]]
            List of metadata dictionaries, one per datapoint. Each dictionary
            can contain multiple key-value pairs.

        Raises
        ------
        ValueError
            If the metadata list length does not match the dataset length.
        """
        if len(metadata_list) != len(self.dataset):
            raise ValueError(
                f"Metadata list has {len(metadata_list)} entries but dataset has {len(self.dataset)} datapoints."
            )
        for i, md_dict in enumerate(
            tqdm(metadata_list, desc="Adding metadata from list")
        ):
            for key, value in md_dict.items():
                self.metadata_store.add_metadata(i, key, value)

    def add_predefined_metadata(self, keys: Union[str, List[str]]):
        """Add predefined metadata using functions from ATTRIBUTE_FUNCTIONS.

        Parameters
        ----------
        keys : Union[str, List[str]]
            Name(s) of predefined metadata function(s) to compute and add.
            Available keys are defined in ATTRIBUTE_FUNCTIONS.

        Raises
        ------
        KeyError
            If any key is not found in ATTRIBUTE_FUNCTIONS.
        """
        if isinstance(keys, str):
            keys = [keys]

        for key in keys:
            if key not in ATTRIBUTE_FUNCTIONS:
                raise KeyError(f"Unknown predefined metadata key: {key}")
            self.add_metadata(key, ATTRIBUTE_FUNCTIONS[key])

    def add_metadata_from_dataframe(self, df):
        """Add metadata from a pandas DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing metadata. Each row corresponds to a datapoint in order
            (first row = first datapoint, etc), and each column becomes a metadata key.

        Raises
        ------
        ValueError
            If DataFrame has more rows than dataset has datapoints.
        """
        if len(df) > len(self.dataset):
            raise ValueError(
                f"DataFrame has {len(df)} rows but dataset only has {len(self.dataset)} datapoints"
            )

        for idx, row in enumerate(df.itertuples(index=False)):
            for col, val in zip(df.columns, row):
                self.metadata_store.add_metadata(idx, col, val)

    # -------------------------------------------------------------------------
    #                                SLICING
    # -------------------------------------------------------------------------
    def slice_by_value(
        self,
        metadata_key: str,
        operator_str: str,
        value: Any,
        slice_name: Optional[str] = None,
    ):
        """Create a slice based on a value comparison of metadata values.

        Parameters
        ----------
        metadata_key : str
            The metadata key to compare.
        operator_str : str
            The comparison operator (">", "<", ">=", "<=", "==", "!=").
        value : Any
            The value to compare against.
        slice_name : str, optional
            Name for the slice. If None, a name will be generated, by default None.

        Returns
        -------
        Slice
            A new slice containing datapoints that meet the value criteria.
        """
        op_func = OPERATOR_DICT[operator_str]
        indices = [
            i
            for i in range(len(self.dataset))
            if op_func(self.metadata_store.get_metadata(i, metadata_key), value)
        ]
        if slice_name is None:
            slice_name = create_filename(self.name, metadata_key, operator_str, value)

        return self._create_new_instance(self.dataset, indices, slice_name)

    def slice_by_percentile(
        self,
        metadata_key: str,
        operator_str: str,
        percentile: float,
        slice_name: Optional[str] = None,
    ):
        """Create a slice based on a percentile threshold of metadata values.

        Parameters
        ----------
        metadata_key : str
            The metadata key to compare.
        operator_str : str
            The comparison operator (">", "<", ">=", "<=", "==", "!=").
        percentile : float
            The percentile value (0-100) to use as threshold.
        slice_name : Optional[str], optional
            Name for the slice. If None, a name will be generated, by default None.

        Returns
        -------
        Slice
            A new slice containing datapoints that meet the percentile criteria.
        """
        op_func = OPERATOR_DICT[operator_str]
        values = [
            self.metadata_store.get_metadata(i, metadata_key)
            for i in range(len(self.dataset))
        ]
        threshold = np.percentile(values, percentile)
        indices = [
            i
            for i in range(len(self.dataset))
            if op_func(self.metadata_store.get_metadata(i, metadata_key), threshold)
        ]
        if slice_name is None:
            slice_name = create_filename(
                self.name, metadata_key, operator_str, percentile
            )

        return self._create_new_instance(self.dataset, indices, slice_name)

    # TODO: We should add a test for this
    def slice_by_groundtruth_class(
        self,
        class_names: Optional[List[str]] = None,
        class_ids: Optional[List[int]] = None,
        slice_name: Optional[str] = None,
    ):
        """Create a slice containing datapoints with specific ground truth classes.

        Parameters
        ----------
        class_names : Optional[List[str]], optional
            List of class names to include. Requires label_to_name mapping,
            by default None.
        class_ids : Optional[List[int]], optional
            List of class IDs to include, by default None.
        slice_name : Optional[str], optional
            Name for the slice. If None, a name will be generated, by default None.

        Returns
        -------
        Slice
            A new slice containing datapoints with the specified classes.
        """
        if not class_names and not class_ids:
            raise ValueError("Must provide either class_names or class_ids")

        # Convert class names to IDs if provided
        if class_names:
            if not self.label_to_name:
                raise ValueError("Class names require label_to_name mapping")

            # Validate all names exist in label mapping
            valid_names = set(self.label_to_name.values())
            invalid_names = set(class_names) - valid_names
            if invalid_names:
                raise ValueError(f"Invalid class names: {sorted(invalid_names)}")

            # Convert to unique class IDs
            class_ids = [
                class_id
                for class_id, name in self.label_to_name.items()
                if name in class_names
            ]

        # Remove duplicates
        class_id_set = set(class_ids) if class_ids else set()

        # Collect matching original indices
        filtered_indices = []
        for i in range(len(self.dataset)):
            ground_truth = self.groundtruth_store.get(i)

            if isinstance(ground_truth, (Labels, BoundingBoxes)):
                if torch.any(
                    torch.isin(ground_truth.labels, torch.tensor(list(class_id_set)))
                ):
                    filtered_indices.append(i)

        # Generate default name if needed
        if not slice_name:
            target_classes = class_names if class_names else sorted(class_id_set)
            class_str = "_".join(map(str, target_classes))
            slice_name = create_filename(self.name, "class", "==", class_str)

        return self._create_new_instance(self.dataset, filtered_indices, slice_name)
