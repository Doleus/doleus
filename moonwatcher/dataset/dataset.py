"""Dataset classes and utilities for model evaluation and analysis."""

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, Subset
from tqdm import tqdm

from moonwatcher.annotations import (BoundingBoxes, GroundTruths, Labels,
                                     Predictions)
from moonwatcher.datapoint import Datapoint
from moonwatcher.dataset.metadata import ATTRIBUTE_FUNCTIONS
from moonwatcher.prediction_store import PredictionStore
from moonwatcher.utils.data import OPERATOR_DICT, TaskType
from moonwatcher.utils.helpers import get_current_timestamp


def find_root_dataset(dataset: Dataset) -> Dataset:
    """Find the root dataset by iteratively traversing dataset wrappers.

    Parameters
    ----------
    dataset : Dataset
        The dataset to find the root of, which may be wrapped in one or more
        dataset wrappers (e.g., Subset).

    Returns
    -------
    Dataset
        The root dataset that contains the actual data.
    """
    current = dataset
    while hasattr(current, "dataset"):
        current = current.dataset
    return current


def _get_raw_image(
    root_dataset: Dataset, index: int
) -> Union[Image.Image, np.ndarray, torch.Tensor]:
    """Retrieve the original image from a dataset bypassing its transforms.

    Parameters
    ----------
    root_dataset : Dataset
        The root dataset to get the image from.
    index : int
        The index of the image to retrieve.

    Returns
    -------
    Union[Image.Image, np.ndarray, torch.Tensor]
        The raw image in its original format, before any transforms are applied.
    """
    if not hasattr(root_dataset, "transform"):
        return root_dataset[index][0]

    original_transform = root_dataset.transform
    root_dataset.transform = None
    data = root_dataset[index]
    image = data[0]
    root_dataset.transform = original_transform
    return image


def _pil_or_numpy_to_tensor(
    image: Union[Image.Image, np.ndarray, torch.Tensor]
) -> torch.Tensor:
    """Convert numpy and PIL image formats to a standardized torch.Tensor format.

    Parameters
    ----------
    image : Union[Image.Image, np.ndarray, torch.Tensor]
        The input image in PIL, numpy array, or torch.Tensor format.

    Returns
    -------
    torch.Tensor
        A tensor of shape [C, H, W] with float values in [0,1].

    Raises
    ------
    TypeError
        If the input image format is not supported.
    """
    if isinstance(image, Image.Image):
        return transforms.ToTensor()(image)

    elif isinstance(image, np.ndarray):
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0
        return torch.from_numpy(image.transpose((2, 0, 1)))

    elif isinstance(image, torch.Tensor):
        if image.dim() == 3 and image.shape[-1] == 3:
            return image.permute(2, 0, 1)
        return image

    else:
        raise TypeError(f"Unsupported image type {type(image)}")


class Moonwatcher(Dataset):
    """Dataset wrapper for model evaluation and analysis.

    This class wraps a PyTorch dataset and provides functionality for:
    - Storing and managing ground truth annotations
    - Managing model predictions
    - Computing and storing metadata
    - Creating dataset slices based on various criteria
    - Evaluating model performance on different slices
    """

    def __init__(
        self,
        dataset: Dataset,
        name: str,
        task_type: str,
        task: Optional[str] = None,
        num_classes: int = None,
        label_to_name: Dict[int, str] = None,
        metadata: Dict[str, Any] = None,
        datapoints_metadata: List[Dict[str, Any]] = None,
    ):
        """Initialize a Moonwatcher dataset wrapper.

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
        num_classes : int, optional
            Number of classes in the dataset, by default None.
        label_to_name : Dict[int, str], optional
            Mapping from class IDs to class names, by default None.
        metadata : Dict[str, Any], optional
            Dataset-level metadata, by default None.
        datapoints_metadata : List[Dict[str, Any]], optional
            Per-datapoint metadata, by default None.
        """
        super().__init__()
        self.name = name
        self.dataset = dataset
        self.task_type = task_type
        self.task = task
        self.num_classes = num_classes
        self.label_to_name = label_to_name
        self.name_to_label = {}
        if label_to_name is not None:
            self.name_to_label = {v: k for k, v in label_to_name.items()}

        # Dataset-level metadata
        self.metadata = metadata if metadata is not None else {}
        self.metadata["_timestamp"] = get_current_timestamp()

        # Initialize datapoints with metadata
        self.datapoints = []
        for i in range(len(self.dataset)):
            md = {}
            if datapoints_metadata is not None and i < len(datapoints_metadata):
                md = datapoints_metadata[i]
            self.datapoints.append(Datapoint(id=i, metadata=md))

        self.groundtruths = GroundTruths(dataset=self)
        self.predictions = Predictions(dataset=self)
        self.add_groundtruths()
        self.prediction_store = PredictionStore()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]

    def __getattr__(self, attr):
        """
        Proxy any attribute lookup to the underlying dataset if not found here.
        """
        return getattr(self.dataset, attr)

    # -------------------------------------------------------------------------
    #                           GROUND TRUTHS
    # -------------------------------------------------------------------------

    def _get_root_index(self, local_idx: int) -> int:
        """Get the corresponding index in the root dataset for a slice index.

        Parameters
        ----------
        local_idx : int
            The local index in the current slice.

        Returns
        -------
        int
            The corresponding index in the root dataset.
        """
        if isinstance(self.dataset, Subset):
            return self.dataset.indices[local_idx]
        else:
            return local_idx

    def add_groundtruths(self):
        """Add ground truth annotations from the underlying dataset.

        This method loops over every item in the underlying dataset and converts
        the outputs into custom annotation objects (Labels or BoundingBoxes). For
        classification tasks, expects (image, label) tuples. For detection tasks,
        expects (image, bounding_boxes, labels) tuples.

        Raises
        ------
        ValueError
            If the dataset returns unexpected data format or has unsupported
            task type.
        """
        self.groundtruths = GroundTruths(dataset=self)

        for idx in tqdm(
            range(len(self.dataset)), desc=f"Building GROUND TRUTHS for {self.name}"
        ):
            data = self.dataset[idx]

            if self.task_type == TaskType.CLASSIFICATION.value:
                # Expect (image, label)
                if len(data) < 2:
                    raise ValueError(
                        "Expected (image, label) from dataset, got fewer elements."
                    )
                _, label = data

                # Convert label to tensor of shape [1] if needed
                if not isinstance(label, torch.Tensor):
                    label = torch.tensor(label)
                if label.dim() == 0:
                    label = label.unsqueeze(0)

                ann = Labels(datapoint_number=idx, labels=label)
                self.groundtruths.add(ann)

            elif self.task_type == TaskType.DETECTION.value:
                # Expect (image, bounding_boxes, labels)
                if len(data) != 3:
                    raise ValueError(
                        "Expected (image, bounding_boxes, labels) for detection."
                    )
                _, bounding_boxes, labels = data

                if not isinstance(bounding_boxes, torch.Tensor):
                    bounding_boxes = torch.tensor(bounding_boxes, dtype=torch.float32)
                if not isinstance(labels, torch.Tensor):
                    labels = torch.tensor(labels, dtype=torch.long)

                ann = BoundingBoxes(
                    datapoint_number=idx, boxes_xyxy=bounding_boxes, labels=labels
                )
                self.groundtruths.add(ann)

            else:
                raise ValueError(f"Unsupported task_type={self.task_type}.")

    # -------------------------------------------------------------------------
    #                           PREDICTIONS
    # -------------------------------------------------------------------------
    def _set_predictions(self, predictions: Any):
        """Add model predictions to the dataset.

        Parameters
        ----------
        predictions : Any
            For classification tasks:
                torch.Tensor of shape [N, num_classes] (logits/probabilities) or
                [N] (label indices).
            For detection tasks:
                List of length N, where each element is a dictionary containing
                'boxes' (M,4), 'labels' (M,), and 'scores' (M,) tensors.

        Raises
        ------
        TypeError
            If predictions are not in the expected format for the task.
        ValueError
            If predictions size doesn't match dataset size or has invalid shape.
        """
        self.predictions = Predictions(dataset=self)

        # Classification case: predictions is typically [N, num_classes]
        if self.task_type == TaskType.CLASSIFICATION.value:
            if not isinstance(predictions, torch.Tensor):
                raise TypeError(
                    "For classification, predictions must be a torch.Tensor."
                )

            num_samples = predictions.shape[0]
            if num_samples != len(self.dataset):
                raise ValueError(
                    "Mismatch between predictions size and dataset length."
                )

            # If shape is [N], assume these are predicted labels (class IDs)
            # If shape is [N, C], assume these are logits or probabilities
            if predictions.dim() == 1:
                # predicted labels
                for i in range(num_samples):
                    label_val = predictions[i].unsqueeze(0)
                    ann = Labels(datapoint_number=i, labels=label_val, scores=None)
                    self.predictions.add(ann)

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
                        scores=scores,  # shape [num_classes]
                    )
                    self.predictions.add(ann)

            else:
                raise ValueError("Classification predictions must be 1D or 2D tensor.")

        # Detection case: predictions is typically a list of length N
        elif self.task_type == TaskType.DETECTION.value:
            if not isinstance(predictions, list):
                raise TypeError(
                    "For detection, predictions must be a list of length N."
                )
            print(f"predictions: {predictions}")
            print(f"len(predictions): {len(predictions)}")
            print(f"self.dataset: {self.dataset}")
            print(f"len(self.dataset): {len(self.dataset)}")
            if len(predictions) != len(self.dataset):
                raise ValueError(
                    "Mismatch between predictions list and dataset length."
                )

            # Each element should look like {"boxes": (M,4), "labels": (M,), "scores": (M,)}
            for i, pred_dict in enumerate(
                tqdm(predictions, desc="Building DETECTION predictions")
            ):
                boxes_xyxy = torch.tensor(pred_dict["boxes"], dtype=torch.float32)
                labels = torch.tensor(pred_dict["labels"], dtype=torch.long)
                scores = torch.tensor(pred_dict["scores"], dtype=torch.float32)

                ann = BoundingBoxes(
                    datapoint_number=i,
                    boxes_xyxy=boxes_xyxy,
                    labels=labels,
                    scores=scores,
                )
                self.predictions.add(ann)

        else:
            raise ValueError(f"Unsupported task_type={self.task_type}.")

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
        # Store predictions in the prediction store
        self.prediction_store.add_predictions(
            predictions=predictions,
            dataset_id=self.name,
            model_id=model_id,
        )

    # -------------------------------------------------------------------------
    #                            METADATA METHODS
    # -------------------------------------------------------------------------
    def _prepare_image_for_metadata(self, index: int) -> np.ndarray:
        """Prepare an image for metadata computation.

        Parameters
        ----------
        index : int
            Index of the image in the dataset.

        Returns
        -------
        np.ndarray
            The image as a numpy array in BGR format.
        """
        root_dataset = find_root_dataset(self.dataset)
        raw_image = _get_raw_image(root_dataset, index)
        if isinstance(raw_image, torch.Tensor):
            raw_image = (raw_image.permute(1, 2, 0).cpu().numpy() * 255).astype(
                np.uint8
            )
        elif isinstance(raw_image, Image.Image):
            raw_image = np.array(raw_image)
        return raw_image

    def add_metadata(
        self, metadata_key: str, value_or_func: Union[Any, Callable[[np.ndarray], Any]]
    ):
        """Applies a custom function to an image to generate metadata and stores the computed metadata values for each datapoint.

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
                image = self._prepare_image_for_metadata(i)
                value = value_or_func(image)
            else:
                value = value_or_func

            self.datapoints[i].add_metadata(metadata_key, value)

    def add_metadata_from_list(self, metadata_list: List[Dict[str, Any]]):
        """Add metadata from a list of dictionaries.

        Parameters
        ----------
        metadata_list : List[Dict[str, Any]]
            List of metadata dictionaries, one per datapoint. Each dictionary
            can contain multiple key-value pairs.

        Notes
        -----
        If the metadata list is shorter than the dataset, only the first
        len(metadata_list) datapoints will receive metadata.
        """
        for i, md_dict in enumerate(
            tqdm(metadata_list, desc="Adding metadata from list")
        ):
            if i >= len(self.datapoints):
                break
            self.datapoints[i].metadata.update(md_dict)

    def add_predefined_metadata(self, keys: Union[str, List[str]]):
        """Add predefined metadata using functions from ATTRIBUTE_FUNCTIONS.

        Parameters
        ----------
        keys : Union[str, List[str]]
            Name(s) of predefined metadata function(s) to compute and add.
            Available keys are defined in ATTRIBUTE_FUNCTIONS.

        Raises
        ------
        ValueError
            If any key is not found in ATTRIBUTE_FUNCTIONS.
        """
        if isinstance(keys, str):
            keys = [keys]

        for key in keys:
            if key not in ATTRIBUTE_FUNCTIONS:
                raise ValueError(f"Unknown predefined metadata key: {key}")
            self.add_metadata(key, ATTRIBUTE_FUNCTIONS[key])

    # -------------------------------------------------------------------------
    #                                SLICING
    # -------------------------------------------------------------------------
    def _generate_filename(
        self, metadata_key: str, operator_str: str, value: Any
    ) -> str:
        """Generate a default filename for a slice based on its criteria.

        Parameters
        ----------
        metadata_key : str
            The metadata key used for slicing.
        operator_str : str
            The operator used for comparison.
        value : Any
            The threshold or target value.

        Returns
        -------
        str
            A generated filename for the slice.
        """
        abbreviations = {
            ">": "gt",
            "<": "lt",
            ">=": "ge",
            "<=": "le",
            "==": "eq",
            "class": "cl",
        }
        return (
            f"{self.name}_{metadata_key}_"
            f"{abbreviations.get(operator_str, operator_str)}_"
            f"{str(value).replace('.', '_')}"
        )

    def slice_by_threshold(
        self,
        metadata_key: str,
        operator_str: str,
        threshold: Any,
        slice_name: str = None,
    ):
        """Create a slice based on a threshold comparison of metadata values.

        Parameters
        ----------
        metadata_key : str
            The metadata key to compare.
        operator_str : str
            The comparison operator (">", "<", ">=", "<=", "==", "!=").
        threshold : Any
            The threshold value to compare against.
        slice_name : str, optional
            Name for the slice. If None, a name will be generated, by default None.

        Returns
        -------
        Slice
            A new slice containing datapoints that meet the threshold criteria.
        """
        op_func = OPERATOR_DICT[operator_str]
        indices = [
            i
            for i, dp in enumerate(self.datapoints)
            if op_func(dp.get_metadata(metadata_key), threshold)
        ]
        if slice_name is None:
            slice_name = self._generate_filename(metadata_key, operator_str, threshold)
        return Slice(name=slice_name, root_dataset=self, indices=indices)

    def slice_by_percentile(
        self,
        metadata_key: str,
        operator_str: str,
        percentile: float,
        slice_name: str = None,
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
        slice_name : str, optional
            Name for the slice. If None, a name will be generated, by default None.

        Returns
        -------
        Slice
            A new slice containing datapoints that meet the percentile criteria.
        """
        op_func = OPERATOR_DICT[operator_str]
        values = [dp.get_metadata(metadata_key) for dp in self.datapoints]
        threshold = np.percentile(values, percentile)
        indices = [
            i
            for i, dp in enumerate(self.datapoints)
            if op_func(dp.get_metadata(metadata_key), threshold)
        ]
        if slice_name is None:
            slice_name = self._generate_filename(metadata_key, operator_str, percentile)
        return Slice(name=slice_name, root_dataset=self, indices=indices)

    def slice_by_metadata_value(
        self,
        metadata_key: str,
        target_value: Any,
        slice_name: Optional[str] = None,
        tolerance: float = 1e-6,
    ) -> "Slice":
        """Create a slice containing datapoints with a specific metadata value.

        Parameters
        ----------
        metadata_key : str
            The metadata key to match.
        target_value : Any
            The value to match against.
        slice_name : Optional[str], optional
            Name for the slice. If None, a name will be generated, by default None.
        tolerance : float, optional
            Tolerance for floating point comparisons, by default 1e-6.

        Returns
        -------
        Slice
            A new slice containing datapoints that match the target value.

        Raises
        ------
        KeyError
            If metadata_key doesn't exist in any datapoint.
        ValueError
            If no datapoints match the target value.
        """

        if not any(metadata_key in dp.metadata for dp in self.datapoints):
            raise KeyError(f"Metadata key '{metadata_key}' not found in any datapoint")

        indices = []
        for i, dp in enumerate(self.datapoints):
            try:
                value = dp.get_metadata(metadata_key)

                # Handle different types of comparisons
                if isinstance(target_value, float) and isinstance(value, (int, float)):
                    # Use tolerance for float comparisons
                    if abs(value - target_value) <= tolerance:
                        indices.append(i)
                elif isinstance(target_value, np.ndarray) and isinstance(
                    value, np.ndarray
                ):
                    # Handle numpy array comparison
                    if np.array_equal(value, target_value):
                        indices.append(i)
                else:
                    # Direct comparison for other types
                    if value == target_value:
                        indices.append(i)
            except KeyError:
                # Skip datapoints that don't have this metadata key
                continue

        if not indices:
            raise ValueError(f"No datapoints found with {metadata_key}={target_value}")

        if not slice_name:
            # Create a safe slice name
            value_str = str(target_value)
            if len(value_str) > 50:
                value_str = value_str[:47] + "..."
            # Replace problematic characters
            value_str = "".join(c if c.isalnum() else "_" for c in value_str)
            slice_name = f"{metadata_key}_{value_str}"

        return Slice(name=slice_name, root_dataset=self, indices=indices)

    def slice_by_groundtruth_class(
        self,
        class_names: Optional[List[str]] = None,
        class_ids: Optional[List[int]] = None,
        slice_name: Optional[str] = None,
    ) -> "Slice":
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

        Raises
        ------
        ValueError
            If neither class_names nor class_ids are provided, or if class names
            are invalid.
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

        # Remove duplicates and convert to set for faster lookups
        class_id_set = set(class_ids) if class_ids else set()

        # Collect matching original indices
        filtered_indices = []
        for datapoint in self.datapoints:
            original_idx = datapoint.id

            try:
                ground_truth = self.groundtruths.get(original_idx)
            except KeyError:
                continue  # Skip datapoints without ground truth

            # Unified handling for both annotation types
            if isinstance(ground_truth, (Labels, BoundingBoxes)):
                # Check if any label matches our target classes
                # Works for:
                # - Classification (single/multi-label): labels tensor shape [N]
                # - Detection: labels tensor shape [M] (per-box labels)
                if torch.any(
                    torch.isin(ground_truth.labels, torch.tensor(list(class_id_set)))
                ):
                    filtered_indices.append(original_idx)

        # Generate default name if needed
        if not slice_name:
            target_classes = class_names if class_names else sorted(class_id_set)
            class_str = "_".join(map(str, target_classes))[:50]  # Limit length
            slice_name = f"gt_class_{class_str}"

        return Slice(name=slice_name, root_dataset=self, indices=filtered_indices)


class Slice(Moonwatcher):
    """A subset of a Moonwatcher dataset containing only selected datapoints.

    A Slice maintains a reference to its parent Moonwatcher dataset and provides
    access to a subset of its datapoints. It inherits all functionality from
    the parent dataset while operating only on the selected subset.
    """

    def __init__(
        self,
        name: str,
        root_dataset: Moonwatcher,
        indices: List[int],
    ):
        """Initialize a Slice instance.

        Parameters
        ----------
        name : str
            Name of the slice.
        root_dataset : Moonwatcher
            The parent dataset this slice is created from.
        indices : List[int]
            List of indices from the parent dataset to include in this slice.
        """
        self.name = name
        self.root_dataset = root_dataset
        self.indices = indices
        self.datapoints = [root_dataset.datapoints[i] for i in indices]

    def __len__(self) -> int:
        """Get the number of datapoints in the slice.

        Returns
        -------
        int
            Number of datapoints in the slice.
        """
        return len(self.indices)

    def __getitem__(self, idx: int):
        """Get a datapoint from the slice by index.

        Parameters
        ----------
        idx : int
            Index in the slice.

        Returns
        -------
        Any
            The datapoint from the parent dataset corresponding to the slice index.
        """
        # Map the local slice index to the corresponding index in the root dataset.
        root_idx = self.indices[idx]
        return self.root_dataset.dataset[root_idx]

    def __getattr__(self, attr: str):
        """Get an attribute from the parent dataset if not found in the slice.

        Parameters
        ----------
        attr : str
            Name of the attribute to get.

        Returns
        -------
        Any
            The attribute value from the parent dataset.
        """
        return getattr(self.root_dataset, attr)


class MoonwatcherClassification(Moonwatcher):
    """Moonwatcher dataset wrapper specialized for classification tasks."""

    def __init__(
        self,
        dataset: Dataset,
        name: str,
        task: str,
        num_classes: int = None,
        label_to_name: Dict[int, str] = None,
        metadata: Dict[str, Any] = None,
        datapoints_metadata: List[Dict[str, Any]] = None,
    ):
        """Initialize a MoonwatcherClassification dataset.

        Parameters
        ----------
        dataset : Dataset
            The PyTorch dataset to wrap.
        name : str
            Name of the dataset.
        task : str
            Specific classification task description.
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
            task_type=TaskType.CLASSIFICATION.value,
            task=task,
            num_classes=num_classes,
            label_to_name=label_to_name,
            metadata=metadata,
            datapoints_metadata=datapoints_metadata,
        )


class MoonwatcherDetection(Moonwatcher):
    """Moonwatcher dataset wrapper specialized for detection tasks."""

    def __init__(
        self,
        dataset: Dataset,
        name: str,
        num_classes: int = None,
        label_to_name: Dict[int, str] = None,
        metadata: Dict[str, Any] = None,
        datapoints_metadata: List[Dict[str, Any]] = None,
    ):
        """Initialize a MoonwatcherDetection dataset.

        Parameters
        ----------
        dataset : Dataset
            The PyTorch dataset to wrap.
        name : str
            Name of the dataset.
        task : str
            Specific detection task description.
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


def get_original_indices(dataset: Union[Moonwatcher, Slice]) -> List[int]:
    """Get the original dataset indices for a dataset or slice.

    Parameters
    ----------
    dataset : Union[Moonwatcher, Slice]
        The dataset or slice to get indices for.

    Returns
    -------
    List[int]
        List of indices in the original dataset.

    Raises
    ------
    TypeError
        If the dataset is not a Moonwatcher or Slice instance.
    """
    if isinstance(dataset, Slice):
        parent_indices = get_original_indices(dataset.root_dataset)
        return [parent_indices[i] for i in dataset.indices]
    elif isinstance(dataset, Moonwatcher):
        return list(range(len(dataset.dataset)))
    else:
        raise TypeError("Unsupported dataset type for get_original_indices.")
