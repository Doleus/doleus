from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from tqdm import tqdm

from doleus.annotations.base import Annotations
from doleus.annotations.classification import Labels
from doleus.annotations.detection import BoundingBoxes
from doleus.storage.metadata_store import MetadataStore
from doleus.storage.prediction_store import PredictionStore
from doleus.utils.data import OPERATOR_DICT, TaskType
from doleus.utils.image_metadata import ATTRIBUTE_FUNCTIONS
from doleus.utils.utils import (find_root_dataset, get_current_timestamp,
                                get_raw_image)


class Doleus(Dataset):
    """Dataset wrapper for Doleus.

    This class provides functionality for:
    - Storing the dataset
    - Storing model predictions
    - Computing and storing metadata
    - Creating dataset slices based on various criteria
    """

    def __init__(
        self,
        dataset: Dataset,
        name: str,
        task_type: str,
        task: Optional[str] = None,
        label_to_name: Dict[int, str] = None,
        metadata: Dict[str, Any] = None,
        datapoints_metadata: List[Dict[str, Any]] = None,
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
        datapoints_metadata : List[Dict[str, Any]], optional
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

        # Initialize datapoints with metadata
        self.datapoints = []
        for i in range(len(self.dataset)):
            md = {}
            if datapoints_metadata is not None and i < len(datapoints_metadata):
                md = datapoints_metadata[i]
            self.datapoints.append(MetadataStore(id=i, metadata=md))

        self.groundtruths = Annotations()
        self.predictions = Annotations()
        self.add_groundtruths()
        self.prediction_store = PredictionStore()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]

    def __getattr__(self, attr):
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
        the outputs into custom annotation objects (Labels or BoundingBoxes).
        """
        self.groundtruths = Annotations()
        for idx in range(len(self.dataset)):
            data = self.dataset[idx]

            if self.task_type == TaskType.CLASSIFICATION.value:
                if len(data) < 2:
                    raise ValueError(
                        "Expected (image, label) from dataset, got fewer elements."
                    )
                _, labels = data #TODO: Rename to ensure it becomes apparent that this could be one or more labels

                # Convert label to tensor of shape [1] if needed
                if not isinstance(labels, torch.Tensor):
                    labels = torch.tensor(labels)
                if labels.dim() == 0:
                    labels = labels.unsqueeze(0)

                ann = Labels(datapoint_number=idx, labels=labels)
                self.groundtruths.add(ann)

            elif self.task_type == TaskType.DETECTION.value:
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
        """
        self.predictions = Annotations()

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
        self.prediction_store.add_predictions(
            predictions=predictions,
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
        raw_image = get_raw_image(root_dataset, index)
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
        if len(df) > len(self.datapoints):
            raise ValueError(
                f"DataFrame has {len(df)} rows but dataset only has {len(self.datapoints)} datapoints"
            )

        for idx, row in enumerate(df.itertuples(index=False)):
            metadata_dict = {col: val for col, val in zip(df.columns, row)}
            self.datapoints[idx].metadata.update(metadata_dict)

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

        # TODO: avoid circular import without this workaround
        from doleus.datasets.slice import Slice

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

        # TODO: avoid circular import without this workaround
        from doleus.datasets.slice import Slice

        return Slice(name=slice_name, root_dataset=self, indices=indices)

    def slice_by_metadata_value(
        self,
        metadata_key: str,
        target_value: Any,
        slice_name: Optional[str] = None,
        tolerance: float = 1e-6,
    ):
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

        # TODO: avoid circular import without this workaround
        from doleus.datasets.slice import Slice

        return Slice(name=slice_name, root_dataset=self, indices=indices)

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

        # Remove duplicates and convert to set for faster lookups
        class_id_set = set(class_ids) if class_ids else set()

        # Collect matching original indices
        filtered_indices = []
        for datapoint in self.datapoints:
            original_idx = datapoint.id
            ground_truth = self.groundtruths.get(original_idx)

            if isinstance(ground_truth, (Labels, BoundingBoxes)):
                if torch.any(
                    torch.isin(ground_truth.labels, torch.tensor(list(class_id_set)))
                ):
                    filtered_indices.append(original_idx)

        # Generate default name if needed
        if not slice_name:
            target_classes = class_names if class_names else sorted(class_id_set)
            class_str = "_".join(map(str, target_classes))[:50]  # Limit length
            slice_name = f"gt_class_{class_str}"

        # TODO: avoid circular import without this workaround
        from doleus.datasets.slice import Slice

        return Slice(name=slice_name, root_dataset=self, indices=filtered_indices)


def get_original_indices(dataset: Union["Doleus", "Slice"]) -> List[int]:
    """Get the original dataset indices for a dataset or slice.

    Parameters
    ----------
    dataset : Union[Doleus, Slice]
        The dataset or slice to get indices for.

    Returns
    -------
    List[int]
        List of indices in the original dataset.

    Raises
    ------
    TypeError
        If the dataset is not a Doleus or Slice instance.
    """
    # TODO: avoid circular import without this workaround
    from doleus.datasets.slice import Slice

    if isinstance(dataset, Slice):
        parent_indices = get_original_indices(dataset.root_dataset)
        return [parent_indices[i] for i in dataset.indices]
    elif isinstance(dataset, Doleus):
        return list(range(len(dataset.dataset)))
    else:
        raise TypeError("Unsupported dataset type for get_original_indices.")
