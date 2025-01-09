# dataset.py

from typing import List, Dict, Any, Callable, Union, Optional
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset

from moonwatcher.datapoint import Datapoint
from moonwatcher.utils.data import OPERATOR_DICT, TaskType
from moonwatcher.dataset.metadata import ATTRIBUTE_FUNCTIONS
from moonwatcher.annotations import GroundTruths, Labels, BoundingBoxes
from moonwatcher.utils.helpers import get_current_timestamp


def find_root_dataset(dataset: Dataset) -> Dataset:
    """
    Recursively find the original underlying dataset in case the user
    has wrapped the dataset in multiple Subset objects, etc.
    """
    if hasattr(dataset, "dataset"):
        return find_root_dataset(dataset.dataset)
    return dataset


def _get_raw_image(
    root_dataset: Dataset,
    index: int
) -> Union[Image.Image, np.ndarray, torch.Tensor]:
    """
    Retrieve the original image from `root_dataset` bypassing its transforms.

    This function assumes the dataset has an attribute `transform`.
    We temporarily set it to None (if it exists), fetch the image,
    then restore it to preserve user transformations for future calls.
    """
    if not hasattr(root_dataset, "transform"):
        # If there's no .transform attribute, nothing to toggle
        return root_dataset[index][0]

    # Temporarily store transform, set to None
    original_transform = root_dataset.transform
    root_dataset.transform = None
    data = root_dataset[index]
    image = data[0]
    # Restore transform
    root_dataset.transform = original_transform

    return image


def _pil_or_numpy_to_tensor(image: Union[Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
    """
    Converts a PIL Image, numpy array, or torch.Tensor into a standard
    torch.Tensor of shape [C, H, W], with float values in [0,1].
    """
    if isinstance(image, Image.Image):
        image = transforms.ToTensor()(image)  # yields float in [0,1]
    elif isinstance(image, np.ndarray):
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0
        # shape HWC -> CHW
        image = torch.from_numpy(image.transpose((2, 0, 1)))
    elif isinstance(image, torch.Tensor):
        # If shape is (H, W, C), permute
        if image.dim() == 3 and image.shape[-1] == 3:
            image = image.permute(2, 0, 1)
        # If shape is already (C, H, W), no change
    else:
        raise TypeError(
            f"Unsupported image type {type(image)} for metadata extraction")
    return image


class Moonwatcher(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        name: str,
        task_type: str,
        task: str,
        num_classes: int = None,
        label_to_name: Dict[int, str] = None,
        metadata: Dict[str, Any] = None,
        description: str = None,
        datapoints_metadata: List[Dict[str, Any]] = None,
    ):
        """
        Creates a Moonwatcher dataset wrapper around an existing dataset.

        :param dataset: the underlying dataset (e.g., PyTorch's ImageFolder)
        :param name: the name of this dataset
        :param task_type: either 'classification' or 'detection'
        :param task: either 'binary', 'multiclass', or 'multilabel'
        :param num_classes: number of classes (for classification tasks)
        :param label_to_name: dictionary mapping label ids to label names
        :param metadata: dictionary of metadata tags for the dataset
        :param description: dataset description
        :param datapoints_metadata: pre-supplied metadata for each datapoint
        """
        super().__init__()
        self.name = name
        self.dataset = dataset
        self.label_to_name = label_to_name
        self.task_type = task_type
        self.task = task
        self.metadata = metadata if metadata is not None else {}
        self.metadata["_timestamp"] = get_current_timestamp()
        self.description = description
        self.num_classes = num_classes

        # Build reverse mapping name->label for faster lookups
        self.name_to_label = {}
        if label_to_name is not None:
            self.name_to_label = {v: k for k, v in label_to_name.items()}

        self.datapoints_metadata = datapoints_metadata
        self.datapoints = []
        for i in range(len(self.dataset)):
            md = datapoints_metadata[i] if (
                datapoints_metadata is not None and i < len(datapoints_metadata)) else {}
            self.datapoints.append(Datapoint(id=i, metadata=md))

        # Build groundtruth annotations
        self.groundtruths = GroundTruths(self)
        self._initialize_groundtruths()

    def _initialize_groundtruths(self):
        """
        Load and store groundtruth annotations for every item in the dataset.
        This handles both classification and detection tasks.
        """
        for index in tqdm(range(len(self.dataset)), desc=f"Saving annotations of dataset {self.name}."):
            data = self.dataset[index]
            if self.task_type == TaskType.CLASSIFICATION.value:
                try:
                    image, label = data
                except ValueError as e:
                    raise ValueError(
                        f"Dataset should return two elements (image, label). Error: {e}"
                    )
                # Ensure label is torch.Tensor
                if not isinstance(label, torch.Tensor):
                    try:
                        label = torch.tensor(label)
                    except Exception as ex:
                        raise TypeError(
                            f"Unable to convert label to torch.Tensor: {ex}")

                # Convert label to (1,) if scalar
                if label.dim() == 0:
                    label = label.unsqueeze(0)

                groundtruth = Labels(datapoint_number=index, labels=label)

            elif self.task_type == TaskType.DETECTION.value:
                try:
                    image, bounding_boxes, labels = data
                except ValueError as e:
                    raise ValueError(
                        f"Dataset should return (image, bounding_boxes, labels). Error: {e}"
                    )
                groundtruth = BoundingBoxes(
                    datapoint_number=index,
                    boxes_xyxy=bounding_boxes,
                    labels=labels
                )
            else:
                raise ValueError(
                    f"Unsupported task type: {self.task_type}. Must be 'classification' or 'detection'.")
            self.groundtruths.add(groundtruth)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]

    def __getattr__(self, attr):
        """
        Proxy any attribute lookup to the underlying dataset.
        This allows the user to directly call self.dataset methods if needed.
        """
        return getattr(self.dataset, attr)

    # -------------------------------------------------------------------------
    #                            METADATA METHODS
    # -------------------------------------------------------------------------

    def add_predefined_metadata(self, predefined_metadata_keys: Union[str, List[str]]):
        """
        Adds one or multiple predefined metadata keys (e.g. brightness, contrast, etc.)
        in a single pass to avoid re-loading images repeatedly.

        :param predefined_metadata_keys: either a single string or a list of strings
                                         from ATTRIBUTE_FUNCTIONS
        """
        # Normalize to a list
        if isinstance(predefined_metadata_keys, str):
            predefined_metadata_keys = [predefined_metadata_keys]

        # We'll do a single loop over the dataset, computing each requested metadata
        # to reduce overhead
        root_dataset = find_root_dataset(self.dataset)

        for i in tqdm(range(len(self.dataset)), desc=f"Adding metadata to {self.name}"):
            # Retrieve raw image (bypassing transforms)
            raw_image = _get_raw_image(root_dataset, i)
            # Convert to tensor -> numpy for attribute functions
            image_tensor = _pil_or_numpy_to_tensor(raw_image)
            # Now shape [C, H, W], scale [0,1]
            image_np = (image_tensor.permute(
                1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            # For each metadata key in the list, compute & store if not present
            for key in predefined_metadata_keys:
                if key not in self.datapoints[i].metadata:
                    val = ATTRIBUTE_FUNCTIONS[key](image_np)
                    self.datapoints[i].add_metadata(key=key, value=val)

    def add_metadata_from_list(self, metadata_list: List[Dict[str, Any]]):
        """
        Add metadata for all data points from an external list of dicts.

        :param metadata_list: each element is a dict of {metadata_key: metadata_value}
        """
        for i, md_dict in enumerate(tqdm(metadata_list, desc="Adding metadata from list")):
            if i < len(self.datapoints):
                for key, value in md_dict.items():
                    self.datapoints[i].add_metadata(key=key, value=value)

    def add_metadata_from_groundtruths(self, class_name: str):
        """
        Add the number of occurrences of a specific class (by name) in each picture as metadata.

        :param class_name: class name to count
        """
        # Check if we already built name_to_label
        if not self.name_to_label:
            # If user never provided label_to_name, fallback to naive approach
            found_key = None
            if self.label_to_name is not None:
                for k, v in self.label_to_name.items():
                    if v == class_name:
                        found_key = k
                        break
                if found_key is None:
                    raise ValueError(
                        f"Class name '{class_name}' not found in label_to_name.")
                class_id = found_key
            else:
                raise ValueError(
                    "No label_to_name mapping provided, cannot match class name.")
        else:
            if class_name not in self.name_to_label:
                raise ValueError(
                    f"Class name '{class_name}' not found in label_to_name mapping.")
            class_id = self.name_to_label[class_name]

        for datapoint in tqdm(self.datapoints, desc=f"Adding metadata for class '{class_name}'"):
            i = datapoint.id
            groundtruth = self.groundtruths.get(i)

            if self.task_type == TaskType.DETECTION.value:
                if isinstance(groundtruth, BoundingBoxes):
                    count = (groundtruth.labels == class_id).sum().item()
                    datapoint.add_metadata(key=class_name, value=count)
            elif self.task_type == TaskType.CLASSIFICATION.value:
                if isinstance(groundtruth, Labels):
                    count = (groundtruth.labels == class_id).sum().item()
                    datapoint.add_metadata(key=class_name, value=count)

    def add_metadata_custom(self, metadata_key: str, metadata_func: Callable):
        """
        Add metadata for all datapoints using a custom function. The function
        must accept a NumPy image and return a single value. For example:

            def my_sharpness_metric(np_image):
                # compute something
                return sharpness_val

            dataset.add_metadata_custom('sharpness', my_sharpness_metric)

        :param metadata_key: name to store the computed metadata under
        :param metadata_func: a function that receives (H,W,C) uint8 array and returns a value
        """
        root_dataset = find_root_dataset(self.dataset)

        for i in tqdm(range(len(self.dataset)), desc=f"Adding custom metadata '{metadata_key}'"):
            raw_image = _get_raw_image(root_dataset, i)
            image_tensor = _pil_or_numpy_to_tensor(raw_image)
            image_np = (image_tensor.permute(
                1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            if metadata_key not in self.datapoints[i].metadata:
                val = metadata_func(image_np)
                self.datapoints[i].add_metadata(key=metadata_key, value=val)

    # -------------------------------------------------------------------------
    #                                SLICING
    # -------------------------------------------------------------------------

    def _generate_filename(self, metadata_key: str, operator_str: str, value: Any):
        abbreviations = {
            ">": "gt",
            "<": "lt",
            ">=": "ge",
            "<=": "le",
            "==": "eq",
            "class": "cl",
        }
        filename = f"{self.name}_{metadata_key}_{abbreviations.get(operator_str, operator_str)}_{str(value).replace('.', '_')}"
        return filename

    def slice_by_threshold(self, metadata_key: str, operator_str: str, threshold: Any, slice_name: str = None):
        """
        Create a slice by thresholding on a particular metadatum. E.g. ("brightness", "<", 0.1).
        """
        op_func = OPERATOR_DICT[operator_str]
        indices = [
            i for i, dp in enumerate(self.datapoints)
            if op_func(dp.get_metadata(metadata_key), threshold)
        ]

        if slice_name is None:
            slice_name = self._generate_filename(
                metadata_key, operator_str, threshold)
        return Slice(self, slice_name, indices, self)

    def slice_by_percentile(self, metadata_key: str, operator_str: str, percentile: float, slice_name: str = None):
        """
        Create a slice by applying an operator to a percentile of the distribution
        of a particular metadatum.
        """
        op_func = OPERATOR_DICT[operator_str]
        values = [dp.get_metadata(metadata_key) for dp in self.datapoints]
        threshold = np.percentile(values, percentile)

        indices = [
            i for i, dp in enumerate(self.datapoints)
            if op_func(dp.get_metadata(metadata_key), threshold)
        ]

        if slice_name is None:
            slice_name = self._generate_filename(
                metadata_key, operator_str, percentile)
        return Slice(self, slice_name, indices, self)

    def slice_by_class(self, metadata_key: str, slice_names: List[str] = None):
        """
        Create slices for each distinct value of a given categorical metadata key.
        E.g. if metadata_key="weather" and it takes values 'sunny','rainy','snowy',
        then create three slices.
        """
        class_indices = {}
        for i, dp in enumerate(self.datapoints):
            class_value = dp.get_metadata(metadata_key)
            class_indices.setdefault(class_value, []).append(i)

        class_values = sorted(class_indices.keys(), key=lambda x: str(x))
        num_classes = len(class_values)

        if not slice_names or len(slice_names) != num_classes:
            slice_names = [
                self._generate_filename(metadata_key, "class", val)
                for val in class_values
            ]

        slices = []
        for val, slice_name in zip(class_values, slice_names):
            indices = class_indices[val]
            slices.append(Slice(self, slice_name, indices, self))
        return slices


class Slice(Moonwatcher):
    """
    A Slice is a lightweight view onto a subset of indices of the parent Moonwatcher dataset.
    Inherits from Moonwatcher so it can reuse methods like add_metadata, etc.
    """

    def __init__(
        self,
        moonwatcher_dataset: Moonwatcher,
        name: str,
        indices: List[int],
        original_dataset: Moonwatcher,
        description: str = None,
    ):
        # Basic identifying info
        self.dataset_name = moonwatcher_dataset.name
        self.name = name
        self.task_type = moonwatcher_dataset.task_type
        self.task = moonwatcher_dataset.task
        self.metadata = moonwatcher_dataset.metadata
        self.datapoints_metadata = moonwatcher_dataset.datapoints_metadata
        self.groundtruths = moonwatcher_dataset.groundtruths
        self.description = description

        self.indices = indices
        self.dataset = Subset(moonwatcher_dataset.dataset, indices)
        self.moonwatcher_dataset = moonwatcher_dataset
        self.original_dataset = original_dataset

        # For each index, we store the corresponding subset of datapoints
        self.datapoints = [moonwatcher_dataset.datapoints[i] for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __getattr__(self, attr):
        return getattr(self.dataset, attr)

    # -------------------------------------------------------------------------
    # Override metadata methods to ensure we add metadata to both the slice
    # and the underlying original dataset if appropriate.
    # -------------------------------------------------------------------------

    def add_predefined_metadata(self, predefined_metadata_key: Union[str, List[str]]):
        super().add_predefined_metadata(predefined_metadata_key)
        self.original_dataset.add_predefined_metadata(predefined_metadata_key)

    def add_metadata_from_groundtruths(self, class_name: str):
        super().add_metadata_from_groundtruths(class_name)
        self.original_dataset.add_metadata_from_groundtruths(class_name)

    def add_metadata_custom(self, metadata_key: str, metadata_func: Callable):
        super().add_metadata_custom(metadata_key, metadata_func)
        self.original_dataset.add_metadata_custom(metadata_key, metadata_func)

    def add_metadata_from_list(self, metadata_list: List[Dict[str, Any]]):
        super().add_metadata_from_list(metadata_list)
        self.original_dataset.add_metadata_from_list(metadata_list)

    # -------------------------------------------------------------------------
    # Slicing from a slice returns a new Slice that references the original dataset
    # (maintaining a consistent reference chain).
    # -------------------------------------------------------------------------

    def slice_by_threshold(self, metadata_key: str, operator_str: str, threshold: Any, slice_name: str = None):
        op_func = OPERATOR_DICT[operator_str]
        filtered_indices = [
            idx for idx, dp in zip(self.indices, self.datapoints)
            if op_func(dp.get_metadata(metadata_key), threshold)
        ]
        if slice_name is None:
            slice_name = self._generate_filename(
                metadata_key, operator_str, threshold)

        return Slice(self, slice_name, filtered_indices, self.original_dataset)

    def slice_by_percentile(self, metadata_key: str, operator_str: str, percentile: float, slice_name: str = None):
        op_func = OPERATOR_DICT[operator_str]
        slice_values = [dp.get_metadata(metadata_key)
                        for dp in self.datapoints]
        threshold = np.percentile(slice_values, percentile)

        filtered_indices = [
            idx for idx, dp in zip(self.indices, self.datapoints)
            if op_func(dp.get_metadata(metadata_key), threshold)
        ]
        if slice_name is None:
            slice_name = self._generate_filename(
                metadata_key, operator_str, percentile)

        return Slice(self, slice_name, filtered_indices, self.original_dataset)

    def slice_by_class(self, metadata_key: str, slice_names: List[str] = None):
        class_indices = {}
        for idx, dp in zip(self.indices, self.datapoints):
            class_value = dp.get_metadata(metadata_key)
            class_indices.setdefault(class_value, []).append(idx)

        class_values = sorted(class_indices.keys(), key=lambda x: str(x))
        num_classes = len(class_values)

        if not slice_names or len(slice_names) != num_classes:
            slice_names = [
                self._generate_filename(metadata_key, "class", val)
                for val in class_values
            ]

        slices = []
        for val, sn in zip(class_values, slice_names):
            idxs = class_indices[val]
            slices.append(Slice(self, sn, idxs, self.original_dataset))
        return slices


class MoonwatcherClassification(Moonwatcher):
    """
    Specialization of Moonwatcher for classification tasks.
    """

    def __init__(
        self,
        dataset: Dataset,
        name: str,
        task: str,
        num_classes: int = None,
        label_to_name: Dict[int, str] = None,
        metadata: Dict[str, Any] = None,
        description: str = None,
        datapoints_metadata: List[Dict[str, Any]] = None,
    ):
        super().__init__(
            dataset=dataset,
            name=name,
            task_type=TaskType.CLASSIFICATION.value,
            task=task,
            num_classes=num_classes,
            label_to_name=label_to_name,
            metadata=metadata,
            description=description,
            datapoints_metadata=datapoints_metadata,
        )


class MoonwatcherDetection(Moonwatcher):
    """
    Specialization of Moonwatcher for detection tasks.
    """

    def __init__(
        self,
        dataset: Dataset,
        name: str,
        task: str,
        num_classes: int = None,
        label_to_name: Dict[int, str] = None,
        metadata: Dict[str, Any] = None,
        description: str = None,
        datapoints_metadata: List[Dict[str, Any]] = None,
    ):
        super().__init__(
            dataset=dataset,
            name=name,
            task_type=TaskType.DETECTION.value,
            task=task,
            num_classes=num_classes,
            label_to_name=label_to_name,
            metadata=metadata,
            description=description,
            datapoints_metadata=datapoints_metadata,
        )
