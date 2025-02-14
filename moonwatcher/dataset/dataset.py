from typing import List, Dict, Any, Callable, Union, Optional
from tqdm import tqdm

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset

from moonwatcher.datapoint import Datapoint
from moonwatcher.utils.data import OPERATOR_DICT, TaskType
from moonwatcher.utils.helpers import get_current_timestamp
from moonwatcher.dataset.metadata import ATTRIBUTE_FUNCTIONS
from moonwatcher.annotations import GroundTruths, Predictions, Labels, BoundingBoxes


# TODO: Find a cleaner way to handle this
def find_root_dataset(dataset: Dataset) -> Dataset:
    """
    Recursively find the root dataset.
    """
    if hasattr(dataset, "dataset"):
        return find_root_dataset(dataset.dataset)
    return dataset


def _get_raw_image(root_dataset: Dataset, index: int) -> Union[Image.Image, np.ndarray, torch.Tensor]:
    """
    Retrieve the original image from `root_dataset` bypassing its transforms.
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
    """
    Converts a PIL Image, numpy array, or torch.Tensor into a standard
    torch.Tensor of shape [C, H, W], with float values in [0,1].
    """
    if isinstance(image, Image.Image):
        return transforms.ToTensor()(image)  # shape [C,H,W], float in [0,1]

    elif isinstance(image, np.ndarray):
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0
        # shape HWC -> CHW
        return torch.from_numpy(image.transpose((2, 0, 1)))

    elif isinstance(image, torch.Tensor):
        # If shape is (H, W, C), permute
        if image.dim() == 3 and image.shape[-1] == 3:
            return image.permute(2, 0, 1)
        return image  # assume already [C,H,W]

    else:
        raise TypeError(f"Unsupported image type {type(image)}")


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
        """
        super().__init__()
        self.name = name
        self.dataset = dataset
        self.task_type = task_type  # e.g. "classification" or "detection"
        self.task = task            # e.g. "binary", "multiclass", etc.
        self.num_classes = num_classes

        self.label_to_name = label_to_name
        self.name_to_label = {}
        if label_to_name is not None:
            self.name_to_label = {v: k for k, v in label_to_name.items()}

        self.metadata = metadata if metadata is not None else {}
        self.metadata["_timestamp"] = get_current_timestamp()
        self.description = description

        # Convert user-supplied datapoints_metadata into Datapoint objects
        self.datapoints_metadata = datapoints_metadata
        self.datapoints = []
        for i in range(len(self.dataset)):
            md = {}
            if datapoints_metadata is not None and i < len(datapoints_metadata):
                md = datapoints_metadata[i]
            self.datapoints.append(Datapoint(id=i, metadata=md))

        # TODO: Is this the right place to initialize these?
        self.groundtruths = GroundTruths(dataset=self)
        self.predictions = Predictions(dataset=self)
        self.add_groundtruths()

    # -------------------------------------------------------------------------
    #                           GROUND TRUTHS
    # -------------------------------------------------------------------------

    def _get_root_index(self, local_idx: int) -> int:
        """
        Return the corresponding index in the root dataset given a local index for a slice.
        """
        if isinstance(self.dataset, Subset):
            return self.dataset.indices[local_idx]  # Root index
        else:
            return local_idx

    def add_groundtruths(self):
        """
        Loops over every item in the underlying dataset and converts the
        second/third outputs into annotation objects (Labels or BoundingBoxes).
        Stores them in self.groundtruths.
        """
        # Clear any existing ground truths (if you want to allow overwriting)
        self.groundtruths = GroundTruths(dataset=self)

        for idx in tqdm(range(len(self.dataset)), desc=f"Building GROUND TRUTHS for {self.name}"):
            data = self.dataset[idx]

            if self.task_type == TaskType.CLASSIFICATION.value:
                # Expect (image, label)
                if len(data) < 2:
                    raise ValueError(
                        "Expected (image, label) from dataset, got fewer elements.")
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
                if len(data) < 3:
                    raise ValueError(
                        "Expected (image, bounding_boxes, labels) for detection.")
                _, bounding_boxes, labels = data

                if not isinstance(bounding_boxes, torch.Tensor):
                    bounding_boxes = torch.tensor(
                        bounding_boxes, dtype=torch.float32)
                if not isinstance(labels, torch.Tensor):
                    labels = torch.tensor(labels, dtype=torch.long)

                ann = BoundingBoxes(datapoint_number=idx,
                                    boxes_xyxy=bounding_boxes, labels=labels)
                self.groundtruths.add(ann)

            else:
                raise ValueError(f"Unsupported task_type={self.task_type}.")

    # -------------------------------------------------------------------------
    #                           PREDICTIONS
    # -------------------------------------------------------------------------
    def add_predictions(self, predictions: Any):
        """
        Expects model outputs (e.g., a tensor of logits for classification, or
        bounding boxes + scores for detection). Converts them into annotation
        objects (Labels w/ scores, or BoundingBoxes w/ scores) stored in `self.predictions`.

        :param predictions:
            For CLASSIFICATION: a torch.Tensor of shape [N, num_classes] or [N]
                                (logits, probabilities, or label indices).
            For DETECTION: a list of length N, each containing bounding_boxes + labels + scores
                           (your structure may vary).
        """
        # Clear existing predictions
        self.predictions = Predictions(dataset=self)

        # Classification case: predictions is typically [N, num_classes]
        if self.task_type == TaskType.CLASSIFICATION.value:
            if not isinstance(predictions, torch.Tensor):
                raise TypeError(
                    "For classification, predictions must be a torch.Tensor.")

            num_samples = predictions.shape[0]
            if num_samples != len(self.dataset):
                raise ValueError(
                    "Mismatch between predictions size and dataset length.")

            # If shape is [N], assume these are predicted labels (class IDs)
            # If shape is [N, C], assume these are logits or probabilities
            if predictions.dim() == 1:
                # predicted labels
                for i in range(num_samples):
                    label_val = predictions[i].unsqueeze(0)
                    ann = Labels(datapoint_number=i,
                                 labels=label_val, scores=None)
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
                        labels=pred_label,       # shape [1]
                        scores=scores           # shape [num_classes]
                    )
                    self.predictions.add(ann)

            else:
                raise ValueError(
                    "Classification predictions must be 1D or 2D tensor.")

        # Detection case: predictions is typically a list of length N
        elif self.task_type == TaskType.DETECTION.value:
            if not isinstance(predictions, list):
                raise TypeError(
                    "For detection, predictions must be a list of length N.")
            if len(predictions) != len(self.dataset):
                raise ValueError(
                    "Mismatch between predictions list and dataset length.")

            # Each element should look like {"boxes": (M,4), "labels": (M,), "scores": (M,)}
            for i, pred_dict in enumerate(tqdm(predictions, desc="Building DETECTION predictions")):
                boxes_xyxy = torch.tensor(
                    pred_dict["boxes"], dtype=torch.float32)
                labels = torch.tensor(pred_dict["labels"], dtype=torch.long)
                scores = torch.tensor(pred_dict["scores"], dtype=torch.float32)

                ann = BoundingBoxes(
                    datapoint_number=i,
                    boxes_xyxy=boxes_xyxy,
                    labels=labels,
                    scores=scores
                )
                self.predictions.add(ann)

        else:
            raise ValueError(f"Unsupported task_type={self.task_type}.")

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
    #                            METADATA METHODS
    # -------------------------------------------------------------------------
    # TODO: Check
    def add_predefined_metadata(self, predefined_metadata_keys: Union[str, List[str]]):
        if isinstance(predefined_metadata_keys, str):
            predefined_metadata_keys = [predefined_metadata_keys]

        root_dataset = find_root_dataset(self.dataset)

        for i in tqdm(range(len(self.dataset)), desc=f"Adding metadata to {self.name}"):
            raw_image = _get_raw_image(root_dataset, i)
            image_tensor = _pil_or_numpy_to_tensor(raw_image)
            # Convert back to numpy [H,W,C], scale 0..255
            image_np = (image_tensor.permute(
                1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            for key in predefined_metadata_keys:
                if key not in self.datapoints[i].metadata:
                    val = ATTRIBUTE_FUNCTIONS[key](image_np)
                    self.datapoints[i].add_metadata(key=key, value=val)

    def add_metadata_from_list(self, metadata_list: List[Dict[str, Any]]):
        for i, md_dict in enumerate(
            tqdm(metadata_list, desc="Adding metadata from list")
        ):
            if i < len(self.datapoints):
                for key, value in md_dict.items():
                    self.datapoints[i].add_metadata(key=key, value=value)

    def add_metadata_custom(self, metadata_key: str, metadata_func: Callable):
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
        return (
            f"{self.name}_{metadata_key}_"
            f"{abbreviations.get(operator_str, operator_str)}_"
            f"{str(value).replace('.', '_')}"
        )

    def slice_by_threshold(self, metadata_key: str, operator_str: str, threshold: Any, slice_name: str = None):
        op_func = OPERATOR_DICT[operator_str]
        indices = [i for i, dp in enumerate(self.datapoints)
                   if op_func(dp.get_metadata(metadata_key), threshold)]
        if slice_name is None:
            slice_name = self._generate_filename(
                metadata_key, operator_str, threshold)
        return Slice(name=slice_name, root_dataset=self, indices=indices)

    def slice_by_percentile(self, metadata_key: str, operator_str: str, percentile: float, slice_name: str = None):
        op_func = OPERATOR_DICT[operator_str]
        values = [dp.get_metadata(metadata_key) for dp in self.datapoints]
        threshold = np.percentile(values, percentile)
        indices = [i for i, dp in enumerate(self.datapoints)
                   if op_func(dp.get_metadata(metadata_key), threshold)]
        if slice_name is None:
            slice_name = self._generate_filename(
                metadata_key, operator_str, percentile)
        return Slice(name=slice_name, root_dataset=self, indices=indices)

    def slice_by_metadata_value(
        self,
        metadata_key: str,
        target_value: Any,
        slice_name: Optional[str] = None
    ) -> "Slice":
        """Create slice where datapoint.metadata[metadata_key] == target_value.

        Args:
            metadata_key: Metadata field to check
            target_value: Value to match exactly
            slice_name: Optional name for the slice

        Returns:
            Slice containing matching datapoints
        """
        indices = [
            i for i, dp in enumerate(self.datapoints)
            if dp.get_metadata(metadata_key) == target_value
        ]

        if not slice_name:
            slice_name = f"{metadata_key}_{str(target_value).replace(' ', '_')}"

        return Slice(name=slice_name, root_dataset=self, indices=indices)

    def slice_by_groundtruth_class(
        self,
        class_names: Optional[List[str]] = None,
        class_ids: Optional[List[int]] = None,
        slice_name: Optional[str] = None
    ) -> "Slice":
        """Create slice containing datapoints with specified classes in their ground truth.

        Args:
            class_names: Class names to include (requires label_to_name mapping)
            class_ids: Class IDs to include (direct identifier)
            slice_name: Optional name for the resulting slice

        Returns:
            Slice containing datapoints with specified classes

        Raises:
            ValueError: If neither class_names nor class_ids are provided
        """
        # Validate at least one argument provided
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
                raise ValueError(
                    f"Invalid class names: {sorted(invalid_names)}")

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
                if torch.any(torch.isin(ground_truth.labels, torch.tensor(list(class_id_set)))):
                    filtered_indices.append(original_idx)

        # Generate default name if needed
        if not slice_name:
            target_classes = class_names if class_names else sorted(
                class_id_set)
            class_str = "_".join(map(str, target_classes))[:50]  # Limit length
            slice_name = f"gt_class_{class_str}"

        return Slice(name=slice_name, root_dataset=self, indices=filtered_indices)


class Slice(Moonwatcher):
    """
    A Slice is a child of the parent Moonwatcher dataset containing only a subset of indices.
    """

    def __init__(
        self,
        name: str,
        root_dataset: Moonwatcher,
        indices: List[int],
        description: str = None,
    ):
        self.name = name
        self.root_dataset = root_dataset
        self.indices = indices
        self.description = description
        self.datapoints = [root_dataset.datapoints[i] for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Map the local slice index to the corresponding index in the root dataset.
        root_idx = self.indices[idx]
        return self.root_dataset.dataset[root_idx]

    def __getattr__(self, attr):
        return getattr(self.root_dataset, attr)  # getattr(self.dataset, attr)


class MoonwatcherClassification(Moonwatcher):
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


def get_original_indices(dataset: Union[Moonwatcher, Slice]) -> List[int]:
    """
    Recursively retrieve the original indices from a dataset or slice.
    """
    if isinstance(dataset, Slice):
        parent_indices = get_original_indices(dataset.root_dataset)
        return [parent_indices[i] for i in dataset.indices]
    elif isinstance(dataset, Moonwatcher):
        return list(range(len(dataset.dataset)))
    else:
        raise TypeError("Unsupported dataset type for get_original_indices.")
