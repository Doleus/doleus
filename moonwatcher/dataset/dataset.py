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
from moonwatcher.annotations import GroundTruths, Predictions, Labels, BoundingBoxes
from moonwatcher.utils.helpers import get_current_timestamp


def find_root_dataset(dataset: Dataset) -> Dataset:
    """
    Recursively find the original underlying dataset in case the user
    has wrapped the dataset in multiple Subset objects.
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

        # Annotations: we store them here, but fill them *after* init as needed
        self.groundtruths = GroundTruths(dataset=self)
        self.predictions = Predictions(dataset=self)
        self.add_groundtruths_from_dataset()

    # -------------------------------------------------------------------------
    #                           GROUND TRUTHS
    # -------------------------------------------------------------------------

    def _get_root_index(self, local_idx: int) -> int:
        """
        Return the 'real' index in the *original* dataset if self.dataset is a Subset.
        Otherwise, return local_idx.
        """
        if isinstance(self.dataset, Subset):
            return self.dataset.indices[local_idx]  # Root index
        else:
            return local_idx

    def add_groundtruths_from_dataset(self):
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
    def add_predictions_from_model_outputs(self, predictions: Any):
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

    def add_metadata_from_groundtruths(self, class_name: str):
        """
        Example: if your dataset is classification and 'class_name' is in label_to_name,
        add to each datapoint how many times that class appears.
        """
        if not self.name_to_label:
            if self.label_to_name is not None:
                self.name_to_label = {v: k for k,
                                      v in self.label_to_name.items()}
            else:
                raise ValueError(
                    "No label_to_name provided, can't match class_name.")

        if class_name not in self.name_to_label:
            raise ValueError(
                f"Class '{class_name}' not found in label_to_name.")
        class_id = self.name_to_label[class_name]

        for datapoint in tqdm(self.datapoints, desc=f"Adding metadata for class '{class_name}'"):
            i = datapoint.id
            if i not in self.groundtruths.datapoint_number_to_annotation_index:
                # no groundtruth for this item?
                continue

            groundtruth = self.groundtruths.get(i)

            if self.task_type == TaskType.DETECTION.value:
                if isinstance(groundtruth, BoundingBoxes):
                    count = (groundtruth.labels == class_id).sum().item()
                    datapoint.add_metadata(key=class_name, value=count)
            elif self.task_type == TaskType.CLASSIFICATION.value:
                if isinstance(groundtruth, Labels):
                    count = (groundtruth.labels == class_id).sum().item()
                    datapoint.add_metadata(key=class_name, value=count)

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
        return Slice(self, slice_name, indices, self)

    def slice_by_percentile(self, metadata_key: str, operator_str: str, percentile: float, slice_name: str = None):
        op_func = OPERATOR_DICT[operator_str]
        values = [dp.get_metadata(metadata_key) for dp in self.datapoints]
        threshold = np.percentile(values, percentile)
        indices = [i for i, dp in enumerate(self.datapoints)
                   if op_func(dp.get_metadata(metadata_key), threshold)]
        if slice_name is None:
            slice_name = self._generate_filename(
                metadata_key, operator_str, percentile)
        return Slice(self, slice_name, indices, self)

    def slice_by_class(self, metadata_key: str, slice_names: List[str] = None):
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
        for val, sn in zip(class_values, slice_names):
            idxs = class_indices[val]
            slices.append(Slice(self, sn, idxs, self))
        return slices


class Slice(Moonwatcher):
    """
    A Slice is a child of the parent Moonwatcher dataset containing only a subset of indices.
    """

    def __init__(
        self,
        moonwatcher_dataset: Moonwatcher,
        name: str,
        indices: List[int],
        original_dataset: Moonwatcher,
        description: str = None,
    ):
        self.dataset_name = moonwatcher_dataset.name
        self.name = name
        self.task_type = moonwatcher_dataset.task_type
        self.task = moonwatcher_dataset.task
        self.metadata = moonwatcher_dataset.metadata
        self.datapoints_metadata = moonwatcher_dataset.datapoints_metadata
        self.groundtruths = moonwatcher_dataset.groundtruths
        self.predictions = moonwatcher_dataset.predictions
        self.description = description

        self.indices = indices
        self.dataset = Subset(moonwatcher_dataset.dataset, indices)
        self.moonwatcher_dataset = moonwatcher_dataset
        self.original_dataset = original_dataset

        # For each index, store the corresponding subset of datapoints
        self.datapoints = [moonwatcher_dataset.datapoints[i] for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __getattr__(self, attr):
        return getattr(self.dataset, attr)

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

    # Slicing from a slice returns a new Slice referencing the original dataset
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
