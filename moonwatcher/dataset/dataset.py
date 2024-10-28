from typing import List, Dict, Any, Callable

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset

from moonwatcher.datapoint import Datapoint
from moonwatcher.utils.data import OPERATOR_DICT
from moonwatcher.utils.data import DataType, TaskType, Task
from moonwatcher.utils.api_connector import is_api_key_and_endpoint_available
from moonwatcher.base.base import MoonwatcherObject
from moonwatcher.dataset.metadata import ATTRIBUTE_FUNCTIONS
from moonwatcher.utils.api_connector import upload_if_possible
from moonwatcher.annotations import GroundTruths, Labels, BoundingBoxes, Predictions, PredictedLabels, PredictedBoundingBoxes
from moonwatcher.utils.helpers import get_current_timestamp, convert_to_list


def find_root_dataset(dataset):
    """Recursively find the root dataset."""
    if hasattr(dataset, "dataset"):
        return find_root_dataset(dataset.dataset)
    return dataset


class Moonwatcher(MoonwatcherObject, Dataset):
    def __init__(
        self,
        dataset: Dataset,
        name: str,
        task_type: str,
        task: str,
        output_transform: Callable,
        predictions: List[Any],
        num_classes: int = None,
        label_to_name: Dict = None,
        metadata: Dict[str, Any] = None,
        description: str = None,
        locators: List[str] = None,
        datapoints_metadata: List[Dict[str, Any]] = None,
    ):
        """
        Creates a moonwatcher dataset and model wrapper around an existing dataset.

        :param dataset: the dataset to be wrapped
        :param name: the name of the dataset
        :param task_type: either classification or detection
        :param task: either binary, multiclass or multilabel
        :param output_transform: necessary to transform dataset output into moonwatcher format
        :param predictions: a list of predictions for the dataset
        :param num_classes: number of classes
        :param label_to_name: dictionary mapping label ids to name
        :param metadata: dictionary of tags for the dataset
        :param description: description of the dataset
        :param locators: URLs for every image to display in webapp
        :param datapoints_metadata: metadata for every datapoint
        """
        MoonwatcherObject.__init__(self, name=name, datatype=DataType.DATASET)
        Dataset.__init__(self)
        self.dataset = dataset
        self.label_to_name = label_to_name
        self.task_type = task_type
        self.task = task
        self.metadata = metadata if metadata is not None else {}
        self.metadata["_timestamp"] = get_current_timestamp()
        self.description = description
        self.locators = locators
        self.num_classes = num_classes
        self.datapoints = []
        self.datapoints_metadata = datapoints_metadata
        self.predictions = predictions
        self.output_transform = output_transform

        if self.locators:
            if not isinstance(self.locators, list):
                raise ValueError("Locators needs to be a list")
            if len(self.locators) != len(dataset):
                raise ValueError(
                    "Locators needs to provide a locator for every image (List with length of dataset)"
                )
            for locator in self.locators:
                if not isinstance(locator, str):
                    raise ValueError("Locators need to be strings")

        for i in range(len(self.dataset)):
            metadata = (
                datapoints_metadata[i]
                if datapoints_metadata is not None and i < len(datapoints_metadata)
                else {}
            )
            if self.locators:
                datapoint = Datapoint(
                    number=i, metadata=metadata, locator=self.locators[i]
                )
            else:
                datapoint = Datapoint(number=i, metadata=metadata)
            self.datapoints.append(datapoint)

        self.groundtruths = GroundTruths(self)

        for index in tqdm(
            list(range(len(self.dataset))),
            desc=f"Saving annotations of dataset {self.name}.",
        ):
            data = self.dataset[index]
            try:
                transformed_data = self.output_transform(data)
            except Exception as e:
                raise Exception(
                    f"Application of output_transform on dataset failed: {e}"
                )

            if self.task_type == TaskType.CLASSIFICATION.value:
                try:
                    image, label = transformed_data
                except ValueError as e:
                    raise ValueError(
                        f"Dataset output_transform should return two elements (image, label): {e}"
                    )
                groundtruth = Labels(datapoint_number=index, labels=label)
            elif self.task_type == TaskType.DETECTION.value:
                try:
                    image, bounding_boxes, labels = transformed_data
                except ValueError as e:
                    raise ValueError(
                        f"Dataset output_transform should return three elements (image, bounding_boxes, labels): {e}"
                    )
                groundtruth = BoundingBoxes(
                    datapoint_number=index, boxes_xyxy=bounding_boxes, labels=labels
                )
            else:
                raise ValueError(
                    f"Unsupported task type: {self.task_type} - Select either 'classification' or 'detection'"
                )

            self.groundtruths.add(groundtruth)

        self.groundtruths.store()
        self.store()
        self.upload_if_not()

        # Handle predictions
        # TODO: Requires cleanup
        if self.predictions is not None:
            if len(self.predictions) != len(self.dataset):
                raise ValueError(
                    f"Number of predictions ({len(self.predictions)}) does not match number of datapoints ({len(self.dataset)})"
                )
            self.predictions_obj = Predictions(self)
            for index, prediction in enumerate(self.predictions):
                if self.task_type == TaskType.CLASSIFICATION.value:
                    scores = torch.tensor(prediction, dtype=torch.float32)
                    # TODO: use torchmetrics function to determine wether we get labels (int) or scores (float)
                    labels = (scores > 0.5).to(torch.int64)
                    pred = PredictedLabels(
                        datapoint_number=index, labels=labels, scores=scores)
                elif self.task_type == TaskType.DETECTION.value:
                    boxes, labels = prediction
                    pred = PredictedBoundingBoxes(
                        datapoint_number=index, boxes_xyxy=boxes, labels=labels, scores=scores
                    )
                else:
                    raise ValueError(
                        f"Unsupported task type: {self.task_type} - Select either 'classification' or 'detection'"
                    )
                self.predictions_obj.add(pred)
            self.predictions_obj.store()

    def _upload(self):
        datapoints = []
        if not is_api_key_and_endpoint_available():
            return False
        if self.datapoints[0].locator is None:
            raise ValueError(
                "Please provide locators for the images if you want to upload the dataset."
            )
        for datapoint in self.datapoints:
            datapoints.append(
                {
                    "locator": datapoint.locator,
                    "metadata": datapoint.metadata,
                }
            )
        data = {
            "name": self.name,
            "description": self.description,
            "timestamp": get_current_timestamp(),
            "metadata": self.metadata,
            "label_to_name": self.label_to_name,
            "datapoints": datapoints,
            "task_type": self.task_type,
        }
        upload_if_possible(datatype=DataType.DATASET.value, data=data)

        groundtruths = []
        for groundtruth in self.groundtruths:
            groundtruths.append(
                {
                    "dataset_name": self.name,
                    "datapoint_number": groundtruth.datapoint_number,
                    "boxes": (
                        [convert_to_list(boxes)
                         for boxes in groundtruth.boxes_xyxy]
                        if hasattr(groundtruth, "boxes_xyxy")
                        else None
                    ),
                    "labels": convert_to_list(groundtruth.labels),
                }
            )
        upload_if_possible(
            datatype=DataType.GROUNDTRUTHS.value, data=groundtruths
        )

        predictions = []
        for prediction in self.predictions_obj:
            predictions.append(
                {
                    "dataset_name": self.name,
                    "datapoint_number": prediction.datapoint_number,
                    "boxes": (
                        [convert_to_list(boxes)
                         for boxes in prediction.boxes_xyxy]
                        if hasattr(prediction, "boxes_xyxy")
                        else None
                    ),
                    "labels": convert_to_list(prediction.labels),
                }
            )
        return upload_if_possible(
            datatype=DataType.PREDICTIONS.value, data=predictions
        )

    def add_predefined_metadata(self, predefined_metadata_key: str):
        """
        Use a predefined metadata creation function to add metadata "brightness", "contrast", "saturation", "resolution"

        :param predefined_metadata_key: the key for the type of metadata to add
        """
        root_dataset = find_root_dataset(self.dataset)
        transform = root_dataset.transform
        root_dataset.transform = None
        for i in tqdm(range(len(self.dataset)), desc="Adding metadata"):
            data = self.dataset[i]
            image = data[0]

            transform_to_tensor = transforms.ToTensor()

            if isinstance(image, Image.Image):
                image = transform_to_tensor(image)
            elif isinstance(image, np.ndarray):
                if image.dtype != np.float32:
                    image = image.astype(np.float32) / 255
                image = torch.from_numpy(image.transpose((2, 0, 1)))
            elif isinstance(image, torch.Tensor):
                if image.shape[-1] == 3:
                    image = image.permute(2, 0, 1)
                else:
                    image = image
            else:
                raise TypeError("Unsupported image type")

            image = image.permute(1, 2, 0).cpu().numpy()
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image

            if predefined_metadata_key not in self.datapoints[i].metadata:
                metadata_value = ATTRIBUTE_FUNCTIONS[predefined_metadata_key](
                    image)
                self.datapoints[i].add_metadata(
                    key=predefined_metadata_key, value=metadata_value
                )

        root_dataset.transform = transform
        self.store(overwrite=True)

    def add_metadata_from_list(self, metadata_list: List[Dict[str, Any]]):
        """
        Add metadata for all data points from a list

        :param metadata_list: metadata dicts for all data points.
        """
        for i, metadata in enumerate(tqdm(metadata_list, desc="Adding metadata")):
            if i < len(self.datapoints):
                for key, value in metadata.items():
                    self.datapoints[i].add_metadata(key=key, value=value)
        self.store(overwrite=True)

    def add_metadata_from_groundtruths(self, class_name: str):
        """
        Add the number of occurrences of a specific class in each picture as metadata.

        :param class_name: Name of the class to count occurrences for.
        """
        class_id = None
        for key, value in self.label_to_name.items():
            if value == class_name:
                class_id = key
                break

        if class_id is None:
            raise ValueError(
                f"Class name '{class_name}' not found in label_to_name mapping."
            )

        for i, datapoint in enumerate(tqdm(self.datapoints, desc=f"Adding metadata")):
            groundtruth = self.groundtruths.get(datapoint.number)

            if self.task_type == TaskType.DETECTION.value:
                if isinstance(groundtruth, BoundingBoxes):
                    count = (groundtruth.labels == class_id).sum().item()
                    datapoint.add_metadata(key=f"{class_name}", value=count)

            elif self.task_type == TaskType.CLASSIFICATION.value:
                if isinstance(groundtruth, Labels):
                    count = (groundtruth.labels == class_id).sum().item()
                    datapoint.add_metadata(key=f"{class_name}", value=count)

        self.store(overwrite=True)

    def add_metadata_custom(self, metadata_key: str, metadata_func: Callable):
        """
        Add metadata for all using a metadata function
        :param metadata_key: name of the metadatum
        :param metadata_func: function that calculates a metadata value given an image input
        :return:
        """
        root_dataset = find_root_dataset(self.dataset)
        transform = root_dataset.transform
        root_dataset.transform = None

        for i in tqdm(range(len(self.dataset)), desc="Adding metadata"):
            data = self.dataset[i]
            image = data[0]

            transform_to_tensor = transforms.ToTensor()

            if isinstance(image, Image.Image):
                image = transform_to_tensor(image)
            elif isinstance(image, np.ndarray):
                if image.dtype != np.float32:
                    image = image.astype(np.float32) / 255
                image = torch.from_numpy(image.transpose((2, 0, 1)))
            elif isinstance(image, torch.Tensor):
                if image.shape[-1] == 3:
                    image = image.permute(2, 0, 1)
                else:
                    image = image
            else:
                raise TypeError("Unsupported image type")

            image = image.permute(1, 2, 0).cpu().numpy()
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image

            if metadata_key not in self.datapoints[i].metadata:
                metadata_value = metadata_func(image)
                self.datapoints[i].add_metadata(
                    key=metadata_key, value=metadata_value
                )

        root_dataset.transform = transform
        self.store(overwrite=True)

    def _generate_filename(self, metadata_key: str, operator_str: str, value: Any):
        abbreviations = {
            ">": "gt",
            "<": "lt",
            ">=": "ge",
            "<=": "le",
            "==": "eq",
            "class": "cl",
        }
        filename = f"{self.name}_{metadata_key}_{abbreviations[operator_str]}_{str(value).replace('.', '_')}"
        return filename

    def slice_by_threshold(
        self,
        metadata_key: str,
        operator_str: str,
        threshold: Any,
        slice_name: str = None,
    ):
        """
        Create a slice, the metadata key has to exist already: e.g. ("brightness", "<", 0.1)

        :param metadata_key: name of the metadatum
        :param operator: compare symbol like >, >= etc.
        :param threshold: threshold for selection of what data should stay inside
        :param slice_name: name of the slice to create
        """
        op_func = OPERATOR_DICT[operator_str]

        indices = [
            i
            for i, datapoint in enumerate(self.datapoints)
            if op_func(datapoint.get_metadata(metadata_key), threshold)
        ]

        if slice_name is None:
            slice_name = self._generate_filename(
                metadata_key, operator_str, threshold)

        return Slice(self, slice_name, indices, self)

    def slice_by_percentile(
        self,
        metadata_key: str,
        operator_str: str,
        percentile: Any,
        slice_name: str = None,
    ):
        """
        Create a slice, the metadata key has to exist already: e.g. ("brightness", "<", 99)

        :param metadata_key: name of the metadatum
        :param operator: compare symbol like >, >= etc.
        :param percentile: value between 0 and 100
        :param slice_name: name of the slice to create
        """
        op_func = OPERATOR_DICT[operator_str]
        values = [datapoint.get_metadata(metadata_key)
                  for datapoint in self.datapoints]
        threshold = np.percentile(values, percentile)

        indices = [
            i
            for i, datapoint in enumerate(self.datapoints)
            if op_func(datapoint.get_metadata(metadata_key), threshold)
        ]

        if slice_name is None:
            slice_name = self._generate_filename(
                metadata_key, operator_str, percentile)

        return Slice(self, slice_name, indices, self)

    def slice_by_class(self, metadata_key: str, slice_names: List[str] = None):
        """
        Create slices based on a categorical metadatum (e.g. weather: "sunny", "rainy" ...)

        :param metadata_key: name of the metadatum
        :param slice_names: list of names for the slices to create (optional)
        """
        # Collect indices by class value
        class_indices = {}
        for i, datapoint in enumerate(self.datapoints):
            class_value = datapoint.get_metadata(metadata_key)

            if class_value not in class_indices:
                class_indices[class_value] = []

            class_indices[class_value].append(i)

        class_values = sorted(class_indices.keys())
        num_classes = len(class_values)

        if slice_names is None or len(slice_names) != num_classes:
            # Generate default slice names
            slice_names = [
                self._generate_filename(metadata_key, "class", class_value)
                for class_value in class_values
            ]

        slices = []
        for class_value, slice_name in zip(class_values, slice_names):
            indices = class_indices[class_value]
            slices.append(Slice(self, slice_name, indices, self))

        return slices

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]

    def __getattr__(self, attr):
        return getattr(self.dataset, attr)

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


class Slice(Moonwatcher, MoonwatcherObject):
    def __init__(
        self,
        moonwatcher_dataset: Moonwatcher,
        name: str,
        indices: List[int],
        original_dataset: Moonwatcher,
        description: str = None,
    ):
        self.dataset_name = (
            moonwatcher_dataset.name
        )
        MoonwatcherObject.__init__(self, name=name, datatype=DataType.SLICE)

        self.task_type = moonwatcher_dataset.task_type
        self.task = moonwatcher_dataset.task
        self.output_transform = moonwatcher_dataset.output_transform
        self.metadata = moonwatcher_dataset.metadata
        self.locators = moonwatcher_dataset.locators
        self.datapoints = [moonwatcher_dataset.datapoints[i] for i in indices]
        self.datapoints_metadata = moonwatcher_dataset.datapoints_metadata
        self.groundtruths = moonwatcher_dataset.groundtruths
        self.predictions = moonwatcher_dataset.predictions
        self.predictions_obj = moonwatcher_dataset.predictions_obj

        self.description = description
        self.indices = indices
        self.dataset = Subset(moonwatcher_dataset.dataset, indices)
        self.moonwatcher_dataset = moonwatcher_dataset
        self.original_dataset = original_dataset

        if self.locators:
            self.locators = [self.locators[i] for i in indices]

        self.store()
        self.upload_if_not()

    def _upload(self):
        data = {
            "dataset_name": self.dataset_name,
            "name": self.name,
            "description": self.description,
            "timestamp": get_current_timestamp(),
            "metadata": self.metadata,
            "datapoint_numbers": self.indices,
        }
        return upload_if_possible(datatype=DataType.SLICE.value, data=data)

    def slice_by_threshold(
        self,
        metadata_key: str,
        operator_str: str,
        threshold: Any,
        slice_name: str = None,
    ):
        op_func = OPERATOR_DICT[operator_str]

        indices = [
            idx
            for idx, datapoint in zip(self.indices, self.datapoints)
            if op_func(datapoint.get_metadata(metadata_key), threshold)
        ]

        if slice_name is None:
            slice_name = self._generate_filename(
                metadata_key, operator_str, threshold
            )

        return Slice(self, slice_name, indices, self.original_dataset)

    def slice_by_percentile(
        self,
        metadata_key: str,
        operator_str: str,
        percentile: Any,
        slice_name: str = None,
    ):
        op_func = OPERATOR_DICT[operator_str]
        values = [datapoint.get_metadata(metadata_key)
                  for datapoint in self.datapoints]
        threshold = np.percentile(values, percentile)

        indices = [
            idx
            for idx, datapoint in zip(self.indices, self.datapoints)
            if op_func(datapoint.get_metadata(metadata_key), threshold)
        ]

        if slice_name is None:
            slice_name = self._generate_filename(
                metadata_key, operator_str, percentile
            )

        return Slice(self, slice_name, indices, self.original_dataset)

    def slice_by_class(self, metadata_key: str, slice_names: List[str] = None):
        """
        Create slices based on a categorical metadatum (e.g. weather: "sunny", "rainy" ...)

        :param metadata_key: name of the metadatum
        :param slice_names: list of names for the slices to create (optional)
        """
        # Collect indices by class value
        class_indices = {}
        for idx, datapoint in zip(self.indices, self.datapoints):
            class_value = datapoint.get_metadata(metadata_key)

            if class_value not in class_indices:
                class_indices[class_value] = []

            class_indices[class_value].append(idx)

        class_values = sorted(class_indices.keys())
        num_classes = len(class_values)

        if slice_names is None or len(slice_names) != num_classes:
            # Generate default slice names
            slice_names = [
                self._generate_filename(metadata_key, "class", class_value)
                for class_value in class_values
            ]

        slices = []
        for class_value, slice_name in zip(class_values, slice_names):
            indices = class_indices[class_value]
            slices.append(Slice(self, slice_name, indices, self))

        return slices

    def add_predefined_metadata(self, predefined_metadata_key: str):
        """
        Use a predefined metadata creation function to add metadata "brightness", "contrast", "saturation", "resolution"

        :param predefined_metadata_key: the key for the predefined metadata to add
        """
        super().add_predefined_metadata(predefined_metadata_key)
        self.original_dataset.add_predefined_metadata(predefined_metadata_key)

    def add_metadata_from_groundtruths(self, class_name: str):
        """
        Add the number of occurrences of a specific class in each picture as metadata.

        :param class_name: Name of the class to count occurrences for.
        """
        super().add_metadata_from_groundtruths(class_name)
        self.original_dataset.add_metadata_from_groundtruths(class_name)

    def add_metadata_custom(self, metadata_key: str, metadata_func: Callable):
        """
        Add metadata for all using a metadata function
        :param metadata_key: name of the metadatum
        :param metadata_func: function that calculates a metadata value given an image input
        :return:
        """
        super().add_metadata_custom(metadata_key, metadata_func)
        self.original_dataset.add_metadata_custom(metadata_key, metadata_func)

    def add_metadata_from_list(self, metadata_list: List[Dict[str, Any]]):
        """
        Add metadata for all data points from a list

        :param metadata_list: metadata dicts for all data points.
        """
        super().add_metadata_from_list(metadata_list)
        self.original_dataset.add_metadata_from_list(metadata_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]

    def __getattr__(self, attr):
        return getattr(self.dataset, attr)

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
