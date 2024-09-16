import uuid
from typing import Dict, List

import pandas as pd
import pytest
import torch
from doleus.datasets import DoleusClassification, DoleusDetection
from doleus.utils import Task
from torch.utils.data import Dataset


class TestGroundTruths:
    """Ground truth labels for predictable testing (10 samples each)."""

    BINARY_LABELS = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

    MULTICLASS_LABELS = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]

    MULTILABEL_LABELS = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
    ]

    DETECTION_LABELS = [
        {"boxes": [[10, 10, 50, 50]], "labels": [0]},
        {"boxes": [[20, 20, 60, 60]], "labels": [1]},
        {"boxes": [[30, 30, 70, 70]], "labels": [2]},
        {"boxes": [[10, 10, 40, 40], [50, 50, 80, 80]], "labels": [0, 1]},
        {"boxes": [[15, 15, 45, 45], [55, 55, 85, 85]], "labels": [0, 2]},
        {"boxes": [[20, 20, 50, 50], [60, 60, 90, 90]], "labels": [1, 2]},
        {
            "boxes": [[10, 10, 30, 30], [40, 40, 60, 60], [70, 70, 90, 90]],
            "labels": [0, 1, 2],
        },
        {
            "boxes": [[15, 15, 35, 35], [45, 45, 65, 65], [75, 75, 95, 95]],
            "labels": [0, 0, 1],
        },
        {"boxes": [[25, 25, 65, 65]], "labels": [0]},
        {"boxes": [[35, 35, 75, 75]], "labels": [1]},
    ]


class TestMetadata:
    """Metadata patterns for predictable testing (10 samples each)."""

    BASIC_METADATA = [
        {"source": "camera_a", "validated": True},  # 0
        {"source": "camera_b", "validated": False},  # 1
        {"source": "camera_a", "validated": True},  # 2
        {"source": "camera_b", "validated": False},  # 3
        {"source": "camera_a", "validated": True},  # 4
        {"source": "camera_b", "validated": False},  # 5
        {"source": "camera_a", "validated": True},  # 6
        {"source": "camera_b", "validated": False},  # 7
        {"source": "camera_a", "validated": True},  # 8
        {"source": "camera_b", "validated": False},  # 9
    ]
    # camera_a: indices 0,2,4,6,8 (5 samples)
    # camera_b: indices 1,3,5,7,9 (5 samples)
    # validated=True: indices 0,2,4,6,8 (5 samples)
    # validated=False: indices 1,3,5,7,9 (5 samples)

    NUMERIC_METADATA = [
        {"confidence_score": 0.95, "batch_id": 1},  # 0
        {"confidence_score": 0.87, "batch_id": 1},  # 1
        {"confidence_score": 0.92, "batch_id": 1},  # 2
        {"confidence_score": 0.78, "batch_id": 2},  # 3
        {"confidence_score": 0.89, "batch_id": 2},  # 4
        {"confidence_score": 0.65, "batch_id": 2},  # 5
        {"confidence_score": 0.91, "batch_id": 3},  # 6
        {"confidence_score": 0.72, "batch_id": 3},  # 7
        {"confidence_score": 0.83, "batch_id": 3},  # 8
        {"confidence_score": 0.76, "batch_id": 3},  # 9
    ]
    # High confidence (>= 0.85): indices 0,1,2,4,6 (5 samples)
    # Low confidence (< 0.85): indices 3,5,7,8,9 (5 samples)
    # batch_id=1: indices 0,1,2 (3 samples)
    # batch_id=2: indices 3,4,5 (3 samples)
    # batch_id=3: indices 6,7,8,9 (4 samples)

    MIXED_METADATA = [
        {
            "environment": "lab",
            "temperature": 22.5,
            "sample_count": 100,
            "corrupted": False,
        },  # 0
        {
            "environment": "field",
            "temperature": 18.3,
            "sample_count": 85,
            "corrupted": True,
        },  # 1
        {
            "environment": "lab",
            "temperature": 23.1,
            "sample_count": 120,
            "corrupted": False,
        },  # 2
        {
            "environment": "field",
            "temperature": 16.7,
            "sample_count": 75,
            "corrupted": False,
        },  # 3
        {
            "environment": "lab",
            "temperature": 21.8,
            "sample_count": 110,
            "corrupted": True,
        },  # 4
        {
            "environment": "field",
            "temperature": 19.2,
            "sample_count": 90,
            "corrupted": False,
        },  # 5
        {
            "environment": "lab",
            "temperature": 24.0,
            "sample_count": 95,
            "corrupted": False,
        },  # 6
        {
            "environment": "field",
            "temperature": 17.5,
            "sample_count": 80,
            "corrupted": True,
        },  # 7
        {
            "environment": "lab",
            "temperature": 22.8,
            "sample_count": 105,
            "corrupted": False,
        },  # 8
        {
            "environment": "field",
            "temperature": 20.1,
            "sample_count": 88,
            "corrupted": False,
        },  # 9
    ]
    # environment="lab": indices 0,2,4,6,8 (5 samples)
    # environment="field": indices 1,3,5,7,9 (5 samples)
    # temperature >= 20.0: indices 0,2,4,6,8,9 (6 samples)
    # temperature < 20.0: indices 1,3,5,7 (4 samples)
    # sample_count >= 100: indices 0,2,4,8 (4 samples)
    # sample_count < 100: indices 1,3,5,6,7,9 (6 samples)
    # corrupted=True: indices 1,4,7 (3 samples)
    # corrupted=False: indices 0,2,3,5,6,8,9 (7 samples)

    STRING_NUMERIC_METADATA = [
        {"priority": "1", "status": "active"},  # 0 - numbers as strings
        {"priority": "2", "status": "inactive"},  # 1
        {"priority": "1", "status": "active"},  # 2
        {"priority": "3", "status": "pending"},  # 3
        {"priority": "2", "status": "active"},  # 4
        {"priority": "1", "status": "inactive"},  # 5
        {"priority": "3", "status": "active"},  # 6
        {"priority": "2", "status": "pending"},  # 7
        {"priority": "1", "status": "active"},  # 8
        {"priority": "3", "status": "inactive"},  # 9
    ]
    # priority="1": indices 0,2,5,8 (4 samples)
    # priority="2": indices 1,4,7 (3 samples)
    # priority="3": indices 3,6,9 (3 samples)
    # status="active": indices 0,2,4,6,8 (5 samples)
    # status="inactive": indices 1,5,9 (3 samples)
    # status="pending": indices 3,7 (2 samples)


class MockClassificationDataset(Dataset):
    """Mock dataset for classification testing with explicit labels (10 samples)."""

    def __init__(self, task_type: str = "binary"):
        self.dataset_size = 10
        self.task_type = task_type

        if task_type == Task.BINARY.value:
            self.num_classes = 2
        elif task_type == Task.MULTICLASS.value:
            self.num_classes = 3
        elif task_type == Task.MULTILABEL.value:
            self.num_classes = 3
        else:
            raise ValueError(f"Unsupported task: {task_type}")

        torch.manual_seed(42)

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, datapoint_idx: int):
        if datapoint_idx >= self.dataset_size:
            raise IndexError(
                f"Index {datapoint_idx} out of range for dataset of size {self.dataset_size}"
            )

        torch.manual_seed(42 + datapoint_idx)
        image_tensor = torch.rand(3, 224, 224)

        if self.task_type == Task.BINARY.value:
            ground_truth_label = torch.tensor(
                TestGroundTruths.BINARY_LABELS[datapoint_idx], dtype=torch.long
            )

        elif self.task_type == Task.MULTICLASS.value:
            ground_truth_label = torch.tensor(
                TestGroundTruths.MULTICLASS_LABELS[datapoint_idx], dtype=torch.long
            )

        elif self.task_type == Task.MULTILABEL.value:
            ground_truth_label = torch.tensor(
                TestGroundTruths.MULTILABEL_LABELS[datapoint_idx], dtype=torch.long
            )

        else:
            raise ValueError(f"Unsupported task: {self.task_type}")

        return image_tensor, ground_truth_label


class MockDetectionDataset(Dataset):
    """Mock dataset for object detection testing with explicit scenarios (10 samples)."""

    def __init__(self):
        self.dataset_size = 10

        torch.manual_seed(42)

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, datapoint_idx: int):
        if datapoint_idx >= self.dataset_size:
            raise IndexError(
                f"Index {datapoint_idx} out of range for dataset of size {self.dataset_size}"
            )

        torch.manual_seed(42 + datapoint_idx)
        image_tensor = torch.rand(3, 416, 416)

        scenario = TestGroundTruths.DETECTION_LABELS[datapoint_idx]

        bounding_boxes = torch.tensor(scenario["boxes"], dtype=torch.float32)
        detection_labels = torch.tensor(scenario["labels"], dtype=torch.long)

        return image_tensor, bounding_boxes, detection_labels


# Classification Fixtures
@pytest.fixture
def mock_binary_classification_dataset() -> MockClassificationDataset:
    """Create a mock binary classification dataset with explicit labels (10 samples)."""
    return MockClassificationDataset(task_type=Task.BINARY.value)


@pytest.fixture
def mock_multiclass_classification_dataset() -> MockClassificationDataset:
    """Create a mock multiclass classification dataset with explicit labels (10 samples)."""
    return MockClassificationDataset(task_type=Task.MULTICLASS.value)


@pytest.fixture
def mock_multilabel_classification_dataset() -> MockClassificationDataset:
    """Create a mock multilabel classification dataset with explicit labels (10 samples)."""
    return MockClassificationDataset(task_type=Task.MULTILABEL.value)


@pytest.fixture
def doleus_binary_classification_dataset(
    mock_binary_classification_dataset,
) -> DoleusClassification:
    """Create a DoleusClassification dataset for binary classification (10 samples)."""
    unique_dataset_name = f"binary_test_{uuid.uuid4().hex[:8]}"

    return DoleusClassification(
        dataset=mock_binary_classification_dataset,
        name=unique_dataset_name,
        task=Task.BINARY.value,
        num_classes=2,
        label_to_name={0: "negative", 1: "positive"},
    )


@pytest.fixture
def doleus_multiclass_classification_dataset(
    mock_multiclass_classification_dataset,
) -> DoleusClassification:
    """Create a DoleusClassification dataset for multiclass classification (10 samples)."""
    unique_dataset_name = f"multiclass_test_{uuid.uuid4().hex[:8]}"

    return DoleusClassification(
        dataset=mock_multiclass_classification_dataset,
        name=unique_dataset_name,
        task=Task.MULTICLASS.value,
        num_classes=3,
        label_to_name={0: "class_0", 1: "class_1", 2: "class_2"},
    )


@pytest.fixture
def doleus_multilabel_classification_dataset(
    mock_multilabel_classification_dataset,
) -> DoleusClassification:
    """Create a DoleusClassification dataset for multilabel classification (10 samples)."""
    unique_dataset_name = f"multilabel_test_{uuid.uuid4().hex[:8]}"

    return DoleusClassification(
        dataset=mock_multilabel_classification_dataset,
        name=unique_dataset_name,
        task=Task.MULTILABEL.value,
        num_classes=3,
        label_to_name={0: "label_0", 1: "label_1", 2: "label_2"},
    )


# Detection Fixtures
@pytest.fixture
def mock_object_detection_dataset() -> MockDetectionDataset:
    """Create a mock object detection dataset with explicit scenarios (10 samples)."""
    return MockDetectionDataset()


@pytest.fixture
def doleus_object_detection_dataset(mock_object_detection_dataset) -> DoleusDetection:
    """Create a DoleusDetection dataset for object detection (10 samples)."""
    unique_dataset_name = f"detection_test_{uuid.uuid4().hex[:8]}"

    return DoleusDetection(
        dataset=mock_object_detection_dataset,
        name=unique_dataset_name,
        label_to_name={0: "person", 1: "car", 2: "bicycle"},
    )


# Metadata Fixtures
@pytest.fixture
def basic_metadata() -> List[Dict]:
    """String and boolean metadata for basic testing."""
    return TestMetadata.BASIC_METADATA.copy()


@pytest.fixture
def numeric_metadata() -> List[Dict]:
    """Float and integer metadata for numeric slicing tests."""
    return TestMetadata.NUMERIC_METADATA.copy()


@pytest.fixture
def mixed_metadata() -> List[Dict]:
    """Mixed data types (string, float, int, bool) for comprehensive testing."""
    return TestMetadata.MIXED_METADATA.copy()


@pytest.fixture
def string_numeric_metadata() -> List[Dict]:
    """String and numeric metadata for testing."""
    return TestMetadata.STRING_NUMERIC_METADATA.copy()


@pytest.fixture
def metadata_dataframe(
    mixed_metadata: List[Dict],
) -> pd.DataFrame:  # Use existing metadata
    """DataFrame from MIXED_METADATA for testing pandas integration."""
    return pd.DataFrame(mixed_metadata)
    # Slicing assertions for this DataFrame should match MIXED_METADATA:
    # environment="lab": indices 0,2,4,6,8 (5 samples)
    # environment="field": indices 1,3,5,7,9 (5 samples)
    # temperature >= 20.0: indices 0,2,4,6,8,9 (6 samples)
    # temperature < 20.0: indices 1,3,5,7 (4 samples)
    # sample_count >= 100: indices 0,2,4,8 (4 samples)
    # sample_count < 100: indices 1,3,5,6,7,9 (6 samples)
    # corrupted=True: indices 1,4,7 (3 samples)
    # corrupted=False: indices 0,2,3,5,6,8,9 (7 samples)
