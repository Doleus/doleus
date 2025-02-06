import pytest
import torch
import numpy as np
from torch.utils.data import Dataset
from moonwatcher.dataset.dataset import MoonwatcherClassification, MoonwatcherDetection
import uuid

class ClassificationMockDataset(Dataset):
    def __len__(self) -> int:
        return 100

    def __getitem__(self, idx: int):
        return np.random.rand(3, 256, 256), idx % 5

@pytest.fixture
def create_classification_mock_dataset() -> ClassificationMockDataset:
    return ClassificationMockDataset()

@pytest.fixture
def moonwatcher_classification_dataset(create_classification_mock_dataset) -> 'MoonwatcherClassification':
    classification_mock_dataset = create_classification_mock_dataset
    unique_name = f"test_dataset_{uuid.uuid4()}"
    num_classes = 5  # Randomly chosen
    num_datapoints = len(classification_mock_dataset)
    predictions = torch.randint(
        low=0, high=num_classes, size=(num_datapoints,)) #Predictions are random integers that represent class numbers

    return MoonwatcherClassification(
        dataset=classification_mock_dataset,
        name=unique_name,
        task="binary",
        num_classes=num_classes,
    )

class DetectionMockDataset(Dataset):
    def __len__(self) -> int:
        return 100

    def __getitem__(self, idx: int):
        # Return a dummy image, bounding boxes, and labels
        dummy_image = np.random.rand(3, 256, 256)
        bounding_boxes = np.array([[10, 10, 50, 50], [60, 60, 100, 100]])
        labels = np.array([0, 1])
        return dummy_image, bounding_boxes, labels

@pytest.fixture
def create_detection_mock_dataset() -> DetectionMockDataset:
    return DetectionMockDataset()

@pytest.fixture
def moonwatcher_detection_dataset(create_detection_mock_dataset) -> 'MoonwatcherDetection':
    detection_mock_dataset = create_detection_mock_dataset
    unique_name = f"test_detection_dataset_{uuid.uuid4()}"
    num_classes = 2  # Randomly chosen

    return MoonwatcherDetection(
        dataset=detection_mock_dataset,
        name=unique_name,
        task="detection",
        num_classes=num_classes,
    ) 