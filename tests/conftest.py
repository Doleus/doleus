import pytest
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from moonwatcher.dataset.dataset import MoonwatcherClassification, MoonwatcherDetection
import uuid

# Classification Dataset Fixtures
class ClassificationMockDataset(Dataset):
    """Mock dataset for classification testing."""
    def __len__(self) -> int:
        return 100

    def __getitem__(self, idx: int):
        return np.random.rand(3, 256, 256), idx % 5

@pytest.fixture
def create_classification_mock_dataset() -> ClassificationMockDataset:
    """Create a basic classification mock dataset."""
    return ClassificationMockDataset()

@pytest.fixture
def moonwatcher_classification_dataset(create_classification_mock_dataset) -> MoonwatcherClassification:
    """Create a MoonwatcherClassification dataset with basic configuration."""
    classification_mock_dataset = create_classification_mock_dataset
    unique_name = f"test_dataset_{uuid.uuid4()}"
    num_classes = 2
    
    return MoonwatcherClassification(
        dataset=classification_mock_dataset,
        name=unique_name,
        task="binary",
        num_classes=num_classes,
    )

@pytest.fixture
def class_metadata_dataset() -> tuple[list[int], list[int], MoonwatcherClassification]:
    """
    Create a classification dataset with metadata and known class distributions.
    Returns:
        tuple containing (class_a_indices, class_b_indices, dataset)
    """
    num_samples = 100
    
    # Create deterministic class assignments
    all_indices = np.random.permutation(num_samples)
    class_a_indices = all_indices[:50].tolist()
    class_b_indices = all_indices[50:].tolist()
    
    # Create labels and metadata
    labels = [1 if i in class_a_indices else 0 for i in range(num_samples)]
    metadata = [{'class': 'A'} if i in class_a_indices else {'class': 'B'} 
                for i in range(num_samples)]
    
    class MockDataset(Dataset):
        def __init__(self, labels):
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return torch.rand(3, 224, 224), self.labels[idx]

    # Create dataset
    dataset = MoonwatcherClassification(
        dataset=MockDataset(labels),
        name="class_metadata_test",
        task="binary",
        num_classes=2
    )
    
    # Add metadata
    dataset.add_metadata_from_list(metadata)
    
    return class_a_indices, class_b_indices, dataset

# Detection Dataset Fixtures
class DetectionMockDataset(Dataset):
    """Mock dataset for object detection testing."""
    def __len__(self) -> int:
        return 100

    def __getitem__(self, idx: int):
        dummy_image = np.random.rand(3, 256, 256)
        bounding_boxes = torch.tensor([[10, 10, 50, 50], [60, 60, 100, 100]], dtype=torch.float32)
        labels = torch.tensor([0, 1], dtype=torch.long)
        return dummy_image, bounding_boxes, labels

@pytest.fixture
def create_detection_mock_dataset() -> DetectionMockDataset:
    """Create a basic detection mock dataset."""
    return DetectionMockDataset()

@pytest.fixture
def moonwatcher_detection_dataset(create_detection_mock_dataset) -> MoonwatcherDetection:
    """Create a MoonwatcherDetection dataset with basic configuration."""
    detection_mock_dataset = create_detection_mock_dataset
    unique_name = f"test_detection_dataset_{uuid.uuid4()}"
    num_classes = 2

    return MoonwatcherDetection(
        dataset=detection_mock_dataset,
        name=unique_name,
        task="detection",
        num_classes=num_classes,
    ) 