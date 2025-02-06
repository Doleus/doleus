import uuid

import torch
import pytest
import numpy as np
from torch.utils.data import Dataset

from moonwatcher.metric import get_original_indices
from moonwatcher.dataset.dataset import Moonwatcher, MoonwatcherClassification, MoonwatcherDetection, Slice

class DetectionMockDataset(Dataset):
    pass


def output_transform(datapoint):
    image, labels = datapoint

    # If labels is a single integer, which can happen in binary classification, convert it to a 1-dimensional tensor
    if isinstance(labels, int):
        labels = torch.tensor(labels)
        # Add an extra dimension at position 0 to make it a 1-dimensional tensor
        labels = torch.tensor([labels])
        print(f"labels shape: {labels.shape}")
    return image, labels


@pytest.fixture
def moonwatcher_classification_dataset(create_classification_mock_dataset) -> 'MoonwatcherClassification':
    from moonwatcher.dataset.dataset import MoonwatcherClassification
    import uuid
    
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


def test_initialization(moonwatcher_classification_dataset):
    assert moonwatcher_classification_dataset.name.startswith(
        "test_dataset_")
    assert len(moonwatcher_classification_dataset) == 100
    assert isinstance(moonwatcher_classification_dataset,
                      MoonwatcherClassification)


def test_add_groundtruths_from_dataset(moonwatcher_classification_dataset):
    moonwatcher_classification_dataset.add_groundtruths_from_dataset()
    assert len(moonwatcher_classification_dataset.groundtruths) == 100


def test_add_metadata_from_list(moonwatcher_classification_dataset):
    metadata_list = [{'key1': 'value1'}, {'key2': 'value2'}]
    moonwatcher_classification_dataset.add_metadata_from_list(metadata_list)
    assert 'key1' in moonwatcher_classification_dataset.datapoints[0].metadata
    assert 'key2' in moonwatcher_classification_dataset.datapoints[1].metadata


def test_slice_by_threshold(moonwatcher_classification_dataset):
    moonwatcher_classification_dataset.add_metadata_from_list(
        [{'brightness': 0.5} for _ in range(100)])
    slice_dataset = moonwatcher_classification_dataset.slice_by_threshold(
        "brightness", ">", 0.4)
    assert len(slice_dataset) == 100


def test_slice_by_metadata_value(moonwatcher_classification_dataset):
    moonwatcher_classification_dataset.add_metadata_from_list(
        [{'class': 'A'} for _ in range(50)] + [{'class': 'B'} for _ in range(50)])
    slice_dataset = moonwatcher_classification_dataset.slice_by_metadata_value(
        "class", "A")
    assert len(slice_dataset) == 50

def test_detection_initialization(moonwatcher_detection_dataset):
    assert moonwatcher_detection_dataset.name.startswith(
        "test_detection_dataset_")
    assert len(moonwatcher_detection_dataset) == 100
    assert isinstance(moonwatcher_detection_dataset,
                      MoonwatcherDetection)


def test_add_groundtruths_from_detection_dataset(moonwatcher_detection_dataset):
    moonwatcher_detection_dataset.add_groundtruths_from_dataset()
    assert len(moonwatcher_detection_dataset.groundtruths) == 100


def test_slice_by_groundtruth_class(moonwatcher_detection_dataset):
    slice_dataset = moonwatcher_detection_dataset.slice_by_groundtruth_class(
        class_ids=[0])
    assert len(slice_dataset) > 0  # Ensure some data points are sliced
