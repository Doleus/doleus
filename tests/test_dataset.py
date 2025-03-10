import random
import uuid

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from moonwatcher.check import Check
from moonwatcher.dataset.dataset import (MoonwatcherClassification,
                                         MoonwatcherDetection,
                                         get_original_indices)


@pytest.fixture
def class_metadata_dataset() -> tuple[list[int], list[int], MoonwatcherClassification]:
    """Create a dataset with known class distributions and metadata for testing.

    Returns:
        tuple containing:
            - class_a_indices: indices of samples in class A
            - class_b_indices: indices of samples in class B
            - dataset: MoonwatcherClassification dataset with metadata
    """
    num_samples = 100

    # Create deterministic class assignments
    all_indices = np.random.permutation(num_samples)
    class_a_indices = all_indices[:50].tolist()
    class_b_indices = all_indices[50:].tolist()

    # Create labels and metadata
    labels = [1 if i in class_a_indices else 0 for i in range(num_samples)]
    metadata = [
        {"class": "A"} if i in class_a_indices else {"class": "B"}
        for i in range(num_samples)
    ]

    # Create predictions with known error rates
    predictions = torch.zeros(num_samples, dtype=torch.long)
    predictions[class_a_indices] = 1  # Set class A predictions to 1

    # Introduce errors: 5 wrong in class A (90% accuracy) and 10 wrong in class B (80% accuracy)
    class_a_indices_wrong = random.sample(class_a_indices, 5)
    class_b_indices_wrong = random.sample(class_b_indices, 10)

    # Flip predictions for error cases
    predictions[class_a_indices_wrong] = 0
    predictions[class_b_indices_wrong] = 1

    # Create dataset with ground truths and metadata
    class MockDataset(Dataset):
        def __init__(self):
            self.labels = labels

        def __len__(self):
            return num_samples

        def __getitem__(self, idx):
            return torch.rand(3, 224, 224), self.labels[idx]

    # Create Moonwatcher dataset
    dataset = MoonwatcherClassification(
        dataset=MockDataset(), name="class_metadata_test", task="binary", num_classes=2
    )

    # Add metadata
    dataset.add_metadata_from_list(metadata)

    # Add predictions using the proper API
    dataset.add_predictions(predictions)

    return class_a_indices, class_b_indices, dataset


class TestClassification:
    """Tests for MoonwatcherClassification functionality."""

    def test_initialization(self, moonwatcher_classification_dataset):
        """Test basic initialization of classification dataset."""
        assert moonwatcher_classification_dataset.name.startswith("test_dataset_")
        assert len(moonwatcher_classification_dataset) == 100
        assert isinstance(moonwatcher_classification_dataset, MoonwatcherClassification)

    def test_add_groundtruths(self, moonwatcher_classification_dataset):
        """Test adding ground truths to classification dataset."""
        moonwatcher_classification_dataset.add_groundtruths()
        assert len(moonwatcher_classification_dataset.groundtruths) == 100

    def test_metadata_operations(self, moonwatcher_classification_dataset):
        """Test metadata addition and retrieval."""
        # Test adding metadata
        metadata_list = [{"key1": "value1"}, {"key2": "value2"}]
        moonwatcher_classification_dataset.add_metadata_from_list(metadata_list)
        assert "key1" in moonwatcher_classification_dataset.datapoints[0].metadata
        assert "key2" in moonwatcher_classification_dataset.datapoints[1].metadata

    def test_slicing_by_metadata(self, moonwatcher_classification_dataset):
        """Test slicing dataset by metadata values."""
        moonwatcher_classification_dataset.add_metadata_from_list(
            [{"class": "A"} for _ in range(50)] + [{"class": "B"} for _ in range(50)]
        )
        slice_dataset = moonwatcher_classification_dataset.slice_by_metadata_value(
            "class", "A"
        )
        assert len(slice_dataset) == 50

    def test_slicing_by_threshold(self, moonwatcher_classification_dataset):
        """Test slicing dataset by threshold values."""
        moonwatcher_classification_dataset.add_metadata_from_list(
            [{"brightness": 0.5} for _ in range(100)]
        )
        slice_dataset = moonwatcher_classification_dataset.slice_by_threshold(
            "brightness", ">", 0.4
        )
        assert len(slice_dataset) == 100

    def test_index_mapping(self, class_metadata_dataset):
        """Test correct index mapping through slicing."""
        class_a_indices, class_b_indices, dataset = class_metadata_dataset

        class_a_slice = dataset.slice_by_metadata_value("class", "A")
        class_b_slice = dataset.slice_by_metadata_value("class", "B")

        assert set(get_original_indices(class_a_slice)) == set(class_a_indices)
        assert set(get_original_indices(class_b_slice)) == set(class_b_indices)

    def test_accuracy_checks(self, class_metadata_dataset):
        """Test accuracy checks for different slices."""
        _, _, dataset = class_metadata_dataset

        # Create slices
        slice_a = dataset.slice_by_metadata_value("class", "A")
        slice_b = dataset.slice_by_metadata_value("class", "B")

        # Get raw predictions tensor
        raw_predictions = torch.zeros(len(dataset), dtype=torch.long)
        for i in range(len(dataset)):
            raw_predictions[i] = dataset.predictions[i].labels[0]

        # Test class A accuracy (expect 90%)
        check_a = Check(
            name="class_a_accuracy",
            dataset=slice_a,
            predictions=raw_predictions,
            metric="Accuracy",
            operator="==",
            value=0.90,
        )

        # Test class B accuracy (expect 80%)
        check_b = Check(
            name="class_b_accuracy",
            dataset=slice_b,
            predictions=raw_predictions,
            metric="Accuracy",
            operator="==",
            value=0.80,
        )

        # Run checks
        check_a_result = check_a.run()
        check_b_result = check_b.run()

        assert check_a_result["result"] == pytest.approx(
            0.90, abs=0.001
        ), f"Class A accuracy check failed: {check_a_result}"
        assert check_b_result["result"] == pytest.approx(
            0.80, abs=0.001
        ), f"Class B accuracy check failed: {check_b_result}"

        # Test overall accuracy (expect 85%)
        overall_check = Check(
            name="overall_accuracy",
            dataset=dataset,
            predictions=raw_predictions,
            metric="Accuracy",
            operator="==",
            value=0.85,
        )

        overall_result = overall_check.run()
        assert overall_result["result"] == pytest.approx(
            0.85, abs=0.001
        ), f"Overall accuracy check failed: {overall_result}"


class TestDetection:
    """Tests for MoonwatcherDetection functionality."""

    def test_initialization(self, moonwatcher_detection_dataset):
        """Test basic initialization of detection dataset."""
        assert moonwatcher_detection_dataset.name.startswith("test_detection_dataset_")
        assert len(moonwatcher_detection_dataset) == 100
        assert isinstance(moonwatcher_detection_dataset, MoonwatcherDetection)

    def test_add_groundtruths(self, moonwatcher_detection_dataset):
        """Test adding ground truths to detection dataset."""
        moonwatcher_detection_dataset.add_groundtruths()
        assert len(moonwatcher_detection_dataset.groundtruths) == 100

    def test_slice_by_groundtruth_class(self, moonwatcher_detection_dataset):
        """Test slicing by ground truth class."""
        moonwatcher_detection_dataset.add_groundtruths()
        slice_dataset = moonwatcher_detection_dataset.slice_by_groundtruth_class(
            class_ids=[0]
        )
        assert len(slice_dataset) > 0
