# SPDX-FileCopyrightText: 2025 Doleus contributors
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
import torch
from doleus.annotations import Annotations
from doleus.annotations.classification import Labels
from doleus.storage.ground_truth_store.classification import (
    ClassificationGroundTruthStore,
)
from doleus.utils import Task


class TestClassificationGroundTruthStore:
    """Test suite for ClassificationGroundTruthStore."""

    def test_initialization_binary_classification(
        self, mock_binary_classification_dataset
    ):
        """Test initialization for binary classification task."""
        # Arrange & Act
        store = ClassificationGroundTruthStore(
            dataset=mock_binary_classification_dataset,
            task=Task.BINARY.value,
            num_classes=2,
        )

        # Assert
        assert store.dataset is mock_binary_classification_dataset
        assert store.task == Task.BINARY.value
        assert store.num_classes == 2
        assert store.groundtruths is not None
        assert isinstance(store.groundtruths, Annotations)

    def test_initialization_multiclass_classification(
        self, mock_multiclass_classification_dataset
    ):
        """Test initialization for multiclass classification task."""
        # Arrange & Act
        store = ClassificationGroundTruthStore(
            dataset=mock_multiclass_classification_dataset,
            task=Task.MULTICLASS.value,
            num_classes=3,
        )

        # Assert
        assert store.dataset is mock_multiclass_classification_dataset
        assert store.task == Task.MULTICLASS.value
        assert store.num_classes == 3
        assert store.groundtruths is not None

    def test_initialization_multilabel_classification(
        self, mock_multilabel_classification_dataset
    ):
        """Test initialization for multilabel classification task."""
        # Arrange & Act
        store = ClassificationGroundTruthStore(
            dataset=mock_multilabel_classification_dataset,
            task=Task.MULTILABEL.value,
            num_classes=3,
        )

        # Assert
        assert store.dataset is mock_multilabel_classification_dataset
        assert store.task == Task.MULTILABEL.value
        assert store.num_classes == 3
        assert store.groundtruths is not None

    def test_process_groundtruths_binary_valid_labels(
        self, mock_binary_classification_dataset
    ):
        """Test processing ground truths for binary classification with valid labels."""
        # Arrange & Act
        store = ClassificationGroundTruthStore(
            dataset=mock_binary_classification_dataset,
            task=Task.BINARY.value,
            num_classes=2,
        )

        # Assert
        assert len(store.groundtruths.annotations) == len(
            mock_binary_classification_dataset
        )

        # Check first few annotations
        for i in range(min(3, len(mock_binary_classification_dataset))):
            annotation = store.get(i)
            assert isinstance(annotation, Labels)
            assert annotation.datapoint_number == i
            assert annotation.labels.shape == (1,)  # Binary should be [1] shape
            assert annotation.labels.dtype == torch.long
            assert annotation.labels.item() in [0, 1]
            assert annotation.scores is None  # Ground truth should have no scores

    def test_process_groundtruths_multiclass_valid_labels(
        self, mock_multiclass_classification_dataset
    ):
        """Test processing ground truths for multiclass classification with valid labels."""
        # Arrange & Act
        store = ClassificationGroundTruthStore(
            dataset=mock_multiclass_classification_dataset,
            task=Task.MULTICLASS.value,
            num_classes=3,
        )

        # Assert
        assert len(store.groundtruths.annotations) == len(
            mock_multiclass_classification_dataset
        )

        # Check first few annotations
        for i in range(min(3, len(mock_multiclass_classification_dataset))):
            annotation = store.get(i)
            assert isinstance(annotation, Labels)
            assert annotation.datapoint_number == i
            assert annotation.labels.shape == (1,)  # Multiclass should be [1] shape
            assert annotation.labels.dtype == torch.long
            assert 0 <= annotation.labels.item() < 3  # Should be within num_classes
            assert annotation.scores is None

    def test_process_groundtruths_multilabel_valid_labels(
        self, mock_multilabel_classification_dataset
    ):
        """Test processing ground truths for multilabel classification with valid labels."""
        # Arrange & Act
        store = ClassificationGroundTruthStore(
            dataset=mock_multilabel_classification_dataset,
            task=Task.MULTILABEL.value,
            num_classes=3,
        )

        # Assert
        assert len(store.groundtruths.annotations) == len(
            mock_multilabel_classification_dataset
        )

        # Check first few annotations
        for i in range(min(3, len(mock_multilabel_classification_dataset))):
            annotation = store.get(i)
            assert isinstance(annotation, Labels)
            assert annotation.datapoint_number == i
            assert annotation.labels.shape == (
                3,
            )  # Multilabel should be [num_classes] shape
            assert annotation.labels.dtype == torch.long
            assert torch.all(
                (annotation.labels == 0) | (annotation.labels == 1)
            )  # Multi-hot encoded
            assert annotation.scores is None

    def test_get_valid_datapoint_numbers(self, mock_binary_classification_dataset):
        """Test get method with valid datapoint numbers."""
        # Arrange
        store = ClassificationGroundTruthStore(
            dataset=mock_binary_classification_dataset,
            task=Task.BINARY.value,
            num_classes=2,
        )

        # Act & Assert
        for i in range(len(mock_binary_classification_dataset)):
            annotation = store.get(i)
            assert annotation is not None
            assert annotation.datapoint_number == i

    def test_get_invalid_datapoint_number(self, mock_binary_classification_dataset):
        """Test get method with invalid datapoint number."""
        # Arrange
        store = ClassificationGroundTruthStore(
            dataset=mock_binary_classification_dataset,
            task=Task.BINARY.value,
            num_classes=2,
        )

        # Act
        annotation = store.get(len(mock_binary_classification_dataset) + 5)

        # Assert
        assert annotation is None

    def test_unsupported_task_raises_error(self, mock_binary_classification_dataset):
        """Test that unsupported task raises ValueError."""
        # Arrange & Act & Assert
        with pytest.raises(
            ValueError, match="Unsupported task for ClassificationGroundTruthStore"
        ):
            ClassificationGroundTruthStore(
                dataset=mock_binary_classification_dataset,
                task="unsupported_task",
                num_classes=2,
            )


class TestClassificationGroundTruthStoreErrorHandling:
    """Test suite for error handling in ClassificationGroundTruthStore."""

    def test_invalid_dataset_format_raises_error(self):
        """Test that invalid dataset format raises ValueError."""
        # Arrange - Create a mock dataset that returns invalid format
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=1)
        mock_dataset.__getitem__ = Mock(
            return_value="invalid_format"
        )  # Not a tuple/list

        # Act & Assert
        with pytest.raises(
            ValueError, match="Dataset item at index 0 is not in the expected format"
        ):
            ClassificationGroundTruthStore(
                dataset=mock_dataset, task=Task.BINARY.value, num_classes=2
            )

    def test_binary_invalid_label_value_raises_error(self):
        """Test that invalid binary label values raise ValueError."""
        # Arrange - Create dataset with invalid binary labels
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=1)
        mock_dataset.__getitem__ = Mock(
            return_value=(torch.rand(3, 224, 224), torch.tensor(2))
        )  # Invalid: should be 0 or 1

        # Act & Assert
        with pytest.raises(
            ValueError, match="Binary ground truth for item 0 must be 0 or 1"
        ):
            ClassificationGroundTruthStore(
                dataset=mock_dataset, task=Task.BINARY.value, num_classes=2
            )

    def test_multiclass_out_of_range_label_raises_error(self):
        """Test that out-of-range multiclass labels raise ValueError."""
        # Arrange - Create dataset with out-of-range labels
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=1)
        mock_dataset.__getitem__ = Mock(
            return_value=(torch.rand(3, 224, 224), torch.tensor(3))
        )  # Invalid: should be 0-2

        # Act & Assert
        with pytest.raises(
            ValueError,
            match="Multiclass ground truth for item 0 must be between 0 and 2",
        ):
            ClassificationGroundTruthStore(
                dataset=mock_dataset, task=Task.MULTICLASS.value, num_classes=3
            )

    def test_multilabel_wrong_shape_raises_error(self):
        """Test that wrong shape multilabel tensors raise ValueError."""
        # Arrange - Create dataset with wrong shape multilabel
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=1)
        mock_dataset.__getitem__ = Mock(
            return_value=(torch.rand(3, 224, 224), torch.tensor([1, 0]))
        )  # Wrong shape: should be [3]

        # Act & Assert
        with pytest.raises(
            ValueError,
            match="Multilabel ground truth tensor shape for item 0 must be \\(3,\\)",
        ):
            ClassificationGroundTruthStore(
                dataset=mock_dataset, task=Task.MULTILABEL.value, num_classes=3
            )

    def test_multilabel_invalid_values_raises_error(self):
        """Test that multilabel tensors with invalid values raise ValueError."""
        # Arrange - Create dataset with invalid multilabel values
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=1)
        mock_dataset.__getitem__ = Mock(
            return_value=(torch.rand(3, 224, 224), torch.tensor([1, 0, 2]))
        )  # Invalid: contains 2

        # Act & Assert
        with pytest.raises(
            ValueError,
            match="Multilabel ground truth tensor for item 0 must be multi-hot encoded",
        ):
            ClassificationGroundTruthStore(
                dataset=mock_dataset, task=Task.MULTILABEL.value, num_classes=3
            )

    def test_multilabel_wrong_dtype_raises_error(self):
        """Test that multilabel tensors with wrong dtype raise ValueError."""
        # Arrange - Create dataset with wrong dtype multilabel
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=1)
        mock_dataset.__getitem__ = Mock(
            return_value=(torch.rand(3, 224, 224), torch.tensor([1.0, 0.0, 1.0]))
        )  # Wrong dtype: float

        # Act & Assert
        with pytest.raises(
            ValueError,
            match="Multilabel ground truth tensor for item 0 must be of integer type",
        ):
            ClassificationGroundTruthStore(
                dataset=mock_dataset, task=Task.MULTILABEL.value, num_classes=3
            )


class TestClassificationGroundTruthStoreIntegration:
    """Integration tests for ClassificationGroundTruthStore with Doleus datasets."""

    def test_integration_with_doleus_binary_dataset(
        self, doleus_binary_classification_dataset
    ):
        """Test integration with DoleusClassification binary dataset."""
        # Arrange
        doleus_dataset = doleus_binary_classification_dataset

        # Act
        ground_truth_store = doleus_dataset.groundtruth_store

        # Assert
        assert isinstance(ground_truth_store, ClassificationGroundTruthStore)
        assert ground_truth_store.task == Task.BINARY.value
        assert ground_truth_store.num_classes == 2

        # Test that all annotations are properly processed
        for i in range(len(doleus_dataset)):
            annotation = ground_truth_store.get(i)
            assert isinstance(annotation, Labels)
            assert annotation.labels.item() in [0, 1]

    def test_integration_with_doleus_multiclass_dataset(
        self, doleus_multiclass_classification_dataset
    ):
        """Test integration with DoleusClassification multiclass dataset."""
        # Arrange
        doleus_dataset = doleus_multiclass_classification_dataset

        # Act
        ground_truth_store = doleus_dataset.groundtruth_store

        # Assert
        assert isinstance(ground_truth_store, ClassificationGroundTruthStore)
        assert ground_truth_store.task == Task.MULTICLASS.value
        assert ground_truth_store.num_classes == 3

        # Test that all annotations are properly processed
        for i in range(min(5, len(doleus_dataset))):  # Test first 5
            annotation = ground_truth_store.get(i)
            assert isinstance(annotation, Labels)
            assert 0 <= annotation.labels.item() < 3

    def test_integration_with_doleus_multilabel_dataset(
        self, doleus_multilabel_classification_dataset
    ):
        """Test integration with DoleusClassification multilabel dataset."""
        # Arrange
        doleus_dataset = doleus_multilabel_classification_dataset

        # Act
        ground_truth_store = doleus_dataset.groundtruth_store

        # Assert
        assert isinstance(ground_truth_store, ClassificationGroundTruthStore)
        assert ground_truth_store.task == Task.MULTILABEL.value
        assert ground_truth_store.num_classes == 3

        # Test that all annotations are properly processed
        for i in range(min(3, len(doleus_dataset))):  # Test first 3
            annotation = ground_truth_store.get(i)
            assert isinstance(annotation, Labels)
            assert annotation.labels.shape == (3,)
            assert torch.all((annotation.labels == 0) | (annotation.labels == 1))
