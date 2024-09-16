from unittest.mock import Mock

import pytest
import torch
from doleus.annotations import Annotations
from doleus.annotations.detection import BoundingBoxes
from doleus.storage.ground_truth_store.detection import DetectionGroundTruthStore


class TestDetectionGroundTruthStore:
    """Test suite for DetectionGroundTruthStore."""

    def test_initialization_with_valid_dataset(self, mock_object_detection_dataset):
        """Test initialization with a valid detection dataset."""
        # Arrange & Act
        store = DetectionGroundTruthStore(dataset=mock_object_detection_dataset)

        # Assert
        assert store.dataset is mock_object_detection_dataset
        assert store.groundtruths is not None
        assert isinstance(store.groundtruths, Annotations)

    def test_process_groundtruths_valid_data(self, mock_object_detection_dataset):
        """Test processing ground truths with valid detection data."""
        # Arrange & Act
        store = DetectionGroundTruthStore(dataset=mock_object_detection_dataset)

        # Assert
        assert len(store.groundtruths.annotations) == len(mock_object_detection_dataset)

        # Check first few annotations
        for i in range(min(3, len(mock_object_detection_dataset))):
            annotation = store.get(i)
            assert isinstance(annotation, BoundingBoxes)
            assert annotation.datapoint_number == i
            assert annotation.boxes_xyxy.dtype == torch.float32
            assert annotation.labels.dtype == torch.long
            assert annotation.scores is None  # Ground truth should have no scores

            # Validate bounding box format (x1, y1, x2, y2)
            assert annotation.boxes_xyxy.shape[1] == 4
            assert (
                annotation.boxes_xyxy.shape[0] == annotation.labels.shape[0]
            )  # Same number of boxes and labels

    def test_get_valid_datapoint_numbers(self, mock_object_detection_dataset):
        """Test get method with valid datapoint numbers."""
        # Arrange
        store = DetectionGroundTruthStore(dataset=mock_object_detection_dataset)

        # Act & Assert
        for i in range(len(mock_object_detection_dataset)):
            annotation = store.get(i)
            assert annotation is not None
            assert annotation.datapoint_number == i
            assert isinstance(annotation, BoundingBoxes)

    def test_get_invalid_datapoint_number(self, mock_object_detection_dataset):
        """Test get method with invalid datapoint number."""
        # Arrange
        store = DetectionGroundTruthStore(dataset=mock_object_detection_dataset)

        # Act
        annotation = store.get(len(mock_object_detection_dataset) + 5)

        # Assert
        assert annotation is None

    def test_bounding_box_shapes_and_types(self, mock_object_detection_dataset):
        """Test that bounding boxes have correct shapes and types."""
        # Arrange
        store = DetectionGroundTruthStore(dataset=mock_object_detection_dataset)

        # Act & Assert
        for i in range(min(3, len(mock_object_detection_dataset))):
            annotation = store.get(i)

            # Check shapes
            num_detections = annotation.boxes_xyxy.shape[0]
            assert annotation.boxes_xyxy.shape == (num_detections, 4)
            assert annotation.labels.shape == (num_detections,)

            # Check types
            assert annotation.boxes_xyxy.dtype == torch.float32
            assert annotation.labels.dtype == torch.long

            # Check that boxes are valid (x1 <= x2, y1 <= y2)
            boxes = annotation.boxes_xyxy
            assert torch.all(boxes[:, 0] <= boxes[:, 2])  # x1 <= x2
            assert torch.all(boxes[:, 1] <= boxes[:, 3])  # y1 <= y2


class TestDetectionGroundTruthStoreErrorHandling:
    """Test suite for error handling in DetectionGroundTruthStore."""

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
            DetectionGroundTruthStore(dataset=mock_dataset)

    def test_wrong_number_of_elements_raises_error(self):
        """Test that wrong number of elements in dataset item raises ValueError."""
        # Arrange - Create dataset with wrong number of elements
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=1)
        mock_dataset.__getitem__ = Mock(
            return_value=(torch.rand(3, 416, 416), torch.tensor([[10, 10, 100, 100]]))
        )  # Missing labels

        # Act & Assert
        with pytest.raises(
            ValueError, match="Dataset item at index 0 is not in the expected format"
        ):
            DetectionGroundTruthStore(dataset=mock_dataset)

    def test_invalid_bounding_box_shape_raises_error(self):
        """Test that invalid bounding box shapes raise ValueError."""
        # Arrange - Create dataset with wrong bounding box shape
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=1)
        mock_dataset.__getitem__ = Mock(
            return_value=(
                torch.rand(3, 416, 416),
                torch.tensor([[10, 10, 100]]),  # Wrong shape: should be (M, 4)
                torch.tensor([0]),
            )
        )

        # Act & Assert
        with pytest.raises(
            ValueError, match="Bounding boxes for item 0 must have shape \\(M, 4\\)"
        ):
            DetectionGroundTruthStore(dataset=mock_dataset)

    def test_mismatched_boxes_labels_shapes_raises_error(self):
        """Test that mismatched bounding boxes and labels shapes raise ValueError."""
        # Arrange - Create dataset with mismatched shapes
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=1)
        mock_dataset.__getitem__ = Mock(
            return_value=(
                torch.rand(3, 416, 416),
                torch.tensor([[10, 10, 100, 100], [20, 20, 200, 200]]),  # 2 boxes
                torch.tensor([0]),  # 1 label - mismatch!
            )
        )

        # Act & Assert
        with pytest.raises(
            ValueError, match="Labels for item 0 must have shape \\(M,\\)"
        ):
            DetectionGroundTruthStore(dataset=mock_dataset)

    def test_invalid_bounding_box_conversion_raises_error(self):
        """Test that invalid bounding box data that can't be converted raises ValueError."""
        # Arrange - Create dataset with unconvertible bounding box data
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=1)
        mock_dataset.__getitem__ = Mock(
            return_value=(
                torch.rand(3, 416, 416),
                "invalid_boxes",  # Can't convert to tensor
                torch.tensor([0]),
            )
        )

        # Act & Assert
        with pytest.raises(
            ValueError, match="Could not convert bounding_boxes for item 0 to tensor"
        ):
            DetectionGroundTruthStore(dataset=mock_dataset)

    def test_invalid_labels_conversion_raises_error(self):
        """Test that invalid labels data that can't be converted raises ValueError."""
        # Arrange - Create dataset with unconvertible labels data
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=1)
        mock_dataset.__getitem__ = Mock(
            return_value=(
                torch.rand(3, 416, 416),
                torch.tensor([[10, 10, 100, 100]]),
                "invalid_labels",  # Can't convert to tensor
            )
        )

        # Act & Assert
        with pytest.raises(
            ValueError, match="Could not convert labels for item 0 to tensor"
        ):
            DetectionGroundTruthStore(dataset=mock_dataset)

    def test_wrong_labels_dimensions_raises_error(self):
        """Test that wrong labels dimensions raise ValueError."""
        # Arrange - Create dataset with wrong labels dimensions
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=1)
        mock_dataset.__getitem__ = Mock(
            return_value=(
                torch.rand(3, 416, 416),
                torch.tensor([[10, 10, 100, 100]]),
                torch.tensor([[0, 1]]),  # Wrong dimensions: should be 1D
            )
        )

        # Act & Assert
        with pytest.raises(
            ValueError, match="Labels for item 0 must have shape \\(M,\\)"
        ):
            DetectionGroundTruthStore(dataset=mock_dataset)


class TestDetectionGroundTruthStoreDataTypeHandling:
    """Test suite for data type handling in DetectionGroundTruthStore."""

    def test_converts_list_bounding_boxes_to_tensor(self):
        """Test that list bounding boxes are converted to tensors."""
        # Arrange - Create dataset with list bounding boxes
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=1)
        mock_dataset.__getitem__ = Mock(
            return_value=(
                torch.rand(3, 416, 416),
                [[10, 10, 100, 100], [20, 20, 200, 200]],  # List format
                [0, 1],  # List format
            )
        )

        # Act
        store = DetectionGroundTruthStore(dataset=mock_dataset)
        annotation = store.get(0)

        # Assert
        assert isinstance(annotation.boxes_xyxy, torch.Tensor)
        assert annotation.boxes_xyxy.dtype == torch.float32
        assert annotation.boxes_xyxy.shape == (2, 4)
        assert isinstance(annotation.labels, torch.Tensor)
        assert annotation.labels.dtype == torch.long

    def test_converts_numpy_arrays_to_tensors(self):
        """Test that numpy arrays are converted to tensors."""
        import numpy as np

        # Arrange - Create dataset with numpy arrays
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=1)
        mock_dataset.__getitem__ = Mock(
            return_value=(
                torch.rand(3, 416, 416),
                np.array([[10, 10, 100, 100]], dtype=np.float32),
                np.array([0], dtype=np.int64),
            )
        )

        # Act
        store = DetectionGroundTruthStore(dataset=mock_dataset)
        annotation = store.get(0)

        # Assert
        assert isinstance(annotation.boxes_xyxy, torch.Tensor)
        assert annotation.boxes_xyxy.dtype == torch.float32
        assert isinstance(annotation.labels, torch.Tensor)
        assert annotation.labels.dtype == torch.long

    def test_ensures_correct_tensor_dtypes(self):
        """Test that tensor dtypes are corrected if needed."""
        # Arrange - Create dataset with wrong dtypes
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=1)
        mock_dataset.__getitem__ = Mock(
            return_value=(
                torch.rand(3, 416, 416),
                torch.tensor([[10, 10, 100, 100]], dtype=torch.int32),  # Wrong dtype
                torch.tensor([0], dtype=torch.float32),  # Wrong dtype
            )
        )

        # Act
        store = DetectionGroundTruthStore(dataset=mock_dataset)
        annotation = store.get(0)

        # Assert
        assert annotation.boxes_xyxy.dtype == torch.float32
        assert annotation.labels.dtype == torch.long


class TestDetectionGroundTruthStoreIntegration:
    """Integration tests for DetectionGroundTruthStore with Doleus datasets."""

    def test_integration_with_doleus_detection_dataset(
        self, doleus_object_detection_dataset
    ):
        """Test integration with DoleusDetection dataset."""
        # Arrange
        doleus_dataset = doleus_object_detection_dataset

        # Act
        ground_truth_store = doleus_dataset.groundtruth_store

        # Assert
        assert isinstance(ground_truth_store, DetectionGroundTruthStore)
        assert ground_truth_store.dataset is doleus_dataset.dataset

        # Test that all annotations are properly processed
        for i in range(min(5, len(doleus_dataset))):  # Test first 5
            annotation = ground_truth_store.get(i)
            assert isinstance(annotation, BoundingBoxes)
            assert annotation.datapoint_number == i
            assert annotation.boxes_xyxy.dtype == torch.float32
            assert annotation.labels.dtype == torch.long
            assert annotation.scores is None

            # Validate that we have valid detections
            num_detections = annotation.boxes_xyxy.shape[0]
            assert num_detections > 0  # Should have at least one detection
            assert annotation.labels.shape[0] == num_detections

    def test_multiple_detections_per_image(self, doleus_object_detection_dataset):
        """Test handling of multiple detections per image."""
        # Arrange
        doleus_dataset = doleus_object_detection_dataset
        ground_truth_store = doleus_dataset.groundtruth_store

        # Act & Assert - Check that some images have multiple detections
        found_multiple_detections = False
        for i in range(len(doleus_dataset)):
            annotation = ground_truth_store.get(i)
            num_detections = annotation.boxes_xyxy.shape[0]
            if num_detections > 1:
                found_multiple_detections = True
                # Verify all detections are valid
                assert annotation.labels.shape[0] == num_detections
                assert torch.all(
                    annotation.boxes_xyxy[:, 0] <= annotation.boxes_xyxy[:, 2]
                )  # x1 <= x2
                assert torch.all(
                    annotation.boxes_xyxy[:, 1] <= annotation.boxes_xyxy[:, 3]
                )  # y1 <= y2
                break

        # Our mock dataset should have some images with multiple detections
        assert (
            found_multiple_detections
        ), "Expected to find at least one image with multiple detections"
