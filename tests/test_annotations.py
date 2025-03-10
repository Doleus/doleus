"""Changes made to tests/test_annotations.py:

1. Modified the test_labels_initialization function to support multi-class classification.
2. Updated the pytest.mark.parametrize decorator with new test cases:
   - Added a test case for multiple labels (tensor([1, 2, 3], dtype=torch.int32))
   - Added a test case for invalid 2D tensor input (tensor([[1, 2]], dtype=torch.int32))
3. Changed the assertion for label shape:
   - Replaced "assert label_obj.labels.shape == (1,)" with "assert label_obj.labels.dim() == 1"
   to ensure the labels are always a 1D tensor, regardless of the number of labels.

These changes allow the test to verify correct handling of both single-label and
multi-label classifications while maintaining the requirement for 1D integer tensors.""

"""

import pytest
import torch

from moonwatcher.annotations import (Annotations, BoundingBoxes, GroundTruths,
                                     Labels, PredictedBoundingBoxes,
                                     PredictedLabels, Predictions)


def tensor(data, dtype=torch.float32):
    return torch.tensor(data, dtype=dtype)


# Tests for BoundingBoxes
@pytest.mark.parametrize(
    "datapoint_id, boxes, labels, valid",
    [
        (1, tensor([[1, 2, 3, 4]]), tensor([1], dtype=torch.int32), True),
        # Invalid boxes
        (2, "not a tensor", tensor([1], dtype=torch.int32), False),
        (3, tensor([[1, 2, 3, 4]]), "not a tensor", False),  # Invalid labels
    ],
)
def test_bounding_boxes_initialization(datapoint_id, boxes, labels, valid):
    if valid:
        bbox = BoundingBoxes(datapoint_id, boxes, labels)
        assert bbox.datapoint_number == datapoint_id
        assert torch.equal(bbox.boxes_xyxy, boxes)
        assert torch.equal(bbox.labels, labels)
    else:
        with pytest.raises(TypeError):
            BoundingBoxes(datapoint_id, boxes, labels)


def test_bounding_boxes_validation():
    # Test invalid box dimensions
    with pytest.raises(ValueError):
        BoundingBoxes(
            1, tensor([1, 2, 3]), tensor([1], dtype=torch.int32)
        )  # Not Nx4 shape

    # Test invalid width
    with pytest.raises(ValueError):
        BoundingBoxes(
            1, tensor([[3, 0, 1, 1]]), tensor([1], dtype=torch.int32)
        )  # x2 < x1

    # Test invalid height
    with pytest.raises(ValueError):
        BoundingBoxes(
            1, tensor([[0, 3, 1, 1]]), tensor([1], dtype=torch.int32)
        )  # y2 < y1

    # Test mismatched number of boxes and labels
    with pytest.raises(ValueError):
        BoundingBoxes(
            1, tensor([[1, 2, 3, 4]]), tensor([1, 2], dtype=torch.int32)
        )  # More labels than boxes

    # Test coordinate range
    with pytest.raises(ValueError):
        # Negative coordinates
        BoundingBoxes(1, tensor([[-1, 0, 1, 1]]), tensor([1], dtype=torch.int32))

    # Test to_dict method
    boxes = tensor([[1, 2, 3, 4]])
    labels = tensor([1], dtype=torch.int32)
    bbox = BoundingBoxes(1, boxes, labels)
    dict_repr = bbox.to_dict()
    assert torch.equal(dict_repr["boxes"], boxes)
    assert torch.equal(dict_repr["labels"], labels)


# Tests for PredictedBoundingBoxes
@pytest.mark.parametrize(
    "datapoint_id, boxes, labels, scores, valid",
    [
        (
            1,
            tensor([[1, 2, 3, 4]]),
            tensor([1], dtype=torch.int32),
            tensor([0.99]),
            True,
        ),
        (
            2,
            tensor([[1, 2, 3, 4]]),
            tensor([1], dtype=torch.int32),
            0.99,
            False,
        ),  # Invalid scores
    ],
)
def test_predicted_bounding_boxes_initialization(
    datapoint_id, boxes, labels, scores, valid
):
    if valid:
        pred_bbox = PredictedBoundingBoxes(datapoint_id, boxes, labels, scores)
        assert pred_bbox.scores is not None
        assert torch.equal(pred_bbox.scores, scores)
    else:
        with pytest.raises(TypeError):
            PredictedBoundingBoxes(datapoint_id, boxes, labels, scores)


@pytest.mark.parametrize(
    "scores, expected_exception",
    [
        # Mismatched number of scores and boxes
        (tensor([0.9, 0.8]), ValueError),
        (tensor([[0.9]]), ValueError),  # Wrong shape (2D)
        (tensor([1.2]), ValueError),  # Score > 1.0
        (tensor([-0.1]), ValueError),  # Score < 0.0
    ],
)
def test_predicted_bounding_boxes_score_validation(scores, expected_exception):
    boxes = tensor([[1, 2, 3, 4]])
    labels = tensor([1], dtype=torch.int32)
    with pytest.raises(expected_exception):
        PredictedBoundingBoxes(1, boxes, labels, scores)


# Tests for Labels
@pytest.mark.parametrize(
    "datapoint_id, labels, expected_exception",
    [
        (1, tensor([1], dtype=torch.int32), None),  # Single label
        (2, tensor([1, 2, 3], dtype=torch.int32), None),  # Multiple labels
        (3, tensor([1.3]), TypeError),  # Non-integer labels
        (4, "not a tensor", TypeError),  # Non-tensor labels
        (5, tensor([[1, 2]], dtype=torch.int32), ValueError),  # 2D tensor (invalid)
        # Scalar tensor. Please note that we expect the labels to be a 1-dimensional tensor, i.e. (x,). If you pass a scalar to a dataset, it will be converted to a 1-dimensional tensor.
        (6, tensor(5, dtype=torch.int32), ValueError),
    ],
)
def test_labels_initialization(datapoint_id, labels, expected_exception):
    if expected_exception:
        with pytest.raises(expected_exception):
            Labels(datapoint_id, labels)
    else:
        label_obj = Labels(datapoint_id, labels)
        assert label_obj.labels.dim() == 1  # Ensure it's a 1D tensor
        assert label_obj.labels.dtype == torch.int32


@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32, torch.int64])
def test_labels_dtypes(dtype):
    labels = Labels(1, tensor([1], dtype=dtype))
    assert labels.labels.dtype == dtype


# Tests for PredictedLabels
@pytest.mark.parametrize(
    "datapoint_id, labels, scores, expected_exception",
    [
        (1, tensor([1], dtype=torch.int32), tensor([0.9]), None),  # Correct case
        (
            2,
            tensor([1], dtype=torch.int32),
            "not a tensor",
            TypeError,
        ),  # Non-tensor scores
        (
            3,
            tensor([1], dtype=torch.int32),
            tensor([0.9, 0.1]),
            ValueError,
        ),  # There are more scores than labels
        (4, "not a tensor", tensor([0.9]), TypeError),  # Non-tensor labels
        (
            5,
            tensor([1, 2], dtype=torch.int32),
            tensor([0.9]),
            ValueError,
        ),  # There are more labels than scores
    ],
)
def test_predicted_labels_initialization(
    datapoint_id, labels, scores, expected_exception
):
    if expected_exception:
        with pytest.raises(expected_exception):
            PredictedLabels(datapoint_id, labels, scores)
    else:
        predicted_label_obj = PredictedLabels(datapoint_id, labels, scores)
        assert predicted_label_obj.labels.shape == (1,)
        assert predicted_label_obj.labels.dtype == torch.int32
        assert predicted_label_obj.scores.shape == (1,)
        assert predicted_label_obj.scores.dtype == torch.float32


# Test for Annotations Class
def test_annotations():
    bbox = BoundingBoxes(1, tensor([[0, 1, 2, 3]]), tensor([1], dtype=torch.int32))
    annotations = Annotations([bbox])
    retrieved = annotations.get(1)
    assert retrieved == bbox
    assert len(annotations) == 1


# Test for annotation methods


def test_annotations_methods():
    bbox1 = BoundingBoxes(1, tensor([[0, 1, 2, 3]]), tensor([1], dtype=torch.int32))
    bbox2 = BoundingBoxes(2, tensor([[1, 2, 3, 4]]), tensor([2], dtype=torch.int32))
    bbox3 = BoundingBoxes(1, tensor([[1, 2, 3, 4]]), tensor([3], dtype=torch.int32))

    annotations = Annotations([bbox1])

    # Test add method
    annotations.add(bbox2)
    assert len(annotations) == 2

    # Test get_datapoint_ids
    assert sorted(annotations.get_datapoint_ids()) == [1, 2]

    # Test __getitem__
    assert annotations[2] == bbox2

    # Test error case
    with pytest.raises(KeyError):
        annotations.get(999)  # Non-existent datapoint

    # Test iteration
    assert list(annotations) == [bbox1, bbox2]

    # Test duplicate datapoint numbers
    annotations = Annotations([bbox1])
    with pytest.raises(KeyError):
        annotations.add(bbox3)

    # Test invalid annotation type
    with pytest.raises(TypeError):
        Annotations([{"invalid": "type"}])

    # Test invalid input types for constructor
    with pytest.raises(TypeError):
        Annotations("not a list")

    # Test adding invalid annotation type
    annotations = Annotations()
    with pytest.raises(TypeError):
        annotations.add("not an annotation")


# Test for Predictions and GroundTruths
def test_predictions_validation():
    class MockDataset:
        name = "test_dataset"

    dataset = MockDataset()
    bbox = BoundingBoxes(1, tensor([[0, 1, 2, 3]]), tensor([1], dtype=torch.int32))
    label = Labels(1, tensor([1], dtype=torch.int32))

    # Test that Predictions only accepts PredictedBoundingBoxes/PredictedLabels
    with pytest.raises(TypeError):
        Predictions(dataset, [bbox])  # Regular BoundingBoxes not allowed

    with pytest.raises(TypeError):
        Predictions(dataset, [label])  # Regular Labels not allowed


def test_groundtruths_validation():
    class MockDataset:
        name = "test_dataset"

    dataset = MockDataset()
    pred_bbox = PredictedBoundingBoxes(
        1, tensor([[0, 1, 2, 3]]), tensor([1], dtype=torch.int32), tensor([0.9])
    )
    pred_label = PredictedLabels(1, tensor([1], dtype=torch.int32), tensor([0.9]))
    # Test that GroundTruths only accepts BoundingBoxes/Labels
    with pytest.raises(TypeError):
        # PredictedBoundingBoxes not allowed
        GroundTruths(dataset, groundtruths=[pred_bbox])
    with pytest.raises(TypeError):
        # PredictedLabels not allowed
        GroundTruths(dataset, groundtruths=[pred_label])
