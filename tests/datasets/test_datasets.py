# SPDX-FileCopyrightText: 2025 Doleus contributors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import torch

from doleus.utils import Task, TaskType


def test_classification_initialization(
    doleus_binary_classification_dataset, doleus_multiclass_classification_dataset
):
    binary_dataset = doleus_binary_classification_dataset
    assert binary_dataset.task_type == TaskType.CLASSIFICATION.value
    assert binary_dataset.task == Task.BINARY.value
    assert binary_dataset.num_classes == 2
    assert binary_dataset.label_to_name == {0: "negative", 1: "positive"}
    assert len(binary_dataset) == 10
    assert len(binary_dataset.metadata_store.metadata) == 10
    assert binary_dataset.groundtruth_store is not None

    multiclass_dataset = doleus_multiclass_classification_dataset
    assert multiclass_dataset.task == Task.MULTICLASS.value
    assert multiclass_dataset.num_classes == 3
    assert multiclass_dataset.label_to_name == {
        0: "class_0",
        1: "class_1",
        2: "class_2",
    }
    assert len(multiclass_dataset) == 10


def test_detection_initialization(doleus_object_detection_dataset):
    dataset = doleus_object_detection_dataset
    assert dataset.task_type == TaskType.DETECTION.value
    assert dataset.label_to_name == {0: "person", 1: "car", 2: "bicycle"}
    assert len(dataset) == 10
    assert len(dataset.metadata_store.metadata) == 10
    assert dataset.groundtruth_store is not None


def test_adding_predictions(
    doleus_binary_classification_dataset, doleus_object_detection_dataset
):
    # Test classification predictions
    classification_dataset = doleus_binary_classification_dataset
    predictions = torch.sigmoid(torch.randn(10))
    classification_dataset.add_model_predictions(predictions, "test_model")

    assert "test_model" in classification_dataset.prediction_store.predictions
    assert (
        len(
            classification_dataset.prediction_store.predictions[
                "test_model"
            ].annotations
        )
        == 10
    )

    # Test detection predictions
    detection_dataset = doleus_object_detection_dataset
    detection_predictions = []
    for i in range(10):
        num_objects = (i % 3) + 1
        detection_predictions.append(
            {
                "boxes": torch.rand(num_objects, 4) * 100,
                "labels": torch.randint(0, 3, (num_objects,)),
                "scores": torch.rand(num_objects) * 0.8 + 0.2,
            }
        )

    detection_dataset.add_model_predictions(detection_predictions, "detection_model")
    assert "detection_model" in detection_dataset.prediction_store.predictions
    assert (
        len(
            detection_dataset.prediction_store.predictions[
                "detection_model"
            ].annotations
        )
        == 10
    )

    # Test multiple model predictions
    model1_preds = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    model2_preds = torch.tensor([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    model3_preds = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    classification_dataset.add_model_predictions(model1_preds, "model_a")
    classification_dataset.add_model_predictions(model2_preds, "model_b")
    classification_dataset.add_model_predictions(model3_preds, "model_c")

    # Verify all models are stored
    assert len(classification_dataset.prediction_store.predictions) == 4
    assert all(
        model in classification_dataset.prediction_store.predictions
        for model in ["test_model", "model_a", "model_b", "model_c"]
    )

    # Check that retrieved predictions match original ones
    for i in range(10):
        retrieved_a = classification_dataset.prediction_store.get_predictions(
            "model_a"
        )[i]
        retrieved_b = classification_dataset.prediction_store.get_predictions(
            "model_b"
        )[i]
        retrieved_c = classification_dataset.prediction_store.get_predictions(
            "model_c"
        )[i]

        assert retrieved_a.scores == model1_preds[i]
        assert retrieved_b.scores == model2_preds[i]
        assert retrieved_c.scores == model3_preds[i]


def test_adding_metadata(doleus_binary_classification_dataset):
    dataset = doleus_binary_classification_dataset

    # Test adding metadata from list
    metadata_list = [
        {"validated": i % 2 == 0, "source": f"camera_{i % 2}"} for i in range(10)
    ]
    dataset.add_metadata_from_list(metadata_list)
    assert "validated" in dataset.metadata_store.metadata[0]
    assert dataset.metadata_store.get_metadata(0, "validated") == True

    # Test adding predefined metadata
    dataset.add_predefined_metadata("brightness")
    assert "brightness" in dataset.metadata_store.metadata[0]

    # Test adding metadata from DataFrame
    df = pd.DataFrame(
        {
            "image_id": [f"img_{i}" for i in range(10)],
            "quality_score": np.random.rand(10),
        }
    )
    dataset.add_metadata_from_dataframe(df)
    assert "image_id" in dataset.metadata_store.metadata[0]
    assert "quality_score" in dataset.metadata_store.metadata[0]


def test_slicing_correctness(doleus_binary_classification_dataset, basic_metadata):
    dataset = doleus_binary_classification_dataset
    dataset.add_metadata_from_list(basic_metadata)

    # Add predictions
    predictions = torch.tensor([0.9, 0.3, 0.8, 0.2, 0.7, 0.1, 0.6, 0.4, 0.5, 0.0])
    dataset.add_model_predictions(predictions, "test_model")

    # Create slice
    sliced_dataset = dataset.slice_by_value("validated", "==", True, "validated_slice")

    # Verify basic slice properties
    assert sliced_dataset.name == "validated_slice"
    assert len(sliced_dataset) == 5
    assert sliced_dataset.task == Task.BINARY.value
    assert sliced_dataset.num_classes == 2

    # Verify metadata is correctly sliced
    for i in range(5):
        assert sliced_dataset.metadata_store.get_metadata(i, "validated") == True
        assert sliced_dataset.metadata_store.get_metadata(i, "source") == "camera_a"

    # Verify ground truth is correctly preserved
    original_indices_in_slice = [0, 2, 4, 6, 8]
    for i, original_idx in enumerate(original_indices_in_slice):
        sliced_gt = sliced_dataset.groundtruth_store.get(i)
        original_gt = dataset.groundtruth_store.get(original_idx)
        assert sliced_gt.labels == original_gt.labels

    # Verify predictions are correctly sliced and re-indexed
    assert "test_model" in sliced_dataset.prediction_store.predictions
    assert (
        len(sliced_dataset.prediction_store.predictions["test_model"].annotations) == 5
    )

    for i, original_idx in enumerate(original_indices_in_slice):
        sliced_pred = sliced_dataset.prediction_store.get_predictions("test_model")[i]
        original_pred = dataset.prediction_store.get_predictions("test_model")[
            original_idx
        ]
        assert abs(sliced_pred.scores - original_pred.scores) < 1e-6
        assert sliced_pred.datapoint_number == i


def test_chained_slicing(
    doleus_binary_classification_dataset, basic_metadata, numeric_metadata
):
    dataset = doleus_binary_classification_dataset
    dataset.add_metadata_from_list(basic_metadata)
    dataset.add_metadata_from_list(numeric_metadata)

    # First slice
    validated_slice = dataset.slice_by_value("validated", "==", True, "validated")
    assert len(validated_slice) == 5

    # Second slice
    high_conf_validated = validated_slice.slice_by_value(
        "confidence_score", ">=", 0.9, "high_conf_validated"
    )
    assert len(high_conf_validated) == 3
    assert high_conf_validated.name == "high_conf_validated"

    # Verify metadata is preserved
    for i in range(len(high_conf_validated)):
        assert high_conf_validated.metadata_store.get_metadata(i, "validated") == True
        assert (
            high_conf_validated.metadata_store.get_metadata(i, "confidence_score")
            >= 0.9
        )


def test_slice_and_operator(
    doleus_binary_classification_dataset, basic_metadata, numeric_metadata
):
    dataset = doleus_binary_classification_dataset
    dataset.add_metadata_from_list(basic_metadata)
    dataset.add_metadata_from_list(numeric_metadata)

    conditions = [("validated", "==", True), ("confidence_score", ">=", 0.9)]

    filtered_slice = dataset.slice_by_conditions(
        conditions, logical_operator="AND", slice_name="validated_high_conf"
    )

    assert len(filtered_slice) == 3
    assert filtered_slice.name == "validated_high_conf"

    for i in range(len(filtered_slice)):
        assert filtered_slice.metadata_store.get_metadata(i, "validated") == True
        assert filtered_slice.metadata_store.get_metadata(i, "confidence_score") >= 0.9


def test_slice_or_operator(
    doleus_binary_classification_dataset, basic_metadata, numeric_metadata
):
    dataset = doleus_binary_classification_dataset
    dataset.add_metadata_from_list(basic_metadata)
    dataset.add_metadata_from_list(numeric_metadata)

    conditions = [("batch_id", "==", 1), ("batch_id", "==", 2)]

    filtered_slice = dataset.slice_by_conditions(
        conditions, logical_operator="OR", slice_name="batch_1_or_2"
    )

    assert len(filtered_slice) == 6
    assert filtered_slice.name == "batch_1_or_2"

    for i in range(len(filtered_slice)):
        batch_id = filtered_slice.metadata_store.get_metadata(i, "batch_id")
        assert batch_id in [1, 2]


def test_complex_slicing(
    doleus_binary_classification_dataset, basic_metadata, numeric_metadata
):
    dataset = doleus_binary_classification_dataset
    dataset.add_metadata_from_list(basic_metadata)
    dataset.add_metadata_from_list(numeric_metadata)

    new_method = dataset.slice_by_conditions(
        [("validated", "==", True), ("confidence_score", ">=", 0.9)],
        logical_operator="AND",
    )

    chained_method = dataset.slice_by_value("validated", "==", True)
    chained_method = chained_method.slice_by_value("confidence_score", ">=", 0.9)

    assert len(new_method) == len(chained_method)

    for i in range(len(new_method)):
        new_validated = new_method.metadata_store.get_metadata(i, "validated")
        new_conf = new_method.metadata_store.get_metadata(i, "confidence_score")

        chained_validated = chained_method.metadata_store.get_metadata(i, "validated")
        chained_conf = chained_method.metadata_store.get_metadata(i, "confidence_score")

        assert new_validated == chained_validated == True
        assert new_conf == chained_conf >= 0.9
