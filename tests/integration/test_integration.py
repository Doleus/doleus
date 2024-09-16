import pytest
import torch
from doleus.checks import Check, CheckSuite


class TestDoleusIntegration:
    """Integration tests for complete Doleus workflows.

    These tests verify complete workflows across multiple components:
    - Dataset creation and wrapping
    - Metadata addition and management
    - Model prediction integration
    - Data slicing and filtering
    - Check creation and execution
    - Check suite reporting
    """

    def test_binary_classification_integration(
        self, doleus_binary_classification_dataset, basic_metadata
    ):
        """Test complete binary classification workflow."""

        # ARRANGE
        dataset = doleus_binary_classification_dataset
        dataset_size = len(dataset)  # 10 samples

        # Add metadata using fixture
        dataset.add_metadata_from_list(basic_metadata)

        # Create binary predictions (1D probabilities for positive class)
        predictions = torch.sigmoid(
            torch.randn(dataset_size)
        )  # Random probabilities [0,1]
        model_id = "test_binary_model"

        # ACT - Execute the main workflow
        dataset.add_model_predictions(predictions=predictions, model_id=model_id)

        # Create slices
        camera_a_slice = dataset.slice_by_value("source", "==", "camera_a")
        validated_slice = dataset.slice_by_value("validated", "==", True)

        # Create checks - mix of threshold and evaluation-only checks
        checks = [
            Check(
                name="overall_accuracy",
                dataset=dataset,
                model_id=model_id,
                metric="Accuracy",
                operator=">=",
                value=0.0,
            ),  # Threshold check
            Check(
                name="camera_a_precision",
                dataset=camera_a_slice,
                model_id=model_id,
                metric="Precision",
            ),  # Evaluation-only
            Check(
                name="validated_recall",
                dataset=validated_slice,
                model_id=model_id,
                metric="Recall",
                operator=">",
                value=0.1,
            ),  # Threshold check
        ]

        check_suite = CheckSuite(name="binary_integration_test", checks=checks)
        results = check_suite.run_all(show=False)

        # ASSERT
        assert isinstance(results, dict)
        assert results["checksuite_name"] == "binary_integration_test"
        assert len(results["checks"]) == 3
        assert "success" in results

        # Verify slicing worked correctly
        assert len(camera_a_slice) == 5  # Indices 0,2,4,6,8
        assert len(validated_slice) == 5  # Indices 0,2,4,6,8

        # Verify all checks executed
        check_names = [check["check_name"] for check in results["checks"]]
        assert "overall_accuracy" in check_names
        assert "camera_a_precision" in check_names
        assert "validated_recall" in check_names

        # CONTRACT CORRECTNESS: Verify report schema matches visualization expectations
        # CheckSuite report schema
        suite_required_keys = ["checksuite_name", "success", "checks"]
        for key in suite_required_keys:
            assert key in results, f"Missing key in suite report: {key}"

        # Verify CheckSuite types
        assert isinstance(results["success"], bool)
        assert isinstance(results["checksuite_name"], str)
        assert isinstance(results["checks"], list)

        # Individual check report schemas - differentiate by check type
        for i, check_report in enumerate(results["checks"]):
            # All checks must have these keys
            common_required_keys = [
                "check_name",
                "dataset_id",
                "metric",
                "operator",
                "value",
                "result",
                "success",
            ]
            for key in common_required_keys:
                assert key in check_report, f"Missing key in check report {i}: {key}"

            # Common type validations
            assert isinstance(check_report["check_name"], str)
            assert isinstance(check_report["dataset_id"], str)
            assert isinstance(check_report["metric"], str)
            assert isinstance(check_report["result"], (int, float))

            # Differentiate contract correctness by check type
            if (
                check_report["operator"] is not None
                and check_report["value"] is not None
            ):
                # THRESHOLD CHECK: operator and value provided, success is boolean
                assert isinstance(check_report["operator"], str)
                assert isinstance(check_report["value"], (int, float))
                assert isinstance(check_report["success"], bool)
            else:
                # EVALUATION-ONLY CHECK: operator and value are None, success is None
                assert check_report["operator"] is None
                assert check_report["value"] is None
                assert check_report["success"] is None

    def test_multiclass_classification_integration(
        self, doleus_multiclass_classification_dataset, numeric_metadata
    ):
        """Test complete multiclass classification workflow."""

        # ARRANGE
        dataset = doleus_multiclass_classification_dataset
        dataset_size = len(dataset)  # 10 samples
        num_classes = dataset.num_classes  # 3 classes

        # Add metadata using fixture
        dataset.add_metadata_from_list(numeric_metadata)

        # Create multiclass predictions (2D logits)
        predictions = torch.randn(dataset_size, num_classes)
        model_id = "test_multiclass_model"

        # ACT
        dataset.add_model_predictions(predictions=predictions, model_id=model_id)

        # Create slices
        high_confidence_slice = dataset.slice_by_value("confidence_score", ">=", 0.85)
        batch_1_slice = dataset.slice_by_value("batch_id", "==", 1)

        # Create checks
        checks = [
            Check(
                name="overall_accuracy",
                dataset=dataset,
                model_id=model_id,
                metric="Accuracy",
            ),
            Check(
                name="high_conf_f1",
                dataset=high_confidence_slice,
                model_id=model_id,
                metric="F1_Score",
            ),
            Check(
                name="batch1_precision",
                dataset=batch_1_slice,
                model_id=model_id,
                metric="Precision",
            ),
        ]

        check_suite = CheckSuite(name="multiclass_integration_test", checks=checks)
        results = check_suite.run_all(show=False)

        # ASSERT
        assert isinstance(results, dict)
        assert results["checksuite_name"] == "multiclass_integration_test"
        assert len(results["checks"]) == 3

        # Verify slices have expected sizes
        # High confidence (>= 0.85): indices 0,1,2,4,6 = 5 samples
        # batch_id=1: indices 0,1,2 = 3 samples
        assert len(high_confidence_slice) == 5
        assert len(batch_1_slice) == 3

    def test_detection_integration(
        self, doleus_object_detection_dataset, mixed_metadata
    ):
        """Test complete object detection workflow."""

        # ARRANGE
        dataset = doleus_object_detection_dataset
        dataset_size = len(dataset)  # 10 samples

        # Add metadata using fixture
        dataset.add_metadata_from_list(mixed_metadata)

        # Create detection predictions
        predictions = []
        for i in range(dataset_size):
            # Each prediction has boxes, labels, and scores
            num_objects = (i % 3) + 1  # 1-3 objects per image
            predictions.append(
                {
                    "boxes": torch.rand(num_objects, 4) * 100,  # Random boxes
                    "labels": torch.randint(0, 3, (num_objects,)),  # Random labels 0-2
                    "scores": torch.rand(num_objects) * 0.8 + 0.2,  # Scores 0.2-1.0
                }
            )

        model_id = "test_detection_model"

        # ACT
        dataset.add_model_predictions(predictions=predictions, model_id=model_id)

        # Create slices
        lab_slice = dataset.slice_by_value("environment", "==", "lab")
        warm_slice = dataset.slice_by_value("temperature", ">=", 20.0)

        # Create checks
        checks = [
            Check(name="overall_map", dataset=dataset, model_id=model_id, metric="mAP"),
            Check(
                name="lab_iou",
                dataset=lab_slice,
                model_id=model_id,
                metric="IntersectionOverUnion",
            ),
            Check(
                name="warm_ciou",
                dataset=warm_slice,
                model_id=model_id,
                metric="CompleteIntersectionOverUnion",
            ),
        ]

        check_suite = CheckSuite(name="detection_integration_test", checks=checks)
        results = check_suite.run_all(show=False)

        # ASSERT
        assert isinstance(results, dict)
        assert results["checksuite_name"] == "detection_integration_test"
        assert len(results["checks"]) == 3

        # Verify slices
        # Lab environment: indices 0,2,4,6,8 = 5 samples
        # Temperature >= 20.0: indices 0,2,4,6,8,9 = 6 samples
        assert len(lab_slice) == 5
        assert len(warm_slice) == 6

    def test_error_handling_integration(self, doleus_binary_classification_dataset):
        """Test error handling in integration scenarios."""

        # ARRANGE
        dataset = doleus_binary_classification_dataset

        # ACT & ASSERT - Test various error conditions

        # Test slicing with non-existent metadata key
        with pytest.raises(KeyError):
            dataset.slice_by_value("non_existent_key", "==", "value")

        # Test check with non-existent model
        with pytest.raises(KeyError):
            check = Check(
                name="invalid_check",
                dataset=dataset,
                model_id="non_existent_model",
                metric="Accuracy",
            )
            check.run(show=False)

    def test_cross_task_metadata_consistency(
        self,
        doleus_binary_classification_dataset,
        doleus_object_detection_dataset,
        string_numeric_metadata,
    ):
        """Test metadata consistency across different task types."""

        # ARRANGE
        binary_dataset = doleus_binary_classification_dataset
        detection_dataset = doleus_object_detection_dataset

        # ACT
        binary_dataset.add_metadata_from_list(string_numeric_metadata)
        detection_dataset.add_metadata_from_list(string_numeric_metadata)

        # Create slices with same criteria
        binary_high_priority = binary_dataset.slice_by_value("priority", "==", "1")
        detection_high_priority = detection_dataset.slice_by_value(
            "priority", "==", "1"
        )

        # ASSERT
        # Both should slice based on same metadata pattern
        # priority="1": indices 0,2,5,8 = 4 samples each
        assert len(binary_high_priority) == 4
        assert len(detection_high_priority) == 4

    def test_dataframe_metadata_integration(
        self, doleus_binary_classification_dataset, metadata_dataframe
    ):
        """Test pandas DataFrame metadata integration."""

        # ARRANGE
        dataset = doleus_binary_classification_dataset

        # ACT - Add metadata from DataFrame
        dataset.add_metadata_from_dataframe(metadata_dataframe)

        # Create slices using DataFrame columns (converted from mixed_metadata)
        lab_slice = dataset.slice_by_value("environment", "==", "lab")
        high_temp_slice = dataset.slice_by_value("temperature", ">=", 20.0)
        high_count_slice = dataset.slice_by_value("sample_count", ">=", 100)
        corrupted_slice = dataset.slice_by_value("corrupted", "==", True)

        # ASSERT - Using mixed_metadata patterns
        # environment="lab": indices 0,2,4,6,8 = 5 samples
        # temperature >= 20.0: indices 0,2,4,6,8,9 = 6 samples
        # sample_count >= 100: indices 0,2,4,8 = 4 samples
        # corrupted=True: indices 1,4,7 = 3 samples
        assert len(lab_slice) == 5
        assert len(high_temp_slice) == 6
        assert len(high_count_slice) == 4
        assert len(corrupted_slice) == 3

    # REPRESENTATIVE NEGATIVE PATHS - Test key integration failure scenarios

    def test_binary_classification_wrong_prediction_dimensions_integration(
        self, doleus_binary_classification_dataset
    ):
        """Test error propagation when binary classification predictions have wrong dimensions."""

        # ARRANGE
        dataset = doleus_binary_classification_dataset
        dataset_size = len(dataset)  # 10 samples

        # ACT & ASSERT - Try to add multiclass-shaped predictions to binary dataset
        wrong_predictions = torch.randn(
            dataset_size, 3
        )  # Wrong: binary should be (N,) or (N,1)

        with pytest.raises(
            ValueError, match="binary classification predictions must be 1D tensor"
        ):
            dataset.add_model_predictions(
                predictions=wrong_predictions, model_id="wrong_model"
            )

    def test_multiclass_classification_wrong_prediction_dtype_integration(
        self, doleus_multiclass_classification_dataset
    ):
        """Test error propagation when multiclass classification predictions have wrong dtype for 1D."""

        # ARRANGE
        dataset = doleus_multiclass_classification_dataset
        dataset_size = len(dataset)  # 10 samples

        # ACT & ASSERT - Try to add 1D float predictions (should be int for 1D multiclass)
        wrong_predictions = torch.randn(
            dataset_size
        )  # Wrong: 1D multiclass should be integer class indices

        with pytest.raises(
            ValueError,
            match="For multiclass with 1D predictions, dtype must be integer",
        ):
            dataset.add_model_predictions(
                predictions=wrong_predictions, model_id="wrong_model"
            )

    def test_detection_mismatched_prediction_shapes_integration(
        self, doleus_object_detection_dataset
    ):
        """Test error propagation when detection predictions have mismatched shapes."""

        # ARRANGE
        dataset = doleus_object_detection_dataset
        dataset_size = len(dataset)  # 10 samples

        # ACT & ASSERT - Try to add detection predictions with mismatched box/label shapes
        wrong_predictions = []
        for i in range(dataset_size):
            wrong_predictions.append(
                {
                    "boxes": torch.rand(3, 4) * 100,  # 3 boxes
                    "labels": torch.randint(
                        0, 3, (2,)
                    ),  # Wrong: only 2 labels (should be 3)
                    "scores": torch.rand(3) * 0.8 + 0.2,  # 3 scores
                }
            )

        with pytest.raises(
            ValueError, match="labels for sample .* must have shape \\(M,\\)"
        ):
            dataset.add_model_predictions(
                predictions=wrong_predictions, model_id="wrong_model"
            )
