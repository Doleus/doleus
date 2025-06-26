# SPDX-FileCopyrightText: 2025 Doleus contributors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import tempfile
from unittest.mock import patch

import pytest
import torch
from doleus.checks import Check, CheckSuite


class TestCheck:
    """Test cases for the Check class."""

    def test_check_initialization_basic(self, doleus_binary_classification_dataset):
        """Test basic Check initialization without threshold."""
        dataset = doleus_binary_classification_dataset
        check = Check(
            name="test_accuracy",
            dataset=dataset,
            model_id="test_model",
            metric="Accuracy",
        )

        assert check.name == "test_accuracy"
        assert check.dataset == dataset
        assert check.model_id == "test_model"
        assert check.metric == "Accuracy"
        assert check.metric_parameters == {}
        assert check.target_class is None
        assert check.operator is None
        assert check.value is None
        assert not check.testing

    def test_check_initialization_with_threshold(
        self, doleus_binary_classification_dataset
    ):
        """Test Check initialization with threshold parameters."""
        dataset = doleus_binary_classification_dataset
        check = Check(
            name="test_accuracy_threshold",
            dataset=dataset,
            model_id="test_model",
            metric="Accuracy",
            operator=">=",
            value=0.8,
        )

        assert check.name == "test_accuracy_threshold"
        assert check.operator == ">="
        assert check.value == 0.8
        assert check.testing

    def test_check_initialization_with_parameters(
        self, doleus_binary_classification_dataset
    ):
        """Test Check initialization with metric parameters."""
        dataset = doleus_binary_classification_dataset
        metric_params = {"average": "micro", "num_classes": 2}
        check = Check(
            name="test_precision",
            dataset=dataset,
            model_id="test_model",
            metric="Precision",
            metric_parameters=metric_params,
            target_class=1,
        )

        assert check.metric_parameters == metric_params
        assert check.target_class == 1

    def test_check_run_basic(self, doleus_binary_classification_dataset):
        """Test basic check execution without threshold."""
        dataset = doleus_binary_classification_dataset

        predictions = torch.tensor([0.9, 0.3, 0.8, 0.2, 0.7, 0.1, 0.6, 0.4, 0.5, 0.0])
        dataset.add_model_predictions(predictions, "test_model")

        check = Check(
            name="test_accuracy",
            dataset=dataset,
            model_id="test_model",
            metric="Accuracy",
        )

        with patch("doleus.checks.base.visualize_report") as mock_visualize:
            report = check.run(show=True)

        # Verify report structure
        assert isinstance(report, dict)
        assert report["check_name"] == "test_accuracy"
        assert report["dataset_id"] == dataset.name
        assert report["metric"] == "Accuracy"
        assert report["operator"] is None
        assert report["value"] is None
        assert report["success"] is None
        assert "timestamp" in report
        assert "result" in report
        assert isinstance(report["result"], (int, float))

        mock_visualize.assert_called_once_with(report)

    def test_check_run_with_threshold_pass(self, doleus_binary_classification_dataset):
        """Test check execution with threshold that passes."""
        dataset = doleus_binary_classification_dataset

        # Ground truth: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        # Add predictions that should give high accuracy (mostly correct)
        predictions = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.1, 0.9, 0.2, 0.8, 0.1, 0.9])
        dataset.add_model_predictions(predictions, "test_model")

        check = Check(
            name="test_accuracy_threshold",
            dataset=dataset,
            model_id="test_model",
            metric="Accuracy",
            operator=">=",
            value=0.5,  # Should pass with these predictions
        )

        with patch("doleus.checks.base.visualize_report") as mock_visualize:
            report = check.run(show=True)

        assert report["operator"] == ">="
        assert report["value"] == 0.5
        assert report["success"] is True
        assert report["result"] >= 0.5

        mock_visualize.assert_called_once_with(report)

    def test_check_run_with_threshold_fail(self, doleus_binary_classification_dataset):
        """Test check execution with threshold that fails."""
        dataset = doleus_binary_classification_dataset

        # Ground truth: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        # Add predictions that should give low accuracy (mostly wrong)
        predictions = torch.tensor([0.9, 0.1, 0.8, 0.2, 0.9, 0.1, 0.8, 0.2, 0.9, 0.1])
        dataset.add_model_predictions(predictions, "test_model")

        check = Check(
            name="test_accuracy_threshold",
            dataset=dataset,
            model_id="test_model",
            metric="Accuracy",
            operator=">=",
            value=0.8,  # Should fail with these predictions
        )

        with patch("doleus.checks.base.visualize_report") as mock_visualize:
            report = check.run(show=True)

        assert report["operator"] == ">="
        assert report["value"] == 0.8
        assert report["success"] is False
        assert report["result"] < 0.8

        mock_visualize.assert_called_once_with(report)

    def test_check_run_without_show(self, doleus_binary_classification_dataset):
        """Test check execution without visualization."""
        dataset = doleus_binary_classification_dataset

        predictions = torch.tensor([0.9, 0.3, 0.8, 0.2, 0.7, 0.1, 0.6, 0.4, 0.5, 0.0])
        dataset.add_model_predictions(predictions, "test_model")

        check = Check(
            name="test_accuracy",
            dataset=dataset,
            model_id="test_model",
            metric="Accuracy",
        )

        with patch("doleus.checks.base.visualize_report") as mock_visualize:
            report = check.run(show=False)

        mock_visualize.assert_not_called()

    def test_check_run_save_report(self, doleus_binary_classification_dataset):
        """Test check execution with report saving."""
        dataset = doleus_binary_classification_dataset

        predictions = torch.tensor([0.9, 0.3, 0.8, 0.2, 0.7, 0.1, 0.6, 0.4, 0.5, 0.0])
        dataset.add_model_predictions(predictions, "test_model")

        check = Check(
            name="test_accuracy",
            dataset=dataset,
            model_id="test_model",
            metric="Accuracy",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            os.chdir(temp_dir)

            try:
                with patch("doleus.checks.base.visualize_report"):
                    report = check.run(show=False, save_report=True)

                # Verify file was created
                expected_filename = "check_test_accuracy_report.json"
                assert os.path.exists(expected_filename)

                # Verify file content
                with open(expected_filename, "r") as f:
                    saved_report = json.load(f)

                assert saved_report["check_name"] == report["check_name"]
                assert saved_report["dataset_id"] == report["dataset_id"]
                assert saved_report["metric"] == report["metric"]
                assert saved_report["result"] == report["result"]

            finally:
                os.chdir(original_cwd)

    def test_check_call_method(self, doleus_binary_classification_dataset):
        """Test the __call__ method convenience."""
        dataset = doleus_binary_classification_dataset

        predictions = torch.tensor([0.9, 0.3, 0.8, 0.2, 0.7, 0.1, 0.6, 0.4, 0.5, 0.0])
        dataset.add_model_predictions(predictions, "test_model")

        check = Check(
            name="test_accuracy",
            dataset=dataset,
            model_id="test_model",
            metric="Accuracy",
        )

        with patch("doleus.checks.base.visualize_report") as mock_visualize:
            report1 = check.run(show=True, save_report=False)
            report2 = check(show=True, save_report=False)

        # Verify both methods return the same result
        assert report1["check_name"] == report2["check_name"]
        assert report1["result"] == report2["result"]
        assert mock_visualize.call_count == 2

    def test_check_different_operators(self, doleus_binary_classification_dataset):
        """Test check execution with different comparison operators."""
        dataset = doleus_binary_classification_dataset

        # Ground truth: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        # These predictions give accuracy of ~0.8 (8 correct out of 10)
        predictions = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.9, 0.1, 0.2, 0.8, 0.1, 0.9])
        dataset.add_model_predictions(predictions, "test_model")

        operators_and_values = [
            (">", 0.5, True),  # Should pass (accuracy > 0.5)
            (">=", 0.5, True),  # Should pass (accuracy >= 0.5)
            ("<", 0.5, False),  # Should fail (accuracy < 0.5 is false)
            ("<=", 0.5, False),  # Should fail (accuracy <= 0.5 is false)
            ("!=", 0.5, True),  # Should pass (accuracy != 0.5)
        ]

        for operator, value, expected_success in operators_and_values:
            check = Check(
                name=f"test_{operator}_{value}",
                dataset=dataset,
                model_id="test_model",
                metric="Accuracy",
                operator=operator,
                value=value,
            )

            with patch("doleus.checks.base.visualize_report"):
                report = check.run(show=False)

            assert report["operator"] == operator
            assert report["value"] == value
            assert report["success"] == expected_success

    def test_check_multiclass_metrics(self, doleus_multiclass_classification_dataset):
        """Test check execution with multiclass metrics."""
        dataset = doleus_multiclass_classification_dataset

        # Add multiclass predictions
        predictions = torch.randn(10, 3)  # 10 samples, 3 classes
        dataset.add_model_predictions(predictions, "test_model")

        metrics_to_test = ["Accuracy", "Precision", "Recall", "F1_Score"]

        for metric in metrics_to_test:
            check = Check(
                name=f"test_{metric}",
                dataset=dataset,
                model_id="test_model",
                metric=metric,
            )

            with patch("doleus.checks.base.visualize_report"):
                report = check.run(show=False)

            assert report["metric"] == metric
            assert isinstance(report["result"], (int, float))
            assert report["success"] is None  # No threshold

    def test_check_detection_metrics(self, doleus_object_detection_dataset):
        """Test check execution with detection metrics."""
        dataset = doleus_object_detection_dataset

        # Add detection predictions
        predictions = []
        for i in range(10):
            num_objects = (i % 3) + 1
            predictions.append(
                {
                    "boxes": torch.rand(num_objects, 4) * 100,
                    "labels": torch.randint(0, 3, (num_objects,)),
                    "scores": torch.rand(num_objects) * 0.8 + 0.2,
                }
            )

        dataset.add_model_predictions(predictions, "test_model")

        detection_metrics = [
            "mAP",
            "IntersectionOverUnion",
            "CompleteIntersectionOverUnion",
        ]

        for metric in detection_metrics:
            check = Check(
                name=f"test_{metric}",
                dataset=dataset,
                model_id="test_model",
                metric=metric,
            )

            with patch("doleus.checks.base.visualize_report"):
                report = check.run(show=False)

            assert report["metric"] == metric
            assert isinstance(report["result"], (int, float))

    def test_check_error_handling(self, doleus_binary_classification_dataset):
        """Test error handling in check execution."""
        dataset = doleus_binary_classification_dataset

        # Test with non-existent model
        check = Check(
            name="test_invalid_model",
            dataset=dataset,
            model_id="non_existent_model",
            metric="Accuracy",
        )

        with pytest.raises(KeyError):
            check.run(show=False)

        # Test with invalid metric
        predictions = torch.tensor([0.9, 0.3, 0.8, 0.2, 0.7, 0.1, 0.6, 0.4, 0.5, 0.0])
        dataset.add_model_predictions(predictions, "test_model")

        check = Check(
            name="test_invalid_metric",
            dataset=dataset,
            model_id="test_model",
            metric="InvalidMetric",
        )

        with pytest.raises(Exception):  # Should raise some exception for invalid metric
            check.run(show=False)

    def test_check_metric_calculation_correctness(
        self, doleus_binary_classification_dataset
    ):
        """Test that metric calculations are mathematically correct."""
        dataset = doleus_binary_classification_dataset

        # Ground truth: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

        # Test case 1: Perfect predictions (all correct)
        # Predictions: [0.1, 0.9, 0.2, 0.8, 0.1, 0.9, 0.2, 0.8, 0.1, 0.9]
        # Result: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] - Perfect match!
        perfect_predictions = torch.tensor(
            [0.1, 0.9, 0.2, 0.8, 0.1, 0.9, 0.2, 0.8, 0.1, 0.9]
        )
        dataset.add_model_predictions(perfect_predictions, "perfect_model")

        check = Check(
            name="perfect_accuracy",
            dataset=dataset,
            model_id="perfect_model",
            metric="Accuracy",
        )

        with patch("doleus.checks.base.visualize_report"):
            report = check.run(show=False)

        assert report["result"] == 1.0  # Should be 100% accurate

        # Test case 2: All wrong predictions
        # Predictions: [0.9, 0.1, 0.8, 0.2, 0.9, 0.1, 0.8, 0.2, 0.9, 0.1]
        # Result: [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] - All wrong!
        wrong_predictions = torch.tensor(
            [0.9, 0.1, 0.8, 0.2, 0.9, 0.1, 0.8, 0.2, 0.9, 0.1]
        )
        dataset.add_model_predictions(wrong_predictions, "wrong_model")

        check = Check(
            name="wrong_accuracy",
            dataset=dataset,
            model_id="wrong_model",
            metric="Accuracy",
        )

        with patch("doleus.checks.base.visualize_report"):
            report = check.run(show=False)

        assert report["result"] == 0.0  # Should be 0% accurate

        # Test case 3: Mixed predictions (8 correct, 2 wrong)
        # Predictions: [0.1, 0.9, 0.2, 0.8, 0.9, 0.1, 0.2, 0.8, 0.1, 0.9]
        # Result: [0, 1, 0, 1, 1, 0, 0, 1, 0, 1]
        # Correct: positions 0,1,2,3,6,7,8,9 (8 correct)
        # Wrong: positions 4,5 (2 wrong)
        # Accuracy: 8/10 = 0.8
        mixed_predictions = torch.tensor(
            [0.1, 0.9, 0.2, 0.8, 0.9, 0.1, 0.2, 0.8, 0.1, 0.9]
        )
        dataset.add_model_predictions(mixed_predictions, "mixed_model")

        check = Check(
            name="mixed_accuracy",
            dataset=dataset,
            model_id="mixed_model",
            metric="Accuracy",
        )

        with patch("doleus.checks.base.visualize_report"):
            report = check.run(show=False)

        assert report["result"] == pytest.approx(0.8)  # Should be 80% accurate

    def test_check_precision_calculation_correctness(
        self, doleus_binary_classification_dataset
    ):
        """Test that precision calculations are mathematically correct."""
        dataset = doleus_binary_classification_dataset

        # Ground truth: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        # Predictions: [0.9, 0.9, 0.1, 0.8, 0.9, 0.1, 0.1, 0.8, 0.1, 0.9]
        # Result: [1, 1, 0, 1, 1, 0, 0, 1, 0, 1] (threshold 0.5)
        # True Positives: positions 1,3,7,9 (4 TP)
        # False Positives: positions 0,4 (2 FP)
        # Precision = TP / (TP + FP) = 4 / (4 + 2) = 4/6 = 0.666...
        precision_predictions = torch.tensor(
            [0.9, 0.9, 0.1, 0.8, 0.9, 0.1, 0.1, 0.8, 0.1, 0.9]
        )
        dataset.add_model_predictions(precision_predictions, "precision_model")

        check = Check(
            name="precision_test",
            dataset=dataset,
            model_id="precision_model",
            metric="Precision",
        )

        with patch("doleus.checks.base.visualize_report"):
            report = check.run(show=False)

        assert report["result"] == pytest.approx(2 / 3, rel=1e-2)  # Should be ~0.667


class TestCheckSuite:
    """Test cases for the CheckSuite class."""

    def test_checksuite_initialization(self, doleus_binary_classification_dataset):
        """Test CheckSuite initialization."""
        dataset = doleus_binary_classification_dataset

        # Create some checks
        predictions = torch.tensor([0.9, 0.3, 0.8, 0.2, 0.7, 0.1, 0.6, 0.4, 0.5, 0.0])
        dataset.add_model_predictions(predictions, "test_model")

        checks = [
            Check(
                name="accuracy_check",
                dataset=dataset,
                model_id="test_model",
                metric="Accuracy",
                operator=">=",
                value=0.5,
            ),
            Check(
                name="precision_check",
                dataset=dataset,
                model_id="test_model",
                metric="Precision",
            ),
        ]

        suite = CheckSuite(name="test_suite", checks=checks)

        assert suite.name == "test_suite"
        assert len(suite.checks) == 2
        assert suite.checks[0].name == "accuracy_check"
        assert suite.checks[1].name == "precision_check"

    def test_checksuite_run_all_basic(self, doleus_binary_classification_dataset):
        """Test basic CheckSuite execution."""
        dataset = doleus_binary_classification_dataset

        predictions = torch.tensor([0.9, 0.3, 0.8, 0.2, 0.7, 0.1, 0.6, 0.4, 0.5, 0.0])
        dataset.add_model_predictions(predictions, "test_model")

        checks = [
            Check(
                name="accuracy_check",
                dataset=dataset,
                model_id="test_model",
                metric="Accuracy",
                operator=">=",
                value=0.5,
            ),
            Check(
                name="precision_check",
                dataset=dataset,
                model_id="test_model",
                metric="Precision",
            ),
        ]

        suite = CheckSuite(name="test_suite", checks=checks)

        with patch("doleus.checks.base.visualize_report") as mock_visualize:
            report = suite.run_all(show=True)

        # Verify suite report structure
        assert isinstance(report, dict)
        assert report["checksuite_name"] == "test_suite"
        assert "success" in report
        assert "checks" in report
        assert "timestamp" in report
        assert len(report["checks"]) == 2

        # Verify individual check reports
        check_names = [check["check_name"] for check in report["checks"]]
        assert "accuracy_check" in check_names
        assert "precision_check" in check_names

        mock_visualize.assert_called_once_with(report)

    def test_checksuite_run_all_empty(self):
        """Test CheckSuite execution with no checks."""
        suite = CheckSuite(name="empty_suite", checks=[])

        with patch("doleus.checks.base.visualize_report") as mock_visualize:
            report = suite.run_all(show=True)

        assert report["checksuite_name"] == "empty_suite"
        assert report["success"] is True  # Empty suite should pass
        assert report["checks"] == []
        mock_visualize.assert_called_once_with(report)

    def test_checksuite_success_logic(self, doleus_binary_classification_dataset):
        """Test CheckSuite success logic with different check outcomes."""
        dataset = doleus_binary_classification_dataset

        # Ground truth: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        # These predictions give accuracy of 0.8 (8 correct out of 10)
        predictions = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.9, 0.1, 0.2, 0.8, 0.1, 0.9])
        dataset.add_model_predictions(predictions, "test_model")

        # Test 1: All checks pass
        checks_all_pass = [
            Check(
                name="accuracy_high",
                dataset=dataset,
                model_id="test_model",
                metric="Accuracy",
                operator=">=",
                value=0.5,  # Should pass (0.8 >= 0.5)
            ),
            Check(
                name="precision_high",
                dataset=dataset,
                model_id="test_model",
                metric="Precision",
                operator=">=",
                value=0.5,  # Should pass
            ),
        ]

        suite_all_pass = CheckSuite(name="all_pass_suite", checks=checks_all_pass)
        with patch("doleus.checks.base.visualize_report"):
            report = suite_all_pass.run_all(show=False)

        assert report["success"] is True

        # Test 2: Some checks fail
        checks_some_fail = [
            Check(
                name="accuracy_high",
                dataset=dataset,
                model_id="test_model",
                metric="Accuracy",
                operator=">=",
                value=0.5,  # Should pass (0.8 >= 0.5)
            ),
            Check(
                name="accuracy_very_high",
                dataset=dataset,
                model_id="test_model",
                metric="Accuracy",
                operator=">=",
                value=0.9,  # Should fail (0.8 < 0.9)
            ),
        ]

        suite_some_fail = CheckSuite(name="some_fail_suite", checks=checks_some_fail)
        with patch("doleus.checks.base.visualize_report"):
            report = suite_some_fail.run_all(show=False)

        assert report["success"] is False

        # Test 3: Mix of threshold and non-threshold checks
        checks_mixed = [
            Check(
                name="accuracy_threshold",
                dataset=dataset,
                model_id="test_model",
                metric="Accuracy",
                operator=">=",
                value=0.5,  # Should pass (0.8 >= 0.5)
            ),
            Check(
                name="precision_eval",
                dataset=dataset,
                model_id="test_model",
                metric="Precision",
                # No threshold - should not affect success
            ),
        ]

        suite_mixed = CheckSuite(name="mixed_suite", checks=checks_mixed)
        with patch("doleus.checks.base.visualize_report"):
            report = suite_mixed.run_all(show=False)

        assert report["success"] is True

    def test_checksuite_run_all_without_show(
        self, doleus_binary_classification_dataset
    ):
        """Test CheckSuite execution without visualization."""
        dataset = doleus_binary_classification_dataset

        predictions = torch.tensor([0.9, 0.3, 0.8, 0.2, 0.7, 0.1, 0.6, 0.4, 0.5, 0.0])
        dataset.add_model_predictions(predictions, "test_model")

        checks = [
            Check(
                name="accuracy_check",
                dataset=dataset,
                model_id="test_model",
                metric="Accuracy",
                operator=">=",
                value=0.5,
            ),
        ]

        suite = CheckSuite(name="test_suite", checks=checks)

        with patch("doleus.checks.base.visualize_report") as mock_visualize:
            report = suite.run_all(show=False)

        mock_visualize.assert_not_called()

    def test_checksuite_run_all_save_report(self, doleus_binary_classification_dataset):
        """Test CheckSuite execution with report saving."""
        dataset = doleus_binary_classification_dataset

        predictions = torch.tensor([0.9, 0.3, 0.8, 0.2, 0.7, 0.1, 0.6, 0.4, 0.5, 0.0])
        dataset.add_model_predictions(predictions, "test_model")

        checks = [
            Check(
                name="accuracy_check",
                dataset=dataset,
                model_id="test_model",
                metric="Accuracy",
                operator=">=",
                value=0.5,
            ),
        ]

        suite = CheckSuite(name="test_suite", checks=checks)

        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            os.chdir(temp_dir)

            try:
                with patch("doleus.checks.base.visualize_report"):
                    report = suite.run_all(show=False, save_report=True)

                # Verify file was created
                expected_filename = "checksuite_test_suite_report.json"
                assert os.path.exists(expected_filename)

                # Verify file content
                with open(expected_filename, "r") as f:
                    saved_report = json.load(f)

                assert saved_report["checksuite_name"] == report["checksuite_name"]
                assert saved_report["success"] == report["success"]
                assert len(saved_report["checks"]) == len(report["checks"])

            finally:
                os.chdir(original_cwd)

    def test_checksuite_call_method(self, doleus_binary_classification_dataset):
        """Test the __call__ method convenience."""
        dataset = doleus_binary_classification_dataset

        predictions = torch.tensor([0.9, 0.3, 0.8, 0.2, 0.7, 0.1, 0.6, 0.4, 0.5, 0.0])
        dataset.add_model_predictions(predictions, "test_model")

        checks = [
            Check(
                name="accuracy_check",
                dataset=dataset,
                model_id="test_model",
                metric="Accuracy",
                operator=">=",
                value=0.5,
            ),
        ]

        suite = CheckSuite(name="test_suite", checks=checks)

        with patch("doleus.checks.base.visualize_report") as mock_visualize:
            report1 = suite.run_all(show=True, save_report=False)
            report2 = suite(show=True, save_report=False)

        # Verify both methods return the same result
        assert report1["checksuite_name"] == report2["checksuite_name"]
        assert report1["success"] == report2["success"]
        assert mock_visualize.call_count == 2

    def test_checksuite_multiple_datasets(
        self,
        doleus_binary_classification_dataset,
        doleus_multiclass_classification_dataset,
    ):
        """Test CheckSuite with checks from different datasets."""
        binary_dataset = doleus_binary_classification_dataset
        multiclass_dataset = doleus_multiclass_classification_dataset

        # Add predictions to both datasets
        binary_predictions = torch.tensor(
            [0.9, 0.3, 0.8, 0.2, 0.7, 0.1, 0.6, 0.4, 0.5, 0.0]
        )
        binary_dataset.add_model_predictions(binary_predictions, "binary_model")

        multiclass_predictions = torch.randn(10, 3)
        multiclass_dataset.add_model_predictions(
            multiclass_predictions, "multiclass_model"
        )

        checks = [
            Check(
                name="binary_accuracy",
                dataset=binary_dataset,
                model_id="binary_model",
                metric="Accuracy",
                operator=">=",
                value=0.5,
            ),
            Check(
                name="multiclass_accuracy",
                dataset=multiclass_dataset,
                model_id="multiclass_model",
                metric="Accuracy",
                operator=">=",
                value=0.3,
            ),
        ]

        suite = CheckSuite(name="multi_dataset_suite", checks=checks)

        with patch("doleus.checks.base.visualize_report"):
            report = suite.run_all(show=False)

        assert len(report["checks"]) == 2
        assert report["checks"][0]["dataset_id"] == binary_dataset.name
        assert report["checks"][1]["dataset_id"] == multiclass_dataset.name

    def test_checksuite_error_handling(self, doleus_binary_classification_dataset):
        """Test error handling in CheckSuite execution."""
        dataset = doleus_binary_classification_dataset

        # Create a check that will fail
        checks = [
            Check(
                name="failing_check",
                dataset=dataset,
                model_id="non_existent_model",
                metric="Accuracy",
            ),
        ]

        suite = CheckSuite(name="error_suite", checks=checks)

        # Should raise an exception when running the suite
        with pytest.raises(KeyError):
            suite.run_all(show=False)


class TestCheckIntegration:
    """Integration tests for Check and CheckSuite functionality."""

    def test_check_with_sliced_dataset(
        self, doleus_binary_classification_dataset, basic_metadata
    ):
        """Test Check execution on sliced datasets."""
        dataset = doleus_binary_classification_dataset
        dataset.add_metadata_from_list(basic_metadata)

        predictions = torch.tensor([0.9, 0.3, 0.8, 0.2, 0.7, 0.1, 0.6, 0.4, 0.5, 0.0])
        dataset.add_model_predictions(predictions, "test_model")

        # Create slice
        validated_slice = dataset.slice_by_value("validated", "==", True)

        check = Check(
            name="validated_accuracy",
            dataset=validated_slice,
            model_id="test_model",
            metric="Accuracy",
            operator=">=",
            value=0.5,
        )

        with patch("doleus.checks.base.visualize_report"):
            report = check.run(show=False)

        assert report["check_name"] == "validated_accuracy"
        assert report["dataset_id"] == validated_slice.name
        assert isinstance(report["result"], (int, float))

    def test_checksuite_with_mixed_check_types(
        self, doleus_binary_classification_dataset, basic_metadata
    ):
        """Test CheckSuite with a mix of threshold and evaluation-only checks."""
        dataset = doleus_binary_classification_dataset
        dataset.add_metadata_from_list(basic_metadata)

        predictions = torch.tensor([0.9, 0.3, 0.8, 0.2, 0.7, 0.1, 0.6, 0.4, 0.5, 0.0])
        dataset.add_model_predictions(predictions, "test_model")

        # Create slices
        validated_slice = dataset.slice_by_value("validated", "==", True)
        camera_a_slice = dataset.slice_by_value("source", "==", "camera_a")

        checks = [
            Check(
                name="overall_accuracy_threshold",
                dataset=dataset,
                model_id="test_model",
                metric="Accuracy",
                operator=">=",
                value=0.5,
            ),
            Check(
                name="validated_precision_eval",
                dataset=validated_slice,
                model_id="test_model",
                metric="Precision",
                # No threshold - evaluation only
            ),
            Check(
                name="camera_a_recall_threshold",
                dataset=camera_a_slice,
                model_id="test_model",
                metric="Recall",
                operator=">",
                value=0.3,
            ),
        ]

        suite = CheckSuite(name="mixed_check_suite", checks=checks)

        with patch("doleus.checks.base.visualize_report"):
            report = suite.run_all(show=False)

        assert len(report["checks"]) == 3

        # Verify different check types
        threshold_checks = [c for c in report["checks"] if c["operator"] is not None]
        eval_checks = [c for c in report["checks"] if c["operator"] is None]

        assert len(threshold_checks) == 2
        assert len(eval_checks) == 1

    def test_check_report_schema_consistency(
        self, doleus_binary_classification_dataset
    ):
        """Test that check reports maintain consistent schema."""
        dataset = doleus_binary_classification_dataset

        predictions = torch.tensor([0.9, 0.3, 0.8, 0.2, 0.7, 0.1, 0.6, 0.4, 0.5, 0.0])
        dataset.add_model_predictions(predictions, "test_model")

        # Test different check configurations
        check_configs = [
            {
                "name": "threshold_check",
                "metric": "Accuracy",
                "operator": ">=",
                "value": 0.5,
            },
            {
                "name": "eval_check",
                "metric": "Precision",
                "operator": None,
                "value": None,
            },
        ]

        for config in check_configs:
            check = Check(
                name=config["name"],
                dataset=dataset,
                model_id="test_model",
                metric=config["metric"],
                operator=config.get("operator"),
                value=config.get("value"),
                target_class=config.get("target_class"),
            )

            with patch("doleus.checks.base.visualize_report"):
                report = check.run(show=False)

            # Verify required keys exist
            required_keys = [
                "check_name",
                "dataset_id",
                "metric",
                "operator",
                "value",
                "result",
                "success",
                "timestamp",
            ]

            for key in required_keys:
                assert key in report, f"Missing key: {key}"

            # Verify data types
            assert isinstance(report["check_name"], str)
            assert isinstance(report["dataset_id"], str)
            assert isinstance(report["metric"], str)
            assert isinstance(report["result"], (int, float))
            assert "timestamp" in report

            # Verify operator and value consistency
            if config["operator"] is not None:
                assert report["operator"] == config["operator"]
                assert report["value"] == config["value"]
                assert isinstance(report["success"], bool)
            else:
                assert report["operator"] is None
                assert report["value"] is None
                assert report["success"] is None
