# SPDX-FileCopyrightText: 2025 Doleus contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from doleus.metrics import calculate_metric
from doleus.utils import Task
from torchmetrics import Accuracy, F1Score, Precision, Recall


class TestBinaryClassificationMetrics:

    BINARY_PREDICTIONS_ALL_CORRECT = torch.tensor(
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long
    )
    BINARY_PREDICTIONS_ALL_INCORRECT = torch.tensor(
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.long
    )
    BINARY_PREDICTIONS_NO_FALSE_POSITIVES = torch.tensor(
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 0], dtype=torch.long
    )
    BINARY_PREDICTIONS_WITH_FALSE_POSITIVES = torch.tensor(
        [1, 1, 1, 1, 0, 1, 0, 1, 0, 0], dtype=torch.long
    )

    BINARY_PREDICTION_SCORES_ALL_CORRECT = torch.tensor(
        [0.1, 0.9, 0.2, 0.8, 0.0, 0.7, 0.3, 0.6, 0.1, 0.9], dtype=torch.float
    )
    BINARY_PREDICTION_SCORES_ALL_INCORRECT = torch.tensor(
        [0.9, 0.1, 0.8, 0.2, 0.7, 0.0, 0.6, 0.3, 0.9, 0.1], dtype=torch.float
    )
    BINARY_PREDICTION_SCORES_MIXED = torch.tensor(
        [0.4, 0.6, 0.4, 0.6, 0.4, 0.6, 0.4, 0.6, 0.4, 0.4], dtype=torch.float
    )

    BINARY_PREDICTION_SCORES_STEPWISE = torch.tensor(
        [0.9, 0.8, 0.7, 0.6, 0.55, 0.5, 0.4, 0.3, 0.2, 0.1], dtype=torch.float
    )

    BINARY_PREDICTION_LOGITS_ALL_CORRECT = torch.tensor(
        [-2.2, 2.2, -1.4, 1.4, -3.0, 1.0, -0.8, 0.4, -2.2, 2.2], dtype=torch.float
    )
    BINARY_PREDICTION_LOGITS_ALL_INCORRECT = torch.tensor(
        [2.2, -2.2, 1.4, -1.4, 1.0, -3.0, 0.4, -0.8, 2.2, -2.2], dtype=torch.float
    )
    BINARY_PREDICTION_LOGITS_MIXED = torch.tensor(
        [-0.4, 0.4, -0.4, 0.4, -0.4, 0.4, -0.4, 0.4, -0.4, -0.4], dtype=torch.float
    )

    @pytest.fixture(autouse=True)
    def setup(self, doleus_binary_classification_dataset):
        self.dataset = doleus_binary_classification_dataset

    def _calculate_accuracy(self, predictions, model_id="test_model"):
        self.dataset.prediction_store.add_predictions(
            predictions, model_id=model_id, task=Task.BINARY.value
        )

        # Follows pattern from Check class. A bit ugly. Should be refactored.
        predictions_list = [
            self.dataset.prediction_store.get(model_id=model_id, datapoint_number=i)
            for i in range(len(self.dataset))
        ]

        return calculate_metric(
            dataset=self.dataset, metric="Accuracy", predictions=predictions_list
        )

    def test_accuracy_perfect(self):
        result = self._calculate_accuracy(self.BINARY_PREDICTIONS_ALL_CORRECT)
        assert result == 1.0

    def test_accuracy_zero(self):
        result = self._calculate_accuracy(self.BINARY_PREDICTIONS_ALL_INCORRECT)
        assert result == 0.0

    def test_accuracy_mixed(self):
        result = self._calculate_accuracy(self.BINARY_PREDICTIONS_NO_FALSE_POSITIVES)
        assert result == pytest.approx(0.9)

    def test_accuracy_float_perfect(self):
        result = self._calculate_accuracy(self.BINARY_PREDICTION_SCORES_ALL_CORRECT)
        assert result == 1.0

    def test_accuracy_float_zero(self):
        result = self._calculate_accuracy(self.BINARY_PREDICTION_SCORES_ALL_INCORRECT)
        assert result == 0.0

    def test_accuracy_float_threshold_boundary(self):
        result = self._calculate_accuracy(self.BINARY_PREDICTION_SCORES_MIXED)
        assert result == pytest.approx(0.9)

    def test_accuracy_logits_perfect(self):
        result = self._calculate_accuracy(self.BINARY_PREDICTION_LOGITS_ALL_CORRECT)
        assert result == 1.0

    def test_accuracy_logits_zero(self):
        result = self._calculate_accuracy(self.BINARY_PREDICTION_LOGITS_ALL_INCORRECT)
        assert result == 0.0

    def test_accuracy_logits_mixed(self):
        result = self._calculate_accuracy(self.BINARY_PREDICTION_LOGITS_MIXED)
        assert result == pytest.approx(0.9)

    def test_accuracy_with_custom_threshold(self):
        self.dataset.prediction_store.add_predictions(
            self.BINARY_PREDICTION_SCORES_MIXED,
            model_id="test_model_threshold",
            task=Task.BINARY.value,
        )

        predictions_list = [
            self.dataset.prediction_store.get(
                model_id="test_model_threshold", datapoint_number=i
            )
            for i in range(len(self.dataset))
        ]

        result = calculate_metric(
            dataset=self.dataset,
            metric="Accuracy",
            predictions=predictions_list,
            metric_parameters={"threshold": 0.3},
        )
        assert result == pytest.approx(0.5)

        result = calculate_metric(
            dataset=self.dataset,
            metric="Accuracy",
            predictions=predictions_list,
            metric_parameters={"threshold": 0.7},
        )
        assert result == pytest.approx(0.5)

    def _calculate_precision(self, predictions, model_id="test_model_precision"):
        self.dataset.prediction_store.add_predictions(
            predictions, model_id=model_id, task=Task.BINARY.value
        )

        predictions_list = [
            self.dataset.prediction_store.get(model_id=model_id, datapoint_number=i)
            for i in range(len(self.dataset))
        ]

        return calculate_metric(
            dataset=self.dataset, metric="Precision", predictions=predictions_list
        )

    def test_precision_perfect(self):
        result = self._calculate_precision(self.BINARY_PREDICTIONS_ALL_CORRECT)
        assert result == 1.0

    def test_precision_zero(self):
        result = self._calculate_precision(self.BINARY_PREDICTIONS_ALL_INCORRECT)
        assert result == 0.0

    def test_precision_no_false_positives(self):
        result = self._calculate_precision(self.BINARY_PREDICTIONS_NO_FALSE_POSITIVES)
        assert result == 1.0  # TP=4, FP=0, so precision = 4/(4+0) = 1.0

    def test_precision_with_false_positives(self):
        # TP=4 (positions 1,3,5,7), FP=2 (positions 0,2) → Precision = 4/(4+2) = 2/3
        result = self._calculate_precision(
            self.BINARY_PREDICTIONS_WITH_FALSE_POSITIVES,
            model_id="test_precision_with_fp",
        )
        assert result == pytest.approx(2 / 3)

    def test_precision_float_perfect(self):
        result = self._calculate_precision(self.BINARY_PREDICTION_SCORES_ALL_CORRECT)
        assert result == 1.0

    def test_precision_float_zero(self):
        result = self._calculate_precision(self.BINARY_PREDICTION_SCORES_ALL_INCORRECT)
        assert result == 0.0

    def _calculate_tpr_at_fpr(self, predictions, fpr_threshold, model_id="test_model_tpr_at_fpr"):
        self.dataset.prediction_store.add_predictions(
            predictions, model_id=model_id, task=Task.BINARY.value
        )

        predictions_list = [
            self.dataset.prediction_store.get(model_id=model_id, datapoint_number=i)
            for i in range(len(self.dataset))
        ]

        return calculate_metric(
            dataset=self.dataset,
            metric="TPR_at_FPR",
            predictions=predictions_list,
            metric_parameters={"fpr_threshold": fpr_threshold},
        )

    def test_tpr_at_fpr_perfect_fpr_zero(self):
        result = self._calculate_tpr_at_fpr(
            self.BINARY_PREDICTION_SCORES_ALL_CORRECT, fpr_threshold=0.0
        )
        # ROC curve starts at (FPR=0.0, TPR=0.0), so at FPR=0.0, TPR is 0.0
        assert result == pytest.approx(0.0, abs=0.01)

    def test_tpr_at_fpr_perfect_fpr_low(self):
        result = self._calculate_tpr_at_fpr(
            self.BINARY_PREDICTION_SCORES_ALL_CORRECT, fpr_threshold=0.1
        )
        # With perfect predictions, TPR should be 1.0 at low FPR thresholds
        assert result == pytest.approx(1.0, abs=0.01)

    def test_tpr_at_fpr_perfect_fpr_high(self):
        result = self._calculate_tpr_at_fpr(
            self.BINARY_PREDICTION_SCORES_ALL_CORRECT, fpr_threshold=0.5
        )
        # With perfect predictions, TPR should be 1.0 at high FPR thresholds
        assert result == pytest.approx(1.0, abs=0.01)

    def test_tpr_at_fpr_perfect_fpr_one(self):
        result = self._calculate_tpr_at_fpr(
            self.BINARY_PREDICTION_SCORES_ALL_CORRECT, fpr_threshold=1.0
        )
        # With perfect predictions, TPR should be 1.0 at FPR=1.0
        assert result == pytest.approx(1.0, abs=0.01)

    def test_tpr_at_fpr_incorrect(self):
        result = self._calculate_tpr_at_fpr(
            self.BINARY_PREDICTION_SCORES_ALL_INCORRECT, fpr_threshold=0.5
        )
        # All positives rank below negatives, yielding no true positives before FPR exceeds 0.5
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_tpr_at_fpr_mixed(self):
        result = self._calculate_tpr_at_fpr(
            self.BINARY_PREDICTION_SCORES_MIXED, fpr_threshold=0.5
        )
        # Threshold 0.5 still sits before any false positives, capturing 4/5 positives → TPR = 0.8
        assert result == pytest.approx(0.8, abs=1e-6)

    def test_tpr_at_fpr_stepwise_midpoint(self):
        result = self._calculate_tpr_at_fpr(
            self.BINARY_PREDICTION_SCORES_STEPWISE, fpr_threshold=0.41
        )
        # The ROC curve advances in 0.2 FPR increments; a threshold slightly above 0.4 captures 2/5 positives → TPR ≈ 0.4
        assert result == pytest.approx(0.4, abs=1e-6)

    def test_tpr_at_fpr_stepwise_low(self):
        result = self._calculate_tpr_at_fpr(
            self.BINARY_PREDICTION_SCORES_STEPWISE, fpr_threshold=0.21
        )
        # Allowing just over one false positive (FPR ≈ 0.2) yields exactly one true positive → TPR ≈ 0.2
        assert result == pytest.approx(0.2, abs=1e-6)

    def _calculate_fpr_at_tpr(self, predictions, tpr_threshold, model_id="test_model_fpr_at_tpr"):
        self.dataset.prediction_store.add_predictions(
            predictions, model_id=model_id, task=Task.BINARY.value
        )

        predictions_list = [
            self.dataset.prediction_store.get(model_id=model_id, datapoint_number=i)
            for i in range(len(self.dataset))
        ]

        return calculate_metric(
            dataset=self.dataset,
            metric="FPR_at_TPR",
            predictions=predictions_list,
            metric_parameters={"tpr_threshold": tpr_threshold},
        )

    def test_fpr_at_tpr_perfect_tpr_zero(self):
        result = self._calculate_fpr_at_tpr(
            self.BINARY_PREDICTION_SCORES_ALL_CORRECT, tpr_threshold=0.0
        )
        # ROC curve starts at (FPR=0.0, TPR=0.0), so at TPR=0.0, FPR is 0.0
        assert result == pytest.approx(0.0, abs=0.01)

    def test_fpr_at_tpr_perfect_tpr_low(self):
        result = self._calculate_fpr_at_tpr(
            self.BINARY_PREDICTION_SCORES_ALL_CORRECT, tpr_threshold=0.5
        )
        # Perfect ranking means no negatives are admitted before 50% recall → FPR = 0.0
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_fpr_at_tpr_perfect_tpr_high(self):
        result = self._calculate_fpr_at_tpr(
            self.BINARY_PREDICTION_SCORES_ALL_CORRECT, tpr_threshold=0.9
        )
        # Even at 90% recall the negatives remain untouched → FPR = 0.0
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_fpr_at_tpr_perfect_tpr_one(self):
        result = self._calculate_fpr_at_tpr(
            self.BINARY_PREDICTION_SCORES_ALL_CORRECT, tpr_threshold=1.0
        )
        # With perfect predictions, FPR should be 0.0 at TPR=1.0
        assert result == pytest.approx(0.0, abs=0.01)

    def test_fpr_at_tpr_incorrect(self):
        result = self._calculate_fpr_at_tpr(
            self.BINARY_PREDICTION_SCORES_ALL_INCORRECT, tpr_threshold=0.5
        )
        # Reaching TPR 0.5 requires accepting every negative first → FPR = 1.0
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_fpr_at_tpr_mixed(self):
        result = self._calculate_fpr_at_tpr(
            self.BINARY_PREDICTION_SCORES_MIXED, tpr_threshold=0.5
        )
        # Half of the positives are recovered before any negative is accepted → FPR = 0.0
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_fpr_at_tpr_stepwise_midpoint(self):
        result = self._calculate_fpr_at_tpr(
            self.BINARY_PREDICTION_SCORES_STEPWISE, tpr_threshold=0.6
        )
        # Achieving TPR 0.6 requires admitting exactly three of the five negatives → FPR = 0.6
        assert result == pytest.approx(0.6, abs=1e-6)

    def test_fpr_at_tpr_stepwise_low(self):
        result = self._calculate_fpr_at_tpr(
            self.BINARY_PREDICTION_SCORES_STEPWISE, tpr_threshold=0.2
        )
        # First non-zero TPR is attained while still keeping FPR at 0.2
        assert result == pytest.approx(0.2, abs=1e-6)

    def test_tpr_at_fpr_missing_fpr_threshold(self):
        """Test that TPR_at_FPR raises RuntimeError when fpr_threshold is missing."""
        self.dataset.prediction_store.add_predictions(
            self.BINARY_PREDICTION_SCORES_ALL_CORRECT,
            model_id="test_missing_fpr_threshold",
            task=Task.BINARY.value,
        )

        predictions_list = [
            self.dataset.prediction_store.get(
                model_id="test_missing_fpr_threshold", datapoint_number=i
            )
            for i in range(len(self.dataset))
        ]

        with pytest.raises(RuntimeError, match="TPR_at_FPR requires 'fpr_threshold' parameter"):
            calculate_metric(
                dataset=self.dataset,
                metric="TPR_at_FPR",
                predictions=predictions_list,
                metric_parameters={},
            )

    def test_fpr_at_tpr_missing_tpr_threshold(self):
        """Test that FPR_at_TPR raises RuntimeError when tpr_threshold is missing."""
        self.dataset.prediction_store.add_predictions(
            self.BINARY_PREDICTION_SCORES_ALL_CORRECT,
            model_id="test_missing_tpr_threshold",
            task=Task.BINARY.value,
        )

        predictions_list = [
            self.dataset.prediction_store.get(
                model_id="test_missing_tpr_threshold", datapoint_number=i
            )
            for i in range(len(self.dataset))
        ]

        with pytest.raises(RuntimeError, match="FPR_at_TPR requires 'tpr_threshold' parameter"):
            calculate_metric(
                dataset=self.dataset,
                metric="FPR_at_TPR",
                predictions=predictions_list,
                metric_parameters={},
            )

    def test_tpr_at_fpr_wrong_task_type(self, doleus_multiclass_classification_dataset):
        """Test that TPR_at_FPR raises RuntimeError for multiclass tasks."""
        multiclass_dataset = doleus_multiclass_classification_dataset
        multiclass_dataset.prediction_store.add_predictions(
            torch.tensor([0.8, 0.2, 0.1] * 10).reshape(10, 3),
            model_id="test_wrong_task",
            task=Task.MULTICLASS.value,
        )

        predictions_list = [
            multiclass_dataset.prediction_store.get(
                model_id="test_wrong_task", datapoint_number=i
            )
            for i in range(len(multiclass_dataset))
        ]

        with pytest.raises(RuntimeError, match="TPR_at_FPR only supports binary classification"):
            calculate_metric(
                dataset=multiclass_dataset,
                metric="TPR_at_FPR",
                predictions=predictions_list,
                metric_parameters={"fpr_threshold": 0.1},
            )

    def test_fpr_at_tpr_wrong_task_type(self, doleus_multiclass_classification_dataset):
        """Test that FPR_at_TPR raises RuntimeError for multiclass tasks."""
        multiclass_dataset = doleus_multiclass_classification_dataset
        multiclass_dataset.prediction_store.add_predictions(
            torch.tensor([0.8, 0.2, 0.1] * 10).reshape(10, 3),
            model_id="test_wrong_task_fpr",
            task=Task.MULTICLASS.value,
        )

        predictions_list = [
            multiclass_dataset.prediction_store.get(
                model_id="test_wrong_task_fpr", datapoint_number=i
            )
            for i in range(len(multiclass_dataset))
        ]

        with pytest.raises(RuntimeError, match="FPR_at_TPR only supports binary classification"):
            calculate_metric(
                dataset=multiclass_dataset,
                metric="FPR_at_TPR",
                predictions=predictions_list,
                metric_parameters={"tpr_threshold": 0.1},
            )

    def test_tpr_at_fpr_missing_scores(self):
        """Test that TPR_at_FPR raises RuntimeError when predictions contain labels instead of scores."""
        self.dataset.prediction_store.add_predictions(
            self.BINARY_PREDICTIONS_ALL_CORRECT,
            model_id="test_missing_scores_tpr",
            task=Task.BINARY.value,
        )

        predictions_list = [
            self.dataset.prediction_store.get(
                model_id="test_missing_scores_tpr", datapoint_number=i
            )
            for i in range(len(self.dataset))
        ]

        with pytest.raises(
            RuntimeError, match="TPR_at_FPR requires prediction scores/logits"
        ):
            calculate_metric(
                dataset=self.dataset,
                metric="TPR_at_FPR",
                predictions=predictions_list,
                metric_parameters={"fpr_threshold": 0.1},
            )

    def test_fpr_at_tpr_missing_scores(self):
        """Test that FPR_at_TPR raises RuntimeError when predictions contain labels instead of scores."""
        self.dataset.prediction_store.add_predictions(
            self.BINARY_PREDICTIONS_ALL_CORRECT,
            model_id="test_missing_scores_fpr",
            task=Task.BINARY.value,
        )

        predictions_list = [
            self.dataset.prediction_store.get(
                model_id="test_missing_scores_fpr", datapoint_number=i
            )
            for i in range(len(self.dataset))
        ]

        with pytest.raises(
            RuntimeError, match="FPR_at_TPR requires prediction scores/logits"
        ):
            calculate_metric(
                dataset=self.dataset,
                metric="FPR_at_TPR",
                predictions=predictions_list,
                metric_parameters={"tpr_threshold": 0.1},
            )


class TestMulticlassClassificationMetrics:

    MULTICLASS_PREDICTIONS_ALL_CORRECT = torch.tensor(
        [0, 0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=torch.long
    )
    MULTICLASS_PREDICTIONS_ALL_INCORRECT = torch.tensor(
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=torch.long
    )
    # Per-class accuracy: Class 0 (2/4), Class 1 (2/3), Class 2 (1/3). Macro avg: 0.5
    MULTICLASS_PREDICTIONS_MACRO_50 = torch.tensor(
        [0, 0, 1, 1, 1, 1, 0, 2, 0, 0], dtype=torch.long
    )

    MULTICLASS_PREDICTION_SCORES_ALL_CORRECT = torch.tensor(
        [
            [0.8, 0.1, 0.1],
            [0.7, 0.2, 0.1],
            [0.9, 0.05, 0.05],
            [0.6, 0.3, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.85, 0.05],
            [0.1, 0.1, 0.8],
            [0.05, 0.15, 0.8],
            [0.2, 0.1, 0.7],
        ],
        dtype=torch.float,
    )
    MULTICLASS_PREDICTION_SCORES_ALL_INCORRECT = torch.tensor(
        [
            [0.1, 0.8, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.85, 0.05],
            [0.3, 0.6, 0.1],
            [0.8, 0.1, 0.1],
            [0.7, 0.2, 0.1],
            [0.9, 0.05, 0.05],
            [0.8, 0.1, 0.1],
            [0.7, 0.2, 0.1],
            [0.8, 0.1, 0.1],
        ],
        dtype=torch.float,
    )
    MULTICLASS_PREDICTION_SCORES_MACRO_50 = torch.tensor(
        [
            [0.8, 0.1, 0.1],
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.7, 0.1],
            [0.8, 0.1, 0.1],
            [0.1, 0.1, 0.8],
            [0.8, 0.1, 0.1],
            [0.7, 0.2, 0.1],
        ],
        dtype=torch.float,
    )
    MULTICLASS_PREDICTION_SCORES_WITH_FALSE_NEGATIVES = torch.tensor(
        [
            [0.8, 0.1, 0.1],
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.1, 0.8, 0.1],
            [0.2, 0.7, 0.1],
            [0.8, 0.1, 0.1],
            [0.1, 0.1, 0.8],
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
        ],
        dtype=torch.float,
    )

    MULTICLASS_PREDICTION_LOGITS_ALL_CORRECT = torch.tensor(
        [
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
        ],
        dtype=torch.float,
    )
    MULTICLASS_LOGITS_MACRO_50 = torch.tensor(
        [
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0],
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=torch.float,
    )

    @pytest.fixture(autouse=True)
    def setup(self, doleus_multiclass_classification_dataset):
        self.dataset = doleus_multiclass_classification_dataset

    def _calculate_accuracy(self, predictions, model_id="test_model"):
        self.dataset.prediction_store.add_predictions(
            predictions, model_id=model_id, task=Task.MULTICLASS.value
        )

        predictions_list = [
            self.dataset.prediction_store.get(model_id=model_id, datapoint_number=i)
            for i in range(len(self.dataset))
        ]

        return calculate_metric(
            dataset=self.dataset, metric="Accuracy", predictions=predictions_list
        )

    def test_accuracy_perfect(self):
        result = self._calculate_accuracy(self.MULTICLASS_PREDICTIONS_ALL_CORRECT)
        assert result == 1.0

    def test_accuracy_zero(self):
        result = self._calculate_accuracy(self.MULTICLASS_PREDICTIONS_ALL_INCORRECT)
        assert result == 0.0

    def test_accuracy_macro_averaging(self):
        result = self._calculate_accuracy(self.MULTICLASS_PREDICTIONS_MACRO_50)
        assert result == pytest.approx(0.5)

    def test_accuracy_scores_perfect(self):
        result = self._calculate_accuracy(self.MULTICLASS_PREDICTION_SCORES_ALL_CORRECT)
        assert result == 1.0

    def test_accuracy_scores_zero(self):
        result = self._calculate_accuracy(
            self.MULTICLASS_PREDICTION_SCORES_ALL_INCORRECT
        )
        assert result == 0.0

    def test_accuracy_scores_macro_averaging(self):
        result = self._calculate_accuracy(self.MULTICLASS_PREDICTION_SCORES_MACRO_50)
        assert result == pytest.approx(0.5)

    def test_accuracy_logits_perfect(self):
        result = self._calculate_accuracy(self.MULTICLASS_PREDICTION_LOGITS_ALL_CORRECT)
        assert result == 1.0

    def test_accuracy_logits_macro_averaging(self):
        result = self._calculate_accuracy(self.MULTICLASS_LOGITS_MACRO_50)
        assert result == pytest.approx(0.5)

    def test_accuracy_with_explicit_macro_average(self):
        """Test that explicitly setting average='macro' gives same result as default."""
        self.dataset.prediction_store.add_predictions(
            self.MULTICLASS_PREDICTIONS_ALL_CORRECT,
            model_id="test_explicit_macro",
            task=Task.MULTICLASS.value,
        )

        predictions_list = [
            self.dataset.prediction_store.get(
                model_id="test_explicit_macro", datapoint_number=i
            )
            for i in range(len(self.dataset))
        ]

        result = calculate_metric(
            dataset=self.dataset,
            metric="Accuracy",
            predictions=predictions_list,
            metric_parameters={"average": "macro"},
        )
        assert result == 1.0

    def _calculate_recall(self, predictions, model_id="test_model_recall"):
        self.dataset.prediction_store.add_predictions(
            predictions, model_id=model_id, task=Task.MULTICLASS.value
        )

        predictions_list = [
            self.dataset.prediction_store.get(model_id=model_id, datapoint_number=i)
            for i in range(len(self.dataset))
        ]

        return calculate_metric(
            dataset=self.dataset, metric="Recall", predictions=predictions_list
        )

    def test_recall_perfect_scores(self):
        result = self._calculate_recall(self.MULTICLASS_PREDICTION_SCORES_ALL_CORRECT)
        assert result == 1.0

    def test_recall_zero_scores(self):
        result = self._calculate_recall(self.MULTICLASS_PREDICTION_SCORES_ALL_INCORRECT)
        assert result == 0.0

    def test_recall_macro_averaging_scores(self):
        # Per-class recall: Class 0 (2/4), Class 1 (2/3), Class 2 (1/3). Macro avg: 0.5
        result = self._calculate_recall(
            self.MULTICLASS_PREDICTION_SCORES_MACRO_50,
            model_id="test_recall_macro_scores",
        )
        assert result == pytest.approx(0.5)

    def test_recall_with_false_negatives_scores(self):
        # Per-class recall: Class 0 (2/4), Class 1 (2/3), Class 2 (1/3). Macro avg: 0.5
        result = self._calculate_recall(
            self.MULTICLASS_PREDICTION_SCORES_WITH_FALSE_NEGATIVES,
            model_id="test_recall_false_negatives_scores",
        )
        assert result == pytest.approx(0.5)


class TestMultilabelClassificationMetrics:

    MULTILABEL_PREDICTIONS_ALL_CORRECT = torch.tensor(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ],
        dtype=torch.float,
    )

    MULTILABEL_PREDICTIONS_ALL_INCORRECT = torch.tensor(
        [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 0],
            [1, 1, 1],
            [0, 1, 1],
            [1, 0, 1],
        ],
        dtype=torch.float,
    )

    # Per-label accuracy: Label 0 (5/10), Label 1 (5/10), Label 2 (5/10). Macro avg: 0.5
    MULTILABEL_PREDICTIONS_MACRO_50 = torch.tensor(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 0, 1],
            [0, 0, 0],
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        dtype=torch.float,
    )

    MULTILABEL_PREDICTION_SCORES_ALL_CORRECT = torch.tensor(
        [
            [0.9, 0.1, 0.1],
            [0.1, 0.9, 0.1],
            [0.1, 0.1, 0.9],
            [0.8, 0.8, 0.1],
            [0.9, 0.1, 0.8],
            [0.1, 0.9, 0.8],
            [0.8, 0.8, 0.8],
            [0.1, 0.1, 0.1],
            [0.9, 0.1, 0.1],
            [0.1, 0.9, 0.1],
        ],
        dtype=torch.float,
    )

    MULTILABEL_PREDICTION_SCORES_ALL_INCORRECT = torch.tensor(
        [
            [0.1, 0.9, 0.9],
            [0.9, 0.1, 0.9],
            [0.9, 0.9, 0.1],
            [0.1, 0.1, 0.9],
            [0.1, 0.9, 0.1],
            [0.9, 0.1, 0.1],
            [0.1, 0.1, 0.1],
            [0.9, 0.9, 0.9],
            [0.1, 0.9, 0.9],
            [0.9, 0.1, 0.9],
        ],
        dtype=torch.float,
    )

    MULTILABEL_PREDICTION_SCORES_MACRO_50 = torch.tensor(
        [
            [0.9, 0.9, 0.9],
            [0.9, 0.1, 0.9],
            [0.9, 0.1, 0.9],
            [0.8, 0.8, 0.9],
            [0.1, 0.1, 0.9],
            [0.1, 0.1, 0.1],
            [0.8, 0.8, 0.1],
            [0.9, 0.9, 0.1],
            [0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1],
        ],
        dtype=torch.float,
    )

    MULTILABEL_PREDICTION_LOGITS_ALL_CORRECT = torch.tensor(
        [
            [2.0, -2.0, -2.0],
            [-2.0, 2.0, -2.0],
            [-2.0, -2.0, 2.0],
            [2.0, 2.0, -2.0],
            [2.0, -2.0, 2.0],
            [-2.0, 2.0, 2.0],
            [2.0, 2.0, 2.0],
            [-2.0, -2.0, -2.0],
            [2.0, -2.0, -2.0],
            [-2.0, 2.0, -2.0],
        ],
        dtype=torch.float,
    )

    MULTILABEL_LOGITS_MACRO_50 = torch.tensor(
        [
            [2.0, 2.0, 2.0],
            [2.0, -2.0, 2.0],
            [2.0, -2.0, 2.0],
            [2.0, 2.0, 2.0],
            [-2.0, -2.0, 2.0],
            [-2.0, -2.0, -2.0],
            [2.0, 2.0, -2.0],
            [2.0, 2.0, -2.0],
            [-2.0, -2.0, -2.0],
            [-2.0, -2.0, -2.0],
        ],
        dtype=torch.float,
    )
    MULTILABEL_LOGITS_F1_MIXED = torch.tensor(
        [
            [2.0, -2.0, -2.0],
            [-2.0, 2.0, -2.0],
            [2.0, -2.0, 2.0],
            [2.0, 2.0, -2.0],
            [2.0, -2.0, 2.0],
            [-2.0, 2.0, 2.0],
            [2.0, 2.0, 2.0],
            [2.0, 2.0, -2.0],
            [-2.0, -2.0, -2.0],
            [-2.0, 2.0, -2.0],
        ],
        dtype=torch.float,
    )
    MULTILABEL_LOGITS_F1_ZERO = torch.tensor(
        [
            [-2.0, 2.0, 2.0],
            [2.0, -2.0, 2.0],
            [2.0, 2.0, -2.0],
            [-2.0, -2.0, 2.0],
            [-2.0, 2.0, -2.0],
            [2.0, -2.0, -2.0],
            [-2.0, -2.0, -2.0],
            [2.0, 2.0, 2.0],
            [-2.0, 2.0, 2.0],
            [2.0, -2.0, 2.0],
        ],
        dtype=torch.float,
    )

    @pytest.fixture(autouse=True)
    def setup(self, doleus_multilabel_classification_dataset):
        self.dataset = doleus_multilabel_classification_dataset

    def _calculate_accuracy(self, predictions, model_id="test_model"):
        self.dataset.prediction_store.add_predictions(
            predictions, model_id=model_id, task=Task.MULTILABEL.value
        )

        predictions_list = [
            self.dataset.prediction_store.get(model_id=model_id, datapoint_number=i)
            for i in range(len(self.dataset))
        ]

        return calculate_metric(
            dataset=self.dataset, metric="Accuracy", predictions=predictions_list
        )

    def test_accuracy_perfect(self):
        result = self._calculate_accuracy(self.MULTILABEL_PREDICTIONS_ALL_CORRECT)
        assert result == 1.0

    def test_accuracy_zero(self):
        result = self._calculate_accuracy(self.MULTILABEL_PREDICTIONS_ALL_INCORRECT)
        assert result == 0.0

    def test_accuracy_macro_averaging(self):
        result = self._calculate_accuracy(self.MULTILABEL_PREDICTIONS_MACRO_50)
        assert result == pytest.approx(0.5)

    def test_accuracy_scores_perfect(self):
        result = self._calculate_accuracy(self.MULTILABEL_PREDICTION_SCORES_ALL_CORRECT)
        assert result == 1.0

    def test_accuracy_scores_zero(self):
        result = self._calculate_accuracy(
            self.MULTILABEL_PREDICTION_SCORES_ALL_INCORRECT
        )
        assert result == 0.0

    def test_accuracy_scores_macro_averaging(self):
        result = self._calculate_accuracy(self.MULTILABEL_PREDICTION_SCORES_MACRO_50)
        assert result == pytest.approx(0.5)

    def test_accuracy_logits_perfect(self):
        result = self._calculate_accuracy(self.MULTILABEL_PREDICTION_LOGITS_ALL_CORRECT)
        assert result == 1.0

    def test_accuracy_logits_macro_averaging(self):
        result = self._calculate_accuracy(self.MULTILABEL_LOGITS_MACRO_50)
        assert result == pytest.approx(0.5)

    def test_accuracy_with_explicit_macro_average(self):
        """Test that explicitly setting average='macro' gives same result as default."""
        self.dataset.prediction_store.add_predictions(
            self.MULTILABEL_PREDICTIONS_ALL_CORRECT,
            model_id="test_explicit_macro",
            task=Task.MULTILABEL.value,
        )

        predictions_list = [
            self.dataset.prediction_store.get(
                model_id="test_explicit_macro", datapoint_number=i
            )
            for i in range(len(self.dataset))
        ]

        result = calculate_metric(
            dataset=self.dataset,
            metric="Accuracy",
            predictions=predictions_list,
            metric_parameters={"average": "macro"},
        )
        assert result == 1.0

    def _calculate_f1_score(self, predictions, model_id="test_model_f1"):
        self.dataset.prediction_store.add_predictions(
            predictions, model_id=model_id, task=Task.MULTILABEL.value
        )

        predictions_list = [
            self.dataset.prediction_store.get(model_id=model_id, datapoint_number=i)
            for i in range(len(self.dataset))
        ]

        return calculate_metric(
            dataset=self.dataset, metric="F1_Score", predictions=predictions_list
        )

    def test_f1_score_perfect_logits(self):
        result = self._calculate_f1_score(self.MULTILABEL_PREDICTION_LOGITS_ALL_CORRECT)
        assert result == 1.0

    def test_f1_score_zero_logits(self):
        # Ground truth: [1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1], [1,1,1], [0,0,0], [1,0,0], [0,1,0]
        # Predicted:    [0,1,1], [1,0,1], [1,1,0], [0,0,1], [0,1,0], [1,0,0], [0,0,0], [1,1,1], [0,1,1], [1,0,1]
        # All predictions are wrong (no true positives) → F1 = 0 for all labels
        result = self._calculate_f1_score(
            self.MULTILABEL_LOGITS_F1_ZERO, model_id="test_f1_zero"
        )
        assert result == 0.0

    def test_f1_score_mixed_logits(self):
        # Ground truth: [1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1], [1,1,1], [0,0,0], [1,0,0], [0,1,0]
        # Predicted:    [1,0,0], [0,1,0], [1,0,1], [1,1,0], [1,0,1], [0,1,1], [1,1,1], [1,1,0], [0,0,0], [0,1,0]
        # Label 0: TP=5, FP=1, FN=1 → Precision=5/6≈0.83, Recall=5/6≈0.83 → F1≈0.83
        # Label 1: TP=5, FP=1, FN=0 → Precision=5/6≈0.83, Recall=5/5=1.0 → F1≈0.91
        # Label 2: TP=3, FP=1, FN=1 → Precision=3/4=0.75, Recall=3/4=0.75 → F1=0.75
        # Macro avg F1 ≈ (0.83 + 0.91 + 0.75) / 3 ≈ 0.83
        result = self._calculate_f1_score(
            self.MULTILABEL_LOGITS_F1_MIXED, model_id="test_f1_mixed_logits"
        )
        assert result == pytest.approx(0.83, abs=0.05)


def test_torchmetrics_micro_averaging_bug_still_exists():
    """
    Canary test to verify that torchmetrics issue #2280 still persists.

    This test verifies the assumption behind our macro averaging default.
    When this test starts failing, it means torchmetrics has fixed the bug
    and we should reconsider our default averaging strategy.
    """
    # Recreating the data from the torchmetrics issue
    num_classes = 3

    list_of_metrics = [
        Accuracy(task="multiclass", num_classes=num_classes, average="micro"),
        F1Score(task="multiclass", num_classes=num_classes),
        Precision(task="multiclass", num_classes=num_classes),
        Recall(task="multiclass", num_classes=num_classes),
    ]

    pred = torch.Tensor(
        [
            [0, 0.1, 0.5],  # 2
            [0, 0.1, 0.5],  # 2
            [0, 0.1, 0.5],  # 2
            [0, 0.1, 0.5],  # 2
            [0, 0.9, 0.1],  # 1
            [0.9, 0.1, 0],
        ]
    )  # 1

    label = torch.Tensor([2, 2, 2, 0, 2, 1])

    # Expected behavior from the issue:
    # Class 2 we recall 3/4 = 0.75
    # Class 1 we recall 0/1 = 0
    # Class 0 we recall 0/1 = 0
    # Macro average recall should be (0.75 + 0 + 0) / 3 = 0.25
    # But torchmetrics incorrectly applies micro averaging, giving 0.5 for all metrics

    # Convert predictions to class indices (argmax)
    pred_classes = pred.argmax(dim=1)

    # Calculate all metrics
    acc_val = list_of_metrics[0](pred_classes, label.long()).item()
    f1_val = list_of_metrics[1](pred_classes, label.long()).item()
    prec_val = list_of_metrics[2](pred_classes, label.long()).item()
    rec_val = list_of_metrics[3](pred_classes, label.long()).item()

    print(
        f"Torchmetrics values: Acc={acc_val:.4f}, Prec={prec_val:.4f}, Rec={rec_val:.4f}, F1={f1_val:.4f}"
    )

    # If the bug still exists, these should all be identical (micro averaging applied incorrectly)
    # When the bug is fixed, these should be different values
    assert acc_val == prec_val == rec_val == f1_val, (
        "Torchmetrics issue #2280 appears to be fixed! "
        f"Metrics are now different: Acc={acc_val:.4f}, Prec={prec_val:.4f}, Rec={rec_val:.4f}, F1={f1_val:.4f}. "
        "Consider removing macro averaging as default in doleus/metrics/calculator.py"
    )
