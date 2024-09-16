import pytest
import torch
from doleus.datasets import DoleusDetection
from doleus.metrics import calculate_metric


class TestDetectionMetrics:
    DETECTION_ALL_CORRECT_NO_SCORES = [
        {"boxes": [[10, 10, 50, 50]], "labels": [0]},
        {"boxes": [[20, 20, 60, 60]], "labels": [1]},
        {"boxes": [[30, 30, 70, 70]], "labels": [2]},
        {"boxes": [[10, 10, 40, 40], [50, 50, 80, 80]], "labels": [0, 1]},
        {"boxes": [[15, 15, 45, 45], [55, 55, 85, 85]], "labels": [0, 2]},
        {"boxes": [[20, 20, 50, 50], [60, 60, 90, 90]], "labels": [1, 2]},
        {
            "boxes": [[10, 10, 30, 30], [40, 40, 60, 60], [70, 70, 90, 90]],
            "labels": [0, 1, 2],
        },
        {
            "boxes": [[15, 15, 35, 35], [45, 45, 65, 65], [75, 75, 95, 95]],
            "labels": [0, 0, 1],
        },
        {"boxes": [[25, 25, 65, 65]], "labels": [0]},
        {"boxes": [[35, 35, 75, 75]], "labels": [1]},
    ]

    DETECTION_ALL_CORRECT = [
        {**d, "scores": [1.0] * len(d["labels"])}
        for d in DETECTION_ALL_CORRECT_NO_SCORES
    ]

    @pytest.fixture(autouse=True)
    def setup(self, doleus_object_detection_dataset):
        self.dataset = doleus_object_detection_dataset

    def _calculate_mAP(self, predictions, model_id="test_model"):
        self.dataset.prediction_store.add_predictions(predictions, model_id=model_id)

        predictions_list = [
            self.dataset.prediction_store.get(model_id=model_id, datapoint_number=i)
            for i in range(len(predictions))
        ]

        return calculate_metric(
            dataset=self.dataset, metric="mAP", predictions=predictions_list
        )

    def test_mAP_perfect(self):
        predictions = self.DETECTION_ALL_CORRECT
        result = self._calculate_mAP(predictions)
        assert result == 1.0

    def _calculate_iou(self, predictions, model_id="test_model_iou"):
        self.dataset.prediction_store.add_predictions(predictions, model_id=model_id)

        predictions_list = [
            self.dataset.prediction_store.get(model_id=model_id, datapoint_number=i)
            for i in range(len(predictions))
        ]

        return calculate_metric(
            dataset=self.dataset,
            metric="IoU",
            predictions=predictions_list,
        )

    def test_iou_perfect(self):
        predictions = self.DETECTION_ALL_CORRECT_NO_SCORES
        result = self._calculate_iou(predictions)
        # The `IntersectionOverUnion` metric does not find the best one-to-one matches.
        # Instead, it averages the IoU of ALL valid prediction-target pairs, where a
        # pair is valid if the prediction and target have the same class label.
        #
        # EXAMPLE:
        # Consider one image with two ground truth boxes of the same class (label=0):
        # GT    = [box_A (label=0), box_B (label=0)]
        # Preds = [box_A (label=0), box_B (label=0)]  <- A perfect prediction
        #
        # The metric calculates the IoU for all 4 valid (same-label) pairs:
        #  - iou(Preds[0], GT[0]) = 1.0
        #  - iou(Preds[0], GT[1]) = 0.0  <- Non-overlapping boxes
        #  - iou(Preds[1], GT[0]) = 0.0  <- Non-overlapping boxes
        #  - iou(Preds[1], GT[1]) = 1.0
        # The score for this image would be (1+0+0+1)/4 = 0.5, not 1.0.
        #
        # HOW THE SCORE IS CALCULATED FOR THIS TEST:
        # Across our 10-image dataset, there are 17 perfect (IoU=1.0) pairs and
        # 2 additional zero-IoU pairs from the single image with two 'label=0' boxes.
        # Total IoU sum     = 17.0
        # Total valid pairs = 19
        # Final Score = 17 / 19 = 0.8947368...
        assert result == pytest.approx(0.8947368)

    def test_iou_is_one_for_perfect_match_with_distinct_labels(self):
        gt_boxes = torch.tensor([[10, 10, 50, 50], [60, 60, 100, 100]])
        gt_labels = torch.tensor([0, 1])

        class MockDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 1

            def __getitem__(self, idx):
                return (None, gt_boxes, gt_labels)

        dataset = DoleusDetection(dataset=MockDataset(), name="test_distinct_labels")

        predictions_data = [{"boxes": gt_boxes.clone(), "labels": gt_labels.clone()}]
        dataset.prediction_store.add_predictions(
            predictions_data, model_id="perfect_model"
        )

        predictions_list = [dataset.prediction_store.get("perfect_model", 0)]
        result = calculate_metric(
            dataset=dataset, metric="IoU", predictions=predictions_list
        )

        assert result == pytest.approx(
            1.0
        )  # Should return 1.0 because the labels are different.

    def test_iou_is_average_for_perfect_match_with_duplicate_labels(self):
        gt_boxes = torch.tensor([[10, 10, 50, 50], [60, 60, 100, 100]])
        gt_labels = torch.tensor([0, 0])

        class MockDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 1

            def __getitem__(self, idx):
                return (None, gt_boxes, gt_labels)

        dataset = DoleusDetection(dataset=MockDataset(), name="test_duplicate_labels")

        predictions_data = [{"boxes": gt_boxes.clone(), "labels": gt_labels.clone()}]
        dataset.prediction_store.add_predictions(
            predictions_data, model_id="half_model"
        )

        predictions_list = [dataset.prediction_store.get("half_model", 0)]
        result = calculate_metric(
            dataset=dataset, metric="IoU", predictions=predictions_list
        )

        assert result == pytest.approx(
            0.5
        )  # Should return 0.5 because the labels are the same.
