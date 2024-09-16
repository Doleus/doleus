import pytest
import torch
from doleus.annotations import Annotations
from doleus.annotations.detection import BoundingBoxes
from doleus.storage.prediction_store.detection import DetectionPredictionStore


class TestDetectionPredictionStore:
    """Test suite for DetectionPredictionStore."""

    def test_initialization(self):
        """Test initialization of the prediction store."""
        store = DetectionPredictionStore()
        assert hasattr(store, "predictions")
        assert isinstance(store.predictions, dict)
        assert len(store.predictions) == 0

    def test_add_predictions_requires_list(self):
        """Test that add_predictions requires a list input."""
        store = DetectionPredictionStore()
        invalid_predictions = torch.tensor([0.8, 0.2, 0.9])

        with pytest.raises(
            TypeError, match="For detection, predictions must be a list"
        ):
            store.add_predictions(invalid_predictions, "test_model")

    def test_add_predictions_requires_dict_elements(self):
        """Test that each prediction must be a dictionary."""
        store = DetectionPredictionStore()
        invalid_predictions = ["not_a_dict", [1, 2, 3]]

        with pytest.raises(
            TypeError,
            match="Each item in detection predictions list must be a dictionary",
        ):
            store.add_predictions(invalid_predictions, "test_model")

    def test_valid_detection_predictions(self):
        """Test adding valid detection predictions."""
        store = DetectionPredictionStore()
        predictions = [
            {
                "boxes": torch.tensor(
                    [[10, 20, 100, 200], [50, 60, 150, 250]], dtype=torch.float32
                ),
                "labels": torch.tensor([0, 1], dtype=torch.long),
                "scores": torch.tensor([0.9, 0.8], dtype=torch.float32),
            },
            {
                "boxes": torch.tensor([[5, 15, 95, 195]], dtype=torch.float32),
                "labels": torch.tensor([2], dtype=torch.long),
                "scores": torch.tensor([0.7], dtype=torch.float32),
            },
        ]

        store.add_predictions(predictions, "detection_model")

        annotations = store.get_predictions("detection_model")
        assert len(annotations.annotations) == 2

        first_pred = store.get("detection_model", 0)
        assert isinstance(first_pred, BoundingBoxes)
        assert first_pred.datapoint_number == 0
        assert torch.equal(
            first_pred.boxes_xyxy,
            torch.tensor([[10, 20, 100, 200], [50, 60, 150, 250]], dtype=torch.float32),
        )
        assert torch.equal(first_pred.labels, torch.tensor([0, 1], dtype=torch.long))
        assert torch.equal(
            first_pred.scores, torch.tensor([0.9, 0.8], dtype=torch.float32)
        )

        second_pred = store.get("detection_model", 1)
        assert isinstance(second_pred, BoundingBoxes)
        assert second_pred.datapoint_number == 1
        assert torch.equal(
            second_pred.boxes_xyxy,
            torch.tensor([[5, 15, 95, 195]], dtype=torch.float32),
        )
        assert torch.equal(second_pred.labels, torch.tensor([2], dtype=torch.long))
        assert torch.equal(second_pred.scores, torch.tensor([0.7], dtype=torch.float32))

    def test_valid_detection_predictions_no_scores(self):
        """Test adding valid detection predictions without scores."""
        store = DetectionPredictionStore()
        predictions = [
            {
                "boxes": torch.tensor(
                    [[10, 20, 100, 200], [50, 60, 150, 250]], dtype=torch.float32
                ),
                "labels": torch.tensor([0, 1], dtype=torch.long),
            },
            {
                "boxes": torch.tensor([[5, 15, 95, 195]], dtype=torch.float32),
                "labels": torch.tensor([2], dtype=torch.long),
            },
        ]

        store.add_predictions(predictions, "detection_model_no_scores")

        annotations = store.get_predictions("detection_model_no_scores")
        assert len(annotations.annotations) == 2

        first_pred = store.get("detection_model_no_scores", 0)
        assert isinstance(first_pred, BoundingBoxes)
        assert first_pred.datapoint_number == 0
        assert torch.equal(
            first_pred.boxes_xyxy,
            torch.tensor([[10, 20, 100, 200], [50, 60, 150, 250]], dtype=torch.float32),
        )
        assert torch.equal(first_pred.labels, torch.tensor([0, 1], dtype=torch.long))
        assert first_pred.scores is None

        second_pred = store.get("detection_model_no_scores", 1)
        assert isinstance(second_pred, BoundingBoxes)
        assert second_pred.datapoint_number == 1
        assert torch.equal(
            second_pred.boxes_xyxy,
            torch.tensor([[5, 15, 95, 195]], dtype=torch.float32),
        )
        assert torch.equal(second_pred.labels, torch.tensor([2], dtype=torch.long))
        assert second_pred.scores is None

    def test_missing_required_keys_raises_error(self):
        """Test that missing required keys in prediction dict raises ValueError."""
        store = DetectionPredictionStore()
        predictions = [
            {
                "boxes": torch.tensor([[10, 20, 100, 200]], dtype=torch.float32),
                # Missing 'labels'
                "scores": torch.tensor([0.9], dtype=torch.float32),
            }
        ]

        with pytest.raises(
            ValueError, match="Detection prediction dict for sample 0 missing keys"
        ):
            store.add_predictions(predictions, "test_model")

    def test_invalid_boxes_shape_raises_error(self):
        """Test that invalid bounding box shapes raise ValueError."""
        store = DetectionPredictionStore()
        predictions = [
            {
                "boxes": torch.tensor(
                    [[10, 20, 100]], dtype=torch.float32
                ),  # Wrong shape: should be (M, 4)
                "labels": torch.tensor([0], dtype=torch.long),
                "scores": torch.tensor([0.9], dtype=torch.float32),
            }
        ]

        with pytest.raises(
            ValueError, match="boxes for sample 0 must have shape \\(M,4\\)"
        ):
            store.add_predictions(predictions, "test_model")

    def test_mismatched_labels_shape_raises_error(self):
        """Test that mismatched labels shape raises ValueError."""
        store = DetectionPredictionStore()
        predictions = [
            {
                "boxes": torch.tensor(
                    [[10, 20, 100, 200], [50, 60, 150, 250]], dtype=torch.float32
                ),  # 2 boxes
                "labels": torch.tensor([0], dtype=torch.long),  # 1 label - mismatch!
                "scores": torch.tensor([0.9, 0.8], dtype=torch.float32),
            }
        ]

        with pytest.raises(
            ValueError, match="labels for sample 0 must have shape \\(M,\\)"
        ):
            store.add_predictions(predictions, "test_model")

    def test_mismatched_scores_shape_raises_error(self):
        """Test that mismatched scores shape raises ValueError."""
        store = DetectionPredictionStore()
        predictions = [
            {
                "boxes": torch.tensor(
                    [[10, 20, 100, 200]], dtype=torch.float32
                ),  # 1 box
                "labels": torch.tensor([0], dtype=torch.long),
                "scores": torch.tensor(
                    [0.9, 0.8], dtype=torch.float32
                ),  # 2 scores - mismatch!
            }
        ]

        with pytest.raises(
            ValueError, match="scores for sample 0 must have shape \\(M,\\)"
        ):
            store.add_predictions(predictions, "test_model")

    def test_converts_list_inputs_to_tensors(self):
        """Test that list inputs are converted to tensors."""
        store = DetectionPredictionStore()
        predictions = [
            {
                "boxes": [[10, 20, 100, 200], [50, 60, 150, 250]],
                "labels": [0, 1],
                "scores": [0.9, 0.8],
            }
        ]

        store.add_predictions(predictions, "detection_model")

        first_pred = store.get("detection_model", 0)
        assert isinstance(first_pred.boxes_xyxy, torch.Tensor)
        assert first_pred.boxes_xyxy.dtype == torch.float32
        assert isinstance(first_pred.labels, torch.Tensor)
        assert first_pred.labels.dtype == torch.long
        assert isinstance(first_pred.scores, torch.Tensor)
        assert first_pred.scores.dtype == torch.float32

    def test_converts_numpy_arrays_to_tensors(self):
        """Test that numpy arrays are converted to tensors."""
        import numpy as np

        store = DetectionPredictionStore()
        predictions = [
            {
                "boxes": np.array([[10, 20, 100, 200]], dtype=np.float32),
                "labels": np.array([0], dtype=np.int64),
                "scores": np.array([0.9], dtype=np.float32),
            }
        ]

        store.add_predictions(predictions, "detection_model")

        first_pred = store.get("detection_model", 0)
        assert isinstance(first_pred.boxes_xyxy, torch.Tensor)
        assert first_pred.boxes_xyxy.dtype == torch.float32
        assert isinstance(first_pred.labels, torch.Tensor)
        assert first_pred.labels.dtype == torch.long
        assert isinstance(first_pred.scores, torch.Tensor)
        assert first_pred.scores.dtype == torch.float32

    def test_ensures_correct_tensor_dtypes(self):
        """Test that tensor dtypes are corrected if needed."""
        store = DetectionPredictionStore()
        predictions = [
            {
                "boxes": torch.tensor(
                    [[10, 20, 100, 200]], dtype=torch.int32
                ),  # Wrong dtype
                "labels": torch.tensor([0], dtype=torch.float32),  # Wrong dtype
                "scores": torch.tensor([0.9], dtype=torch.int32),  # Wrong dtype
            }
        ]

        store.add_predictions(predictions, "detection_model")

        first_pred = store.get("detection_model", 0)
        assert first_pred.boxes_xyxy.dtype == torch.float32
        assert first_pred.labels.dtype == torch.long
        assert first_pred.scores.dtype == torch.float32

    def test_get_nonexistent_model_raises_keyerror(self):
        """Test that getting predictions for nonexistent model raises KeyError."""
        store = DetectionPredictionStore()

        with pytest.raises(
            KeyError, match="No predictions found for model: nonexistent"
        ):
            store.get("nonexistent", 0)

    def test_get_nonexistent_datapoint_raises_keyerror(self):
        """Test that getting nonexistent datapoint raises KeyError."""
        store = DetectionPredictionStore()
        predictions = [
            {
                "boxes": torch.tensor([[10, 20, 100, 200]], dtype=torch.float32),
                "labels": torch.tensor([0], dtype=torch.long),
                "scores": torch.tensor([0.9], dtype=torch.float32),
            }
        ]
        store.add_predictions(predictions, "model")

        with pytest.raises(
            KeyError, match="No annotation found for datapoint_number: 999"
        ):
            store.get("model", 999)

    def test_get_subset_valid_indices(self):
        """Test getting a subset of predictions with valid indices."""
        store = DetectionPredictionStore()
        predictions = [
            {
                "boxes": torch.tensor([[10, 20, 100, 200]], dtype=torch.float32),
                "labels": torch.tensor([0], dtype=torch.long),
                "scores": torch.tensor([0.9], dtype=torch.float32),
            },
            {
                "boxes": torch.tensor([[30, 40, 130, 240]], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.long),
                "scores": torch.tensor([0.8], dtype=torch.float32),
            },
            {
                "boxes": torch.tensor([[50, 60, 150, 260]], dtype=torch.float32),
                "labels": torch.tensor([2], dtype=torch.long),
                "scores": torch.tensor([0.7], dtype=torch.float32),
            },
        ]
        store.add_predictions(predictions, "model")

        subset = store.get_subset("model", [0, 2])

        assert isinstance(subset, Annotations)
        assert len(subset.annotations) == 2

        # Check that datapoint numbers are re-indexed
        assert 0 in subset.datapoint_number_to_annotation_index
        assert 1 in subset.datapoint_number_to_annotation_index

        assert torch.equal(
            subset[0].boxes_xyxy,
            torch.tensor([[10, 20, 100, 200]], dtype=torch.float32),
        )
        assert torch.equal(subset[0].labels, torch.tensor([0], dtype=torch.long))
        assert torch.equal(
            subset[1].boxes_xyxy,
            torch.tensor([[50, 60, 150, 260]], dtype=torch.float32),
        )
        assert torch.equal(subset[1].labels, torch.tensor([2], dtype=torch.long))

    def test_get_subset_empty_indices(self):
        """Test getting a subset with empty indices."""
        store = DetectionPredictionStore()
        predictions = [
            {
                "boxes": torch.tensor([[10, 20, 100, 200]], dtype=torch.float32),
                "labels": torch.tensor([0], dtype=torch.long),
                "scores": torch.tensor([0.9], dtype=torch.float32),
            }
        ]
        store.add_predictions(predictions, "model")

        subset = store.get_subset("model", [])

        assert isinstance(subset, Annotations)
        assert len(subset.annotations) == 0

    def test_get_subset_nonexistent_model_raises_keyerror(self):
        """Test that get_subset for nonexistent model raises KeyError."""
        store = DetectionPredictionStore()

        with pytest.raises(
            KeyError, match="No predictions found for model: nonexistent"
        ):
            store.get_subset("nonexistent", [0, 1])

    def test_multiple_models_stored_independently(self):
        """Test that multiple models are stored independently."""
        store = DetectionPredictionStore()
        model1_preds = [
            {
                "boxes": torch.tensor([[10, 20, 100, 200]], dtype=torch.float32),
                "labels": torch.tensor([0], dtype=torch.long),
                "scores": torch.tensor([0.9], dtype=torch.float32),
            }
        ]
        model2_preds = [
            {
                "boxes": torch.tensor([[30, 40, 130, 240]], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.long),
                "scores": torch.tensor([0.8], dtype=torch.float32),
            }
        ]

        store.add_predictions(model1_preds, "model1")
        store.add_predictions(model2_preds, "model2")

        assert "model1" in store.predictions
        assert "model2" in store.predictions
        assert len(store.predictions) == 2

        model1_pred = store.get("model1", 0)
        model2_pred = store.get("model2", 0)

        assert torch.equal(model1_pred.labels, torch.tensor([0], dtype=torch.long))
        assert torch.equal(model2_pred.labels, torch.tensor([1], dtype=torch.long))

    def test_overwrite_existing_model_predictions(self):
        """Test that existing model predictions are overwritten."""
        store = DetectionPredictionStore()
        original_preds = [
            {
                "boxes": torch.tensor([[10, 20, 100, 200]], dtype=torch.float32),
                "labels": torch.tensor([0], dtype=torch.long),
                "scores": torch.tensor([0.9], dtype=torch.float32),
            }
        ]
        new_preds = [
            {
                "boxes": torch.tensor([[30, 40, 130, 240]], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.long),
                "scores": torch.tensor([0.8], dtype=torch.float32),
            },
            {
                "boxes": torch.tensor([[50, 60, 150, 260]], dtype=torch.float32),
                "labels": torch.tensor([2], dtype=torch.long),
                "scores": torch.tensor([0.7], dtype=torch.float32),
            },
        ]

        store.add_predictions(original_preds, "model")
        store.add_predictions(new_preds, "model")

        annotations = store.get_predictions("model")
        assert len(annotations.annotations) == 2

        first_pred = store.get("model", 0)
        assert torch.equal(first_pred.labels, new_preds[0]["labels"])

    def test_empty_predictions_list(self):
        """Test handling of empty predictions list."""
        store = DetectionPredictionStore()

        empty_predictions = []
        store.add_predictions(empty_predictions, "empty_model")

        annotations = store.get_predictions("empty_model")
        assert len(annotations.annotations) == 0

    def test_single_prediction(self):
        """Test handling of single prediction."""
        store = DetectionPredictionStore()

        single_prediction = [
            {
                "boxes": torch.tensor([[10, 20, 100, 200]], dtype=torch.float32),
                "labels": torch.tensor([0], dtype=torch.long),
                "scores": torch.tensor([0.9], dtype=torch.float32),
            }
        ]

        store.add_predictions(single_prediction, "single_model")

        annotations = store.get_predictions("single_model")
        assert len(annotations.annotations) == 1

        pred = store.get("single_model", 0)
        assert torch.equal(
            pred.boxes_xyxy, torch.tensor([[10, 20, 100, 200]], dtype=torch.float32)
        )
        assert torch.equal(pred.labels, torch.tensor([0], dtype=torch.long))
        assert torch.equal(pred.scores, torch.tensor([0.9], dtype=torch.float32))

    def test_zero_detections_per_image(self):
        """Test handling of images with zero detections."""
        store = DetectionPredictionStore()

        zero_detection_predictions = [
            {
                "boxes": torch.empty((0, 4), dtype=torch.float32),
                "labels": torch.empty((0,), dtype=torch.long),
                "scores": torch.empty((0,), dtype=torch.float32),
            }
        ]

        store.add_predictions(zero_detection_predictions, "zero_model")

        annotations = store.get_predictions("zero_model")
        assert len(annotations.annotations) == 1

        pred = store.get("zero_model", 0)
        assert pred.boxes_xyxy.shape == (0, 4)
        assert pred.labels.shape == (0,)
        assert pred.scores.shape == (0,)

    def test_multiple_detections_per_image(self):
        """Test handling of images with multiple detections."""
        store = DetectionPredictionStore()

        multi_detection_predictions = [
            {
                "boxes": torch.tensor(
                    [[10, 20, 100, 200], [30, 40, 130, 240], [50, 60, 150, 260]],
                    dtype=torch.float32,
                ),
                "labels": torch.tensor([0, 1, 2], dtype=torch.long),
                "scores": torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32),
            }
        ]

        store.add_predictions(multi_detection_predictions, "multi_model")

        pred = store.get("multi_model", 0)
        assert pred.boxes_xyxy.shape == (3, 4)
        assert pred.labels.shape == (3,)
        assert pred.scores.shape == (3,)

        assert torch.equal(
            pred.boxes_xyxy[0], torch.tensor([10, 20, 100, 200], dtype=torch.float32)
        )
        assert torch.equal(
            pred.boxes_xyxy[1], torch.tensor([30, 40, 130, 240], dtype=torch.float32)
        )
        assert torch.equal(
            pred.boxes_xyxy[2], torch.tensor([50, 60, 150, 260], dtype=torch.float32)
        )
        assert torch.equal(pred.labels, torch.tensor([0, 1, 2], dtype=torch.long))
        assert torch.equal(
            pred.scores, torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32)
        )

    def test_edge_case_very_small_boxes(self):
        """Test handling of very small bounding boxes."""
        store = DetectionPredictionStore()

        small_box_predictions = [
            {
                "boxes": torch.tensor(
                    [[100.0, 100.0, 100.1, 100.1]], dtype=torch.float32
                ),
                "labels": torch.tensor([0], dtype=torch.long),
                "scores": torch.tensor([0.9], dtype=torch.float32),
            }
        ]

        store.add_predictions(small_box_predictions, "small_model")

        pred = store.get("small_model", 0)
        assert torch.equal(
            pred.boxes_xyxy,
            torch.tensor([[100.0, 100.0, 100.1, 100.1]], dtype=torch.float32),
        )

    def test_edge_case_very_large_boxes(self):
        """Test handling of very large bounding boxes."""
        store = DetectionPredictionStore()

        large_box_predictions = [
            {
                "boxes": torch.tensor(
                    [[0.0, 0.0, 10000.0, 10000.0]], dtype=torch.float32
                ),
                "labels": torch.tensor([0], dtype=torch.long),
                "scores": torch.tensor([0.9], dtype=torch.float32),
            }
        ]

        store.add_predictions(large_box_predictions, "large_model")

        pred = store.get("large_model", 0)
        assert torch.equal(
            pred.boxes_xyxy,
            torch.tensor([[0.0, 0.0, 10000.0, 10000.0]], dtype=torch.float32),
        )

    def test_edge_case_score_values(self):
        """Test handling of various score values."""
        store = DetectionPredictionStore()

        score_edge_cases = [
            {
                "boxes": torch.tensor(
                    [
                        [10, 20, 100, 200],
                        [30, 40, 130, 240],
                        [50, 60, 150, 260],
                        [70, 80, 170, 280],
                    ],
                    dtype=torch.float32,
                ),
                "labels": torch.tensor([0, 1, 2, 3], dtype=torch.long),
                "scores": torch.tensor([0.0, 0.5, 1.0, 0.999999], dtype=torch.float32),
            }
        ]

        store.add_predictions(score_edge_cases, "score_model")

        pred = store.get("score_model", 0)
        assert torch.equal(
            pred.scores, torch.tensor([0.0, 0.5, 1.0, 0.999999], dtype=torch.float32)
        )
