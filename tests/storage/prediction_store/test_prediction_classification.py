import pytest
import torch
from doleus.annotations import Annotations, Labels
from doleus.storage.prediction_store.classification import ClassificationPredictionStore
from doleus.utils import Task


class TestClassificationPredictionStore:

    def test_initialization(self):
        store = ClassificationPredictionStore()
        assert hasattr(store, "predictions")
        assert isinstance(store.predictions, dict)
        assert len(store.predictions) == 0

    def test_add_predictions_requires_tensor(self):
        store = ClassificationPredictionStore()
        invalid_predictions = [0.8, 0.2, 0.9]

        with pytest.raises(
            TypeError, match="For classification, predictions must be a torch.Tensor"
        ):
            store.add_predictions(
                invalid_predictions, "test_model", task=Task.BINARY.value
            )

    def test_binary_1d_float_scores(self):
        store = ClassificationPredictionStore()
        predictions = torch.tensor([0.8, 0.3, 0.9, 0.1], dtype=torch.float32)

        store.add_predictions(predictions, "binary_model", task=Task.BINARY.value)

        annotations = store.get_predictions("binary_model")
        assert len(annotations.annotations) == 4

        first_pred = store.get("binary_model", 0)
        assert isinstance(first_pred, Labels)
        assert first_pred.datapoint_number == 0
        assert first_pred.scores is not None
        assert torch.equal(first_pred.scores, torch.tensor([0.8]))
        assert first_pred.labels is None

    def test_binary_1d_int_labels(self):
        store = ClassificationPredictionStore()
        predictions = torch.tensor([1, 0, 1, 0], dtype=torch.long)

        store.add_predictions(predictions, "binary_model", task=Task.BINARY.value)

        annotations = store.get_predictions("binary_model")
        assert len(annotations.annotations) == 4

        first_pred = store.get("binary_model", 0)
        assert first_pred.labels is not None
        assert torch.equal(first_pred.labels, torch.tensor([1]))
        assert first_pred.scores is None

    def test_binary_2d_raises_error(self):
        store = ClassificationPredictionStore()
        predictions = torch.tensor([[0.8, 0.2], [0.3, 0.7]], dtype=torch.float32)

        with pytest.raises(
            ValueError, match="binary classification predictions must be 1D tensor"
        ):
            store.add_predictions(predictions, "model", task=Task.BINARY.value)

    def test_binary_float_labels_edge_case(self):
        """Binary labels as floats are treated as scores, not labels."""
        store = ClassificationPredictionStore()
        predictions = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float32)

        store.add_predictions(predictions, "binary_model", task=Task.BINARY.value)

        first_pred = store.get("binary_model", 0)
        assert first_pred.scores is not None
        assert torch.equal(first_pred.scores, torch.tensor([0.0]))
        assert first_pred.labels is None

    def test_multiclass_1d_int_labels(self):
        store = ClassificationPredictionStore()
        predictions = torch.tensor([0, 1, 2, 0, 1], dtype=torch.long)

        store.add_predictions(
            predictions, "multiclass_model", task=Task.MULTICLASS.value
        )

        annotations = store.get_predictions("multiclass_model")
        assert len(annotations.annotations) == 5

        for i, expected_class in enumerate([0, 1, 2, 0, 1]):
            pred = store.get("multiclass_model", i)
            assert torch.equal(pred.labels, torch.tensor([expected_class]))
            assert pred.scores is None

    def test_multiclass_1d_float_raises_error(self):
        store = ClassificationPredictionStore()
        predictions = torch.tensor([0.8, 0.3, 0.9], dtype=torch.float32)

        with pytest.raises(
            ValueError,
            match="For multiclass with 1D predictions, dtype must be integer",
        ):
            store.add_predictions(predictions, "model", task=Task.MULTICLASS.value)

    def test_multiclass_2d_float_logits(self):
        store = ClassificationPredictionStore()
        predictions = torch.tensor(
            [
                [2.0, 1.0, 0.5],
                [0.1, 3.0, 1.0],
            ],
            dtype=torch.float32,
        )

        store.add_predictions(
            predictions, "multiclass_model", task=Task.MULTICLASS.value
        )

        annotations = store.get_predictions("multiclass_model")
        assert len(annotations.annotations) == 2

        first_pred = store.get("multiclass_model", 0)
        assert torch.equal(first_pred.labels, torch.tensor([0]))
        assert first_pred.scores is not None
        expected_scores = torch.tensor([2.0, 1.0, 0.5])
        assert torch.allclose(first_pred.scores, expected_scores, atol=1e-6)

    def test_multiclass_2d_float_probabilities(self):
        store = ClassificationPredictionStore()
        predictions = torch.tensor(
            [
                [0.7, 0.2, 0.1],
                [0.1, 0.8, 0.1],
            ],
            dtype=torch.float32,
        )

        store.add_predictions(
            predictions, "multiclass_model", task=Task.MULTICLASS.value
        )

        first_pred = store.get("multiclass_model", 0)
        assert torch.equal(first_pred.labels, torch.tensor([0]))
        assert torch.allclose(
            first_pred.scores, torch.tensor([0.7, 0.2, 0.1]), atol=1e-6
        )

    def test_multiclass_2d_int_raises_error(self):
        store = ClassificationPredictionStore()
        predictions = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.long)

        with pytest.raises(
            ValueError, match="For multiclass with 2D predictions, dtype must be float"
        ):
            store.add_predictions(predictions, "model", task=Task.MULTICLASS.value)

    def test_multilabel_2d_float_logits(self):
        store = ClassificationPredictionStore()
        predictions = torch.tensor(
            [
                [2.0, -1.0, 0.5],
                [-0.5, 3.0, 1.0],
            ],
            dtype=torch.float32,
        )

        store.add_predictions(
            predictions, "multilabel_model", task=Task.MULTILABEL.value
        )

        first_pred = store.get("multilabel_model", 0)
        assert first_pred.labels is None
        assert first_pred.scores is not None
        expected_scores = torch.tensor([2.0, -1.0, 0.5])
        assert torch.allclose(first_pred.scores, expected_scores, atol=1e-6)

    def test_multilabel_2d_float_probabilities(self):
        store = ClassificationPredictionStore()
        predictions = torch.tensor(
            [
                [0.9, 0.1, 0.6],
                [0.2, 0.8, 0.4],
            ],
            dtype=torch.float32,
        )

        store.add_predictions(
            predictions, "multilabel_model", task=Task.MULTILABEL.value
        )

        first_pred = store.get("multilabel_model", 0)
        assert first_pred.scores is not None
        assert torch.allclose(
            first_pred.scores, torch.tensor([0.9, 0.1, 0.6]), atol=1e-6
        )

    def test_multilabel_2d_int_multihot(self):
        store = ClassificationPredictionStore()
        predictions = torch.tensor(
            [
                [1, 0, 1],
                [0, 1, 1],
            ],
            dtype=torch.long,
        )

        store.add_predictions(
            predictions, "multilabel_model", task=Task.MULTILABEL.value
        )

        first_pred = store.get("multilabel_model", 0)
        assert first_pred.scores is None
        assert first_pred.labels is not None
        assert torch.equal(
            first_pred.labels, torch.tensor([1, 0, 1], dtype=torch.int32)
        )

    def test_multilabel_1d_raises_error(self):
        store = ClassificationPredictionStore()
        predictions = torch.tensor([0.8, 0.3, 0.9], dtype=torch.float32)

        with pytest.raises(
            ValueError,
            match="multilabel classification predictions must be a 2D tensor",
        ):
            store.add_predictions(predictions, "model", task=Task.MULTILABEL.value)

    def test_unsupported_task_raises_error(self):
        store = ClassificationPredictionStore()
        predictions = torch.tensor([0.8, 0.3], dtype=torch.float32)

        with pytest.raises(ValueError, match="Unsupported task: invalid_task"):
            store.add_predictions(predictions, "model", task="invalid_task")

    def test_get_nonexistent_model_raises_keyerror(self):
        store = ClassificationPredictionStore()

        with pytest.raises(
            KeyError, match="No predictions found for model: nonexistent"
        ):
            store.get("nonexistent", 0)

    def test_get_nonexistent_datapoint_raises_keyerror(self):
        store = ClassificationPredictionStore()
        predictions = torch.tensor([0.8, 0.3], dtype=torch.float32)
        store.add_predictions(predictions, "model", task=Task.BINARY.value)

        with pytest.raises(
            KeyError, match="No annotation found for datapoint_number: 999"
        ):
            store.get("model", 999)

    def test_get_subset_valid_indices(self):
        store = ClassificationPredictionStore()
        predictions = torch.tensor([0, 1, 2, 0, 1], dtype=torch.long)
        store.add_predictions(predictions, "model", task=Task.MULTICLASS.value)

        subset = store.get_subset("model", [0, 2, 4])

        assert isinstance(subset, Annotations)
        assert len(subset.annotations) == 3

        assert 0 in subset.datapoint_number_to_annotation_index
        assert 1 in subset.datapoint_number_to_annotation_index
        assert 2 in subset.datapoint_number_to_annotation_index

        assert torch.equal(subset[0].labels, torch.tensor([0]))
        assert torch.equal(subset[1].labels, torch.tensor([2]))
        assert torch.equal(subset[2].labels, torch.tensor([1]))

    def test_get_subset_empty_indices(self):
        store = ClassificationPredictionStore()
        predictions = torch.tensor([0.8, 0.3], dtype=torch.float32)
        store.add_predictions(predictions, "model", task=Task.BINARY.value)

        subset = store.get_subset("model", [])

        assert isinstance(subset, Annotations)
        assert len(subset.annotations) == 0

    def test_get_subset_nonexistent_model_raises_keyerror(self):
        store = ClassificationPredictionStore()

        with pytest.raises(
            KeyError, match="No predictions found for model: nonexistent"
        ):
            store.get_subset("nonexistent", [0, 1])

    def test_multiple_models_stored_independently(self):
        store = ClassificationPredictionStore()
        binary_preds = torch.tensor([0.8, 0.3], dtype=torch.float32)
        multiclass_preds = torch.tensor([0, 1], dtype=torch.long)

        store.add_predictions(binary_preds, "binary_model", task=Task.BINARY.value)
        store.add_predictions(
            multiclass_preds, "multiclass_model", task=Task.MULTICLASS.value
        )

        assert "binary_model" in store.predictions
        assert "multiclass_model" in store.predictions
        assert len(store.predictions) == 2

        binary_pred = store.get("binary_model", 0)
        assert binary_pred.scores is not None
        assert binary_pred.labels is None

        multiclass_pred = store.get("multiclass_model", 0)
        assert multiclass_pred.labels is not None
        assert multiclass_pred.scores is None

    def test_overwrite_existing_model_predictions(self):
        store = ClassificationPredictionStore()
        original_preds = torch.tensor([0.8], dtype=torch.float32)
        new_preds = torch.tensor([0.3, 0.9], dtype=torch.float32)

        store.add_predictions(original_preds, "model", task=Task.BINARY.value)
        store.add_predictions(new_preds, "model", task=Task.BINARY.value)

        annotations = store.get_predictions("model")
        assert len(annotations.annotations) == 2

        first_pred = store.get("model", 0)
        assert torch.equal(first_pred.scores, torch.tensor([0.3]))

    def test_binary_task_with_multilabel_predictions_raises_error(self):
        store = ClassificationPredictionStore()
        multilabel_predictions = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.long)

        with pytest.raises(
            ValueError, match="binary classification predictions must be 1D tensor"
        ):
            store.add_predictions(
                multilabel_predictions, "model", task=Task.BINARY.value
            )

    def test_multilabel_task_with_binary_predictions_raises_error(self):
        store = ClassificationPredictionStore()
        binary_predictions = torch.tensor([0.8, 0.3, 0.9], dtype=torch.float32)

        with pytest.raises(
            ValueError,
            match="multilabel classification predictions must be a 2D tensor",
        ):
            store.add_predictions(
                binary_predictions, "model", task=Task.MULTILABEL.value
            )

    def test_multiclass_task_with_wrong_float_shape_raises_error(self):
        store = ClassificationPredictionStore()
        wrong_predictions = torch.tensor(
            [[[0.5, 0.5], [0.3, 0.7]]], dtype=torch.float32
        )

        with pytest.raises(
            ValueError,
            match="multiclass classification predictions must be 1D or 2D tensor",
        ):
            store.add_predictions(
                wrong_predictions, "model", task=Task.MULTICLASS.value
            )

    def test_binary_task_with_3d_tensor_raises_error(self):
        store = ClassificationPredictionStore()
        wrong_predictions = torch.tensor([[[0.8]], [[0.3]]], dtype=torch.float32)

        with pytest.raises(
            ValueError,
            match="binary classification predictions must be 1D or 2D tensor",
        ):
            store.add_predictions(wrong_predictions, "model", task=Task.BINARY.value)

    def test_multiclass_task_with_inconsistent_dimensions(self):
        store = ClassificationPredictionStore()
        inconsistent_predictions = torch.tensor([0.5, 1.5, 2.3], dtype=torch.float32)

        with pytest.raises(
            ValueError,
            match="For multiclass with 1D predictions, dtype must be integer",
        ):
            store.add_predictions(
                inconsistent_predictions, "model", task=Task.MULTICLASS.value
            )

    def test_bool_dtype_explicitly_rejected(self):
        store = ClassificationPredictionStore()
        bool_predictions = torch.tensor(
            [[True, False, True], [False, True, False]], dtype=torch.bool
        )

        with pytest.raises(
            TypeError,
            match="Bool tensors are not supported. Please use int or float tensors.",
        ):
            store.add_predictions(bool_predictions, "model", task=Task.MULTILABEL.value)

    def test_helpful_error_messages_for_common_mistakes(self):
        store = ClassificationPredictionStore()
        multiclass_style = torch.tensor([[0.3, 0.7], [0.8, 0.2]], dtype=torch.float32)

        with pytest.raises(ValueError) as exc_info:
            store.add_predictions(multiclass_style, "model", task=Task.BINARY.value)

        error_message = str(exc_info.value)
        assert "binary" in error_message.lower()
        assert "1D" in error_message or "tensor" in error_message

    def test_edge_case_empty_predictions_tensor(self):
        store = ClassificationPredictionStore()

        empty_1d = torch.tensor([], dtype=torch.float32)
        store.add_predictions(empty_1d, "binary_model", task=Task.BINARY.value)

        empty_2d = torch.tensor([], dtype=torch.float32).reshape(0, 3)
        store.add_predictions(empty_2d, "multilabel_model", task=Task.MULTILABEL.value)

        binary_annotations = store.get_predictions("binary_model")
        multilabel_annotations = store.get_predictions("multilabel_model")

        assert len(binary_annotations.annotations) == 0
        assert len(multilabel_annotations.annotations) == 0

    def test_single_sample_predictions(self):
        store = ClassificationPredictionStore()

        binary_single = torch.tensor([0.8], dtype=torch.float32)
        multiclass_single = torch.tensor([1], dtype=torch.long)
        multilabel_single = torch.tensor([[1, 0, 1]], dtype=torch.long)

        store.add_predictions(binary_single, "binary_model", task=Task.BINARY.value)
        store.add_predictions(
            multiclass_single, "multiclass_model", task=Task.MULTICLASS.value
        )
        store.add_predictions(
            multilabel_single, "multilabel_model", task=Task.MULTILABEL.value
        )

        assert len(store.get_predictions("binary_model").annotations) == 1
        assert len(store.get_predictions("multiclass_model").annotations) == 1
        assert len(store.get_predictions("multilabel_model").annotations) == 1

        binary_pred = store.get("binary_model", 0)
        multiclass_pred = store.get("multiclass_model", 0)
        multilabel_pred = store.get("multilabel_model", 0)

        assert torch.equal(binary_pred.scores, torch.tensor([0.8]))
        assert torch.equal(multiclass_pred.labels, torch.tensor([1]))
        assert torch.equal(
            multilabel_pred.labels, torch.tensor([1, 0, 1], dtype=torch.int32)
        )

    def test_binary_task_with_multiclass_values(self):
        """Binary task should reject multiclass class indices."""
        store = ClassificationPredictionStore()
        multiclass_values = torch.tensor([1, 2, 3, 4], dtype=torch.long)

        with pytest.raises(ValueError, match="Binary prediction labels must be 0 or 1"):
            store.add_predictions(multiclass_values, "model", task=Task.BINARY.value)

    def test_multiclass_task_with_negative_values_raises_error(self):
        store = ClassificationPredictionStore()
        negative_classes = torch.tensor([0, 1, -1, 2], dtype=torch.long)

        with pytest.raises(
            ValueError,
            match="Multiclass prediction labels must be non-negative. Got: -1 at sample 2",
        ):
            store.add_predictions(negative_classes, "model", task=Task.MULTICLASS.value)

    def test_multilabel_task_with_invalid_multihot_values_raises_error(self):
        store = ClassificationPredictionStore()
        invalid_multihot = torch.tensor([[1, 0, 1], [0, 2, 1]], dtype=torch.long)

        with pytest.raises(
            ValueError,
            match="Multilabel prediction labels must be multi-hot encoded \\(0s and 1s only\\). Got: .* at sample 1",
        ):
            store.add_predictions(invalid_multihot, "model", task=Task.MULTILABEL.value)

    def test_binary_task_with_out_of_range_scores(self):
        """Scores outside [0,1] range should be accepted (can be logits)."""
        store = ClassificationPredictionStore()
        weird_scores = torch.tensor([-2.5, 3.8, 10.0, -0.5], dtype=torch.float32)

        store.add_predictions(weird_scores, "model", task=Task.BINARY.value)

        annotations = store.get_predictions("model")
        assert len(annotations.annotations) == 4

        first_pred = store.get("model", 0)
        assert torch.equal(first_pred.scores, torch.tensor([-2.5]))

    def test_multiclass_task_with_out_of_range_class_indices(self):
        """Class indices outside typical range are stored without validation."""
        store = ClassificationPredictionStore()
        out_of_range_classes = torch.tensor([0, 1, 2, 5], dtype=torch.long)

        store.add_predictions(out_of_range_classes, "model", task=Task.MULTICLASS.value)

        annotations = store.get_predictions("model")
        assert len(annotations.annotations) == 4

        out_of_range_pred = store.get("model", 3)
        assert torch.equal(out_of_range_pred.labels, torch.tensor([5]))

    def test_multilabel_task_with_scores_outside_01_range(self):
        """Out-of-range scores are stored as raw logits/scores."""
        store = ClassificationPredictionStore()
        weird_probs = torch.tensor(
            [[-0.5, 1.5, 2.0], [3.0, -1.0, 0.5]], dtype=torch.float32
        )

        store.add_predictions(weird_probs, "model", task=Task.MULTILABEL.value)

        first_pred = store.get("model", 0)
        assert first_pred.scores is not None
        expected_scores = torch.tensor([-0.5, 1.5, 2.0])
        assert torch.allclose(first_pred.scores, expected_scores, atol=1e-6)

    def test_multiclass_to_binary_raises_error(self):
        """Multiclass predictions should be rejected for binary task."""
        store = ClassificationPredictionStore()
        cifar10_predictions = torch.tensor(
            [3, 7, 2, 9, 1, 5, 8, 0, 4, 6], dtype=torch.long
        )

        with pytest.raises(
            ValueError,
            match="Binary prediction labels must be 0 or 1. Got: 3 at sample 0",
        ):
            store.add_predictions(
                cifar10_predictions, "binary_model", task=Task.BINARY.value
            )

    def test_multiclass_with_binary_looking_values_accepted(self):
        store = ClassificationPredictionStore()
        binary_looking_predictions = torch.tensor([0, 1, 0, 1, 1, 0], dtype=torch.long)

        store.add_predictions(
            binary_looking_predictions, "multiclass_model", task=Task.MULTICLASS.value
        )

        annotations = store.get_predictions("multiclass_model")
        assert len(annotations.annotations) == 6

        for i, expected_class in enumerate([0, 1, 0, 1, 1, 0]):
            pred = store.get("multiclass_model", i)
            assert torch.equal(pred.labels, torch.tensor([expected_class]))
            assert pred.scores is None
