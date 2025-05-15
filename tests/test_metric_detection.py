import pytest
import torch
from torch.utils.data import Dataset as TorchDataset
from doleus.datasets.detection import DoleusDetection
from doleus.annotations import Annotations, BoundingBoxes
from doleus.utils.data import TaskType
from doleus.metrics.calculator import calculate_metric

# Define a simple dummy dataset for detection
class DummyDetectionDataset(TorchDataset):
    def __init__(self, num_samples=4):
        self.num_samples = num_samples
        # Predefined data (image placeholder, boxes, labels)
        self.data = [
            ( 
                torch.randn(3, 100, 100),
                torch.tensor([[10, 10, 50, 50], [60, 60, 90, 90]]),
                torch.tensor([0, 1])
            ),
            (
                torch.randn(3, 100, 100),
                torch.tensor([[20, 20, 70, 70]]),
                torch.tensor([2])
            ),
            (
                torch.randn(3, 100, 100),
                torch.tensor([[30, 30, 80, 80], [50, 50, 95, 95]]),
                torch.tensor([1, 3])
            ),
            (
                torch.randn(3, 100, 100),
                torch.tensor([[40, 40, 90, 90]]),
                torch.tensor([0])
            ),
        ]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if idx >= self.num_samples:
            raise IndexError("Index out of bounds")
        return self.data[idx]


# --- Test Setup ---

@pytest.fixture(scope="module")
def detection_data():
    """Prepares a DoleusDetection dataset with sample ground truths and predictions."""

    # 1. Create DoleusDetection dataset
    doleus_dataset = DoleusDetection(
        dataset=DummyDetectionDataset(),
        name="test_detection_dataset",
    )

    # 2. Define sample predictions. Datapoint number corresponds to the index of the datapoint in the underlying dataset.
    predictions = [
        BoundingBoxes( 
            datapoint_number=0,
            boxes_xyxy=torch.tensor([[12, 12, 48, 48], [65, 65, 88, 88]], dtype=torch.float32),
            labels=torch.tensor([0, 1]), # Correct labels predicted
            scores=torch.tensor([0.9, 0.85]),
        ),
        BoundingBoxes( 
            datapoint_number=1,
            boxes_xyxy=torch.tensor([[25, 25, 75, 75], [5, 5, 15, 15]], dtype=torch.float32),
            labels=torch.tensor([2, 0]), # Correct label + a false positive (class 0)
            scores=torch.tensor([0.8, 0.5]),
        ),
        BoundingBoxes( 
            datapoint_number=2,
            boxes_xyxy=torch.tensor([[30, 30, 80, 80]], dtype=torch.float32),
            labels=torch.tensor([1]), # Predicts only label 1 (misses label 3)
            scores=torch.tensor([0.75]),
        ),
        BoundingBoxes( 
            datapoint_number=3,
            boxes_xyxy=torch.tensor([[42, 42, 88, 88], [10, 60, 30, 80]], dtype=torch.float32),
            labels=torch.tensor([3, 2]), # Incorrect labels predicted (FP)
            scores=torch.tensor([0.85, 0.6]),
        ),
    ]

    # 3. Add predictions
    doleus_dataset.add_model_predictions(predictions, model_id="test_model")
    return doleus_dataset

# --- Placeholder for Actual Tests ---
# (Tests will be added in the next step)


# Example of how to use the fixture (will be replaced by actual tests):
# def test_setup_works(detection_data):
#     doleus_dataset, relevant_ids = detection_data
#     assert doleus_dataset.name == "test_detection_dataset"
#     assert len(relevant_ids) == 4
#     assert len(doleus_dataset.groundtruths) == 4
#     assert len(doleus_dataset.predictions) == 4
#     assert doleus_dataset.task_type == TaskType.DETECTION.value
#     print("\nSetup seems okay.")


if __name__ == "__main__":
    # You can run pytest directly:
    pytest.main([__file__])
    # Or manually invoke the fixture for debugging:
    # print("Debugging fixture setup:")
    # data, ids = detection_data()
    # test_setup_works((data, ids))
