import pytest
import torch
import numpy as np
from torch.utils.data import Dataset, Subset
from moonwatcher.utils.data import Task, TaskType
from moonwatcher.dataset.dataset import MoonwatcherClassification, Slice
from moonwatcher.check import Check, CheckSuite

class DummyDataset(Dataset):
    def __init__(self, size=100):
        self.size = size
        self.data = torch.randn(size, 3, 32, 32)  # Random images
        self.targets = torch.randint(0, 10, (size,))  # Random labels
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx].item()

@pytest.fixture
def dummy_dataset():
    return DummyDataset()

@pytest.fixture
def moonwatcher_dataset(dummy_dataset):
    return MoonwatcherClassification(
        name="test_dataset",
        dataset=dummy_dataset,
        task=Task.MULTICLASS.value,
        num_classes=10
    )

def test_moonwatcher_dataset_creation(dummy_dataset):
    """Test basic dataset creation"""
    mw_dataset = MoonwatcherClassification(
        name="test_dataset",
        dataset=dummy_dataset,
        task=Task.MULTICLASS.value,
        num_classes=10
    )
    assert len(mw_dataset) == len(dummy_dataset)
    assert mw_dataset.task == Task.MULTICLASS.value
    assert mw_dataset.num_classes == 10

def test_metadata_addition(moonwatcher_dataset):
    """Test adding and retrieving metadata"""
    # Test predefined metadata
    moonwatcher_dataset.add_predefined_metadata("brightness")
    
    # Get metadata directly from datapoints
    brightness_values = [dp.get_metadata("brightness") for dp in moonwatcher_dataset.datapoints]
    assert all(isinstance(v, (float, int, np.float32, np.float64)) for v in brightness_values)
    
    # Test custom metadata
    custom_values = [float(x) for x in np.random.rand(len(moonwatcher_dataset))]
    moonwatcher_dataset.add_metadata("custom", custom_values)
    
    # Verify custom metadata
    retrieved_values = [dp.get_metadata("custom") for dp in moonwatcher_dataset.datapoints]
    assert np.allclose(retrieved_values, custom_values)

def test_model_predictions(moonwatcher_dataset):
    """Test adding and retrieving model predictions"""
    # Create dummy predictions
    predictions = torch.randn(len(moonwatcher_dataset), 10)
    
    # Add predictions
    model_id = moonwatcher_dataset.add_model_predictions(
        predictions=predictions,
        model_name="test_model",
        model_metadata={"architecture": "test"}
    )
    
    # Verify predictions were added correctly
    stored_preds = moonwatcher_dataset.prediction_store.get_predictions(
        dataset_name=moonwatcher_dataset.name,
        model_id=model_id
    )
    assert torch.allclose(stored_preds, predictions)

def test_slicing(moonwatcher_dataset):
    """Test dataset slicing functionality"""
    # Add metadata for slicing using scalar values
    metadata_values = [float(x) for x in np.random.rand(len(moonwatcher_dataset))]
    moonwatcher_dataset.add_metadata("test_metric", metadata_values)
    
    # Create slices using numpy operations to avoid ambiguity
    median = float(np.median(metadata_values))
    
    # Create slices using slice_by_metadata_value for exact matches
    high_indices = [i for i, v in enumerate(metadata_values) if v >= median]
    low_indices = [i for i, v in enumerate(metadata_values) if v < median]
    
    # Add categorical metadata for slicing
    moonwatcher_dataset.add_metadata_from_list([
        {"category": "high" if i in high_indices else "low"}
        for i in range(len(moonwatcher_dataset))
    ])
    
    # Test slicing by metadata value
    high_slice = moonwatcher_dataset.slice_by_metadata_value("category", "high")
    low_slice = moonwatcher_dataset.slice_by_metadata_value("category", "low")
    
    # Verify basic slice properties
    assert len(high_slice) + len(low_slice) == len(moonwatcher_dataset)
    assert isinstance(high_slice, Slice)
    assert isinstance(low_slice, Slice)
    assert len(high_slice) == len(high_indices)
    assert len(low_slice) == len(low_indices)
    
    # Verify slice contents and metadata
    assert all(dp.get_metadata("category") == "high" for dp in high_slice.datapoints)
    assert all(dp.get_metadata("category") == "low" for dp in low_slice.datapoints)
    
    # Verify that the actual data points in the slices match the original indices
    for idx, original_idx in enumerate(high_slice.indices):
        assert original_idx in high_indices
        assert metadata_values[original_idx] >= median
        # Verify the actual data matches
        slice_data, _ = high_slice[idx]
        original_data, _ = moonwatcher_dataset[original_idx]
        assert torch.equal(slice_data, original_data)
    
    for idx, original_idx in enumerate(low_slice.indices):
        assert original_idx in low_indices
        assert metadata_values[original_idx] < median
        # Verify the actual data matches
        slice_data, _ = low_slice[idx]
        original_data, _ = moonwatcher_dataset[original_idx]
        assert torch.equal(slice_data, original_data)
    
    # Test percentile-based slicing
    # Convert metadata values to float to ensure scalar comparison
    moonwatcher_dataset.add_metadata_from_list([
        {"test_metric": float(v)} for v in metadata_values
    ])
    
    bright_slice = moonwatcher_dataset.slice_by_percentile("test_metric", ">=", 50)
    dim_slice = moonwatcher_dataset.slice_by_percentile("test_metric", "<", 50)
    
    # Verify percentile slices match threshold-based slices
    assert len(bright_slice) == len(high_slice)
    assert len(dim_slice) == len(low_slice)
    assert set(bright_slice.indices) == set(high_slice.indices)
    assert set(dim_slice.indices) == set(low_slice.indices)

def test_check_creation_and_execution(moonwatcher_dataset):
    """Test Check and CheckSuite functionality"""
    # Add predictions
    predictions = torch.randn(len(moonwatcher_dataset), 10)
    model_id = moonwatcher_dataset.add_model_predictions(
        predictions=predictions,
        model_name="test_model",
        model_metadata={"architecture": "test"}
    )
    
    # Create check
    check = Check(
        name="test_check",
        dataset=moonwatcher_dataset,
        model_id=model_id,
        metric="Accuracy",
        operator=">",
        value=0.0  # Should always pass since we're just testing functionality
    )
    
    # Create and run check suite
    suite = CheckSuite(
        name="test_suite",
        checks=[check]
    )
    
    results = suite.run_all(show=False)
    # Verify the structure of results
    assert isinstance(results, dict)
    assert "checks" in results
    assert len(results["checks"]) == 1
    assert results["checks"][0]["check_name"] == "test_check"
    assert "success" in results

def test_task_handling():
    """Test different task types"""
    dataset = DummyDataset(size=50)
    
    # Test binary classification
    binary_dataset = MoonwatcherClassification(
        name="binary_test",
        dataset=dataset,
        task=Task.BINARY.value,
        num_classes=2
    )
    assert binary_dataset.task == Task.BINARY.value
    
    # Test multiclass classification
    multiclass_dataset = MoonwatcherClassification(
        name="multiclass_test",
        dataset=dataset,
        task=Task.MULTICLASS.value,
        num_classes=10
    )
    assert multiclass_dataset.task == Task.MULTICLASS.value

def test_invalid_inputs(dummy_dataset):
    """Test error handling for invalid inputs"""
    # Test missing required arguments
    with pytest.raises((ValueError, TypeError)):
        MoonwatcherClassification(
            dataset=dummy_dataset,  # Missing name
            task=Task.MULTICLASS.value,
            num_classes=10
        )
    
    # Test missing dataset
    with pytest.raises((ValueError, TypeError)):
        MoonwatcherClassification(
            name="test",
            task=Task.MULTICLASS.value,
            num_classes=10
        )

def test_slice_accuracy_mapping():
    """Test that accuracy calculations on slices match manual calculations"""
    # Create a small dataset with known labels
    class SmallDataset(Dataset):
        def __init__(self):
            # 10 datapoints: 5 class 0, 5 class 1
            self.data = torch.randn(10, 3, 32, 32)
            self.targets = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            # Return the label as a tensor, not a scalar
            return self.data[idx], self.targets[idx]
    
    dataset = SmallDataset()
    mw_dataset = MoonwatcherClassification(
        name="test_accuracy",
        dataset=dataset,
        task=Task.MULTICLASS.value,
        num_classes=2
    )
    
    # Add metadata to create two groups:
    # Group A (indices 0-4): all class 0
    # Group B (indices 5-9): all class 1
    metadata_values = ["A"] * 5 + ["B"] * 5
    mw_dataset.add_metadata_from_list([{"group": v} for v in metadata_values])
    
    # Create predictions with known accuracy:
    # Group A: 4 correct (class 0), 1 wrong (predicted as class 1)
    # Group B: 3 correct (class 1), 2 wrong (predicted as class 0)
    predictions = torch.tensor([
        [0.9, 0.1],  # correct class 0
        [0.8, 0.2],  # correct class 0
        [0.7, 0.3],  # correct class 0
        [0.6, 0.4],  # correct class 0
        [0.3, 0.7],  # wrong: predicted 1, true 0
        [0.2, 0.8],  # correct class 1
        [0.1, 0.9],  # correct class 1
        [0.3, 0.7],  # correct class 1
        [0.8, 0.2],  # wrong: predicted 0, true 1
        [0.7, 0.3],  # wrong: predicted 0, true 1
    ])
    
    # Add predictions
    model_id = mw_dataset.add_model_predictions(
        predictions=predictions,
        model_name="test_model",
        model_metadata={"architecture": "test"}
    )
    
    # Create slices
    slice_a = mw_dataset.slice_by_metadata_value("group", "A")
    slice_b = mw_dataset.slice_by_metadata_value("group", "B")

    # Create and run checks
    check_a = Check(
        name="group_a_accuracy",
        dataset=slice_a,
        model_id=model_id,
        metric="Accuracy",
        operator=">",
        value=0.0
    )
    
    check_b = Check(
        name="group_b_accuracy",
        dataset=slice_b,
        model_id=model_id,
        metric="Accuracy",
        operator=">",
        value=0.0
    )
    
    # Get results
    result_a = check_a.run()
    result_b = check_b.run()
    
    # Expected accuracies:
    # Group A: 4/5 = 0.8 (4 correct out of 5)
    # Group B: 3/5 = 0.6 (3 correct out of 5)
    assert result_a["result"] == pytest.approx(0.8, abs=1e-6)
    assert result_b["result"] == pytest.approx(0.6, abs=1e-6)
    
    # Test overall accuracy
    check_all = Check(
        name="overall_accuracy",
        dataset=mw_dataset,
        model_id=model_id,
        metric="Accuracy",
        operator=">",
        value=0.0
    )
    result_all = check_all.run()
    
    # Expected overall accuracy: 7/10 = 0.7 (7 correct out of 10)
    assert result_all["result"] == pytest.approx(0.7, abs=1e-6) 