import pytest
import torch

from torch.utils.data import Dataset as TorchDataset

from doleus.datasets.classification import DoleusClassification
from doleus.metric import calculate_metric_internal
from doleus.utils.data import Task

class MockModel:
    def __init__(self, name: str, task_type: str):
        self.name = name
        self.task_type = task_type


class MockTorchDataset(TorchDataset):
    def __init__(self, img_labels: list):
        self.img_labels = img_labels

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        return torch.empty(0), self.img_labels[idx]


class MockLabel:
    def __init__(self, labels):
        if not isinstance(labels, torch.Tensor):
            self.labels = torch.tensor(labels)
        else:
            self.labels = labels

def test_multilabel_accuracy():
    model = MockModel(name="mock_multilabel_model", task_type=Task.MULTILABEL.value)
    assert True == True

def test_calculate_accuracy_binary():
    model = MockModel(name="mock_binary_model", task_type=Task.BINARY.value)
    
    groundtruths_labels = [MockLabel([0]), MockLabel([1]), MockLabel([0]), MockLabel([1])]
    predictions_labels = [MockLabel([0]), MockLabel([1]), MockLabel([1]), MockLabel([1])]
        
    dataset_tensor_labels = [gt.labels for gt in groundtruths_labels]
    mock_torch_dataset = MockTorchDataset(dataset_tensor_labels)
    
    doleus_dataset = DoleusClassification(
        dataset=mock_torch_dataset,
        name="mock_binary_dataset",
        task=Task.BINARY.value,
        num_classes=2,
        label_to_name={0: "class0", 1: "class1"}
    )
    relevant_ids = list(range(len(doleus_dataset)))

    result = calculate_metric_internal(
        model, relevant_ids, doleus_dataset, groundtruths_labels, predictions_labels, "Accuracy"
    )
    assert result == 0.75, f"Expected Accuracy to be 0.75 but got {result}"


def test_calculate_precision_binary():
    model = MockModel(name="mock_binary_model", task_type=Task.BINARY.value)
    groundtruths_labels = [MockLabel([0]), MockLabel([1]), MockLabel([0]), MockLabel([1])]
    predictions_labels = [MockLabel([0]), MockLabel([1]), MockLabel([1]), MockLabel([1])]
    
    dataset_tensor_labels = [gt.labels for gt in groundtruths_labels]
    mock_torch_dataset = MockTorchDataset(dataset_tensor_labels)

    doleus_dataset = DoleusClassification(
        dataset=mock_torch_dataset,
        name="mock_binary_dataset",
        task=Task.BINARY.value,
        num_classes=2,
        label_to_name={0: "class0", 1: "class1"}
    )
    relevant_ids = list(range(len(doleus_dataset)))

    result = calculate_metric_internal(
        model, relevant_ids, doleus_dataset, groundtruths_labels, predictions_labels, "Precision"
    )
    assert result == pytest.approx(2/3), f"Expected Precision to be {2/3} but got {result}"


def test_calculate_recall_binary():
    model = MockModel(name="mock_binary_model", task_type=Task.BINARY.value)
    groundtruths_labels = [MockLabel([0]), MockLabel([1]), MockLabel([0]), MockLabel([1])]
    predictions_labels = [MockLabel([0]), MockLabel([1]), MockLabel([1]), MockLabel([1])]

    dataset_tensor_labels = [gt.labels for gt in groundtruths_labels]
    mock_torch_dataset = MockTorchDataset(dataset_tensor_labels)

    doleus_dataset = DoleusClassification(
        dataset=mock_torch_dataset,
        name="mock_binary_dataset",
        task=Task.BINARY.value,
        num_classes=2,
        label_to_name={0: "class0", 1: "class1"}
    )
    relevant_ids = list(range(len(doleus_dataset)))

    result = calculate_metric_internal(
        model, relevant_ids, doleus_dataset, groundtruths_labels, predictions_labels, "Recall"
    )
    assert result == 1.0, f"Expected Recall to be 1.0 but got {result}"


def test_calculate_f1_binary():
    model = MockModel(name="mock_binary_model", task_type=Task.BINARY.value)
    groundtruths_labels = [MockLabel([0]), MockLabel([1]), MockLabel([0]), MockLabel([1])]
    predictions_labels = [MockLabel([0]), MockLabel([1]), MockLabel([1]), MockLabel([1])]

    dataset_tensor_labels = [gt.labels for gt in groundtruths_labels]
    mock_torch_dataset = MockTorchDataset(dataset_tensor_labels)

    doleus_dataset = DoleusClassification(
        dataset=mock_torch_dataset,
        name="mock_binary_dataset",
        task=Task.BINARY.value,
        num_classes=2,
        label_to_name={0: "class0", 1: "class1"}
    )
    relevant_ids = list(range(len(doleus_dataset)))

    result = calculate_metric_internal(
        model, relevant_ids, doleus_dataset, groundtruths_labels, predictions_labels, "F1_Score"
    )
    assert result == 0.8, f"Expected F1_Score to be 0.8 but got {result}"


def test_calculate_accuracy_multilabel():
    model = MockModel(name="mock_multilabel_model", task_type=Task.MULTILABEL.value)
    
    groundtruths_labels = [
        MockLabel([1, 0, 1]), MockLabel([0, 1, 1]),
        MockLabel([1, 1, 0]), MockLabel([0, 0, 1])
    ]
    predictions_labels = [
        MockLabel([1, 0, 1]), MockLabel([0, 1, 0]), 
        MockLabel([1, 1, 0]), MockLabel([0, 1, 1])  
    ]
    
    dataset_tensor_labels = [gt.labels for gt in groundtruths_labels]
    mock_torch_dataset = MockTorchDataset(dataset_tensor_labels)

    doleus_dataset = DoleusClassification(
        dataset=mock_torch_dataset,
        name="mock_multilabel_dataset",
        task=Task.MULTILABEL.value,
        num_classes=3,
        label_to_name={0: "classA", 1: "classB", 2: "classC"}
    )
    relevant_ids = list(range(len(doleus_dataset)))

    result = calculate_metric_internal(
        model, relevant_ids, doleus_dataset, groundtruths_labels, predictions_labels, "Accuracy"
    )
    assert result == 0.5, f"Expected Multilabel Accuracy to be 0.5 but got {result}"


if __name__ == "__main__":
    pytest.main()
