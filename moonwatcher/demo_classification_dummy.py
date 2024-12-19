import torch
from torchvision import transforms
from torch.utils.data import Dataset

from moonwatcher.utils.data import Task
from moonwatcher.check import Check, CheckSuite, automated_checking
from moonwatcher.dataset.dataset import MoonwatcherClassification


class DummyImageNetDataset(Dataset):
    def __init__(self, num_samples=100, num_classes=5, image_size=(3, 224, 224)):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.rand(*self.image_size)
        label = torch.randint(0, self.num_classes, (1,))
        return image, label


# Define inputs
dataset = DummyImageNetDataset()
predictions = torch.randint(0, 5, (100,)).unsqueeze(1)

# Create Moonwatcher dataset
moonwatcher_dataset = MoonwatcherClassification(
    name="dummy_dataset", dataset=dataset, task=Task.MULTICLASS.value, num_classes=5)

# Add Metadata
moonwatcher_dataset.add_predefined_metadata("brightness")

# Create Slices
slice_bright = moonwatcher_dataset.slice_by_percentile("brightness", ">", 90)
slice_dim = moonwatcher_dataset.slice_by_percentile("brightness", "<", 10)

# Create Checks
check_bright = Check(
    name="accuracy_bright",
    dataset_or_slice=slice_bright,
    predictions=predictions,
    metric="Accuracy",
    operator=">",
    value=0.5,
)
check_dim = Check(
    name="accuracy_dim",
    dataset_or_slice=slice_dim,
    predictions=predictions,
    metric="Accuracy",
    operator=">",
    value=0.5,
)
check_suite = CheckSuite(
    name=f"test_brightness", checks=[check_bright, check_dim]
)
results = check_suite(show=True)
