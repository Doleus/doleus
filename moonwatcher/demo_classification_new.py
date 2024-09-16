import json
import random
from pathlib import Path

import torch
import torchvision.datasets
from torchvision import transforms
from torch.utils.data import Subset

from moonwatcher.dataset.dataset import MoonwatcherDataset
from moonwatcher.utils.data_storage import load_dataset
from moonwatcher.utils.data import Task
from moonwatcher.check import automated_checking
from moonwatcher.utils.imagenet_to_stl import stl10_classes

# TODO 1) Load Predictions from a JSON file
predictions_path = "predictions.json"
with open(predictions_path, 'r') as f:
    predictions_data = json.load(f)
    predictions = predictions_data['predictions']

# TODO 2) Choose a Dataset and apply transformations
image_folder = "."
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ]
)
_dataset = torchvision.datasets.STL10(
    root=image_folder,
    transform=transform,
    split="test",
    download=True,
)

# TODO 3) Select a subset of the dataset
n_samples = 100
random.seed(42)
random_indices = random.sample(range(len(_dataset)), n_samples)
subset_dataset = Subset(_dataset, random_indices)

# Adjust predictions to match the subset (assuming they are in the same order)
subset_predictions = predictions  # [predictions[i] for i in random_indices]

# TODO 4) Define dataset output transformation


def dataset_output_transform(data):
    x, label = data
    return x, torch.tensor(label)


# TODO 5) Create Moonwatcher Dataset with predictions
dataset_name = "STL10"

try:
    mw_dataset = load_dataset(dataset_name)
except:
    # Mapping from numerical labels to strings
    label_to_name = dict(enumerate(stl10_classes))

    mw_dataset = MoonwatcherDataset(
        dataset=subset_dataset,
        name=dataset_name,
        task=Task.CLASSIFICATION.value,
        output_transform=dataset_output_transform,
        label_to_name=label_to_name,
    )

# Assign the loaded predictions to the MoonwatcherDataset
mw_dataset.predictions = subset_predictions

# TODO 6) Automated Checking
automated_checking(
    mw_dataset=mw_dataset,
)
