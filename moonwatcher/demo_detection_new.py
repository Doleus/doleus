import os
import json
from pathlib import Path

import torch
import numpy as np
import torchvision.datasets
from torch.utils.data import Subset
from torchvision.transforms import functional as F

from moonwatcher.utils.data import Task
from moonwatcher.check import automated_checking
from moonwatcher.dataset.dataset import MoonwatcherDataset
from moonwatcher.utils.bbox_utils import box_xywh_abs_to_xyxy_abs
from moonwatcher.utils.data_storage import load_dataset


# TODO 1) Load Predictions from a JSON file
predictions_path = "moonwatcher/predictions_detection.json"
with open(predictions_path, 'r') as f:
    predictions_data = json.load(f)
    predictions = predictions_data['predictions']

# TODO 2) Choose a Dataset
cur_filepath = Path(__file__)
coco_path = Path("coco")
coco_sh_path = cur_filepath.parent / "coco.sh"
if not coco_path.exists() or not any(coco_path.iterdir()):
    os.system(f"sh {coco_sh_path}")

image_folder = "coco/images/val2017/"
annotations_file = "coco/annotations/instances_val2017.json"

_dataset = torchvision.datasets.CocoDetection(
    root=image_folder,
    annFile=annotations_file,
)


# TODO 3) Write transformations for the dataset
def dataset_output_transform(data):
    pil_image, annotation_list = data
    boxes = []
    labels = []
    for annotation in annotation_list:
        box_xywh_abs = annotation["bbox"]
        box_xyxy_abs = box_xywh_abs_to_xyxy_abs(box_xywh_abs)
        label = annotation["category_id"]
        boxes.append(box_xyxy_abs)
        labels.append(label)
    x = F.to_tensor(pil_image)
    boxes = torch.tensor(boxes)
    labels = torch.tensor(labels, dtype=torch.int64)
    return x, boxes, labels


# TODO 4) Create Moonwatcher Dataset
dataset_name = "COCO_val2017_subset"

try:
    mw_dataset = load_dataset(dataset_name)
except:
    # Mapping from numerical labels to strings
    label_to_name = {
        key: _dataset.coco.cats[key]["name"] for key in _dataset.coco.cats}

    # Select Subset of Dataset
    n_samples = 20
    _dataset = Subset(_dataset, [i for i in range(n_samples)])

    mw_dataset = MoonwatcherDataset(
        dataset=_dataset,
        name=dataset_name,
        task=Task.DETECTION.value,
        output_transform=dataset_output_transform,
        label_to_name=label_to_name,
    )

# Assign the loaded predictions to the MoonwatcherDataset
mw_dataset.predictions = predictions

# TODO 5) Automated Checking
automated_checking(
    mw_dataset,
)
