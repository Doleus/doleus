import os

import numpy as np
import torch
from torch.utils.data import Subset
from torchvision.datasets import CocoDetection
from torchvision.models.detection import (FasterRCNN_ResNet50_FPN_V2_Weights,
                                          fasterrcnn_resnet50_fpn_v2)
from torchvision.ops import box_convert

from moonwatcher.check import Check, CheckSuite
# Import Moonwatcher components for evaluation
from moonwatcher.dataset.dataset import MoonwatcherDetection

# TODO: Check if I use the correct transform
# TODO: Make the dataset downloadable


# ================================
# Step 1: Define a Simplified Dataset Class
# ================================
class TinyCocoDataset(CocoDetection):
    """
    A simplified dataset class for Tiny COCO.
    Inherits from torchvision's CocoDetection.
    """

    def __init__(
        self, root: str, split: str = "train", transform=None, target_transform=None
    ) -> None:
        # Define paths for images and annotations
        ann_file = os.path.join(root, "annotations", f"instances_{split}2017.json")
        img_folder = os.path.join(root, f"{split}2017")
        super().__init__(
            img_folder, ann_file, transform=transform, target_transform=target_transform
        )

    def __getitem__(self, idx: int):
        # Retrieve image and its annotation(s)
        img, target = super().__getitem__(idx)

        # Extract bounding boxes (if any) and convert from [x,y,w,h] to [x1,y1,x2,y2]
        if target:
            boxes = torch.tensor([ann["bbox"] for ann in target], dtype=torch.float32)
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)  # No boxes present
        boxes = box_convert(boxes, in_fmt="xywh", out_fmt="xyxy")

        # Extract category labels from annotations
        labels = [ann["category_id"] for ann in target]
        return img, boxes, labels


# ================================
# Step 2: Load and Prepare the Dataset
# ================================
dataset = TinyCocoDataset(root="tiny_coco", split="val")

# For a quick demo, select a random subset of images (e.g., 5 images)
subset_size = 5
indices = torch.randperm(len(dataset))[:subset_size].tolist()
dataset_subset = Subset(dataset, indices)


# ================================
# Step 3: Initialize the Model and Preprocess Transform
# ================================
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights)
model.eval()

# Get the preprocessing transform from the weights
preprocess = weights.transforms()


# ================================
# Step 4: Run Predictions on the Dataset Subset
# ================================
predictions_list = []
with torch.no_grad():
    for idx in range(len(dataset_subset)):
        # Retrieve an image (and ground truth, if needed)
        img, gt_boxes, gt_labels = dataset_subset[idx]

        # Preprocess the image and perform prediction
        img_processed = preprocess(img)
        prediction = model([img_processed])[0]

        # Save predictions in the format expected by Moonwatcher
        pred_entry = {
            "boxes": prediction["boxes"].cpu(),
            "labels": prediction["labels"].cpu(),
            "scores": prediction["scores"].cpu(),
        }
        predictions_list.append(pred_entry)


# ================================
# Step 5: Create a Moonwatcher Dataset Wrapper and Add Metadata
# ================================
# Wrap the dataset subset for evaluation with Moonwatcher
moonwatcher_dataset = MoonwatcherDetection(
    name="tiny-coco-val-subset",
    dataset=dataset_subset,
    num_classes=91,  # COCO has 91 classes
)

# Add predefined metadata ("brightness") to allow slicing based on image properties
moonwatcher_dataset.add_predefined_metadata("brightness")

# Add model predictions to the dataset
moonwatcher_dataset.add_model_predictions(
    predictions=predictions_list,
    model_id="faster_rcnn",
)

# Create slices based on brightness percentile
slice_bright = moonwatcher_dataset.slice_by_percentile("brightness", ">=", 50)
slice_dim = moonwatcher_dataset.slice_by_percentile("brightness", "<", 50)


# ================================
# Step 6: Define and Run Evaluation Checks
# ================================
# Define checks for mean Average Precision (mAP) and Intersection over Union (IoU)
checks = [
    Check(
        name="mAP_overall",
        dataset=moonwatcher_dataset,
        model_id="faster_rcnn",
        metric="mAP",
    ),
    Check(
        name="mAP_bright_images",
        dataset=slice_bright,
        model_id="faster_rcnn",
        metric="mAP",
    ),
    Check(
        name="mAP_dim_images",
        dataset=slice_dim,
        model_id="faster_rcnn",
        metric="mAP",
    ),
    Check(
        name="IoU_overall",
        dataset=moonwatcher_dataset,
        model_id="faster_rcnn",
        metric="IntersectionOverUnion",
    ),
]

# Create a check suite to run all evaluation checks
check_suite = CheckSuite(name="tiny-coco-evaluation", checks=checks)

# Run the evaluation and print results
print("\nRunning evaluation checks...")
results = check_suite.run_all(show=True)
