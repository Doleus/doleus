# SPDX-FileCopyrightText: 2025 Doleus contributors
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Union

import numpy as np
import torch
import torchmetrics

from doleus.datasets import Doleus

def _tpr_at_fpr(preds, target, task, num_classes, **kwargs):
    """Compute TPR at a fixed FPR threshold using ROC curve.
    
    Finds the largest FPR <= threshold and returns the associated TPR value.
    
    Parameters
    ----------
    preds : torch.Tensor
        Prediction scores/logits.
    target : torch.Tensor
        Ground truth labels.
    task : str
        Task type (must be "binary").
    num_classes : int
        Number of classes (ignored for binary).
    **kwargs
        Must contain 'fpr_threshold' (float between 0 and 1).
        May also contain valid torchmetrics ROC parameters: 'thresholds', 'ignore_index', 'validate_args'.
    
    Returns
    -------
    float
        TPR value at the largest FPR <= threshold.
    """
    fpr_threshold = kwargs.pop("fpr_threshold")
    
    # Compute ROC curve (remaining kwargs are passed to torchmetrics)
    fpr, tpr, _ = torchmetrics.functional.classification.binary_roc(
        preds, target, **kwargs
    )
    
    # Convert to numpy for easier indexing
    fpr_np = fpr.cpu().numpy()
    tpr_np = tpr.cpu().numpy()
    
    # Handle edge cases
    if len(fpr_np) == 0:
        raise ValueError("ROC curve is empty")
    
    # If threshold is exactly at a point
    exact_matches = np.where(fpr_np == fpr_threshold)[0]
    if len(exact_matches) > 0:
        return float(tpr_np[exact_matches[0]])
    
    # Find the largest FPR <= threshold (next smaller value)
    valid_indices = np.where(fpr_np <= fpr_threshold)[0]
    
    if len(valid_indices) == 0:
        # All FPR values are greater than threshold, return TPR at first point
        return float(tpr_np[0])
    
    # Get the point with largest FPR <= threshold
    idx_max = valid_indices[-1]
    return float(tpr_np[idx_max])


def _fpr_at_tpr(preds, target, task, num_classes, **kwargs):
    """Compute FPR at a fixed TPR threshold using ROC curve.
    
    Finds the smallest TPR >= threshold and returns the associated FPR value.
    
    Parameters
    ----------
    preds : torch.Tensor
        Prediction scores/logits.
    target : torch.Tensor
        Ground truth labels.
    task : str
        Task type (must be "binary").
    num_classes : int
        Number of classes (ignored for binary).
    **kwargs
        Must contain 'tpr_threshold' (float between 0 and 1).
        May also contain valid torchmetrics ROC parameters: 'thresholds', 'ignore_index', 'validate_args'.
    
    Returns
    -------
    float
        FPR value at the smallest TPR >= threshold.
    """
    tpr_threshold = kwargs.pop("tpr_threshold")
    
    # Compute ROC curve (remaining kwargs are passed to torchmetrics)
    fpr, tpr, _ = torchmetrics.functional.classification.binary_roc(
        preds, target, **kwargs
    )
    
    # Convert to numpy for easier indexing
    fpr_np = fpr.cpu().numpy()
    tpr_np = tpr.cpu().numpy()
    
    # Handle edge cases
    if len(tpr_np) == 0:
        raise ValueError("ROC curve is empty")
    
    # If threshold is exactly at a point
    exact_matches = np.where(tpr_np == tpr_threshold)[0]
    if len(exact_matches) > 0:
        return float(fpr_np[exact_matches[0]])
    
    # Find the smallest TPR >= threshold (next bigger value)
    valid_indices = np.where(tpr_np >= tpr_threshold)[0]
    
    if len(valid_indices) == 0:
        # All TPR values are less than threshold, return FPR at last point
        return float(fpr_np[-1])
    
    # Get the point with smallest TPR >= threshold
    idx_min = valid_indices[0]
    return float(fpr_np[idx_min])


METRIC_FUNCTIONS = {
    "Accuracy": torchmetrics.functional.accuracy,
    "Precision": torchmetrics.functional.precision,
    "Recall": torchmetrics.functional.recall,
    "F1_Score": torchmetrics.functional.f1_score,
    "HammingDistance": torchmetrics.functional.hamming_distance,
    "TPR_at_FPR": _tpr_at_fpr,
    "FPR_at_TPR": _fpr_at_tpr,
    "mAP": torchmetrics.detection.MeanAveragePrecision,
    "mAP_small": torchmetrics.detection.MeanAveragePrecision,
    "mAP_medium": torchmetrics.detection.MeanAveragePrecision,
    "mAP_large": torchmetrics.detection.MeanAveragePrecision,
    "CompleteIntersectionOverUnion": torchmetrics.detection.CompleteIntersectionOverUnion,
    "DistanceIntersectionOverUnion": torchmetrics.detection.DistanceIntersectionOverUnion,
    "GeneralizedIntersectionOverUnion": torchmetrics.detection.GeneralizedIntersectionOverUnion,
    "IntersectionOverUnion": torchmetrics.detection.IntersectionOverUnion,
    "IoU": torchmetrics.detection.IntersectionOverUnion,
    "Specificity": torchmetrics.functional.specificity,
}

METRIC_KEYS = {
    "mAP": "map",
    "mAP_small": "map_small",
    "mAP_medium": "map_medium",
    "mAP_large": "map_large",
    "CompleteIntersectionOverUnion": "ciou",
    "DistanceIntersectionOverUnion": "diou",
    "GeneralizedIntersectionOverUnion": "giou",
    "IntersectionOverUnion": "iou",
    "IoU": "iou",
}


def get_class_id(
    target_class: Optional[Union[int, str]], dataset: Doleus
) -> Optional[int]:
    """Get the numerical class ID for a given target class.

    Parameters
    ----------
    target_class : Optional[Union[int, str]]
        The target class to parse. Can be an integer (class ID), string (class name), or None.
    dataset : Doleus
        The dataset containing the label_to_name mapping.

    Returns
    -------
    Optional[int]
        The numerical class ID (None if target_class was not specified).
    """
    if target_class is None:
        return None

    if dataset.label_to_name is None:
        raise AttributeError(
            "label_to_name must be provided as a parameter to the Doleus Dataset when specifying a `target_class` in the Check!"
        )

    if isinstance(target_class, int):
        return target_class

    if isinstance(target_class, str):
        if target_class not in dataset.label_to_name.values():
            raise KeyError(
                f"Class name '{target_class}' not found in label_to_name mapping. Existing classes are: {list(dataset.label_to_name.values())}"
            )
        for k, v in dataset.label_to_name.items():
            if v == target_class:
                return int(k)

    raise TypeError(f"Unsupported type for target_class: {type(target_class)}")
