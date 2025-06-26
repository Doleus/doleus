# SPDX-FileCopyrightText: 2025 Doleus contributors
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Union

import torchmetrics

from doleus.datasets import Doleus

METRIC_FUNCTIONS = {
    "Accuracy": torchmetrics.functional.accuracy,
    "Precision": torchmetrics.functional.precision,
    "Recall": torchmetrics.functional.recall,
    "F1_Score": torchmetrics.functional.f1_score,
    "HammingDistance": torchmetrics.functional.hamming_distance,
    "mAP": torchmetrics.detection.MeanAveragePrecision,
    "mAP_small": torchmetrics.detection.MeanAveragePrecision,
    "mAP_medium": torchmetrics.detection.MeanAveragePrecision,
    "mAP_large": torchmetrics.detection.MeanAveragePrecision,
    "CompleteIntersectionOverUnion": torchmetrics.detection.CompleteIntersectionOverUnion,
    "DistanceIntersectionOverUnion": torchmetrics.detection.DistanceIntersectionOverUnion,
    "GeneralizedIntersectionOverUnion": torchmetrics.detection.GeneralizedIntersectionOverUnion,
    "IntersectionOverUnion": torchmetrics.detection.IntersectionOverUnion,
    "IoU": torchmetrics.detection.IntersectionOverUnion,
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
