from typing import Any, Dict, Optional, Union

import torch
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
}


def parse_target_class(
    target_class: Optional[Union[int, str]], dataset: Doleus
) -> Optional[int]:
    """Parse and validate the target class parameter.

    Parameters
    ----------
    target_class : Optional[Union[int, str]]
        The target class to parse. Can be an integer (class ID) or string (class name).
    dataset : Doleus
        The dataset containing the label_to_name mapping.

    Returns
    -------
    Optional[int]
        The parsed class ID, or None if target_class was None.
    """
    if target_class is None:
        return None

    if dataset.label_to_name is None:
        raise ValueError(
            "label_to_name mapping not provided; cannot parse target_class."
        )

    if isinstance(target_class, int):
        return target_class

    if isinstance(target_class, str):
        if target_class not in dataset.label_to_name.values():
            raise ValueError(
                f"Class name '{target_class}' not found in label_to_name mapping."
            )
        # Invert the mapping: {id: name} becomes {name: id}
        for k, v in dataset.label_to_name.items():
            if v == target_class:
                return int(k)

    raise TypeError(f"Unsupported type for target_class: {type(target_class)}")


def convert_detection_dicts(annotations_list: list[Dict[str, Any]]) -> None:
    """Convert bounding box dictionary values to torch.Tensor format.

    Parameters
    ----------
    annotations_list : list[Dict[str, Any]]
        List of annotation dictionaries containing bounding box information.
    """
    for ann in annotations_list:
        ann["boxes"] = torch.tensor(ann["boxes"], dtype=torch.float32)
        ann["labels"] = torch.tensor(ann["labels"], dtype=torch.int64)
        if "scores" in ann:
            ann["scores"] = torch.tensor(ann["scores"], dtype=torch.float32)
