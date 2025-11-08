# SPDX-FileCopyrightText: 2025 Doleus contributors
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Union

import torch

from doleus.annotations import Annotation, BoundingBoxes, Labels
from doleus.datasets import Doleus
from doleus.metrics.metric_utils import METRIC_FUNCTIONS, METRIC_KEYS, get_class_id
from doleus.utils import TaskType


class MetricCalculator:
    """Metric calculator for classification and detection tasks."""

    def __init__(
        self,
        dataset: Doleus,
        metric: str,
        predictions: List[Annotation],
        metric_parameters: Optional[Dict[str, Any]] = None,
        target_class: Optional[Union[int, str]] = None,
    ):
        """Initialize the metric calculator.

        Parameters
        ----------
        dataset : Doleus
            The dataset to compute metrics on.
        metric : str
            Name of the metric to compute.
        predictions : List[Annotation]
            List of prediction annotations to evaluate against ground truths.
        metric_parameters : Optional[Dict[str, Any]], optional
            Optional parameters to pass directly to the corresponding torchmetrics function, by default None.
            For ROC-based metrics (TPR_at_FPR, FPR_at_TPR), must include 'fpr_threshold' or 'tpr_threshold' respectively.
        target_class : Optional[Union[int, str]], optional
            Optional class ID or name to compute class-specific metrics.
        """
        self.dataset = dataset
        self.metric = metric
        self.predictions = predictions
        self.metric_parameters = metric_parameters or {}
        self.target_class_raw = target_class
        self.target_class_id = get_class_id(target_class, self.dataset)
        self.groundtruths = [
            self.dataset.groundtruth_store.get(i) for i in range(len(self.dataset))
        ]

    def calculate(self) -> float:
        """Calculate the metric.

        Returns
        -------
        float
            The calculated metric value.
        """
        if self.dataset.task_type == TaskType.CLASSIFICATION.value:
            return self._calculate_classification(self.groundtruths, self.predictions)
        elif self.dataset.task_type == TaskType.DETECTION.value:
            return self._calculate_detection(self.groundtruths, self.predictions)
        else:
            raise ValueError(f"Unsupported task type: {self.dataset.task_type}")

    def _calculate_classification(
        self, groundtruths: List[Labels], predictions: List[Labels]
    ) -> float:
        """Compute a classification metric.

        Parameters
        ----------
        groundtruths : List[Labels]
            List of ground truth label annotations.
        predictions : List[Labels]
            List of predicted label annotations. For ROC-based metrics (TPR_at_FPR, FPR_at_TPR),
            predictions must contain scores/logits (not labels).

        Returns
        -------
        float
            The computed metric value.
        """
        try:
            gt_tensor = torch.stack([ann.labels.squeeze() for ann in groundtruths])

            # Special handling for ROC-based metrics
            if self.metric in ["TPR_at_FPR", "FPR_at_TPR"]:
                # Validate binary classification
                if self.dataset.task != "binary":
                    raise ValueError(
                        f"{self.metric} only supports binary classification, "
                        f"got task: {self.dataset.task}"
                    )
                
                # Validate required threshold parameter
                if self.metric == "TPR_at_FPR":
                    if "fpr_threshold" not in self.metric_parameters:
                        raise ValueError(
                            f"{self.metric} requires 'fpr_threshold' parameter in metric_parameters"
                        )
                elif self.metric == "FPR_at_TPR":
                    if "tpr_threshold" not in self.metric_parameters:
                        raise ValueError(
                            f"{self.metric} requires 'tpr_threshold' parameter in metric_parameters"
                        )
                
                # ROC metrics require scores/logits, not labels
                pred_list = []
                for ann in predictions:
                    if ann.scores is None:
                        raise ValueError(
                            f"{self.metric} requires prediction scores/logits, "
                            f"but prediction annotation has no scores. "
                            f"Please provide float predictions (scores/logits) instead of integer labels."
                        )
                    pred_list.append(ann.scores.squeeze())
                
                if not pred_list:
                    raise ValueError("No predictions provided to compute the metric.")
                pred_tensor = torch.stack(pred_list)
                
                metric_fn = METRIC_FUNCTIONS[self.metric]
                metric_value = metric_fn(
                    pred_tensor,
                    gt_tensor,
                    task=self.dataset.task,
                    num_classes=self.dataset.num_classes,
                    **self.metric_parameters,
                )
                
                return float(metric_value)
            
            # Standard classification metrics
            pred_list = []
            for ann in predictions:
                if ann.labels is not None:
                    pred_list.append(ann.labels.squeeze())
                elif ann.scores is not None:
                    pred_list.append(ann.scores.squeeze())
                else:
                    raise ValueError(
                        f"Prediction annotation has neither labels nor scores: {ann}"
                    )

            if not pred_list:
                raise ValueError("No predictions provided to compute the metric.")
            pred_tensor = torch.stack(pred_list)

            # Set macro averaging as the default (see: https://github.com/Lightning-AI/torchmetrics/issues/2280)
            if "average" not in self.metric_parameters:
                self.metric_parameters["average"] = "macro"

            # If a specific class is requested, override averaging
            if self.target_class_id is not None:
                self.metric_parameters["average"] = "none"

            metric_fn = METRIC_FUNCTIONS[self.metric]

            # Torchmetrics expects num_labels for multilabel tasks and num_classes for other tasks
            if self.dataset.task == "multilabel":
                metric_value = metric_fn(
                    pred_tensor,
                    gt_tensor,
                    task=self.dataset.task,
                    num_labels=self.dataset.num_classes,
                    **self.metric_parameters,
                )
            else:
                metric_value = metric_fn(
                    pred_tensor,
                    gt_tensor,
                    task=self.dataset.task,
                    num_classes=self.dataset.num_classes,
                    **self.metric_parameters,
                )

            if self.target_class_id is not None:
                metric_value = metric_value[self.target_class_id]

            return (
                float(metric_value.item())
                if hasattr(metric_value, "item")
                else float(metric_value)
            )
        except Exception as e:
            raise RuntimeError(
                f"Error in classification metric computation: {str(e)}"
            ) from e

    def _calculate_detection(
        self,
        groundtruths: List[BoundingBoxes],
        predictions: List[BoundingBoxes],
    ) -> float:
        """Compute a detection metric.

        Parameters
        ----------
        groundtruths : List[BoundingBoxes]
            List of ground truth bounding box annotations.
        predictions : List[BoundingBoxes]
            List of predicted bounding box annotations.

        Returns
        -------
        float
            The computed metric value.
        """
        try:
            gt_list = [ann.to_dict() for ann in groundtruths]
            pred_list = [ann.to_dict() for ann in predictions]

            if self.target_class_id is not None:
                self.metric_parameters["class_metrics"] = True

            if self.metric in ["mAP", "mAP_small", "mAP_medium", "mAP_large"]:
                self.metric_parameters["iou_type"] = "bbox"

            metric_fn = METRIC_FUNCTIONS[self.metric](**self.metric_parameters)
            metric_fn.update(pred_list, gt_list)
            metric_value_dict = metric_fn.compute()

            if self.target_class_id is not None:
                if self.metric in ["mAP", "mAP_small", "mAP_medium", "mAP_large"]:
                    classes = metric_value_dict.get("classes", None)
                    if classes is not None:
                        index = torch.where(classes == self.target_class_id)[0]
                        result = (
                            metric_value_dict["map_per_class"][index].item()
                            if index.numel() > 0
                            else 0.0
                        )
                    else:
                        result = 0.0
                else:
                    key = f"{METRIC_KEYS[self.metric]}/cl_{self.target_class_id}"
                    result = metric_value_dict.get(key, 0.0)
            else:
                result = metric_value_dict[METRIC_KEYS[self.metric]]

            return float(result.item()) if hasattr(result, "item") else float(result)
        except Exception as e:
            raise RuntimeError(
                f"Error in detection metric computation: {str(e)}"
            ) from e


def calculate_metric(
    dataset: Doleus,
    metric: str,
    predictions: List[Annotation],
    metric_parameters: Optional[Dict[str, Any]] = None,
    target_class: Optional[Union[int, str]] = None,
) -> float:
    """Compute a metric on a dataset.

    Parameters
    ----------
    dataset : Doleus
        The dataset to compute metrics on.
    metric : str
        Name of the metric to compute.
    predictions : List[Annotation]
        List of prediction annotations to evaluate against ground truths.
    metric_parameters : Optional[Dict[str, Any]], optional
        Optional parameters to pass directly to the corresponding torchmetrics function, by default None.
        For ROC-based metrics (TPR_at_FPR, FPR_at_TPR), must include 'fpr_threshold' or 'tpr_threshold' respectively.
        ROC-based metrics only support binary classification and require prediction scores/logits (not labels).
    target_class : Optional[Union[int, str]], optional
        Optional class ID or name to compute class-specific metrics, by default None.

    Returns
    -------
    float
        The computed metric value.
    """
    calculator = MetricCalculator(
        dataset=dataset,
        metric=metric,
        predictions=predictions,
        metric_parameters=metric_parameters,
        target_class=target_class,
    )
    return calculator.calculate()
