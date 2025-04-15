from typing import Any, Dict, List, Optional, Union

import torch

from doleus.annotations.classification import Labels
from doleus.annotations.detection import BoundingBoxes
from doleus.datasets import Doleus, Slice
from doleus.metrics.metric_utils import (METRIC_FUNCTIONS, METRIC_KEYS,
                                         get_class_id)
from doleus.utils.data import TaskType


class MetricCalculator:
    """Metric calculator for classification and detection tasks."""

    def __init__(
        self,
        dataset: Union[Doleus, Slice],
        metric: str,
        metric_parameters: Optional[Dict[str, Any]] = None,
        target_class: Optional[Union[int, str]] = None,
    ):
        """Initialize the metric calculator.

        Parameters
        ----------
        dataset : Union[Doleus, Slice]
            The dataset or slice to compute metrics on.
        metric : str
            Name of the metric to compute.
        metric_parameters : Optional[Dict[str, Any]], optional
            Additional parameters for the metric computation, by default None.
        target_class : Optional[Union[int, str]], optional
            Optional class ID or name to compute class-specific metrics.
        """
        self.dataset = dataset
        self.metric = metric
        self.metric_parameters = metric_parameters or {}
        self.target_class_raw = target_class

        if isinstance(dataset, Slice):
            self.root_dataset = dataset.root_dataset
        else:
            self.root_dataset = dataset

        self.target_class_id = get_class_id(target_class, self.root_dataset)

    def calculate(self, indices: List[int]) -> float:
        """Calculate the metric for the specified indices.

        Parameters
        ----------
        indices : List[int]
            List of indices to calculate the metric for.

        Returns
        -------
        float
            The calculated metric value.
        """
        groundtruths_loaded = [self.dataset.groundtruths[i] for i in indices]
        predictions_loaded = [self.dataset.predictions[i] for i in indices]

        if self.root_dataset.task_type == TaskType.CLASSIFICATION.value:
            return self._calculate_classification(
                groundtruths_loaded, predictions_loaded
            )
        elif self.root_dataset.task_type == TaskType.DETECTION.value:
            return self._calculate_detection(groundtruths_loaded, predictions_loaded)
        else:
            raise ValueError(f"Unsupported task type: {self.root_dataset.task_type}")

    def _calculate_classification(
        self, groundtruths_loaded: List[Labels], predictions_loaded: List[Labels]
    ) -> float:
        """Compute a classification metric.

        Parameters
        ----------
        groundtruths_loaded : List[Labels]
            List of ground truth label annotations.
        predictions_loaded : List[Labels]
            List of predicted label annotations.

        Returns
        -------
        float
            The computed metric value.
        """
        try:
            gt_tensor = torch.stack(
                [ann.labels.squeeze() for ann in groundtruths_loaded]
            )

            pred_list = [
                ann.scores if ann.scores is not None else ann.labels.squeeze()
                for ann in predictions_loaded
            ]
            if not pred_list:
                raise ValueError("No predictions provided to compute the metric.")
            pred_tensor = torch.stack(pred_list)

            # Set default averaging if not specified
            if "average" not in self.metric_parameters:
                self.metric_parameters["average"] = "macro"

            # If a specific class is requested, override averaging
            if self.target_class_id is not None:
                self.metric_parameters["average"] = "none"

            metric_fn = METRIC_FUNCTIONS[self.metric]
            metric_value = metric_fn(
                pred_tensor,
                gt_tensor,
                task=self.root_dataset.task,
                num_classes=self.root_dataset.num_classes,
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
        groundtruths_loaded: List[BoundingBoxes],
        predictions_loaded: List[BoundingBoxes],
    ) -> float:
        """Compute a detection metric.

        Parameters
        ----------
        groundtruths_loaded : List[BoundingBoxes]
            List of ground truth bounding box annotations.
        predictions_loaded : List[BoundingBoxes]
            List of predicted bounding box annotations.

        Returns
        -------
        float
            The computed metric value.
        """
        try:
            gt_list = [ann.to_dict() for ann in groundtruths_loaded]
            pred_list = [ann.to_dict() for ann in predictions_loaded]

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
    dataset: Union[Doleus, Slice],
    indices: List[int],
    metric: str,
    metric_parameters: Optional[Dict[str, Any]] = None,
    target_class: Optional[Union[int, str]] = None,
) -> float:
    """Compute a metric on a dataset or slice.

    Parameters
    ----------
    dataset : Union[Doleus, Slice]
        The dataset or slice to compute metrics on.
    indices : List[int]
        List of indices to compute the metric for.
    metric : str
        Name of the metric to compute.
    metric_parameters : Optional[Dict[str, Any]], optional
        Additional parameters for the metric computation, by default None.
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
        metric_parameters=metric_parameters,
        target_class=target_class,
    )
    return calculator.calculate(indices)
