from typing import List, Dict, Any, Union
from abc import ABC, abstractmethod

from torch.nn import Module

from moonwatcher.utils.data import DataType, TaskType
from moonwatcher.base.base import MoonwatcherObject
from moonwatcher.utils.helpers import get_current_timestamp
from moonwatcher.utils.api_connector import upload_if_possible


class ModelOutputInputTransformation(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def transform_input(self, inputs):
        """
        Transform input data into a format that can be directly passed to the model.

        :param inputs: An image from your specified dataset.
        :return:
            The transformed input data, formatted to be passed directly to the model.
            The output can be structured either as positional arguments (*args),
            or as both positional and keyword arguments (*args and **kwargs).

        Internally, the arguments are directly passed to the model either as model(*args) or model(*args, **kwargs).
        """
        pass

    @abstractmethod
    def transform_output(self, outputs):
        """
        Transform the output of the model into the required format.

        :param outputs: The output from the model.

        :return:
            A tuple containing the model outputs formatted as required:
            - Classification:
                labels (torch.Tensor): A 1-dimensional integer tensor of shape (1) representing the label.
                scores (optional, torch.Tensor): A float tensor of shape (num_classes) representing the confidence scores for each class.

            - Detection:
                boxes_xyxy (torch.Tensor): A tensor of shape (num_boxes, 4) representing bounding box coordinates.
                labels (torch.Tensor): An integer tensor of shape (num_boxes) representing labels for each bounding box.
                scores (optional, torch.Tensor): A float tensor of shape (num_boxes) representing the confidence score for each bounding box.
        """
        pass

    # CHANGE: Added predictions transform to simplify how users pass their model predictions
    def transform_prediction(self, predictions: List[Any]):
        """
        Transforms user-provided predictions into a Predictions object.

        :param predictions: A list containing user predictions in various formats.
        :return: A Predictions object standardized for the Moonwatcher framework.
        """
        pass
    # TODO: Add the function
# CHANGE: Simplified MoonwatcherModel class by removing attributes that are not necessary anymore since model doesn't perform inference.


class MoonwatcherModel(MoonwatcherObject, Module):
    def __init__(
        self,
        name: str,
        task_type: str,
        predictions: List[Any],
    ):
        """
        Creates a moonwatcher model wrapper around an existing model that can be used with the moonwatcher framework

        :param name: the name you want to give this model
        :param task: either classification or detection
        :param predictions: a list of predictions provided by the users
        """
        MoonwatcherObject.__init__(self, name=name, datatype=DataType.MODEL)

        self.name = name
        self.task_type = task_type
        self.predictions = predictions

    # CHANGE: Deleted delete and _upload function. Both aren't requried anymore
