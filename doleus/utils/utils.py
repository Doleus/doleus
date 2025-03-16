"""Utility functions for the Doleus project."""

import datetime
from typing import Union

import numpy as np
import pytz
import torch
from PIL import Image
from torch.utils.data import Dataset


def find_root_dataset(dataset: Dataset) -> Dataset:
    """Find the root dataset by iteratively traversing dataset wrappers.

    Parameters
    ----------
    dataset : Dataset
        The dataset to find the root of, which may be wrapped in one or more
        dataset wrappers (e.g., Subset).

    Returns
    -------
    Dataset
        The root dataset that contains the actual data.
    """
    current = dataset
    while hasattr(current, "dataset"):
        current = current.dataset
    return current


def get_raw_image(
    root_dataset: Dataset, index: int
) -> Union[Image.Image, np.ndarray, torch.Tensor]:
    """Retrieve the original image from a dataset bypassing its transforms.

    Parameters
    ----------
    root_dataset : Dataset
        The root dataset to get the image from.
    index : int
        The index of the image to retrieve.

    Returns
    -------
    Union[Image.Image, np.ndarray, torch.Tensor]
        The raw image in its original format, before any transforms are applied.
    """
    if not hasattr(root_dataset, "transform"):
        return root_dataset[index][0]

    original_transform = root_dataset.transform
    root_dataset.transform = None
    data = root_dataset[index]
    image = data[0]
    root_dataset.transform = original_transform
    return image


def get_current_timestamp() -> str:
    """Get the current timestamp in ISO format with Europe/Berlin timezone.

    Returns
    -------
    str
        The current timestamp in ISO format.
    """
    tz = pytz.timezone("Europe/Berlin")
    timestamp = datetime.datetime.now(tz=tz).isoformat()
    return timestamp
