import datetime
from typing import List, Union

import numpy as np
import pytz
import torch
from PIL import Image
from torch.utils.data import Dataset


# TODO: Since we only allow a depth of 1 for dataset wrappers, this function can be made redundant
def find_root_dataset(dataset: Dataset) -> Dataset:
    """Find the root dataset by iteratively traversing dataset wrappers.

    Parameters
    ----------
    dataset : Dataset
        A dataset which may be wrapped in one or more dataset wrappers
        (e.g. Subset).

    Returns
    -------
    Dataset
        The root dataset.
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


def to_numpy_image(root_dataset: Dataset, index: int) -> np.ndarray:
    """Convert an image to a numpy array format.

    Parameters
    ----------
    root_dataset : Dataset
        The root dataset to get the image from.
    index : int
        Index of the image in the dataset.

    Returns
    -------
    np.ndarray
        The image as a numpy array in BGR format.
    """
    raw_image = get_raw_image(root_dataset, index)
    if isinstance(raw_image, torch.Tensor):
        raw_image = (raw_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    elif isinstance(raw_image, Image.Image):
        raw_image = np.array(raw_image)
    return raw_image


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
