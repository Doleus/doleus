"""Image metadata computation functions for dataset analysis."""

from typing import Any, Dict

import cv2
import numpy as np


def compute_image_metadata(image: np.ndarray) -> Dict[str, Any]:
    """Compute various metadata attributes from an image.

    Parameters
    ----------
    image : np.ndarray
        Input image in BGR format.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing various image metadata attributes.
    """
    return {
        "brightness": compute_brightness(image),
        "contrast": compute_contrast(image),
        "saturation": compute_saturation(image),
        "resolution": compute_resolution(image),
    }


def compute_brightness(image: np.ndarray) -> float:
    """Compute the average brightness of an image.

    Parameters
    ----------
    image : np.ndarray
        Input image in BGR format.

    Returns
    -------
    float
        Average brightness value from the HSV representation.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 2]))


def compute_contrast(image: np.ndarray) -> float:
    """Compute the contrast of an image using standard deviation of intensity.

    Parameters
    ----------
    image : np.ndarray
        Input image in BGR format.

    Returns
    -------
    float
        Contrast calculated as standard deviation of intensity values.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.std(gray))


def compute_saturation(image: np.ndarray) -> float:
    """Compute the average saturation of an image.

    Parameters
    ----------
    image : np.ndarray
        Input image in BGR format.

    Returns
    -------
    float
        Average saturation value from the HSV representation.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 1]))


def compute_resolution(image: np.ndarray) -> int:
    """Compute the total number of pixels in an image.

    Parameters
    ----------
    image : np.ndarray
        Input image in BGR format.

    Returns
    -------
    int
        Total number of pixels (height * width).
    """
    height, width = image.shape[:2]
    return height * width


# Registry of available metadata computation functions
ATTRIBUTE_FUNCTIONS = {
    "brightness": compute_brightness,
    "contrast": compute_contrast,
    "saturation": compute_saturation,
    "resolution": compute_resolution,
}
