"""Image metadata computation functions for dataset analysis."""

import cv2
import numpy as np


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
    return np.mean(hsv[:, :, 2])


def compute_contrast(image: np.ndarray) -> float:
    """Compute the contrast of an image.

    Parameters
    ----------
    image : np.ndarray
        Input image in BGR format.

    Returns
    -------
    float
        Standard deviation of the grayscale image values.
    """
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale.std()


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
    return np.mean(hsv[:, :, 1])


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
