# SPDX-FileCopyrightText: 2025 Doleus contributors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from doleus.utils.image_metadata import (
    ATTRIBUTE_FUNCTIONS,
    compute_brightness,
    compute_contrast,
    compute_image_metadata,
    compute_resolution,
    compute_saturation,
)


class TestComputeBrightness:
    """Test cases for compute_brightness function."""

    def test_black_image(self):
        """Test brightness of completely black image."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        brightness = compute_brightness(image)
        assert brightness == 0.0

    def test_white_image(self):
        """Test brightness of completely white image."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        brightness = compute_brightness(image)
        assert brightness == 255.0

    def test_gray_image(self):
        """Test brightness of uniform gray image."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        brightness = compute_brightness(image)
        assert brightness == pytest.approx(128.0, abs=1.0)

    def test_returns_float(self):
        """Test that brightness returns a float value."""
        image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        brightness = compute_brightness(image)
        assert isinstance(brightness, float)

    def test_brightness_range(self):
        """Test that brightness is within valid range."""
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        brightness = compute_brightness(image)
        assert 0.0 <= brightness <= 255.0


class TestComputeContrast:
    """Test cases for compute_contrast function."""

    def test_uniform_image_zero_contrast(self):
        """Test contrast of uniform image is zero."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        contrast = compute_contrast(image)
        assert contrast == 0.0

    def test_high_contrast_image(self):
        """Test contrast of black and white checkerboard."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[::2, ::2] = 255
        image[1::2, 1::2] = 255
        contrast = compute_contrast(image)
        assert contrast > 0.0

    def test_returns_float(self):
        """Test that contrast returns a float value."""
        image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        contrast = compute_contrast(image)
        assert isinstance(contrast, float)

    def test_contrast_non_negative(self):
        """Test that contrast is always non-negative."""
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        contrast = compute_contrast(image)
        assert contrast >= 0.0


class TestComputeSaturation:
    """Test cases for compute_saturation function."""

    def test_grayscale_image_zero_saturation(self):
        """Test saturation of grayscale image is zero."""
        gray_value = 128
        image = np.ones((100, 100, 3), dtype=np.uint8) * gray_value
        saturation = compute_saturation(image)
        assert saturation == 0.0

    def test_fully_saturated_red(self):
        """Test saturation of fully saturated red image."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:, :, 2] = 255  # BGR format, so red is channel 2
        saturation = compute_saturation(image)
        assert saturation == 255.0

    def test_returns_float(self):
        """Test that saturation returns a float value."""
        image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        saturation = compute_saturation(image)
        assert isinstance(saturation, float)

    def test_saturation_range(self):
        """Test that saturation is within valid range."""
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        saturation = compute_saturation(image)
        assert 0.0 <= saturation <= 255.0


class TestComputeResolution:
    """Test cases for compute_resolution function."""

    def test_resolution_calculation(self):
        """Test resolution calculation for known dimensions."""
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        resolution = compute_resolution(image)
        assert resolution == 20000

    def test_square_image(self):
        """Test resolution of square image."""
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        resolution = compute_resolution(image)
        assert resolution == 2500

    def test_single_pixel(self):
        """Test resolution of single pixel image."""
        image = np.zeros((1, 1, 3), dtype=np.uint8)
        resolution = compute_resolution(image)
        assert resolution == 1

    def test_returns_int(self):
        """Test that resolution returns an integer value."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        resolution = compute_resolution(image)
        assert isinstance(resolution, int)

    def test_grayscale_image_resolution(self):
        """Test resolution calculation for grayscale image."""
        image = np.zeros((100, 200), dtype=np.uint8)
        resolution = compute_resolution(image)
        assert resolution == 20000


class TestComputeImageMetadata:
    """Test cases for compute_image_metadata function."""

    def test_returns_dict(self):
        """Test that compute_image_metadata returns a dictionary."""
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        metadata = compute_image_metadata(image)
        assert isinstance(metadata, dict)

    def test_contains_all_attributes(self):
        """Test that all expected attributes are present."""
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        metadata = compute_image_metadata(image)
        expected_keys = ["brightness", "contrast", "saturation", "resolution"]
        assert set(metadata.keys()) == set(expected_keys)

    def test_all_values_are_numeric(self):
        """Test that all metadata values are numeric."""
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        metadata = compute_image_metadata(image)
        for key, value in metadata.items():
            assert isinstance(value, (int, float))

    def test_black_image_metadata(self):
        """Test metadata values for completely black image."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        metadata = compute_image_metadata(image)
        assert metadata["brightness"] == 0.0
        assert metadata["contrast"] == 0.0
        assert metadata["saturation"] == 0.0
        assert metadata["resolution"] == 10000

    def test_white_image_metadata(self):
        """Test metadata values for completely white image."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        metadata = compute_image_metadata(image)
        assert metadata["brightness"] == 255.0
        assert metadata["contrast"] == 0.0
        assert metadata["saturation"] == 0.0
        assert metadata["resolution"] == 10000


class TestAttributeFunctions:
    """Test cases for ATTRIBUTE_FUNCTIONS dictionary."""

    def test_contains_all_functions(self):
        """Test that ATTRIBUTE_FUNCTIONS contains all expected functions."""
        expected_keys = ["brightness", "contrast", "saturation", "resolution"]
        assert set(ATTRIBUTE_FUNCTIONS.keys()) == set(expected_keys)

    def test_functions_are_callable(self):
        """Test that all functions in ATTRIBUTE_FUNCTIONS are callable."""
        for key, func in ATTRIBUTE_FUNCTIONS.items():
            assert callable(func)

    def test_brightness_function_mapping(self):
        """Test that brightness function is correctly mapped."""
        assert ATTRIBUTE_FUNCTIONS["brightness"] == compute_brightness

    def test_contrast_function_mapping(self):
        """Test that contrast function is correctly mapped."""
        assert ATTRIBUTE_FUNCTIONS["contrast"] == compute_contrast

    def test_saturation_function_mapping(self):
        """Test that saturation function is correctly mapped."""
        assert ATTRIBUTE_FUNCTIONS["saturation"] == compute_saturation

    def test_resolution_function_mapping(self):
        """Test that resolution function is correctly mapped."""
        assert ATTRIBUTE_FUNCTIONS["resolution"] == compute_resolution

    def test_functions_work_through_dictionary(self):
        """Test that functions can be called through the dictionary."""
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        for key, func in ATTRIBUTE_FUNCTIONS.items():
            result = func(image)
            assert result is not None
            assert isinstance(result, (int, float))

