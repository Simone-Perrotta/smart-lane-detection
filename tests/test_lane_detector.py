import numpy as np
import pytest

from src.smart_lane_detection.lane_detector import (
    convert_to_grayscale,
    apply_gaussian_blur
    )

def test_convert_to_grayscale_returns_2d_image():
    # Create a dummy BGR image 
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    grayscale = convert_to_grayscale(image)

    assert len(grayscale.shape) == 2, "Output image should be 2D."

def test_convert_to_grayscale_invalid_input():
    with pytest.raises(ValueError):
        convert_to_grayscale(None) # Input image is None

    with pytest.raises(ValueError):
        convert_to_grayscale(np.zeros((100, 100), dtype=np.uint8))  # Not a 3-channel image

    with pytest.raises(ValueError):
        convert_to_grayscale(np.zeros((100, 100, 4), dtype=np.uint8))  # Not a 3-channel image

def test_apply_gaussian_blur_invalid_input():
    with pytest.raises(ValueError):
        apply_gaussian_blur(None)  # Input image is None

    with pytest.raises(ValueError):
        apply_gaussian_blur(np.zeros((100, 100), dtype=np.uint8), kernel_size=-1)  # Invalid kernel size

    with pytest.raises(ValueError):
        apply_gaussian_blur(np.zeros((100, 100), dtype=np.uint8), kernel_size=4)  # Invalid kernel size (not odd)

def test_apply_gaussian_blur_output_shape():
    image = np.zeros((100, 100), dtype=np.uint8)
    blurred = apply_gaussian_blur(image, kernel_size=5)

    assert blurred.shape == image.shape, "Output image should have the same shape as input image."