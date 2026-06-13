import cv2
import numpy as np
import pytest

from src.smart_lane_detection.lane_detector import (
    convert_to_grayscale,
    apply_gaussian_blur,
    detect_edges,
    region_of_interest,
    detect_lines
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

def test_detect_edges_invalid_input():
    with pytest.raises(ValueError):
        detect_edges(None)  # Input image is None

    with pytest.raises(ValueError):
        detect_edges(np.zeros((100, 100, 3), dtype=np.uint8))  # Not a single-channel image

    with pytest.raises(ValueError):
        detect_edges(np.zeros((100, 100), dtype=np.uint8), low_threshold=-1)  # Invalid low threshold

    with pytest.raises(ValueError):
        detect_edges(np.zeros((100, 100), dtype=np.uint8), high_threshold=-1)  # Invalid high threshold

    with pytest.raises(ValueError):
        detect_edges(np.zeros((100, 100), dtype=np.uint8), low_threshold=150, high_threshold=50)  # Low threshold >= high threshold

def test_detect_edges_output_shape():
    image = np.zeros((100, 100), dtype=np.uint8)
    edges = detect_edges(image, low_threshold=50, high_threshold=150)

    assert edges.shape == image.shape, "Output image should have the same shape as input image."

def test_region_of_interest_invalid_input():
    valid_vertices = np.array([
        [
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1),
        ]
    ], dtype=np.int32)

    with pytest.raises(ValueError):
        region_of_interest(None, valid_vertices)

    with pytest.raises(ValueError):
        region_of_interest(np.zeros((100, 100), dtype=np.uint8), None)

    with pytest.raises(ValueError):
        region_of_interest(np.zeros((100, 100), dtype=np.uint8), np.array([]))


def test_region_of_interest_output_shape():
    image = np.zeros((100, 100), dtype=np.uint8)

    vertices = np.array([
        [
            (0, 0),
            (99, 0),
            (99, 99),
            (0, 99),
        ]
    ], dtype=np.int32)

    masked_image = region_of_interest(image, vertices)

    assert masked_image.shape == image.shape, "Output image should have the same shape as input image."


def test_region_of_interest_removes_pixels_outside_region():
    image = np.ones((100, 100), dtype=np.uint8) * 255

    vertices = np.array([
        [
            (0, 100),
            (50, 50),
            (100, 100),
        ]
    ], dtype=np.int32)

    masked_image = region_of_interest(image, vertices)

    assert masked_image[10, 50] == 0


def test_region_of_interest_keeps_pixels_inside_region():
    image = np.ones((100, 100), dtype=np.uint8) * 255

    vertices = np.array([
        [
            (0, 100),
            (50, 50),
            (100, 100),
        ]
    ], dtype=np.int32)

    masked_image = region_of_interest(image, vertices)

    assert masked_image[90, 50] == 255

def test_detect_lines_invalid_input():
    image = np.zeros((100, 100), dtype=np.uint8)

    with pytest.raises(ValueError):
        detect_lines(None)

    with pytest.raises(ValueError):
        detect_lines(np.zeros((100, 100, 3), dtype=np.uint8))

    with pytest.raises(ValueError):
        detect_lines(image, rho=0)

    with pytest.raises(ValueError):
        detect_lines(image, theta=0)

    with pytest.raises(ValueError):
        detect_lines(image, threshold=0)

    with pytest.raises(ValueError):
        detect_lines(image, min_line_length=0)

    with pytest.raises(ValueError):
        detect_lines(image, max_line_gap=-1)


def test_detect_lines_returns_empty_array_when_no_lines_are_found():
    image = np.zeros((100, 100), dtype=np.uint8)

    lines = detect_lines(image)

    assert isinstance(lines, np.ndarray)
    assert len(lines) == 0


def test_detect_lines_finds_line_segments():
    image = np.zeros((100, 100), dtype=np.uint8)

    cv2.line(image, (10, 10), (90, 90), 255, 2)

    lines = detect_lines(image)

    assert isinstance(lines, np.ndarray)
    assert len(lines) > 0
    assert lines.shape[2] == 4