import cv2
import numpy as np
import pytest

from src.smart_lane_detection.lane_detector import (
    convert_to_grayscale,
    apply_gaussian_blur,
    detect_edges,
    region_of_interest,
    detect_lines,
    draw_lines,
    detect_lanes,
)


def test_convert_to_grayscale_returns_2d_image():
    # Create a dummy BGR image.
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    grayscale = convert_to_grayscale(image)

    assert len(grayscale.shape) == 2, "Output image should be 2D."


def test_convert_to_grayscale_invalid_input():
    with pytest.raises(ValueError):
        convert_to_grayscale(None)

    with pytest.raises(ValueError):
        convert_to_grayscale(np.zeros((100, 100), dtype=np.uint8))

    with pytest.raises(ValueError):
        convert_to_grayscale(np.zeros((100, 100, 4), dtype=np.uint8))


def test_apply_gaussian_blur_invalid_input():
    image = np.zeros((100, 100), dtype=np.uint8)

    with pytest.raises(ValueError):
        apply_gaussian_blur(None)

    with pytest.raises(ValueError):
        apply_gaussian_blur(image, kernel_size=-1)

    with pytest.raises(ValueError):
        apply_gaussian_blur(image, kernel_size=4)


def test_apply_gaussian_blur_output_shape():
    image = np.zeros((100, 100), dtype=np.uint8)

    blurred = apply_gaussian_blur(image, kernel_size=5)

    assert blurred.shape == image.shape, "Output image should have the same shape."


def test_detect_edges_invalid_input():
    image = np.zeros((100, 100), dtype=np.uint8)

    with pytest.raises(ValueError):
        detect_edges(None)

    with pytest.raises(ValueError):
        detect_edges(np.zeros((100, 100, 3), dtype=np.uint8))

    with pytest.raises(ValueError):
        detect_edges(image, low_threshold=-1)

    with pytest.raises(ValueError):
        detect_edges(image, high_threshold=-1)

    with pytest.raises(ValueError):
        detect_edges(image, low_threshold=150, high_threshold=50)


def test_detect_edges_output_shape():
    image = np.zeros((100, 100), dtype=np.uint8)

    edges = detect_edges(image, low_threshold=50, high_threshold=150)

    assert edges.shape == image.shape, "Output image should have the same shape."


def test_region_of_interest_invalid_input():
    image = np.zeros((100, 100), dtype=np.uint8)

    valid_vertices = np.array(
        [
            [
                (0, 0),
                (1, 0),
                (1, 1),
                (0, 1),
            ]
        ],
        dtype=np.int32,
    )

    with pytest.raises(ValueError):
        region_of_interest(None, valid_vertices)

    with pytest.raises(ValueError):
        region_of_interest(image, None)

    with pytest.raises(ValueError):
        region_of_interest(image, np.array([]))


def test_region_of_interest_output_shape():
    image = np.zeros((100, 100), dtype=np.uint8)

    vertices = np.array(
        [
            [
                (0, 0),
                (99, 0),
                (99, 99),
                (0, 99),
            ]
        ],
        dtype=np.int32,
    )

    masked_image = region_of_interest(image, vertices)

    assert masked_image.shape == image.shape, "Output image should have the same shape."


def test_region_of_interest_removes_pixels_outside_region():
    image = np.ones((100, 100), dtype=np.uint8) * 255

    vertices = np.array(
        [
            [
                (0, 100),
                (50, 50),
                (100, 100),
            ]
        ],
        dtype=np.int32,
    )

    masked_image = region_of_interest(image, vertices)

    assert masked_image[10, 50] == 0


def test_region_of_interest_keeps_pixels_inside_region():
    image = np.ones((100, 100), dtype=np.uint8) * 255

    vertices = np.array(
        [
            [
                (0, 100),
                (50, 50),
                (100, 100),
            ]
        ],
        dtype=np.int32,
    )

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


def test_draw_lines_draws_on_image():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    lines = np.array([[[10, 10, 90, 90]]], dtype=np.int32)

    output = draw_lines(image, lines, color=(255, 0, 0), thickness=2)

    assert np.any(output[10:12, 10:12] == [255, 0, 0])
    assert np.any(output[88:90, 88:90] == [255, 0, 0])


def test_draw_lines_keeps_original_image_unchanged():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    original = image.copy()
    lines = np.array([[[10, 10, 90, 90]]], dtype=np.int32)

    draw_lines(image, lines)

    assert np.array_equal(image, original)


def test_draw_lines_handles_empty_lines():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    lines = np.array([])

    output = draw_lines(image, lines)

    assert np.array_equal(output, image)


def test_detect_lanes_raises_error_when_image_is_none():
    with pytest.raises(ValueError):
        detect_lanes(None)


def test_detect_lanes_returns_image_with_same_shape():
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    output = detect_lanes(image)

    assert isinstance(output, np.ndarray)
    assert output.shape == image.shape


def test_detect_lanes_does_not_modify_original_image():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    original = image.copy()

    detect_lanes(image)

    assert np.array_equal(image, original)