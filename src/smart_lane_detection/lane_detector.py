import cv2
import numpy as np


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert a BGR image with 3 channels to grayscale.
    """
    if image is None:
        raise ValueError("Input image is None.")

    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be a BGR image with 3 channels.")

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Apply Gaussian blur to reduce image noise.
    """
    if image is None:
        raise ValueError("Input image is None.")

    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("Kernel size must be a positive odd integer.")

    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def detect_edges(
    image: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150,
) -> np.ndarray:
    """
    Apply Canny edge detection to a grayscale image.
    """
    if image is None:
        raise ValueError("Input image is None.")

    if len(image.shape) != 2:
        raise ValueError("Input image must be a single-channel grayscale image.")

    if low_threshold < 0 or high_threshold < 0:
        raise ValueError("Threshold values must be non-negative.")

    if low_threshold >= high_threshold:
        raise ValueError("Low threshold must be less than high threshold.")

    return cv2.Canny(image, low_threshold, high_threshold)


def region_of_interest(image: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """
    Keep only the image area defined by the polygon vertices.
    """
    if image is None:
        raise ValueError("Input image is None.")

    if vertices is None or len(vertices) == 0:
        raise ValueError("Vertices must be a non-empty array.")

    mask = np.zeros_like(image)

    if len(image.shape) > 2:
        channel_count = image.shape[2]
        mask_color = (255,) * channel_count
    else:
        mask_color = 255

    cv2.fillPoly(mask, vertices, mask_color)

    return cv2.bitwise_and(image, mask)


def detect_lines(
    image: np.ndarray,
    rho: float = 1,
    theta: float = np.pi / 180,
    threshold: int = 20,
    min_line_length: int = 30,
    max_line_gap: int = 20,
) -> np.ndarray:
    """
    Detect line segments using the Probabilistic Hough Transform.
    """
    if image is None:
        raise ValueError("Input image is None.")

    if len(image.shape) != 2:
        raise ValueError("Input image must be a single-channel edge-detected image.")

    if rho <= 0 or theta <= 0 or threshold <= 0:
        raise ValueError("Rho, theta, and threshold must be positive values.")

    if min_line_length <= 0 or max_line_gap < 0:
        raise ValueError(
            "Min line length must be positive and max line gap must be non-negative."
        )

    lines = cv2.HoughLinesP(
        image,
        rho,
        theta,
        threshold,
        np.array([]),
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    if lines is None:
        return np.array([])

    return lines


def draw_lines(
    image: np.ndarray,
    lines: np.ndarray,
    color: tuple = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw detected line segments on a copy of the original image.
    """
    if image is None:
        raise ValueError("Input image is None.")

    if lines is None:
        lines = np.array([])

    if len(color) != 3:
        raise ValueError("Color must be a tuple of 3 elements: B, G, R.")

    if thickness <= 0:
        raise ValueError("Thickness must be a positive integer.")

    line_image = np.copy(image)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)

    return line_image


def detect_lanes(image: np.ndarray) -> np.ndarray:
    """
    Run the basic lane detection pipeline on a single image.
    """
    if image is None:
        raise ValueError("Input image is None.")

    # Step 1: Convert the original BGR image to grayscale.
    grayscale = convert_to_grayscale(image)

    # Step 2: Apply a stronger blur to reduce small road markings and noise.
    blurred = apply_gaussian_blur(grayscale, kernel_size=7)

    # Step 3: Detect edges using slightly stricter Canny thresholds.
    edges = detect_edges(
        blurred,
        low_threshold=80,
        high_threshold=180,
    )

    # Step 4: Define a tighter region of interest.
    # We focus more on the lower road area where lane lines usually appear.
    height, width = edges.shape

    vertices = np.array(
        [
            [
                (int(width * 0.12), height),
                (int(width * 0.42), int(height * 0.76)),
                (int(width * 0.58), int(height * 0.76)),
                (int(width * 0.88), height),
            ]
        ],
        dtype=np.int32,
    )

    # Step 5: Keep only the edges inside the road region.
    masked_edges = region_of_interest(edges, vertices)

    # Step 6: Detect only longer and more relevant line segments.
    lines = detect_lines(
        masked_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=25,
        min_line_length=45,
        max_line_gap=30,
    )

    # Step 7: Draw the detected line segments on the original image.
    line_image = draw_lines(
        image,
        lines,
        color=(0, 255, 0),
        thickness=4,
    )

    return line_image