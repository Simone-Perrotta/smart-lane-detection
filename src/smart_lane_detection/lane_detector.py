import cv2
import numpy as np


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert a BGR image with 3 channels to grayscale with a single channel.
    """

    if image is None:
        raise ValueError("Input image is None.")
    
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be a BGR image with 3 channels.")
    
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Apply Gaussian blur to the input image to reduce noise.
    """

    if image is None:
        raise ValueError("Input image is None.")
    
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("Kernel size must be a positive odd integer.")
    
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def detect_edges(image: np.ndarray, low_threshold: int = 50, high_threshold: int = 150) -> np.ndarray:
    """
    Apply Canny edge detection to the input image.
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
    Apply a mask to the input image, keeping only the region defined by the vertices.
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

    masked_image = cv2.bitwise_and(image, mask)

    return masked_image


def detect_lines(
    image: np.ndarray,
    rho: float = 1,
    theta: float = np.pi / 180,
    threshold: int = 25,
    min_line_length: int = 60,
    max_line_gap: int = 30,
) -> np.ndarray:
    """
    Detect lines in the input image using the Hough Transform.
    """

    if image is None:
        raise ValueError("Input image is None.")
    
    if len(image.shape) != 2:
        raise ValueError("Input image must be a single-channel edge-detected image.")
    
    if rho <= 0 or theta <= 0 or threshold <= 0:
        raise ValueError("Rho, theta, and threshold must be positive values.")
    
    if min_line_length <= 0 or max_line_gap < 0:
        raise ValueError("Min line length must be positive and max line gap must be non-negative.")
    
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
        return np.array([])  # Return an empty array if no lines are detected.
    
    return lines


def filter_lane_lines(lines: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    """
    Filter Hough line segments and keep only lines that are likely to be lane markings.
    """

    if lines is None or len(lines) == 0:
        return np.array([])

    if image_width <= 0:
        raise ValueError("Image width must be a positive integer.")
    
    if image_height <= 0:
        raise ValueError("Image height must be a positive integer.")

    left_candidates = []
    right_candidates = []

    image_center = image_width // 2
    left_lane_end = int(image_width * 0.42)
    right_lane_start = int(image_width * 0.58)

    # Lane markings should usually reach the lower part of the image.
    # This helps remove road arrows and markings that are more central or higher up.
    min_y = int(image_height * 0.70)

    for line in lines:
        for x1, y1, x2, y2 in line:
            # Avoid division by zero for vertical lines.
            if x2 == x1:
                continue

            # Keep only lines that reach the lower part of the image.
            if max(y1, y2) < min_y:
                continue

            slope = (y2 - y1) / (x2 - x1)
            abs_slope = abs(slope)

            # Lane markings are usually diagonal.
            # Very horizontal lines are often arrows or road signs.
            if abs_slope < 0.5:
                continue

            # Very vertical lines are usually not lane markings in this camera view.
            if abs_slope > 2.5:
                continue

            # Left lane: negative slope and clearly on the left side of the image.
            if slope < 0 and x1 < left_lane_end and x2 < image_center:
                left_candidates.append([[x1, y1, x2, y2]])

            # Right lane: positive slope and clearly on the right side of the image.
            elif slope > 0 and x1 > image_center and x2 > right_lane_start:
                right_candidates.append([[x1, y1, x2, y2]])

    filtered_lines = []

    # Keep the left line that is more external, meaning more to the left.
    if left_candidates:
        best_left_line = min(
            left_candidates,
            key=lambda line: min(line[0][0], line[0][2]),
        )
        filtered_lines.append(best_left_line)

    # Keep the right line that is more external, meaning more to the right.
    if right_candidates:
        best_right_line = max(
            right_candidates,
            key=lambda line: max(line[0][0], line[0][2]),
        )
        filtered_lines.append(best_right_line)

    return np.array(filtered_lines, dtype=np.int32)


def draw_lines(
    image: np.ndarray,
    lines: np.ndarray,
    color: tuple = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw lines on the input image.
    """

    if image is None:
        raise ValueError("Input image is None.")
    
    if lines is None:
        lines = np.array([])  # Treat missing lines as an empty array.
    
    if len(color) != 3:
        raise ValueError("Color must be a tuple of 3 elements (B, G, R).")
    
    if thickness <= 0:
        raise ValueError("Thickness must be a positive integer.")
    
    line_image = np.copy(image)
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)
    
    return line_image


def detect_lanes(image: np.ndarray) -> np.ndarray:
    """
    Run the full lane detection pipeline.
    """

    # Validate the input image before starting the pipeline.
    if image is None:
        raise ValueError("Input image is None.")
    
    # Step 1: Convert the image from BGR to grayscale.
    grayscale = convert_to_grayscale(image)

    # Step 2: Apply Gaussian blur to reduce image noise.
    blurred = apply_gaussian_blur(grayscale)

    # Step 3: Detect edges using the Canny algorithm.
    edges = detect_edges(blurred)

    # Step 4: Define the region of interest.
    height, width = edges.shape
    vertices = np.array(
        [[
            (int(width * 0.05), height),              # Bottom-left area
            (int(width * 0.42), int(height * 0.70)),  # Upper-left area
            (int(width * 0.58), int(height * 0.70)),  # Upper-right area
            (int(width * 0.98), height),              # Bottom-right area
        ]],
        dtype=np.int32,
    )

    # Step 5: Apply the region mask to keep only the road area.
    masked_edges = region_of_interest(edges, vertices)

    # Step 6: Detect line segments inside the masked edge image
    # using the Probabilistic Hough Transform.
    lines = detect_lines(masked_edges)

    # Step 7: Keep only line segments that are likely to be lane markings.
    # This helps reduce false positives such as arrows, road signs, and horizontal markings.
    filtered_lines = filter_lane_lines(lines, width, height)

    # Step 8: Draw the filtered line segments on a copy of the original image.
    line_image = draw_lines(image, filtered_lines)

    return line_image