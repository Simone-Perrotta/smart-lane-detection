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

