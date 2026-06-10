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

