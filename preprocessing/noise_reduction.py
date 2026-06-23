"""
preprocessing/noise_reduction.py
──────────────────────────────────
Stage 1 of the preprocessing pipeline: noise reduction.

Industrial cameras often produce images with Gaussian sensor noise,
salt-and-pepper noise from bit errors, or motion blur.  We offer
three filtering strategies depending on the type of noise expected:

  • gaussian  — smooth Gaussian blur; good all-around denoiser
  • median    — excellent against salt-and-pepper noise (non-linear)
  • bilateral — edge-preserving denoiser; best for crack detection
                because it smooths flat regions while keeping edges sharp

The choice is configurable in configs/config.yaml under
`preprocessing.noise_reduction.method`.
"""

import logging
from typing import Union

import cv2
import numpy as np

log = logging.getLogger(__name__)


def gaussian_denoise(
    image: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Applies a Gaussian blur to reduce high-frequency noise.

    Args:
        image:       BGR or grayscale image (uint8 or float32).
        kernel_size: Must be an odd positive integer.
        sigma:       Standard deviation for Gaussian kernel.
                     Higher values = more blurring.

    Returns:
        Denoised image of the same shape and dtype.
    """
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}")

    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma)


def median_denoise(
    image: np.ndarray,
    kernel_size: int = 5,
) -> np.ndarray:
    """
    Applies a median filter — very effective against salt-and-pepper noise.

    Unlike Gaussian blur, the median filter replaces each pixel with the
    median of its neighbourhood, so it does not average-in the outlier
    (noise) values.

    Args:
        image:       BGR or grayscale uint8 image.
        kernel_size: Must be an odd positive integer.

    Returns:
        Denoised image.
    """
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}")

    return cv2.medianBlur(image, kernel_size)


def bilateral_denoise(
    image: np.ndarray,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    """
    Applies a bilateral filter — smooths homogeneous regions while
    preserving hard edges (like cracks and scratches).

    The bilateral filter is slower than Gaussian but significantly better
    for defect detection because the defect edges themselves are signal,
    not noise.

    Args:
        image:       BGR uint8 image.
        d:           Diameter of each pixel neighbourhood.
        sigma_color: Range sigma; large values mix colours across larger
                     intensity differences.
        sigma_space: Spatial sigma; large values mix pixels farther apart.

    Returns:
        Edge-preserved denoised image.
    """
    # cv2.bilateralFilter requires uint8 or float32
    if image.dtype not in (np.uint8, np.float32):
        image = image.astype(np.uint8)

    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def denoise(
    image: np.ndarray,
    method: str = "gaussian",
    **kwargs,
) -> np.ndarray:
    """
    Unified entry point for noise reduction.  Selects the correct
    filter based on the `method` string from config.

    Args:
        image:  Input image array.
        method: One of 'gaussian', 'median', 'bilateral'.
        **kwargs: Method-specific keyword arguments forwarded directly.

    Returns:
        Denoised image.

    Raises:
        ValueError: If method is not recognised.
    """
    # Validate input before we do anything
    if image is None or image.size == 0:
        raise ValueError("Received empty or None image for denoising")

    method = method.lower()

    if method == "gaussian":
        return gaussian_denoise(image, **kwargs)
    elif method == "median":
        return median_denoise(image, **kwargs)
    elif method == "bilateral":
        return bilateral_denoise(image, **kwargs)
    else:
        raise ValueError(
            f"Unknown denoising method '{method}'. "
            f"Choose from: gaussian, median, bilateral"
        )
