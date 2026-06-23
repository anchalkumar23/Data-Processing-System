"""
preprocessing/normalizer.py
────────────────────────────
Stage 2 of the preprocessing pipeline: pixel normalization.

Raw images from cameras have pixel values in [0, 255].  Neural networks
train more stably when the input distribution is zero-centred and has
unit variance.  We support two strategies:

  • minmax     — scales each image independently to [0.0, 1.0]
  • standardize — subtracts the ImageNet channel means and divides by
                  their standard deviations; use this when fine-tuning
                  a pretrained backbone (ResNet, EfficientNet, etc.)

All functions operate on NumPy arrays and return float32 tensors ready
for PyTorch / ONNX consumption.
"""

import logging
from typing import Optional, Tuple, List

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ImageNet channel statistics (RGB order).
# These are the means/stds used when pretraining ResNet on ImageNet.
# We apply them here even for our synthetic data because the backbone
# weights were trained with this normalisation.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resize(
    image: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Resizes image to `target_size` (width, height).

    Uses INTER_AREA for downscaling (produces sharper results)
    and INTER_LINEAR for upscaling.
    """
    h, w = image.shape[:2]
    tw, th = target_size

    if w == tw and h == th:
        return image  # already the right size, skip the copy

    interp = cv2.INTER_AREA if (w > tw or h > th) else cv2.INTER_LINEAR
    return cv2.resize(image, (tw, th), interpolation=interp)


def minmax_normalize(image: np.ndarray) -> np.ndarray:
    """
    Scales pixel values to [0.0, 1.0] per image.

    Simple but effective — preserves relative intensities within
    an image but does not account for cross-image distribution shift.
    """
    img = image.astype(np.float32)
    min_val = img.min()
    max_val = img.max()

    # Avoid division by zero on fully uniform images
    if max_val - min_val < 1e-6:
        return np.zeros_like(img)

    return (img - min_val) / (max_val - min_val)


def standardize(
    image: np.ndarray,
    mean: Optional[List[float]] = None,
    std:  Optional[List[float]] = None,
) -> np.ndarray:
    """
    Normalises an image to zero mean / unit variance using channel-wise
    statistics, following the ImageNet preprocessing convention.

    The input is expected to be a BGR uint8 image (as returned by OpenCV).
    We convert to RGB float32 before subtracting the mean.

    Args:
        image: BGR uint8 image array of shape (H, W, 3).
        mean:  Per-channel means (RGB order).  Defaults to ImageNet means.
        std:   Per-channel stds  (RGB order).  Defaults to ImageNet stds.

    Returns:
        float32 array of shape (H, W, 3) with values roughly in [-3, 3].
    """
    _mean = np.array(mean, dtype=np.float32) if mean is not None else IMAGENET_MEAN
    _std  = np.array(std,  dtype=np.float32) if std  is not None else IMAGENET_STD

    # OpenCV loads as BGR — convert to RGB before applying ImageNet stats
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_f   = img_rgb.astype(np.float32) / 255.0       # [0, 1]
    img_norm = (img_f - _mean) / (_std + 1e-7)         # prevent div-by-zero

    return img_norm


def normalize(
    image: np.ndarray,
    method: str = "standardize",
    target_size: Tuple[int, int] = (224, 224),
    mean: Optional[List[float]] = None,
    std:  Optional[List[float]] = None,
) -> np.ndarray:
    """
    Unified normalisation entry point used by the pipeline.

    Resizes the image first, then applies the chosen normalisation.

    Args:
        image:       BGR uint8 input image.
        method:      'minmax' or 'standardize'.
        target_size: (width, height) to resize to.
        mean:        Channel means (only used with 'standardize').
        std:         Channel stds  (only used with 'standardize').

    Returns:
        float32 numpy array ready for model input.
    """
    if image is None or image.size == 0:
        raise ValueError("Empty image passed to normalizer")

    img = resize(image, target_size)

    if method == "minmax":
        return minmax_normalize(img)
    elif method == "standardize":
        return standardize(img, mean=mean, std=std)
    else:
        raise ValueError(f"Unknown normalisation method '{method}'. Use: minmax, standardize")


def to_chw(image: np.ndarray) -> np.ndarray:
    """
    Converts a (H, W, C) array to (C, H, W) format expected by PyTorch.
    Also works for grayscale by adding a channel dimension.
    """
    if image.ndim == 2:
        return image[np.newaxis, ...]          # (1, H, W)
    return np.transpose(image, (2, 0, 1))     # (H, W, C) → (C, H, W)
