"""
preprocessing/augmentation.py
───────────────────────────────
Stage 3 of the preprocessing pipeline: data augmentation.

Augmentation artificially expands the training set by applying
randomised transformations to each image during training.  This
regularises the model and makes it robust to real-world variation
in lighting, orientation, and sensor noise.

We use `albumentations` rather than torchvision transforms because
albumentations is significantly faster (OpenCV backend), supports a
wider variety of transforms, and makes it easy to keep spatial labels
(bounding boxes / masks) consistent — useful if we extend this project
to segmentation later.

Augmentation is only applied during *training*, never during val/test.
The pipeline.py module handles this split automatically.
"""

import logging
from typing import Optional

import cv2
import numpy as np

# albumentations is the de-facto standard for fast image augmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2

log = logging.getLogger(__name__)


def build_train_transform(
    image_size: int = 224,
    horizontal_flip: bool = True,
    vertical_flip: bool = False,
    rotation_limit: int = 15,
    brightness_limit: float = 0.2,
    contrast_limit: float = 0.2,
    p_flip: float = 0.5,
    p_rotate: float = 0.4,
    p_color_jitter: float = 0.3,
) -> A.Compose:
    """
    Constructs the augmentation pipeline used during training.

    Each transform is gated by its own probability so we can easily
    tune the augmentation strength from config.yaml without changing
    this file.

    Args:
        image_size:       Output image size (square).
        horizontal_flip:  Whether to include horizontal flips.
        vertical_flip:    Whether to include vertical flips.
        rotation_limit:   Max rotation angle (degrees) in each direction.
        brightness_limit: Max fractional brightness change.
        contrast_limit:   Max fractional contrast change.
        p_flip:           Probability to apply flips.
        p_rotate:         Probability to apply rotation.
        p_color_jitter:   Probability to apply brightness/contrast shift.

    Returns:
        An albumentations Compose object ready to be called on images.
    """
    transforms = []

    # Always resize first so every transform works on a consistent shape
    transforms.append(A.Resize(image_size, image_size))

    if horizontal_flip:
        transforms.append(A.HorizontalFlip(p=p_flip))

    if vertical_flip:
        transforms.append(A.VerticalFlip(p=p_flip))

    # Slight rotation — surfaces can be captured at small angles
    transforms.append(
        A.Rotate(limit=rotation_limit, border_mode=cv2.BORDER_REFLECT_101, p=p_rotate)
    )

    # Colour jitter simulates different lighting conditions
    transforms.append(
        A.RandomBrightnessContrast(
            brightness_limit=brightness_limit,
            contrast_limit=contrast_limit,
            p=p_color_jitter,
        )
    )

    # Gaussian noise simulates camera sensor variation
    transforms.append(A.GaussNoise(var_limit=(5.0, 25.0), p=0.2))

    # Slight blur simulates focus variation
    transforms.append(A.OneOf([
        A.MotionBlur(blur_limit=3, p=1.0),
        A.MedianBlur(blur_limit=3, p=1.0),
    ], p=0.15))

    # Normalise to ImageNet statistics (must match the normalizer.py values)
    transforms.append(
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        )
    )

    # Convert HWC numpy → CHW PyTorch tensor
    transforms.append(ToTensorV2())

    return A.Compose(transforms)


def build_val_transform(image_size: int = 224) -> A.Compose:
    """
    Validation / test transform — NO random augmentation, just resize
    and normalise.  This gives us a deterministic, reproducible evaluation.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def augment_numpy(
    image: np.ndarray,
    transform: A.Compose,
) -> np.ndarray:
    """
    Applies an albumentations transform to a raw BGR numpy image and
    returns the result as a (C, H, W) float32 numpy array.

    We keep this separate from the PyTorch dataset so that the
    preprocessing pipeline can also augment images independently
    of the training loop (e.g. during offline dataset preparation).

    Args:
        image:     BGR uint8 numpy array (H, W, 3).
        transform: albumentations Compose transform.

    Returns:
        float32 numpy array of shape (C, H, W).
    """
    # albumentations expects RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = transform(image=rgb)
    # ToTensorV2 returns a torch.Tensor — convert back to numpy if needed
    tensor = result["image"]
    return tensor.numpy()
