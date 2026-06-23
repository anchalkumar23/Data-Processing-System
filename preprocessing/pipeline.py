"""
preprocessing/pipeline.py
──────────────────────────
Orchestrates the three-stage preprocessing pipeline:

    Raw image → [1] Noise Reduction → [2] Normalisation → [3] Augmentation

This module is the single entry point for preprocessing — everything
else in the codebase that needs to preprocess an image calls
`ImagePreprocessor` rather than invoking the individual stages directly.

That way, if we change a preprocessing step, we only change it in one
place and all consumers automatically pick up the change.
"""

import logging
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import yaml

from preprocessing.noise_reduction import denoise
from preprocessing.normalizer import normalize, to_chw
from preprocessing.augmentation import build_train_transform, build_val_transform, augment_numpy

log = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Stateful preprocessing pipeline that reads its configuration from
    config.yaml.  Create one instance and call it repeatedly — the
    albumentations transforms are built once and reused.

    Example usage:
        preprocessor = ImagePreprocessor("configs/config.yaml")
        tensor = preprocessor.process(image, mode="train")
    """

    def __init__(self, config_path: str = "configs/config.yaml") -> None:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        self.cfg_nr   = cfg["preprocessing"]["noise_reduction"]
        self.cfg_norm = cfg["preprocessing"]["normalization"]
        self.cfg_aug  = cfg["preprocessing"]["augmentation"]
        img_size      = cfg["data_gen"]["image_size"][0]  # assume square

        # Build transforms once; they're stateless and thread-safe
        self.train_transform = build_train_transform(
            image_size      = img_size,
            horizontal_flip = self.cfg_aug.get("horizontal_flip", True),
            vertical_flip   = self.cfg_aug.get("vertical_flip", False),
            rotation_limit  = self.cfg_aug.get("rotation_limit", 15),
            brightness_limit= self.cfg_aug.get("brightness_limit", 0.2),
            contrast_limit  = self.cfg_aug.get("contrast_limit", 0.2),
            p_flip          = self.cfg_aug.get("p_flip", 0.5),
            p_rotate        = self.cfg_aug.get("p_rotate", 0.4),
            p_color_jitter  = self.cfg_aug.get("p_color_jitter", 0.3),
        )
        self.val_transform = build_val_transform(image_size=img_size)

        log.debug(f"ImagePreprocessor initialised (size={img_size}, "
                  f"denoise={self.cfg_nr['method']}, norm={self.cfg_norm['method']})")

    def process(
        self,
        image: np.ndarray,
        mode: str = "val",
    ) -> np.ndarray:
        """
        Full preprocessing pass on a single image.

        Args:
            image: BGR uint8 numpy array as returned by cv2.imread().
            mode:  'train' applies augmentation; 'val'/'test' does not.

        Returns:
            float32 numpy array of shape (C, H, W), ready for PyTorch.
        """
        if image is None or image.size == 0:
            raise ValueError("process() received an empty image")

        # Stage 1 — Noise reduction
        nr_method = self.cfg_nr["method"]
        nr_kwargs = self._noise_reduction_kwargs()
        denoised = denoise(image, method=nr_method, **nr_kwargs)

        # Stage 2 — Normalisation (only used for the raw numpy path;
        #            albumentations handles this internally for train/val)
        # We skip the standalone normalise call here because the albumentations
        # transforms embed the A.Normalize step, which avoids double normalisation.

        # Stage 3 — Augmentation (train) or simple resize+normalise (val/test)
        transform = self.train_transform if mode == "train" else self.val_transform
        tensor = augment_numpy(denoised, transform)

        return tensor

    def process_batch(
        self,
        images: list,
        mode: str = "val",
    ) -> np.ndarray:
        """
        Processes a list of images and stacks them into a batch array
        of shape (N, C, H, W).  Useful for offline bulk preprocessing.
        """
        processed = [self.process(img, mode=mode) for img in images]
        return np.stack(processed, axis=0)

    def _noise_reduction_kwargs(self) -> dict:
        """Extracts the relevant kwargs for the configured denoising method."""
        method = self.cfg_nr["method"]
        if method == "gaussian":
            return {
                "kernel_size": self.cfg_nr.get("kernel_size", 5),
                "sigma":       self.cfg_nr.get("sigma", 1.0),
            }
        elif method == "median":
            return {"kernel_size": self.cfg_nr.get("kernel_size", 5)}
        elif method == "bilateral":
            return {
                "d":           self.cfg_nr.get("bilateral_d", 9),
                "sigma_color": self.cfg_nr.get("bilateral_sigma_color", 75),
                "sigma_space": self.cfg_nr.get("bilateral_sigma_space", 75),
            }
        return {}


def load_image(path: Union[str, Path]) -> np.ndarray:
    """
    Loads an image from disk, raising a clear error if it fails.
    Wrapping cv2.imread is important because OpenCV returns None
    (rather than raising) when a file can't be read.
    """
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {path}")
    return img


def preprocess_directory(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    config_path: str = "configs/config.yaml",
    mode: str = "val",
) -> None:
    """
    Batch-processes all images in `input_dir` and saves the results
    as .npy files (float32 tensors) in `output_dir`.

    This is useful for an offline preprocessing step that prepares
    the dataset before training, so the training loop doesn't have
    to preprocess on the fly.

    Args:
        input_dir:   Directory containing raw .jpg / .png images.
        output_dir:  Where to save the preprocessed .npy tensors.
        config_path: Path to config.yaml.
        mode:        'train' or 'val'.
    """
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = ImagePreprocessor(config_path)
    image_paths  = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))

    if not image_paths:
        log.warning(f"No images found in {input_dir}")
        return

    log.info(f"Preprocessing {len(image_paths)} images from {input_dir} → {output_dir}")
    t_start = time.perf_counter()

    for img_path in image_paths:
        try:
            img    = load_image(img_path)
            tensor = preprocessor.process(img, mode=mode)
            out_path = output_dir / (img_path.stem + ".npy")
            np.save(str(out_path), tensor)
        except Exception as e:
            log.warning(f"Skipping {img_path.name}: {e}")

    elapsed = time.perf_counter() - t_start
    fps = len(image_paths) / elapsed
    log.info(f"Done — {len(image_paths)} images in {elapsed:.1f}s  ({fps:.1f} img/s)")
