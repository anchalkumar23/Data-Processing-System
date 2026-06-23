"""
training/dataset.py
────────────────────
PyTorch Dataset for the defect detection task.

Reads images from the standard ImageFolder directory layout:

    data/processed/train/
    ├── normal/
    │   ├── normal_0000.jpg
    │   └── ...
    └── defective/
        ├── defective_0000.jpg
        └── ...

We implement a custom Dataset rather than using torchvision.ImageFolder
so we have full control over:
  - which preprocessing pipeline is applied (our own 3-stage pipeline)
  - error handling for corrupted files
  - per-sample metadata we might want later (filename, defect subtype)
"""

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import torch
from torch.utils.data import Dataset, DataLoader

from preprocessing.pipeline import ImagePreprocessor, load_image

log = logging.getLogger(__name__)


class DefectDataset(Dataset):
    """
    Loads surface images and returns (tensor, label) pairs.

    Args:
        root_dir:     Path to the split directory (e.g. data/processed/train).
        preprocessor: An `ImagePreprocessor` instance.
        mode:         'train' enables augmentation; 'val'/'test' disables it.
    """

    CLASS_TO_IDX: Dict[str, int] = {"normal": 0, "defective": 1}

    def __init__(
        self,
        root_dir: str,
        preprocessor: ImagePreprocessor,
        mode: str = "train",
    ) -> None:
        self.root_dir     = Path(root_dir)
        self.preprocessor = preprocessor
        self.mode         = mode

        # Collect all (image_path, label_idx) pairs
        self.samples: List[Tuple[Path, int]] = []
        self._scan_directory()

        if not self.samples:
            raise RuntimeError(
                f"No images found under {self.root_dir}. "
                f"Run data_gen/generate_dataset.py first."
            )

        log.info(
            f"DefectDataset ({mode}): {len(self.samples)} images from {self.root_dir}"
        )

    def _scan_directory(self) -> None:
        """Walks the class subdirectories and records image paths + labels."""
        for cls_name, cls_idx in self.CLASS_TO_IDX.items():
            cls_dir = self.root_dir / cls_name
            if not cls_dir.exists():
                log.warning(f"Class directory not found: {cls_dir}")
                continue

            imgs = sorted(cls_dir.glob("*.jpg")) + sorted(cls_dir.glob("*.png"))
            for img_path in imgs:
                self.samples.append((img_path, cls_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        try:
            image = load_image(img_path)
        except FileNotFoundError:
            # If an image is missing, return its neighbour so training
            # doesn't crash — log it so we can investigate offline
            log.warning(f"Missing image at index {idx}: {img_path}. Using idx 0.")
            image = load_image(self.samples[0][0])
            label = self.samples[0][1]

        # process() returns a (C, H, W) float32 numpy array
        tensor_np = self.preprocessor.process(image, mode=self.mode)

        # Convert to torch.Tensor — no copy needed because numpy array is C-contiguous
        tensor = torch.from_numpy(tensor_np).float()

        return tensor, label

    def class_counts(self) -> Dict[str, int]:
        """Returns per-class image counts — useful to check for class imbalance."""
        counts: Dict[str, int] = {cls: 0 for cls in self.CLASS_TO_IDX}
        for _, label in self.samples:
            cls_name = {v: k for k, v in self.CLASS_TO_IDX.items()}[label]
            counts[cls_name] += 1
        return counts


def build_dataloaders(
    config: dict,
    preprocessor: ImagePreprocessor,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Builds train, val, and test DataLoaders from the config.

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    paths   = config["paths"]
    tr_cfg  = config["training"]

    train_ds = DefectDataset(paths["train_dir"], preprocessor, mode="train")
    val_ds   = DefectDataset(paths["val_dir"],   preprocessor, mode="val")
    test_ds  = DefectDataset(paths["test_dir"],  preprocessor, mode="test")

    # Log class distribution so we catch imbalance early
    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        log.info(f"{name} class counts: {ds.class_counts()}")

    common = dict(
        num_workers = tr_cfg.get("num_workers", 4),
        pin_memory  = tr_cfg.get("pin_memory", True),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size = tr_cfg["batch_size"],
        shuffle    = True,   # shuffle every epoch
        drop_last  = True,   # avoid partial batches messing up BatchNorm
        **common,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size = tr_cfg["batch_size"],
        shuffle    = False,
        **common,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size = tr_cfg["batch_size"],
        shuffle    = False,
        **common,
    )

    return train_loader, val_loader, test_loader
