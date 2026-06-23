"""
data_gen/generate_dataset.py
─────────────────────────────
Generates a synthetic dataset of surface images for defect detection.

Instead of relying on a real labelled dataset (which would require
proprietary industrial data), we procedurally create:
  - "normal" images  → uniform textured surfaces with random noise
  - "defective" images → same surfaces with synthesised scratches,
                         cracks, or stains painted on top

The result is a directory tree that looks exactly like a real dataset:

    data/raw/
    ├── train/
    │   ├── normal/     (1600 images)
    │   └── defective/  (1600 images)
    ├── val/
    │   ├── normal/     (200 images)
    │   └── defective/  (200 images)
    └── test/
        ├── normal/     (200 images)
        └── defective/  (200 images)

Run:
    python -m data_gen.generate_dataset
"""

import os
import json
import random
import argparse
import logging
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import yaml
from tqdm import tqdm

# ── Logging setup ────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def make_surface_texture(size: Tuple[int, int], seed: int) -> np.ndarray:
    """
    Creates a realistic-looking metal/plastic surface texture by layering
    Perlin-like noise at different scales.  We approximate Perlin noise
    with a sum of rescaled Gaussian blurs applied to random arrays.
    """
    rng = np.random.default_rng(seed)
    h, w = size

    # Start with random static
    base = rng.integers(180, 230, (h, w), dtype=np.uint8)

    # Add coarse variation
    coarse = rng.integers(0, 30, (h // 4, w // 4), dtype=np.uint8)
    coarse = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    coarse = cv2.GaussianBlur(coarse, (21, 21), 0)

    # Add fine grain
    fine = rng.integers(0, 15, (h, w), dtype=np.uint8)
    fine = cv2.GaussianBlur(fine, (3, 3), 0)

    surface = np.clip(base.astype(np.int32) + coarse - 15 + fine - 7, 0, 255).astype(np.uint8)

    # Convert to 3-channel BGR so we can add colour defects later
    surface_bgr = cv2.cvtColor(surface, cv2.COLOR_GRAY2BGR)

    # Slight warm tint to mimic bare metal / plastic
    surface_bgr[:, :, 2] = np.clip(surface_bgr[:, :, 2].astype(np.int32) + 8, 0, 255)

    return surface_bgr


def add_scratch(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Draws a thin, jagged line to simulate a surface scratch.
    Real scratches rarely go in perfectly straight lines, so we use
    a random-walk approach to make it look natural.
    """
    h, w = img.shape[:2]
    out = img.copy()

    # Pick a random starting edge and direction
    x = int(rng.integers(w // 4, 3 * w // 4))
    y = int(rng.integers(0, h // 3))
    length = int(rng.integers(60, 140))
    angle = rng.uniform(70, 110)  # mostly vertical scratches

    dx = np.cos(np.radians(angle))
    dy = np.sin(np.radians(angle))

    # Scratch colour: slightly darker than the surface
    colour = (int(rng.integers(60, 100)),) * 3

    for _ in range(length):
        # Small random jitter on each step → jagged appearance
        jx = x + int(rng.integers(-1, 2))
        jy = y + int(rng.integers(-1, 2))
        cv2.circle(out, (jx, jy), 1, colour, -1)
        x = int(x + dx + rng.uniform(-0.5, 0.5))
        y = int(y + dy + rng.uniform(-0.5, 0.5))
        if not (0 <= x < w and 0 <= y < h):
            break

    return out


def add_crack(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Simulates a crack using a branching polyline — wider and darker than
    a scratch, with optional secondary branches.
    """
    h, w = img.shape[:2]
    out = img.copy()

    # Main crack
    num_pts = int(rng.integers(5, 10))
    pts = []
    x = int(rng.integers(w // 5, 4 * w // 5))
    y = int(rng.integers(h // 5, 4 * h // 5))
    pts.append([x, y])

    for _ in range(num_pts):
        x += int(rng.integers(-20, 20))
        y += int(rng.integers(5, 20))
        pts.append([x, y])

    pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(out, [pts], False, (50, 45, 40), thickness=2)

    # Occasionally add a short branch
    if rng.random() > 0.4:
        mid_idx = len(pts) // 2
        bx, by = pts[mid_idx][0]
        branch = [[bx, by]]
        for _ in range(3):
            bx += int(rng.integers(-15, 15))
            by += int(rng.integers(5, 15))
            branch.append([bx, by])
        branch_pts = np.array(branch, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [branch_pts], False, (55, 50, 45), thickness=1)

    return out


def add_stain(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Adds an irregular blob to simulate oil, rust, or contamination stains.
    We use an ellipse with a slightly transparent overlay for realism.
    """
    h, w = img.shape[:2]
    overlay = img.copy()

    cx = int(rng.integers(w // 4, 3 * w // 4))
    cy = int(rng.integers(h // 4, 3 * h // 4))
    ax = int(rng.integers(15, 50))
    ay = int(rng.integers(10, 35))
    angle = int(rng.integers(0, 180))

    # Pick a stain colour — brownish/orange for rust, dark for oil
    stain_colours = [
        (20, 60, 120),   # rust-brown
        (30, 30, 30),    # oil-black
        (100, 130, 180), # water stain (greyish-blue)
    ]
    colour = stain_colours[int(rng.integers(0, len(stain_colours)))]

    cv2.ellipse(overlay, (cx, cy), (ax, ay), angle, 0, 360, colour, -1)

    # Blend with original for a semi-transparent look
    alpha = rng.uniform(0.35, 0.6)
    out = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    return out


DEFECT_FNS = {
    "scratch": add_scratch,
    "crack":   add_crack,
    "stain":   add_stain,
}


def generate_image(
    size: Tuple[int, int],
    label: str,
    defect_types: list,
    seed: int,
) -> np.ndarray:
    """
    Produces a single image.  Normal images are just surface textures;
    defective images have one randomly chosen defect applied.
    """
    rng = np.random.default_rng(seed)
    h, w = size[1], size[0]

    img = make_surface_texture((h, w), seed=seed)

    if label == "defective":
        defect_type = rng.choice(defect_types)
        img = DEFECT_FNS[defect_type](img, rng)

    return img


def generate_split(
    split: str,
    num_per_class: int,
    img_size: Tuple[int, int],
    defect_types: list,
    output_root: Path,
    seed_offset: int,
) -> None:
    """Generates all images for one split (train / val / test)."""
    classes = ["normal", "defective"]

    for cls in classes:
        out_dir = output_root / split / cls
        out_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"  Generating {num_per_class} '{cls}' images → {out_dir}")

        for i in tqdm(range(num_per_class), desc=f"{split}/{cls}", leave=False):
            seed = seed_offset + hash(f"{split}-{cls}-{i}") % (2**31)
            img = generate_image(img_size, cls, defect_types, seed)
            filename = out_dir / f"{cls}_{i:04d}.jpg"
            cv2.imwrite(str(filename), img, [cv2.IMWRITE_JPEG_QUALITY, 92])


# ── Main ─────────────────────────────────────────────────────────────────────

def main(config_path: str = "configs/config.yaml") -> None:
    cfg = load_config(config_path)
    dg  = cfg["data_gen"]

    img_size    = tuple(dg["image_size"])          # (w, h)
    num_train   = dg["num_train"]
    num_val     = dg["num_val"]
    num_test    = dg["num_test"]
    defect_types = dg["defect_types"]
    seed        = dg["random_seed"]
    output_root = Path(cfg["paths"]["raw_data"])

    log.info("=" * 55)
    log.info("  Synthetic Dataset Generator")
    log.info(f"  Image size : {img_size}")
    log.info(f"  Train/Val/Test per class: {num_train}/{num_val}/{num_test}")
    log.info(f"  Output root: {output_root}")
    log.info("=" * 55)

    splits = {
        "train": num_train,
        "val":   num_val,
        "test":  num_test,
    }

    for i, (split, count) in enumerate(splits.items()):
        log.info(f"[{i+1}/3] Split: {split}")
        generate_split(
            split       = split,
            num_per_class = count,
            img_size    = img_size,
            defect_types = defect_types,
            output_root = output_root,
            seed_offset = seed * (i + 1),
        )

    # Save class map alongside the data for easy reference
    class_map_path = output_root / "class_map.json"
    with open("data_gen/class_map.json") as src:
        class_map = json.load(src)
    with open(class_map_path, "w") as dst:
        json.dump(class_map, dst, indent=2)

    total = sum(v for v in splits.values()) * 2
    log.info(f"\n✓ Done — generated {total} images total.")
    log.info(f"  Dataset root: {output_root.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic defect dataset")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
