"""
tests/test_preprocessing.py
────────────────────────────
Unit tests for the preprocessing pipeline.

Tests validate that:
  - Each denoising method returns the correct shape and dtype
  - Normalisation produces values in the expected range
  - The ImagePreprocessor orchestrates the pipeline end-to-end correctly
  - Edge cases (empty images, single-channel, very small images) are handled

Run:
    pytest tests/test_preprocessing.py -v
"""

import numpy as np
import pytest

from preprocessing.noise_reduction import denoise, gaussian_denoise, median_denoise, bilateral_denoise
from preprocessing.normalizer import normalize, standardize, minmax_normalize, to_chw
from preprocessing.pipeline import ImagePreprocessor


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_bgr_image():
    """224×224 RGB image with random uint8 values — mimics a camera frame."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)


@pytest.fixture
def small_image():
    """Tiny 32×32 image for edge case tests."""
    return np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)


# ── Noise reduction tests ─────────────────────────────────────────────────────

class TestNoiseReduction:

    def test_gaussian_output_shape(self, sample_bgr_image):
        """Gaussian blur should not change image dimensions."""
        result = gaussian_denoise(sample_bgr_image, kernel_size=5)
        assert result.shape == sample_bgr_image.shape

    def test_median_output_shape(self, sample_bgr_image):
        """Median filter should preserve shape."""
        result = median_denoise(sample_bgr_image, kernel_size=5)
        assert result.shape == sample_bgr_image.shape

    def test_bilateral_output_shape(self, sample_bgr_image):
        """Bilateral filter should preserve shape."""
        result = bilateral_denoise(sample_bgr_image)
        assert result.shape == sample_bgr_image.shape

    def test_gaussian_dtype_preserved(self, sample_bgr_image):
        """Output dtype should remain uint8 for uint8 input."""
        result = gaussian_denoise(sample_bgr_image)
        assert result.dtype == np.uint8

    def test_gaussian_reduces_noise(self, sample_bgr_image):
        """After blurring, variance should be lower than the original."""
        result = gaussian_denoise(sample_bgr_image, kernel_size=11, sigma=3.0)
        assert result.std() < sample_bgr_image.std()

    def test_invalid_kernel_size_raises(self, sample_bgr_image):
        """Even kernel sizes should raise ValueError."""
        with pytest.raises(ValueError, match="must be odd"):
            gaussian_denoise(sample_bgr_image, kernel_size=4)

    def test_dispatch_all_methods(self, sample_bgr_image):
        """denoise() should work for all three methods."""
        for method in ("gaussian", "median", "bilateral"):
            result = denoise(sample_bgr_image, method=method)
            assert result.shape == sample_bgr_image.shape

    def test_unknown_method_raises(self, sample_bgr_image):
        with pytest.raises(ValueError, match="Unknown denoising method"):
            denoise(sample_bgr_image, method="magic")

    def test_empty_image_raises(self):
        with pytest.raises(ValueError, match="empty"):
            denoise(np.array([]), method="gaussian")


# ── Normalisation tests ───────────────────────────────────────────────────────

class TestNormalizer:

    def test_minmax_range(self, sample_bgr_image):
        """MinMax normalisation should produce values in [0, 1]."""
        result = minmax_normalize(sample_bgr_image)
        assert result.min() >= 0.0
        assert result.max() <= 1.0 + 1e-6

    def test_minmax_dtype(self, sample_bgr_image):
        result = minmax_normalize(sample_bgr_image)
        assert result.dtype == np.float32

    def test_standardize_dtype(self, sample_bgr_image):
        result = standardize(sample_bgr_image)
        assert result.dtype == np.float32

    def test_standardize_shape(self, sample_bgr_image):
        """Standardize should return (H, W, 3)."""
        result = standardize(sample_bgr_image)
        assert result.shape == sample_bgr_image.shape

    def test_standardize_roughly_zero_centred(self, sample_bgr_image):
        """After ImageNet normalisation, mean should be close to 0."""
        result = standardize(sample_bgr_image)
        # Not exactly 0 because we're using random input, not ImageNet images
        assert abs(result.mean()) < 2.0

    def test_normalize_resize(self, small_image):
        """normalize() should resize to target_size regardless of input."""
        result = normalize(small_image, target_size=(224, 224), method="minmax")
        assert result.shape == (224, 224, 3)

    def test_to_chw(self, sample_bgr_image):
        """to_chw should transpose (H,W,C) → (C,H,W)."""
        chw = to_chw(sample_bgr_image)
        assert chw.shape == (3, 224, 224)

    def test_to_chw_grayscale(self):
        """Grayscale (H,W) should become (1,H,W)."""
        gray = np.zeros((100, 100), dtype=np.uint8)
        chw  = to_chw(gray)
        assert chw.shape == (1, 100, 100)


# ── Pipeline integration test ─────────────────────────────────────────────────

class TestImagePreprocessor:

    def test_process_returns_correct_shape(self, sample_bgr_image, tmp_path):
        """
        End-to-end: processing a raw image should return a (3, 224, 224)
        float32 tensor — exactly what the model expects.
        """
        # Write a minimal config to a temp file
        config = _minimal_config(tmp_path)
        preprocessor = ImagePreprocessor(str(config))
        result = preprocessor.process(sample_bgr_image, mode="val")

        assert result.shape == (3, 224, 224)
        assert result.dtype == np.float32

    def test_process_train_mode(self, sample_bgr_image, tmp_path):
        """Train mode should also return (3, 224, 224) but with augmentation applied."""
        config = _minimal_config(tmp_path)
        preprocessor = ImagePreprocessor(str(config))
        result = preprocessor.process(sample_bgr_image, mode="train")

        assert result.shape == (3, 224, 224)
        assert result.dtype == np.float32

    def test_process_batch(self, sample_bgr_image, tmp_path):
        """process_batch should return (N, 3, 224, 224)."""
        config = _minimal_config(tmp_path)
        preprocessor = ImagePreprocessor(str(config))
        batch = preprocessor.process_batch([sample_bgr_image, sample_bgr_image], mode="val")

        assert batch.shape == (2, 3, 224, 224)

    def test_empty_image_raises(self, tmp_path):
        config = _minimal_config(tmp_path)
        preprocessor = ImagePreprocessor(str(config))
        with pytest.raises(ValueError):
            preprocessor.process(np.array([]), mode="val")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _minimal_config(tmp_path) -> "Path":
    """Writes a minimal config.yaml to a temp directory for testing."""
    import yaml
    from pathlib import Path

    config = {
        "data_gen":     {"image_size": [224, 224], "random_seed": 42},
        "preprocessing": {
            "noise_reduction": {"method": "gaussian", "kernel_size": 5, "sigma": 1.0},
            "normalization":   {"method": "standardize", "mean": [0.485, 0.456, 0.406],
                                "std": [0.229, 0.224, 0.225]},
            "augmentation":    {
                "enabled": True, "horizontal_flip": True, "vertical_flip": False,
                "rotation_limit": 15, "brightness_limit": 0.2, "contrast_limit": 0.2,
                "p_flip": 0.5, "p_rotate": 0.4, "p_color_jitter": 0.3,
            },
        },
        "api": {"metrics_buffer_size": 100},
    }

    path = tmp_path / "config.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    return path
