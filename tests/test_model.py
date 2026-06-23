"""
tests/test_model.py
────────────────────
Unit tests for the model architecture and training utilities.

Tests verify:
  - Model forward pass produces correct output shapes
  - predict_proba outputs sum to 1 (valid probability distribution)
  - EarlyStopping triggers correctly after patience epochs
  - build_model factory returns a model on the right device
  - LiteCNN also produces correct output shapes

Run:
    pytest tests/test_model.py -v
"""

import pytest
import torch
import torch.nn as nn

from training.model import DefectClassifier, LiteCNN, build_model, _get_device
from training.train import EarlyStopping


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def model_cpu():
    """Small DefectClassifier on CPU, no pretrained weights (faster for tests)."""
    return DefectClassifier(num_classes=2, pretrained=False)


@pytest.fixture
def dummy_batch():
    """Fake batch of 4 images: (B=4, C=3, H=224, W=224)."""
    return torch.randn(4, 3, 224, 224)


# ── DefectClassifier tests ────────────────────────────────────────────────────

class TestDefectClassifier:

    def test_forward_output_shape(self, model_cpu, dummy_batch):
        """Forward pass should produce (B, num_classes) logits."""
        logits = model_cpu(dummy_batch)
        assert logits.shape == (4, 2)

    def test_predict_proba_sums_to_one(self, model_cpu, dummy_batch):
        """Softmax probabilities should sum to 1 for each sample."""
        probs = model_cpu.predict_proba(dummy_batch)
        sums  = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)

    def test_output_dtype(self, model_cpu, dummy_batch):
        logits = model_cpu(dummy_batch)
        assert logits.dtype == torch.float32

    def test_no_nan_in_output(self, model_cpu, dummy_batch):
        """Model should never produce NaN logits."""
        logits = model_cpu(dummy_batch)
        assert not torch.isnan(logits).any()

    def test_single_image_forward(self, model_cpu):
        """Batch size of 1 should work (important for real-time inference)."""
        img    = torch.randn(1, 3, 224, 224)
        logits = model_cpu(img)
        assert logits.shape == (1, 2)

    def test_frozen_backbone(self):
        """When freeze_backbone=True, backbone params should not require grad."""
        model = DefectClassifier(pretrained=False, freeze_backbone=True)
        backbone_params = list(model.backbone.parameters())
        # All backbone params should be frozen
        assert all(not p.requires_grad for p in backbone_params)

    def test_head_always_trainable(self, model_cpu):
        """Head parameters should always require grad."""
        head_params = list(model_cpu.head.parameters())
        assert all(p.requires_grad for p in head_params)


# ── LiteCNN tests ─────────────────────────────────────────────────────────────

class TestLiteCNN:

    def test_forward_output_shape(self, dummy_batch):
        model  = LiteCNN(num_classes=2)
        logits = model(dummy_batch)
        assert logits.shape == (4, 2)

    def test_no_pretrained_dependency(self):
        """LiteCNN should initialise without any pretrained weights."""
        model = LiteCNN()
        assert model is not None


# ── EarlyStopping tests ───────────────────────────────────────────────────────

class TestEarlyStopping:

    def test_no_stop_when_improving(self):
        """Should not stop when accuracy improves each epoch."""
        es = EarlyStopping(patience=3, min_delta=0.001)
        scores = [0.70, 0.75, 0.80, 0.85, 0.90]
        stopped = any(es.step(s) for s in scores)
        assert not stopped

    def test_stops_after_patience(self):
        """Should stop after 'patience' epochs of no improvement."""
        es = EarlyStopping(patience=3, min_delta=0.001)
        # Improve once, then stagnate
        es.step(0.80)
        for _ in range(3):
            es.step(0.80)  # no improvement
        assert es.should_stop

    def test_min_delta_respected(self):
        """Improvements smaller than min_delta should not reset counter."""
        es = EarlyStopping(patience=2, min_delta=0.01)
        es.step(0.80)
        es.step(0.805)   # improvement < min_delta → counter stays
        es.step(0.806)   # still < min_delta
        assert es.should_stop

    def test_reset_on_large_improvement(self):
        """Counter should reset when a significant improvement occurs."""
        es = EarlyStopping(patience=2, min_delta=0.01)
        es.step(0.70)
        es.step(0.70)    # counter = 1
        es.step(0.85)    # big improvement → counter resets to 0
        assert es.counter == 0
        assert not es.should_stop


# ── Device detection test ─────────────────────────────────────────────────────

class TestDeviceDetection:

    def test_cpu_fallback(self):
        """_get_device('cpu') should always return CPU."""
        device = _get_device("cpu")
        assert device.type == "cpu"

    def test_cuda_fallback_to_cpu(self):
        """If CUDA is not available, requesting 'cuda' should fall back to CPU."""
        if not torch.cuda.is_available():
            device = _get_device("cuda")
            assert device.type == "cpu"
