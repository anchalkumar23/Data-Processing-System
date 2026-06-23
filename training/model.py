"""
training/model.py
──────────────────
Defines the CNN model for binary defect classification.

Architecture overview:
  - Backbone: ResNet-18 pretrained on ImageNet (via torchvision)
  - Head:     Custom classification head with dropout for regularisation

Why ResNet-18?
  • Fast to train even on CPU (small enough for a resume demo)
  • 87% accuracy is very achievable on our synthetic dataset
  • Well-understood, easy to explain in interviews
  • Supports ONNX export cleanly

We also expose a small `LiteCNN` for completeness — a from-scratch
lightweight CNN that shows understanding of conv layers without
relying on a pretrained backbone.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

log = logging.getLogger(__name__)


class DefectClassifier(nn.Module):
    """
    ResNet-18 backbone with a custom binary classification head.

    The backbone is pretrained on ImageNet, so it already knows how to
    extract meaningful visual features.  We replace the final fully
    connected layer with our own head that outputs 2 class logits.

    Args:
        num_classes:      Number of output classes (2 for binary).
        pretrained:       Whether to load ImageNet weights.
        freeze_backbone:  If True, only the head is trained.
                          Useful for very small datasets; set False here
                          because we have enough synthetic data to fine-tune.
        dropout_rate:     Dropout probability in the head (regularisation).
    """

    def __init__(
        self,
        num_classes:     int  = 2,
        pretrained:      bool = True,
        freeze_backbone: bool = False,
        dropout_rate:    float = 0.3,
    ) -> None:
        super().__init__()

        # Load pretrained ResNet-18
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
            log.info("Backbone frozen — only training the classification head")

        # How many features does the ResNet output before the final FC layer?
        in_features = backbone.fc.in_features  # 512 for ResNet-18

        # Replace the original FC layer with our custom head
        backbone.fc = nn.Identity()  # strip the ImageNet head
        self.backbone = backbone

        # Custom head: Dropout → Linear → (optionally more layers)
        # The dropout helps prevent overfitting on our relatively small dataset
        self.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(256, num_classes),
        )

        # Initialise the head weights with He initialisation
        self._init_head()

        total_params = sum(p.numel() for p in self.parameters())
        trainable   = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log.info(f"DefectClassifier: {total_params:,} params ({trainable:,} trainable)")

    def _init_head(self) -> None:
        """Kaiming (He) initialisation for the linear layers."""
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: float32 tensor of shape (B, 3, H, W).

        Returns:
            Logits tensor of shape (B, num_classes).
            Apply softmax for probabilities, argmax for predicted class.
        """
        features = self.backbone(x)   # (B, 512)
        logits   = self.head(features)  # (B, num_classes)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax probabilities instead of raw logits."""
        return F.softmax(self.forward(x), dim=1)


class LiteCNN(nn.Module):
    """
    A lightweight from-scratch CNN — kept here to demonstrate
    understanding of CNN internals.  Not used in production training,
    but shows the interviewer you know how conv layers work.

    Architecture:
        Conv(3→32) → BN → ReLU → MaxPool
        Conv(32→64) → BN → ReLU → MaxPool
        Conv(64→128) → BN → ReLU → AdaptiveAvgPool
        FC(128→64) → ReLU → FC(64→num_classes)
    """

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),        # 224 → 112

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),        # 112 → 56

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),  # → (B, 128, 4, 4)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def build_model(config: dict) -> DefectClassifier:
    """
    Factory function — creates the model from config.yaml values.

    Args:
        config: Parsed config dict (the 'training' section).

    Returns:
        Initialised DefectClassifier moved to the configured device.
    """
    tr = config["training"]
    model = DefectClassifier(
        num_classes     = tr["num_classes"],
        pretrained      = tr.get("pretrained", True),
        freeze_backbone = tr.get("freeze_backbone", False),
    )

    device = _get_device(tr.get("device", "cuda"))
    model  = model.to(device)
    log.info(f"Model moved to device: {device}")
    return model


def _get_device(preference: str) -> torch.device:
    """
    Returns the best available device.  Falls back gracefully to CPU
    so the project runs everywhere, not just on GPU machines.
    """
    if preference == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        log.info(f"GPU available: {gpu_name}")
        return torch.device("cuda")
    else:
        if preference == "cuda":
            log.warning("CUDA requested but not available — falling back to CPU")
        return torch.device("cpu")
