"""
training/evaluate.py
─────────────────────
Model evaluation on the held-out test set.

Computes:
  - Overall accuracy
  - Per-class precision, recall, F1-score
  - Confusion matrix (printed + saved as PNG)
  - ROC AUC score

Run after training:
    python -m training.evaluate --checkpoint checkpoints/best_model.pth
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for servers without a display
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from tqdm import tqdm

from training.model import build_model, _get_device, load_checkpoint
from training.dataset import build_dataloaders, DefectDataset
from preprocessing.pipeline import ImagePreprocessor

log = logging.getLogger(__name__)

CLASS_NAMES = ["normal", "defective"]


@torch.no_grad()
def run_inference(
    model:   torch.nn.Module,
    loader,
    device:  torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs the model over a DataLoader and collects predictions.

    Returns:
        all_labels:  Ground truth class indices.
        all_preds:   Predicted class indices.
        all_probs:   Softmax probability for the 'defective' class (class 1).
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs  = torch.softmax(logits, dim=1)

        all_labels.extend(labels.numpy())
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())  # prob of 'defective'

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )


def plot_confusion_matrix(
    cm:         np.ndarray,
    class_names: List[str],
    output_path: Path,
) -> None:
    """Saves a styled confusion matrix heatmap as a PNG."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label",      fontsize=12)
    ax.set_title("Confusion Matrix — Test Set", fontsize=14)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info(f"Confusion matrix saved → {output_path}")


def evaluate(
    config_path: str = "configs/config.yaml",
    checkpoint:  str = "checkpoints/best_model.pth",
) -> Dict:
    """
    Full evaluation pipeline — loads the best checkpoint, runs the
    test set, prints a report, and saves the confusion matrix.

    Returns:
        dict with 'accuracy', 'auc', and sklearn classification_report.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = _get_device(config["training"].get("device", "cuda"))

    # Load model + checkpoint
    model = build_model(config)
    load_checkpoint(Path(checkpoint), model)
    model = model.to(device)

    # Build test loader
    preprocessor = ImagePreprocessor(config_path)
    _, _, test_loader = build_dataloaders(config, preprocessor)

    # Run inference
    labels, preds, probs = run_inference(model, test_loader, device)

    # Metrics
    accuracy = accuracy_score(labels, preds)
    auc      = roc_auc_score(labels, probs)
    cm       = confusion_matrix(labels, preds)
    report   = classification_report(labels, preds, target_names=CLASS_NAMES)

    log.info("\n" + "=" * 55)
    log.info(f"  Test Accuracy : {accuracy:.4f}  ({accuracy*100:.2f}%)")
    log.info(f"  ROC AUC       : {auc:.4f}")
    log.info("\n" + report)
    log.info("=" * 55)

    # Save confusion matrix
    logs_dir = Path(config["paths"]["logs"])
    plot_confusion_matrix(cm, CLASS_NAMES, logs_dir / "confusion_matrix.png")

    return {
        "accuracy":             accuracy,
        "auc":                  auc,
        "classification_report": report,
        "confusion_matrix":     cm.tolist(),
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Evaluate the trained model on the test set")
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint)
