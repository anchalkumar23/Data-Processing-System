"""
training/train.py
──────────────────
Main training loop for the defect classification model.

Features:
  - GPU-accelerated training with automatic CPU fallback
  - Learning rate schedulers (cosine annealing, step, ReduceLROnPlateau)
  - Early stopping to prevent overfitting
  - Best-model checkpointing (saves only when val accuracy improves)
  - Per-epoch logging with training loss, val loss, and val accuracy
  - Weights-and-Biases / TensorBoard compatible loss curves (via tqdm logs)

Run:
    python -m training.train
    python -m training.train --config configs/config.yaml --resume checkpoints/last.pth
"""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from training.model import build_model, _get_device
from training.dataset import build_dataloaders
from preprocessing.pipeline import ImagePreprocessor

log = logging.getLogger(__name__)


# ── Training utilities ───────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stops training when validation accuracy stops improving.

    We track the best metric seen so far, and count how many epochs
    have passed without improvement.  If that count exceeds `patience`,
    we signal to stop training.
    """

    def __init__(self, patience: int = 8, min_delta: float = 0.001) -> None:
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_score = None
        self.counter    = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        """
        Call once per epoch with the current validation metric.
        Returns True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            return False

        improvement = score - self.best_score
        if improvement > self.min_delta:
            self.best_score = score
            self.counter    = 0
        else:
            self.counter += 1
            log.info(f"EarlyStopping: no improvement for {self.counter}/{self.patience} epochs")
            if self.counter >= self.patience:
                self.should_stop = True
                log.info("EarlyStopping triggered — stopping training")

        return self.should_stop


def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device:    torch.device,
    epoch:     int,
) -> Tuple[float, float]:
    """
    Runs one full pass over the training set.

    Returns:
        (avg_loss, accuracy) for the epoch.
    """
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [train]", leave=False)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # slightly faster than zero_grad()

        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()

        # Gradient clipping prevents exploding gradients during fine-tuning
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        batch_size  = images.size(0)
        total_loss += loss.item() * batch_size
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += batch_size

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    split:     str = "val",
) -> Tuple[float, float]:
    """
    Evaluates the model on a data split without updating weights.

    Returns:
        (avg_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for images, labels in tqdm(loader, desc=f"  [{split}]", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss   = criterion(logits, labels)

        batch_size  = images.size(0)
        total_loss += loss.item() * batch_size
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += batch_size

    return total_loss / total, correct / total


def build_scheduler(optimizer: optim.Optimizer, config: dict, num_epochs: int):
    """Creates the LR scheduler specified in config."""
    sched_name = config["training"].get("lr_scheduler", "cosine")

    if sched_name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    elif sched_name == "step":
        return StepLR(optimizer, step_size=10, gamma=0.5)
    elif sched_name == "plateau":
        patience = config["training"].get("lr_patience", 5)
        return ReduceLROnPlateau(optimizer, mode="max", patience=patience, verbose=True)
    else:
        raise ValueError(f"Unknown scheduler: {sched_name}")


# ── Checkpoint helpers ───────────────────────────────────────────────────────

def save_checkpoint(
    model:    nn.Module,
    optimizer: optim.Optimizer,
    epoch:    int,
    metrics:  dict,
    path:     Path,
) -> None:
    """Saves model weights + training state to a .pth file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch":      epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics":    metrics,
    }, path)
    log.info(f"Checkpoint saved → {path}  (val_acc={metrics.get('val_accuracy', 0):.4f})")


def load_checkpoint(
    path:      Path,
    model:     nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
) -> Tuple[int, dict]:
    """Loads a checkpoint and restores model/optimizer state."""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    log.info(f"Resumed from checkpoint: {path} (epoch {ckpt['epoch']})")
    return ckpt["epoch"], ckpt.get("metrics", {})


# ── Main training loop ───────────────────────────────────────────────────────

def train(config_path: str = "configs/config.yaml", resume: Optional[str] = None) -> None:
    """
    Full training run: dataset → model → train loop → save best model.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    tr  = config["training"]
    paths = config["paths"]

    device      = _get_device(tr.get("device", "cuda"))
    num_epochs  = tr["num_epochs"]
    lr          = tr["learning_rate"]
    wd          = tr["weight_decay"]
    ckpt_dir    = Path(paths["checkpoints"])

    # ── Data ────────────────────────────────────────────────
    preprocessor = ImagePreprocessor(config_path)
    train_loader, val_loader, _ = build_dataloaders(config, preprocessor)

    # ── Model ───────────────────────────────────────────────
    model = build_model(config)

    # ── Optimiser ───────────────────────────────────────────
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=wd,
    )

    scheduler = build_scheduler(optimizer, config, num_epochs)
    criterion = nn.CrossEntropyLoss()

    # ── Resume from checkpoint ───────────────────────────────
    start_epoch = 0
    best_val_acc = 0.0

    if resume:
        start_epoch, prev_metrics = load_checkpoint(Path(resume), model, optimizer)
        best_val_acc = prev_metrics.get("val_accuracy", 0.0)
        start_epoch += 1  # continue from the next epoch

    early_stopper = EarlyStopping(
        patience  = tr.get("patience", 8),
        min_delta = tr.get("min_delta", 0.001),
    ) if tr.get("early_stopping", True) else None

    # ── Training loop ────────────────────────────────────────
    log.info("=" * 60)
    log.info(f"  Starting training: {num_epochs} epochs on {device}")
    log.info(f"  Batch size: {tr['batch_size']} | LR: {lr} | WD: {wd}")
    log.info("=" * 60)

    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(start_epoch, start_epoch + num_epochs):
        t0 = time.perf_counter()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, "val")

        # Update LR scheduler
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_acc)
        else:
            scheduler.step()

        elapsed = time.perf_counter() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        log.info(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f} | "
            f"lr={current_lr:.2e}  time={elapsed:.1f}s"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch,
                {"val_accuracy": val_acc, "val_loss": val_loss},
                ckpt_dir / "best_model.pth",
            )

        # Always save the last checkpoint for resuming
        save_checkpoint(
            model, optimizer, epoch,
            {"val_accuracy": val_acc, "val_loss": val_loss},
            ckpt_dir / "last.pth",
        )

        # Early stopping check
        if early_stopper and early_stopper.step(val_acc):
            break

    log.info(f"\n✓ Training complete.  Best val accuracy: {best_val_acc:.4f}")
    log.info(f"  Best model saved to: {ckpt_dir / 'best_model.pth'}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Train defect detection model")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    train(args.config, args.resume)
