from __future__ import annotations

"""Training and evaluation loops plus checkpoint helpers."""

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .losses import segmentation_loss
from .metrics import compute_binary_metrics
from .visualization import save_prediction_grid


def _move_batch_to_device(batch: tuple[torch.Tensor, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Move one `(images, masks)` batch to the selected device."""
    images, masks = batch
    return images.to(device), masks.to(device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    threshold: float,
    log_every_n_steps: int = 0,
    epoch: int | None = None,
) -> dict[str, float]:
    """Run one training epoch and return mean loss and segmentation metrics."""
    model.train()
    total_loss = 0.0
    total_metrics = {"dice": 0.0, "iou": 0.0, "pixel_accuracy": 0.0}
    num_batches = max(1, len(loader))

    for step, batch in enumerate(loader, start=1):
        images, masks = _move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = segmentation_loss(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_metrics = compute_binary_metrics(logits.detach(), masks, threshold=threshold)
        for key, value in batch_metrics.items():
            total_metrics[key] += value

        if log_every_n_steps > 0 and (step == 1 or step % log_every_n_steps == 0 or step == num_batches):
            epoch_label = f" epoch={epoch:03d}" if epoch is not None else ""
            print(
                f"[train]{epoch_label} step={step}/{num_batches} "
                f"loss={loss.item():.4f} dice={batch_metrics['dice']:.4f} "
                f"iou={batch_metrics['iou']:.4f}"
            )

    return {
        "loss": total_loss / num_batches,
        **{key: value / num_batches for key, value in total_metrics.items()},
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    visualization_path: str | Path | None = None,
    max_visualizations: int = 4,
    split_name: str = "eval",
    log_every_n_steps: int = 0,
    epoch: int | None = None,
) -> dict[str, float]:
    """Run evaluation without gradients and optionally save one preview grid."""
    model.eval()
    total_loss = 0.0
    total_metrics = {"dice": 0.0, "iou": 0.0, "pixel_accuracy": 0.0}
    example_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    num_batches = max(1, len(loader))

    for step, batch in enumerate(loader, start=1):
        images, masks = _move_batch_to_device(batch, device)
        logits = model(images)
        loss = segmentation_loss(logits, masks)

        total_loss += loss.item()
        batch_metrics = compute_binary_metrics(logits, masks, threshold=threshold)
        for key, value in batch_metrics.items():
            total_metrics[key] += value

        # Save only the first batch for visualization to keep validation cheap
        # and predictable even when the validation set is large.
        if example_batch is None:
            example_batch = (images.detach().cpu(), masks.detach().cpu(), logits.detach().cpu())

        if log_every_n_steps > 0 and (step == 1 or step % log_every_n_steps == 0 or step == num_batches):
            epoch_label = f" epoch={epoch:03d}" if epoch is not None else ""
            print(
                f"[{split_name}]{epoch_label} step={step}/{num_batches} "
                f"loss={loss.item():.4f} dice={batch_metrics['dice']:.4f} "
                f"iou={batch_metrics['iou']:.4f}"
            )

    metrics = {
        "loss": total_loss / num_batches,
        **{key: value / num_batches for key, value in total_metrics.items()},
    }

    if visualization_path is not None and example_batch is not None:
        images, masks, logits = example_batch
        save_prediction_grid(images, masks, logits, visualization_path, max_items=max_visualizations)

    return metrics


def save_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save model/optimizer state together with epoch and metrics."""
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "metadata": metadata or {},
    }
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)


def load_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint into the model and optionally restore optimizer state."""
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
