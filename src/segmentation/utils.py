from __future__ import annotations

"""Utility helpers shared across training, evaluation, and prediction."""

import csv
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig

IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int) -> None:
    """Seed all relevant RNGs so train/val splits and training are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    """Resolve the runtime device from config, preferring accelerators when available."""
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_run_dir(config: ExperimentConfig) -> Path:
    """Create a timestamped run directory for checkpoints, metrics, and previews."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config.training.run_name}_{timestamp}"
    run_dir = Path(config.training.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def append_metrics_row(csv_path: str | Path, row: dict[str, float | int]) -> None:
    """Append one epoch of metrics to a CSV file, creating the header on first write."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def denormalize_image(image_tensor: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet-style normalization for human-readable visualizations."""
    mean = torch.tensor(IMAGE_MEAN, device=image_tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGE_STD, device=image_tensor.device).view(3, 1, 1)
    return (image_tensor * std + mean).clamp(0.0, 1.0)
