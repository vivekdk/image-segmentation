from __future__ import annotations

"""Visualization helpers for validation previews and standalone predictions."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .utils import denormalize_image


def save_prediction_grid(
    images: torch.Tensor,
    targets: torch.Tensor,
    logits: torch.Tensor,
    output_path: str | Path,
    max_items: int = 4,
) -> None:
    """Save a grid of image, ground-truth mask, and predicted mask triplets."""
    count = min(max_items, images.size(0))
    fig, axes = plt.subplots(count, 3, figsize=(9, 3 * count))
    if count == 1:
        axes = np.expand_dims(axes, axis=0)

    probs = torch.sigmoid(logits[:count]).detach().cpu()
    for idx in range(count):
        image = denormalize_image(images[idx].detach().cpu()).permute(1, 2, 0).numpy()
        target = targets[idx, 0].detach().cpu().numpy()
        pred = (probs[idx, 0].numpy() >= 0.5).astype(np.float32)

        axes[idx, 0].imshow(image)
        axes[idx, 0].set_title("Image")
        axes[idx, 1].imshow(target, cmap="gray")
        axes[idx, 1].set_title("Target")
        axes[idx, 2].imshow(pred, cmap="gray")
        axes[idx, 2].set_title("Prediction")

        for axis in axes[idx]:
            axis.axis("off")

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_mask_and_overlay(
    original: Image.Image,
    pred_mask: np.ndarray,
    mask_output_path: str | Path,
    overlay_output_path: str | Path,
) -> None:
    """Persist a binary mask and a red overlay for easy qualitative inspection."""
    mask_image = Image.fromarray((pred_mask * 255).astype(np.uint8), mode="L")
    mask_output_path = Path(mask_output_path)
    overlay_output_path = Path(overlay_output_path)
    mask_output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_image.save(mask_output_path)

    rgb = original.convert("RGB")
    rgb_array = np.asarray(rgb, dtype=np.float32) / 255.0
    overlay_color = np.zeros_like(rgb_array)
    overlay_color[..., 0] = 1.0
    # The overlay uses a constant red tint so the segmentation is obvious even
    # against varied pet colors and backgrounds.
    alpha = 0.35 * pred_mask[..., None]
    overlay = rgb_array * (1.0 - alpha) + overlay_color * alpha
    overlay_image = Image.fromarray((overlay * 255).astype(np.uint8))
    overlay_image.save(overlay_output_path)
