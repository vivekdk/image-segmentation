from __future__ import annotations

"""Evaluation metrics for binary segmentation outputs."""

import torch


def compute_binary_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    """Compute Dice, IoU, and pixel accuracy from raw logits."""
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).to(targets.dtype)
    targets = (targets >= 0.5).to(targets.dtype)

    preds = preds.flatten(1)
    targets = targets.flatten(1)

    intersection = (preds * targets).sum(dim=1)
    pred_sum = preds.sum(dim=1)
    target_sum = targets.sum(dim=1)
    union = pred_sum + target_sum - intersection

    dice = ((2.0 * intersection + 1e-6) / (pred_sum + target_sum + 1e-6)).mean().item()
    iou = ((intersection + 1e-6) / (union + 1e-6)).mean().item()
    pixel_accuracy = (preds.eq(targets).to(torch.float32).mean()).item()

    return {
        "dice": dice,
        "iou": iou,
        "pixel_accuracy": pixel_accuracy,
    }
