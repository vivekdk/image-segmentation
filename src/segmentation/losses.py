from __future__ import annotations

"""Loss functions for binary segmentation."""

import torch
import torch.nn.functional as F


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute soft Dice loss directly from logits.

    Dice complements BCE because it measures overlap at the mask level rather
    than only per-pixel classification, which usually helps segmentation models
    focus on shape quality.
    """
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    targets = targets.flatten(1)
    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def segmentation_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Blend BCE and Dice so optimization sees both local and global errors."""
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_loss_from_logits(logits, targets)
    return 0.5 * bce + 0.5 * dice
