from __future__ import annotations

import torch

from segmentation.metrics import compute_binary_metrics


def test_binary_metrics_are_perfect_for_exact_match() -> None:
    targets = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    logits = torch.tensor([[[[10.0, -10.0], [-10.0, 10.0]]]])
    metrics = compute_binary_metrics(logits, targets, threshold=0.5)
    assert metrics["dice"] > 0.999
    assert metrics["iou"] > 0.999
    assert metrics["pixel_accuracy"] > 0.999
