from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from segmentation.engine import evaluate, train_one_epoch
from segmentation.losses import segmentation_loss
from segmentation.model import UNet


def test_unet_forward_shape() -> None:
    model = UNet(in_channels=3, out_channels=1, base_channels=16)
    inputs = torch.randn(2, 3, 128, 128)
    outputs = model(inputs)
    assert outputs.shape == (2, 1, 128, 128)


def test_segmentation_loss_is_finite() -> None:
    logits = torch.randn(2, 1, 32, 32)
    targets = torch.randint(0, 2, (2, 1, 32, 32), dtype=torch.float32)
    loss = segmentation_loss(logits, targets)
    assert torch.isfinite(loss)


def test_train_and_evaluate_smoke() -> None:
    images = torch.randn(4, 3, 64, 64)
    masks = (torch.rand(4, 1, 64, 64) > 0.5).to(torch.float32)
    loader = DataLoader(TensorDataset(images, masks), batch_size=2, shuffle=False)

    model = UNet(in_channels=3, out_channels=1, base_channels=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    train_metrics = train_one_epoch(model, loader, optimizer, device=torch.device("cpu"), threshold=0.5)
    eval_metrics = evaluate(model, loader, device=torch.device("cpu"), threshold=0.5)

    assert train_metrics["loss"] >= 0.0
    assert 0.0 <= eval_metrics["dice"] <= 1.0
    assert 0.0 <= eval_metrics["iou"] <= 1.0
