from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from segmentation.engine import evaluate, load_checkpoint, save_checkpoint, train_one_epoch
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


def test_checkpoint_round_trip_restores_model_and_optimizer(tmp_path: Path) -> None:
    model = UNet(in_channels=3, out_channels=1, base_channels=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    inputs = torch.randn(2, 3, 64, 64)
    targets = (torch.rand(2, 1, 64, 64) > 0.5).to(torch.float32)
    loss = segmentation_loss(model(inputs), targets)
    loss.backward()
    optimizer.step()

    saved_model_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    saved_optimizer_state = optimizer.state_dict()
    checkpoint_path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=3,
        metrics={"dice": 0.7, "iou": 0.55, "pixel_accuracy": 0.9, "loss": 0.4},
        metadata={"config_path": "configs/oxford_pet_unet.yaml"},
    )

    restored_model = UNet(in_channels=3, out_channels=1, base_channels=8)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    checkpoint = load_checkpoint(checkpoint_path, restored_model, optimizer=restored_optimizer)

    assert checkpoint["epoch"] == 3
    assert checkpoint["metrics"]["dice"] == 0.7
    assert checkpoint["metadata"]["config_path"] == "configs/oxford_pet_unet.yaml"

    for key, value in restored_model.state_dict().items():
        assert torch.allclose(value, saved_model_state[key])

    restored_optimizer_state = restored_optimizer.state_dict()
    assert restored_optimizer_state["param_groups"] == saved_optimizer_state["param_groups"]
    assert restored_optimizer_state["state"].keys() == saved_optimizer_state["state"].keys()

    for param_id, state in restored_optimizer_state["state"].items():
        saved_state = saved_optimizer_state["state"][param_id]
        assert state.keys() == saved_state.keys()
        for state_key, state_value in state.items():
            saved_value = saved_state[state_key]
            if isinstance(state_value, torch.Tensor):
                assert torch.allclose(state_value, saved_value)
            else:
                assert state_value == saved_value
