from __future__ import annotations

"""CLI entrypoint for evaluating a trained checkpoint on the test split."""

import argparse

from .config import load_config
from .dataset import build_dataloaders
from .engine import evaluate, load_checkpoint
from .model import UNet
from .utils import resolve_device


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for checkpoint evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a segmentation checkpoint.")
    parser.add_argument("--config", type=str, required=True, help="Path to the experiment YAML config.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(config.training.device)
    loaders = build_dataloaders(config)

    # Evaluation reconstructs the model from config to ensure the checkpoint is
    # compatible with the declared experiment shape.
    model = UNet(
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        base_channels=config.model.base_channels,
    ).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)

    metrics = evaluate(
        model=model,
        loader=loaders["test"],
        device=device,
        threshold=config.training.threshold,
        split_name="test",
        log_every_n_steps=config.training.log_every_n_steps,
    )
    print(
        "Test metrics "
        f"loss={metrics['loss']:.4f} "
        f"dice={metrics['dice']:.4f} "
        f"iou={metrics['iou']:.4f} "
        f"pixel_accuracy={metrics['pixel_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
