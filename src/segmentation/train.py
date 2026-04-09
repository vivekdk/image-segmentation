from __future__ import annotations

"""CLI entrypoint for model training.

This script wires together config loading, dataset creation, model
initialization, training/evaluation loops, early stopping, checkpointing, and
artifact logging. Keeping it separate from the lower-level engine functions
makes the training flow easier to read and easier to extend later.
"""

import argparse
from pathlib import Path

import torch

from .config import load_config, save_config
from .dataset import build_dataloaders
from .engine import evaluate, load_checkpoint, save_checkpoint, train_one_epoch
from .model import UNet
from .utils import append_metrics_row, create_run_dir, resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for training."""
    parser = argparse.ArgumentParser(description="Train a U-Net for binary segmentation.")
    parser.add_argument("--config", type=str, required=True, help="Path to the experiment YAML config.")
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint path to resume from.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.data.seed)
    device = resolve_device(config.training.device)

    # Each run gets its own output directory so checkpoints, configs, and
    # visualizations stay grouped and reproducible.
    run_dir = create_run_dir(config)
    save_config(config, run_dir / "config.yaml")
    loaders = build_dataloaders(config)
    train_size = len(loaders["train"].dataset)
    val_size = len(loaders["val"].dataset)

    model = UNet(
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        base_channels=config.model.base_channels,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
    )

    print("Starting training run")
    print(f"  config: {Path(args.config).resolve()}")
    print(f"  run_dir: {run_dir}")
    print(f"  device: {device}")
    print(f"  train_samples: {train_size}")
    print(f"  val_samples: {val_size}")
    print(f"  batch_size: {config.data.batch_size}")
    print(f"  image_size: {config.data.image_size}")
    print(f"  epochs: {config.optimizer.epochs}")
    print(f"  learning_rate: {config.optimizer.lr}")
    print(f"  log_every_n_steps: {config.training.log_every_n_steps}")

    start_epoch = 1
    best_dice = -1.0
    epochs_without_improvement = 0
    if args.resume:
        # Resuming restores both weights and optimizer state so LR schedules and
        # momentum-like optimizer buffers continue from the previous run.
        checkpoint = load_checkpoint(args.resume, model, optimizer=optimizer, map_location=device)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_dice = float(checkpoint.get("metrics", {}).get("dice", -1.0))

    metrics_csv = run_dir / "metrics.csv"
    for epoch in range(start_epoch, config.optimizer.epochs + 1):
        print(f"\nEpoch {epoch:03d}/{config.optimizer.epochs}")
        train_metrics = train_one_epoch(
            model=model,
            loader=loaders["train"],
            optimizer=optimizer,
            device=device,
            threshold=config.training.threshold,
            log_every_n_steps=config.training.log_every_n_steps,
            epoch=epoch,
        )
        visualization_path = None
        if config.training.save_visualizations:
            visualization_path = run_dir / "visualizations" / f"epoch_{epoch:03d}.png"
        val_metrics = evaluate(
            model=model,
            loader=loaders["val"],
            device=device,
            threshold=config.training.threshold,
            visualization_path=visualization_path,
            max_visualizations=config.training.num_visualization_samples,
            split_name="val",
            log_every_n_steps=config.training.log_every_n_steps,
            epoch=epoch,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "train_iou": train_metrics["iou"],
            "train_pixel_accuracy": train_metrics["pixel_accuracy"],
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "val_pixel_accuracy": val_metrics["pixel_accuracy"],
        }
        append_metrics_row(metrics_csv, row)
        print(
            f"Epoch {epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_dice={val_metrics['dice']:.4f} val_iou={val_metrics['iou']:.4f}"
        )

        # `last.pt` is updated every epoch for resumability even if the model is
        # currently worse than the best checkpoint seen so far.
        save_checkpoint(
            run_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=val_metrics,
            metadata={"config_path": str(Path(args.config).resolve())},
        )

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            epochs_without_improvement = 0
            # `best.pt` tracks the checkpoint that maximizes validation Dice,
            # which is a stronger segmentation signal than raw loss alone.
            save_checkpoint(
                run_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                metadata={"config_path": str(Path(args.config).resolve())},
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.training.early_stopping_patience:
                print(f"Stopping early after {epoch} epochs without validation Dice improvement.")
                break

    print(f"Training complete. Best validation Dice: {best_dice:.4f}")
    print(f"Artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
