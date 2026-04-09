from __future__ import annotations

"""CLI entrypoint for standalone inference on one image or a directory."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .config import load_config
from .engine import load_checkpoint
from .model import UNet
from .utils import IMAGE_MEAN, IMAGE_STD, resolve_device
from .visualization import save_mask_and_overlay


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for batch or single-image prediction."""
    parser = argparse.ArgumentParser(description="Run segmentation inference on images.")
    parser.add_argument("--config", type=str, required=True, help="Path to the experiment YAML config.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--input", type=str, required=True, help="Path to an image file or directory.")
    parser.add_argument("--output", type=str, required=True, help="Directory for output masks and overlays.")
    return parser.parse_args()


def preprocess_image(image: Image.Image, image_size: int) -> torch.Tensor:
    """Resize and normalize an image so it matches training-time preprocessing."""
    resized = TF.resize(image.convert("RGB"), [image_size, image_size], interpolation=InterpolationMode.BILINEAR)
    tensor = TF.to_tensor(resized)
    return TF.normalize(tensor, IMAGE_MEAN, IMAGE_STD)


def iter_input_images(input_path: str | Path) -> list[Path]:
    """Return a sorted list of image files from a path or directory."""
    path = Path(input_path)
    if path.is_file():
        return [path]
    valid_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted([candidate for candidate in path.iterdir() if candidate.suffix.lower() in valid_suffixes])


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(config.training.device)

    # Prediction rebuilds the architecture from config, then loads learned
    # weights from the checkpoint exactly like evaluation does.
    model = UNet(
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        base_channels=config.model.base_channels,
    ).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    output_dir = Path(args.output)
    images = iter_input_images(args.input)
    if not images:
        raise FileNotFoundError("No input images found.")

    with torch.no_grad():
        for image_path in images:
            original = Image.open(image_path).convert("RGB")
            input_tensor = preprocess_image(original, config.data.image_size).unsqueeze(0).to(device)
            logits = model(input_tensor)
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
            pred_mask = (probs >= config.training.threshold).astype(np.float32)
            # Resize the predicted mask back to the original image resolution so
            # saved outputs line up with the input image on disk.
            pred_mask = np.array(
                Image.fromarray((pred_mask * 255).astype(np.uint8), mode="L").resize(
                    original.size, resample=Image.Resampling.NEAREST
                ),
                dtype=np.float32,
            )
            pred_mask = (pred_mask > 127).astype(np.float32)

            stem = image_path.stem
            save_mask_and_overlay(
                original=original,
                pred_mask=pred_mask,
                mask_output_path=output_dir / f"{stem}_mask.png",
                overlay_output_path=output_dir / f"{stem}_overlay.png",
            )
            print(f"Saved predictions for {image_path.name}")


if __name__ == "__main__":
    main()
