from __future__ import annotations

"""Dataset and transform utilities for binary Oxford-IIIT Pet segmentation.

This module performs two important jobs:
1. It adapts the Oxford-IIIT Pet trimap annotations into a binary foreground
   versus background task.
2. It keeps image transforms and mask transforms perfectly aligned so the model
   sees consistent supervision after resizing and augmentation.
"""

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .config import ExperimentConfig
from .utils import IMAGE_MEAN, IMAGE_STD


def trimap_to_binary_mask(mask: Image.Image) -> torch.Tensor:
    """Convert the Oxford-IIIT Pet trimap into a binary segmentation mask.

    The dataset provides trimaps with three labels:
    - 1: foreground
    - 2: background
    - 3: ambiguous boundary / outline

    For a simple binary baseline we treat both foreground and outline as
    positive pixels, which gives the model a slightly more forgiving target at
    object borders and avoids throwing away supervision.
    """
    mask_array = np.array(mask, dtype=np.uint8)
    binary_mask = (mask_array != 2).astype(np.float32)
    return torch.from_numpy(binary_mask).unsqueeze(0)


@dataclass(slots=True)
class SegmentationPairTransform:
    image_size: int
    augment: bool = False

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply paired preprocessing so image and mask stay spatially aligned."""
        image = image.convert("RGB")
        mask_tensor = trimap_to_binary_mask(mask)

        # Masks must always use nearest-neighbor interpolation. Bilinear
        # interpolation would invent fractional class ids and corrupt labels.
        image = TF.resize(image, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
        mask_tensor = TF.resize(mask_tensor, [self.image_size, self.image_size], interpolation=InterpolationMode.NEAREST)

        if self.augment:
            # Every geometric transform must be mirrored on the mask. Any drift
            # between image and label would silently poison training.
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask_tensor = TF.hflip(mask_tensor)

            # Keep augmentation mild because this baseline is designed for local
            # training and quick iteration rather than aggressive regularization.
            angle = random.uniform(-10.0, 10.0)
            scale = random.uniform(0.95, 1.05)
            image = TF.affine(
                image,
                angle=angle,
                translate=[0, 0],
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )
            mask_tensor = TF.affine(
                mask_tensor,
                angle=angle,
                translate=[0, 0],
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.NEAREST,
                fill=0,
            )

            brightness = random.uniform(0.9, 1.1)
            contrast = random.uniform(0.9, 1.1)
            saturation = random.uniform(0.9, 1.1)
            image = TF.adjust_brightness(image, brightness)
            image = TF.adjust_contrast(image, contrast)
            image = TF.adjust_saturation(image, saturation)

        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, IMAGE_MEAN, IMAGE_STD)
        # After resizing and geometric transforms, convert back to a hard binary
        # target so the loss sees only 0/1 labels.
        mask_tensor = (mask_tensor > 0.5).to(torch.float32)
        return image_tensor, mask_tensor


class OxfordPetBinaryDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Thin dataset wrapper that exposes train/val/test splits for binary masks."""

    def __init__(
        self,
        root_dir: str | Path,
        split: str,
        image_size: int = 256,
        val_fraction: float = 0.15,
        seed: int = 42,
        download: bool = True,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        self.split = split
        self.transform = SegmentationPairTransform(image_size=image_size, augment=split == "train")
        root_dir = str(root_dir)

        if split == "test":
            # The official test split is kept intact for a final evaluation pass.
            self.dataset = OxfordIIITPet(root=root_dir, split="test", target_types="segmentation", download=download)
            self.indices = None
            return

        # Torchvision exposes a combined `trainval` split, so we make a stable
        # deterministic train/val split ourselves.
        full_dataset = OxfordIIITPet(root=root_dir, split="trainval", target_types="segmentation", download=download)
        total_size = len(full_dataset)
        indices = list(range(total_size))
        rng = random.Random(seed)
        rng.shuffle(indices)

        val_size = max(1, int(total_size * val_fraction))
        train_size = total_size - val_size
        if train_size <= 0:
            raise ValueError("Validation fraction leaves no training samples.")

        if split == "train":
            self.indices = indices[:train_size]
        else:
            self.indices = indices[train_size:]
        self.dataset = full_dataset

    def __len__(self) -> int:
        if self.indices is None:
            return len(self.dataset)
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        dataset_index = self.indices[index] if self.indices is not None else index
        image, mask = self.dataset[dataset_index]
        image_tensor, mask_tensor = self.transform(image, mask)
        return image_tensor, mask_tensor


def build_dataloaders(config: ExperimentConfig) -> dict[str, DataLoader]:
    """Build the standard train/val/test dataloaders from experiment config."""
    root_dir = config.data.root_dir
    common_kwargs = {
        "root_dir": root_dir,
        "image_size": config.data.image_size,
        "val_fraction": config.data.val_fraction,
        "seed": config.data.seed,
        "download": config.data.download,
    }
    train_dataset = OxfordPetBinaryDataset(split="train", **common_kwargs)
    val_dataset = OxfordPetBinaryDataset(split="val", **common_kwargs)
    test_dataset = OxfordPetBinaryDataset(split="test", **common_kwargs)

    loader_kwargs = {
        "batch_size": config.data.batch_size,
        "num_workers": config.data.num_workers,
        # Pinning memory only helps on CUDA transfers. Keeping it disabled on
        # CPU-only and MPS runs avoids unnecessary complexity.
        "pin_memory": torch.cuda.is_available(),
    }
    return {
        "train": DataLoader(train_dataset, shuffle=True, **loader_kwargs),
        "val": DataLoader(val_dataset, shuffle=False, **loader_kwargs),
        "test": DataLoader(test_dataset, shuffle=False, **loader_kwargs),
    }
