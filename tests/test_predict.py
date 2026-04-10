from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from segmentation.predict import iter_input_images, preprocess_image


def test_iter_input_images_returns_single_file_as_list(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(image_path)

    assert iter_input_images(image_path) == [image_path]


def test_iter_input_images_filters_supported_images_and_sorts(tmp_path: Path) -> None:
    valid_names = ["b.JPG", "a.png", "c.tiff"]
    invalid_names = ["notes.txt", "mask.npy"]

    for name in valid_names:
        Image.new("RGB", (8, 8), color=(255, 255, 255)).save(tmp_path / name)
    for name in invalid_names:
        (tmp_path / name).write_text("ignore me", encoding="utf-8")

    paths = iter_input_images(tmp_path)

    assert [path.name for path in paths] == ["a.png", "b.JPG", "c.tiff"]


def test_preprocess_image_resizes_and_normalizes() -> None:
    image = Image.new("RGB", (12, 10), color=(255, 255, 255))

    tensor = preprocess_image(image, image_size=32)

    assert tensor.shape == (3, 32, 32)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()
