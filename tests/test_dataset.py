from __future__ import annotations

import numpy as np
from PIL import Image

from segmentation.dataset import SegmentationPairTransform, trimap_to_binary_mask


def test_trimap_to_binary_mask_marks_non_background_pixels() -> None:
    mask = Image.fromarray(np.array([[1, 2], [3, 2]], dtype=np.uint8), mode="L")
    binary = trimap_to_binary_mask(mask)
    assert binary.shape == (1, 2, 2)
    assert binary.tolist() == [[[1.0, 0.0], [1.0, 0.0]]]


def test_pair_transform_returns_normalized_image_and_binary_mask() -> None:
    image = Image.fromarray(np.full((16, 16, 3), 255, dtype=np.uint8), mode="RGB")
    mask = Image.fromarray(np.array([[1] * 16 for _ in range(16)], dtype=np.uint8), mode="L")
    transform = SegmentationPairTransform(image_size=32, augment=False)

    image_tensor, mask_tensor = transform(image, mask)

    assert image_tensor.shape == (3, 32, 32)
    assert mask_tensor.shape == (1, 32, 32)
    assert set(mask_tensor.unique().tolist()) == {1.0}
