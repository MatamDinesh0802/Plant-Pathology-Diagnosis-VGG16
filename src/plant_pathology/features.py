"""Image preprocessing for VGG16."""
from __future__ import annotations

import numpy as np

from .config import IMG_SIZE

# VGG16 was trained with caffe-style preprocessing — BGR + mean subtraction.
# Mean RGB values from ImageNet (in RGB order here; we'll reverse to BGR).
_VGG_MEAN_RGB = np.array([123.68, 116.779, 103.939], dtype=np.float32)


def preprocess_pil(img, size: int = IMG_SIZE) -> np.ndarray:
    """Convert a PIL image to a (1, H, W, 3) float32 tensor in VGG16 format."""
    img = img.convert("RGB").resize((size, size))
    arr = np.asarray(img, dtype=np.float32)            # H, W, 3 (RGB)
    arr = arr[..., ::-1]                                # RGB → BGR
    arr -= _VGG_MEAN_RGB[::-1]                          # subtract BGR mean
    return arr[None, ...]                               # add batch dim


def preprocess_bytes(image_bytes: bytes, size: int = IMG_SIZE) -> np.ndarray:
    from PIL import Image
    import io

    return preprocess_pil(Image.open(io.BytesIO(image_bytes)), size=size)
