"""Smoke tests for the image preprocessing pipeline."""
import numpy as np
from PIL import Image

from src.plant_pathology.config import IMG_SIZE
from src.plant_pathology.features import preprocess_pil


def _synth_image() -> Image.Image:
    """A random RGB image (e.g. simulating a leaf photo)."""
    arr = (np.random.rand(300, 400, 3) * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def test_preprocess_shape():
    img = _synth_image()
    x = preprocess_pil(img)
    assert x.shape == (1, IMG_SIZE, IMG_SIZE, 3)
    assert x.dtype == np.float32


def test_preprocess_mean_subtraction():
    # All-zero input → values should be the negative BGR mean
    arr = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    x = preprocess_pil(img)
    # After BGR mean subtraction, all values should be < 0
    assert x.max() < 0
