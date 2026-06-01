"""Image inference using the ONNX-exported VGG16."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import CLASS_NAMES, INDEX_TO_CLASS, MODELS_DIR
from .features import preprocess_bytes, preprocess_pil


@dataclass
class PathologyPrediction:
    label: str
    label_index: int
    probability: float
    proba_vector: np.ndarray


class PathologyPredictor:
    """Loads the VGG16 ONNX export and classifies an apple leaf image."""

    def __init__(self, models_dir: Path = MODELS_DIR) -> None:
        import onnxruntime as ort

        onnx_path = Path(models_dir) / "vgg16_model.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"Missing model: {onnx_path}\n"
                "Train the model in Colab (notebooks/01_train_vgg16.ipynb), "
                "then drop vgg16_model.onnx into models/."
            )
        self.session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict_from_pil(self, image) -> PathologyPrediction:
        x = preprocess_pil(image).astype(np.float32)
        return self._run(x)

    def predict_from_bytes(self, image_bytes: bytes) -> PathologyPrediction:
        x = preprocess_bytes(image_bytes).astype(np.float32)
        return self._run(x)

    def _run(self, x: np.ndarray) -> PathologyPrediction:
        outputs = self.session.run(None, {self.input_name: x})
        proba = outputs[0][0]
        # If outputs are logits (no softmax), apply softmax
        if proba.sum() < 0.999 or proba.max() > 1.0001:
            e = np.exp(proba - proba.max())
            proba = e / e.sum()
        idx = int(np.argmax(proba))
        return PathologyPrediction(
            label=INDEX_TO_CLASS[idx],
            label_index=idx,
            probability=float(proba[idx]),
            proba_vector=proba,
        )
