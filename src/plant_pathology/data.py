"""Plant Pathology 2020 dataset loader."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import CLASS_NAMES, IMAGES_DIR, TRAIN_CSV


def load_train_dataframe(train_csv: Path = TRAIN_CSV,
                         images_dir: Path = IMAGES_DIR) -> pd.DataFrame:
    """Return a DataFrame with `path`, `label`, `label_idx` from train.csv."""
    if not Path(train_csv).exists():
        raise FileNotFoundError(
            f"train.csv not found at {train_csv}. "
            "Download Plant Pathology 2020 — see data/README.md."
        )
    df = pd.read_csv(train_csv)
    # one-hot → single label
    df["label_idx"] = df[CLASS_NAMES].values.argmax(axis=1)
    df["label"] = df["label_idx"].map(dict(enumerate(CLASS_NAMES)))
    df["path"] = df["image_id"].apply(lambda x: str(Path(images_dir) / f"{x}.jpg"))
    return df[["image_id", "path", "label", "label_idx"]]
