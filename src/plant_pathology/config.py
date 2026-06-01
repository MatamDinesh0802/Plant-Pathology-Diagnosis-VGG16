"""Paths, hyperparameters, and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SAMPLE_DIR = DATA_DIR / "samples"
TRAIN_CSV = RAW_DIR / "train.csv"
IMAGES_DIR = RAW_DIR / "images"

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_PATH = REPORTS_DIR / "metrics.json"

RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# VGG16 input size
IMG_SIZE = 224

CLASS_NAMES = ["healthy", "multiple_diseases", "rust", "scab"]
CLASS_TO_INDEX = {c: i for i, c in enumerate(CLASS_NAMES)}
INDEX_TO_CLASS = {i: c for i, c in enumerate(CLASS_NAMES)}

CLASS_COLORS = {
    "healthy":          "#10B981",
    "multiple_diseases": "#F59E0B",
    "rust":             "#EF4444",
    "scab":             "#8B5CF6",
}

CLASS_EMOJI = {
    "healthy":          "🌿",
    "multiple_diseases": "🍃",
    "rust":             "🍂",
    "scab":             "🍁",
}

CLASS_DESCRIPTION = {
    "healthy":          "No disease — vibrant green leaf.",
    "multiple_diseases": "Multiple pathologies present on the same leaf.",
    "rust":             "Cedar-apple rust — yellow/orange spots.",
    "scab":             "Apple scab — dark olive lesions.",
}
