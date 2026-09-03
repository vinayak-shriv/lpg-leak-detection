"""Project paths, resolved relative to the repository root.

Every script imports paths from here instead of hardcoding them, so scripts
work no matter which directory you run them from.
"""
from pathlib import Path

# src/lpgdetect/paths.py -> src/lpgdetect -> src -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = REPO_ROOT / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"

WEIGHTS_DIR = REPO_ROOT / "weights"
DEFAULT_WEIGHTS = WEIGHTS_DIR / "best.pt"

MEDIA_DIR = REPO_ROOT / "media"
DEFAULT_VIDEO = MEDIA_DIR / "test.mp4"

CONFIGS_DIR = REPO_ROOT / "configs"
DATASET_YAML = CONFIGS_DIR / "bubble_dataset.yaml"

RUNS_DIR = REPO_ROOT / "runs"
OUTPUTS_DIR = REPO_ROOT / "outputs"


def ensure_dir(path: Path) -> Path:
    """Create a directory (and parents) if missing, then return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
