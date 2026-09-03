"""LPG water bath leak detection: bubble detection and temporal filtering."""

__version__ = "0.2.0"

from .paths import REPO_ROOT, DEFAULT_WEIGHTS, DEFAULT_VIDEO, DATASET_YAML
from .tracking import BubbleTracker, Detection, compute_confidence
from .overlays import CylinderMonitor, LeakHeatmap, status_for
from .pipeline import DetectorConfig, FrameResult, run

__all__ = [
    "REPO_ROOT", "DEFAULT_WEIGHTS", "DEFAULT_VIDEO", "DATASET_YAML",
    "BubbleTracker", "Detection", "compute_confidence",
    "CylinderMonitor", "LeakHeatmap", "status_for",
    "DetectorConfig", "FrameResult", "run",
]
