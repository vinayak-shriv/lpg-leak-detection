"""Visual overlays: leak-origin heatmap and per-cylinder status bars."""
from typing import List, Sequence, Tuple

import cv2
import numpy as np

# BGR (OpenCV ordering)
COLOR_LEAK = (0, 0, 255)
COLOR_SUSPECT = (0, 165, 255)
COLOR_CLEAR = (0, 200, 80)

SUSPECT_THRESHOLD = 0.35
LEAK_THRESHOLD = 0.65


def status_for(score: float) -> Tuple[str, Tuple[int, int, int]]:
    """Map a 0-1 score to a status label and its display colour."""
    if score > LEAK_THRESHOLD:
        return "LEAK", COLOR_LEAK
    if score > SUSPECT_THRESHOLD:
        return "SUSPECT", COLOR_SUSPECT
    return "CLEAR", COLOR_CLEAR


class LeakHeatmap:
    """Accumulates heat where bubble trails originate.

    Bubbles are tracked from where they first appear, so the start of each
    trail approximates the leak source on the cylinder. Heat accumulates
    there and decays each frame, so a persistent leak glows while one-off
    false positives fade.
    """

    def __init__(self, shape: Tuple[int, int], decay_rate: float = 0.97,
                 render_threshold: float = 0.08):
        self.map = np.zeros(shape[:2], dtype=np.float32)
        self.decay_rate = decay_rate
        self.render_threshold = render_threshold

    def add(self, cx: int, cy: int, strength: float = 0.15,
            radius: int = 20) -> None:
        h, w = self.map.shape
        # Bounded window keeps the mask cheap on large frames instead of
        # allocating a full-frame grid for every bubble.
        x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
        y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
        if x0 >= x1 or y0 >= y1:
            return

        ys, xs = np.ogrid[y0:y1, x0:x1]
        mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius ** 2
        self.map[y0:y1, x0:x1][mask] += strength
        np.clip(self.map, 0.0, 1.0, out=self.map)

    def decay(self) -> None:
        self.map *= self.decay_rate

    def render(self, frame: np.ndarray) -> np.ndarray:
        heat = (self.map * 255).astype(np.uint8)
        coloured = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        mask = self.map > self.render_threshold
        if mask.any():
            blended = cv2.addWeighted(frame, 0.45, coloured, 0.55, 0)
            frame[mask] = blended[mask]
        return frame


class CylinderMonitor:
    """Per-cylinder leak scoring by horizontal zone.

    The frame width is divided into equal vertical slices, one per cylinder
    on the conveyor. Each confirmed bubble raises the score of the slice it
    sits in; scores decay every frame so a cleared cylinder returns to green.
    """

    def __init__(self, frame_width: int, labels: Sequence[str],
                 decay_rate: float = 0.98, gain: float = 0.1):
        self.labels = list(labels)
        self.num = len(self.labels)
        if self.num == 0:
            raise ValueError("CylinderMonitor needs at least one label")
        self.zone_w = max(1, frame_width // self.num)
        self.scores: List[float] = [0.0] * self.num
        self.decay_rate = decay_rate
        self.gain = gain

    def update(self, cx: int, confidence: float) -> None:
        idx = min(int(cx // self.zone_w), self.num - 1)
        self.scores[idx] = min(self.scores[idx] + confidence * self.gain, 1.0)

    def decay(self) -> None:
        self.scores = [s * self.decay_rate for s in self.scores]

    def draw(self, frame: np.ndarray, y_bottom: int) -> np.ndarray:
        for i, (score, label) in enumerate(zip(self.scores, self.labels)):
            x1 = i * self.zone_w
            x2 = x1 + self.zone_w
            status, colour = status_for(score)

            bar_h = int(score * 30)
            cv2.rectangle(frame, (x1 + 2, y_bottom - bar_h),
                          (x2 - 2, y_bottom), colour, -1)
            cv2.rectangle(frame, (x1 + 2, y_bottom - 30),
                          (x2 - 2, y_bottom), colour, 1)
            cv2.putText(frame, label, (x1 + 5, y_bottom + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1)
            cv2.putText(frame, status, (x1 + 5, y_bottom - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1)
        return frame
