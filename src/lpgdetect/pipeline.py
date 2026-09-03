"""End-to-end leak detection over a video source."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .overlays import CylinderMonitor, LeakHeatmap, status_for
from .tracking import BubbleTracker, Detection, compute_confidence


@dataclass
class DetectorConfig:
    """Tunable parameters for a detection run.

    The ROI fractions and cylinder labels are camera- and plant-specific;
    they are exposed here rather than hardcoded so a different tank layout
    only needs a config change.
    """

    weights: Path
    source: Path
    conf_threshold: float = 0.35
    iou_threshold: float = 0.4
    min_confidence: float = 0.30
    min_radius: int = 3
    max_radius: int = 60
    roi_top: float = 0.25
    roi_bottom: float = 0.92
    max_trail: int = 30
    cylinder_labels: Sequence[str] = field(
        default_factory=lambda: ["10E-01", "10E-02", "10E-03", "10E-04",
                                 "10E-05"]
    )


@dataclass
class FrameResult:
    """Per-frame summary, independent of any drawing."""

    index: int
    overall_score: float
    status: str
    confirmed_bubbles: int
    cylinder_scores: List[float]


def _roi_bounds(cfg: DetectorConfig, height: int,
                width: int) -> Tuple[int, int, int, int]:
    return 0, int(height * cfg.roi_top), width, int(height * cfg.roi_bottom)


def _extract_detections(results, cfg: DetectorConfig, x_off: int,
                        y_off: int) -> List[Detection]:
    """Convert YOLO boxes into full-frame Detections, filtering by size."""
    detections: List[Detection] = []
    for box in results.boxes:
        bx1, by1, bx2, by2 = map(int, box.xyxy[0])
        radius = max(bx2 - bx1, by2 - by1) // 2
        # Size gate: sub-pixel specks and tank-sized blobs are not bubbles.
        if radius < cfg.min_radius or radius > cfg.max_radius:
            continue
        detections.append(
            Detection(
                cx=(bx1 + bx2) // 2 + x_off,
                cy=(by1 + by2) // 2 + y_off,
                r=radius,
                conf=float(box.conf[0]),
                box=(bx1 + x_off, by1 + y_off, bx2 + x_off, by2 + y_off),
            )
        )
    return detections


def _draw_bubble(frame: np.ndarray, det: Detection, conf: float,
                 trail: Sequence[Tuple[int, int]]) -> None:
    for i in range(1, len(trail)):
        fade = i / len(trail)
        cv2.line(frame, trail[i - 1], trail[i],
                 (int(50 * fade), int(220 * fade), int(255 * fade)), 1)

    if conf > 0.65:
        colour = (0, 255, 80)
    elif conf > 0.40:
        colour = (0, 210, 255)
    else:
        colour = (80, 80, 255)

    cv2.circle(frame, (det.cx, det.cy), det.r, colour, 2)
    cv2.putText(frame, f"{conf * 100:.0f}%", (det.cx - det.r,
                det.cy - det.r - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                colour, 1)


def run(cfg: DetectorConfig, display: bool = True,
        output_path: Optional[Path] = None,
        annotate: bool = True) -> Iterator[FrameResult]:
    """Run detection frame by frame, yielding a result per frame.

    Yielding rather than returning lets callers stream results to a log, a
    UI, or a plant alarm without holding the whole video in memory.
    """
    from ultralytics import YOLO  # imported lazily so --help stays fast

    model = YOLO(str(cfg.weights))
    cap = cv2.VideoCapture(str(cfg.source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {cfg.source}")

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    tracker = BubbleTracker(max_trail=cfg.max_trail)
    heatmap = LeakHeatmap((height, width))
    monitor = CylinderMonitor(width, cfg.cylinder_labels)
    x1r, y1r, x2r, y2r = _roi_bounds(cfg, height, width)

    writer = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
            (width, height)
        )

    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            roi = frame[y1r:y2r, x1r:x2r]
            results = model(roi, conf=cfg.conf_threshold,
                            iou=cfg.iou_threshold, verbose=False)[0]
            detections = _extract_detections(results, cfg, x1r, y1r)
            tracked = tracker.update(detections)

            confidences: List[float] = []
            for tid, det in tracked.items():
                # A non-rising track is rejected inside compute_confidence,
                # which returns 0.0 and fails the min_confidence gate below.
                conf = compute_confidence(det, tracker, tid)
                if conf < cfg.min_confidence:
                    continue

                confidences.append(conf)
                monitor.update(det.cx, conf)

                trail = list(tracker.trails[tid])
                if annotate:
                    _draw_bubble(frame, det, conf, trail)
                if conf > 0.4 and len(trail) >= 2:
                    heatmap.add(trail[0][0], trail[0][1], conf * 0.15,
                                radius=18)

            heatmap.decay()
            monitor.decay()

            overall = float(np.mean(confidences)) if confidences else 0.0
            status, colour = status_for(overall)

            if annotate:
                frame = heatmap.render(frame)
                frame = monitor.draw(frame, y2r)
                cv2.putText(frame, f"{status}  {overall * 100:.1f}%",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                            colour, 2)
                cv2.putText(frame, f"Rising bubbles: {len(confidences)}",
                            (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (180, 180, 180), 1)
                cv2.rectangle(frame, (x1r, y1r), (x2r, y2r),
                              (0, 140, 255), 2)

            if writer is not None:
                writer.write(frame)

            if display:
                cv2.imshow("LPG Leak Detector", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            yield FrameResult(
                index=frame_index,
                overall_score=overall,
                status=status,
                confirmed_bubbles=len(confidences),
                cylinder_scores=list(monitor.scores),
            )
            frame_index += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()
