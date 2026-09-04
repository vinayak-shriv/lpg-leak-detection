#!/usr/bin/env python3
"""Find candidate rising-bubble frames using motion, not a YOLO model.

scripts/auto_label.py drafts labels using the existing YOLO weights -- but
those weights themselves learned to fire on static glare (see
docs/DATASET.md and the 2026-09-04 investigation notes), so bootstrapping
new labels from them just repeats the mistake. This script instead finds
genuinely *moving* bright blobs via background subtraction and the same
rise/straightness logic BubbleTracker already uses for inference, so static
objects are excluded by construction rather than by a learned model.

Candidates are saved as annotated preview crops for human review -- nothing
here is written back into dataset/labels/ automatically.

    python scripts/find_bubble_candidates.py --max-candidates 30
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lpgdetect.paths import REPO_ROOT  # noqa: E402
from lpgdetect.tracking import BubbleTracker, Detection, compute_confidence  # noqa: E402


def find_candidates(video_path: Path, roi_top: float, roi_bottom: float,
                    min_radius: int, max_radius: int, min_trail: int,
                    max_candidates: int, out_dir: Path):
    cap = cv2.VideoCapture(str(video_path))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    y1r, y2r = int(height * roi_top), int(height * roi_bottom)

    backsub = cv2.createBackgroundSubtractorMOG2(
        history=800, varThreshold=24, detectShadows=False)

    tracker = BubbleTracker(max_trail=30, min_trail_for_motion=min_trail)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    frame_index = 0
    reported_ids = set()
    frame_cache = {}

    while saved < max_candidates:
        ok, frame = cap.read()
        if not ok:
            break
        roi = frame[y1r:y2r, :]

        fg = backsub.apply(roi)
        # Warm-up period: MOG2 needs frames before its background model is
        # useful, otherwise the whole first second reads as "foreground".
        if frame_index > 60:
            fg = cv2.medianBlur(fg, 3)
            contours, _ = cv2.findContours(
                fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            detections = []
            for c in contours:
                (cx, cy), r = cv2.minEnclosingCircle(c)
                if min_radius <= r <= max_radius:
                    detections.append(Detection(
                        cx=int(cx), cy=int(cy) + y1r, r=int(r), conf=1.0))

            tracked = tracker.update(detections)
            frame_cache[frame_index] = frame.copy()
            if len(frame_cache) > 40:
                frame_cache.pop(min(frame_cache), None)

            for tid, det in tracked.items():
                if tid in reported_ids:
                    continue
                if not tracker.is_rising(tid):
                    continue
                trail = list(tracker.trails[tid])
                if len(trail) < min_trail:
                    continue
                straight = tracker.trail_straightness(tid)
                if straight < 0.5:
                    continue

                reported_ids.add(tid)
                annotated = frame.copy()
                for i in range(1, len(trail)):
                    cv2.line(annotated, trail[i - 1], trail[i],
                            (0, 220, 255), 2)
                cv2.circle(annotated, (det.cx, det.cy), max(det.r, 8),
                          (0, 255, 0), 2)
                cv2.rectangle(annotated, (0, y1r), (width, y2r),
                             (0, 140, 255), 1)
                out_path = out_dir / f"candidate_{frame_index:06d}_id{tid}.jpg"
                cv2.imwrite(str(out_path), annotated)
                print(f"  saved {out_path.name}  "
                     f"pos=({det.cx},{det.cy})  trail_len={len(trail)}  "
                     f"straightness={straight:.2f}")
                saved += 1
                if saved >= max_candidates:
                    break

        frame_index += 1

    cap.release()
    return saved, frame_index


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", type=Path, default=REPO_ROOT / "media" / "test.mp4")
    p.add_argument("--roi-top", type=float, default=0.25)
    p.add_argument("--roi-bottom", type=float, default=0.92)
    p.add_argument("--min-radius", type=int, default=3)
    p.add_argument("--max-radius", type=int, default=60)
    p.add_argument("--min-trail", type=int, default=5,
                   help="Frames a candidate must be tracked before counting")
    p.add_argument("--max-candidates", type=int, default=30)
    p.add_argument("--out", type=Path,
                   default=REPO_ROOT / "outputs" / "bubble_candidates")
    args = p.parse_args()

    if not args.video.exists():
        print(f"Video not found: {args.video}", file=sys.stderr)
        return 1

    print(f"Scanning {args.video} for motion-based bubble candidates...")
    saved, total_frames = find_candidates(
        args.video, args.roi_top, args.roi_bottom, args.min_radius,
        args.max_radius, args.min_trail, args.max_candidates, args.out)
    print(f"\nScanned {total_frames} frames, saved {saved} candidate(s) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
