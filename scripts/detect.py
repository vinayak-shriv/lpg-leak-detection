#!/usr/bin/env python3
"""Run leak detection over a video, with optional overlay video and log.

    python scripts/detect.py                          # interactive window
    python scripts/detect.py --no-display --save-video outputs/annotated.mp4
    python scripts/detect.py --no-display --log outputs/frames.csv

``--no-display`` matters for anything without a desktop session — a server,
a container, or an SSH shell — where cv2.imshow raises rather than opening
a window.
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lpgdetect.paths import DEFAULT_VIDEO, DEFAULT_WEIGHTS  # noqa: E402
from lpgdetect.pipeline import DetectorConfig, run  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--source", type=Path, default=DEFAULT_VIDEO,
                   help="Video file to process")
    p.add_argument("--conf", type=float, default=0.35,
                   help="YOLO confidence threshold per frame")
    p.add_argument("--iou", type=float, default=0.4,
                   help="NMS IoU threshold")
    p.add_argument("--min-confidence", type=float, default=0.30,
                   help="Combined confidence needed to confirm a bubble")
    p.add_argument("--roi-top", type=float, default=0.25,
                   help="Top of the water bath as a fraction of frame height")
    p.add_argument("--roi-bottom", type=float, default=0.92,
                   help="Bottom of the water bath, same units")
    p.add_argument("--cylinders", nargs="*",
                   default=["10E-01", "10E-02", "10E-03", "10E-04", "10E-05"],
                   help="Cylinder labels, left to right across the frame")
    p.add_argument("--no-display", action="store_true",
                   help="Do not open a window (required on headless hosts)")
    p.add_argument("--save-video", type=Path,
                   help="Write the annotated video to this path")
    p.add_argument("--log", type=Path,
                   help="Write per-frame results to this CSV")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    for path, what in ((args.weights, "Weights"), (args.source, "Source")):
        if not path.exists():
            print(f"{what} not found: {path}", file=sys.stderr)
            return 1

    cfg = DetectorConfig(
        weights=args.weights,
        source=args.source,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        min_confidence=args.min_confidence,
        roi_top=args.roi_top,
        roi_bottom=args.roi_bottom,
        cylinder_labels=args.cylinders,
    )

    writer = None
    log_file = None
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        log_file = args.log.open("w", newline="")
        writer = csv.writer(log_file)
        writer.writerow(["frame", "status", "overall_score",
                         "confirmed_bubbles"] + list(args.cylinders))

    peak = 0.0
    frames = flagged = 0
    try:
        for result in run(cfg, display=not args.no_display,
                          output_path=args.save_video):
            frames += 1
            peak = max(peak, result.overall_score)
            if result.status != "CLEAR":
                flagged += 1
            if writer is not None:
                writer.writerow([
                    result.index, result.status,
                    f"{result.overall_score:.4f}", result.confirmed_bubbles,
                    *[f"{s:.4f}" for s in result.cylinder_scores],
                ])
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if log_file is not None:
            log_file.close()

    print(f"\nProcessed {frames} frames")
    print(f"  Peak score:    {peak * 100:.1f}%")
    print(f"  Frames flagged: {flagged}")
    if args.save_video:
        print(f"  Annotated video: {args.save_video}")
    if args.log:
        print(f"  Frame log: {args.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
