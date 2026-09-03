#!/usr/bin/env python3
"""Evaluate trained weights: dataset metrics and/or an annotated video.

    python scripts/validate.py --metrics              # mAP, precision, recall
    python scripts/validate.py --video                # annotated prediction video

Dataset metrics answer "how good is the detector"; the video answers "does it
look right on real footage". They fail in different ways, so both are useful.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lpgdetect.paths import (DATASET_YAML, DEFAULT_VIDEO,  # noqa: E402
                             DEFAULT_WEIGHTS, REPO_ROOT)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--data", type=Path, default=DATASET_YAML)
    p.add_argument("--source", type=Path, default=DEFAULT_VIDEO)
    p.add_argument("--metrics", action="store_true",
                   help="Compute detection metrics on the val split")
    p.add_argument("--video", action="store_true",
                   help="Write an annotated prediction video")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--iou", type=float, default=0.4)
    p.add_argument("--imgsz", type=int, default=640)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.weights.exists():
        print(f"Weights not found: {args.weights}", file=sys.stderr)
        return 1
    if not (args.metrics or args.video):
        args.metrics = args.video = True  # neither flag given: do both

    from ultralytics import YOLO
    model = YOLO(str(args.weights))

    if args.metrics:
        if not args.data.exists():
            print(f"Dataset config not found: {args.data}", file=sys.stderr)
            return 1
        metrics = model.val(data=str(args.data), imgsz=args.imgsz)
        print("\nDetection metrics")
        print(f"  mAP50:     {metrics.box.map50:.3f}")
        print(f"  mAP50-95:  {metrics.box.map:.3f}")
        print(f"  Precision: {metrics.box.mp:.3f}")
        print(f"  Recall:    {metrics.box.mr:.3f}")

    if args.video:
        if not args.source.exists():
            print(f"Source not found: {args.source}", file=sys.stderr)
            return 1
        # stream=True yields frame by frame instead of holding every result
        # in memory, which matters for anything longer than a short clip.
        results = model.predict(
            source=str(args.source), conf=args.conf, iou=args.iou,
            imgsz=args.imgsz, save=True, save_txt=True, stream=True,
            project=str(REPO_ROOT / "runs" / "detect"), name="predict",
        )
        frames = sum(1 for _ in results)
        print(f"\nAnnotated {frames} frames -> runs/detect/predict/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
