#!/usr/bin/env python3
"""Extract frames from water bath footage for labelling.

Consecutive video frames are nearly identical, so sampling at a fixed time
interval rather than every frame avoids a dataset full of duplicates.

    python scripts/export_frames.py --every 0.5
"""
import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lpgdetect.paths import DEFAULT_VIDEO, IMAGES_DIR, ensure_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", type=Path, default=DEFAULT_VIDEO,
                   help="Source video (default: media/test.mp4)")
    p.add_argument("--out", type=Path, default=IMAGES_DIR / "raw",
                   help="Output directory for extracted frames")
    p.add_argument("--every", type=float, default=0.5,
                   help="Seconds between saved frames (default: 0.5)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.video.exists():
        print(f"Video not found: {args.video}", file=sys.stderr)
        return 1

    out_dir = ensure_dir(args.out)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"Cannot open video: {args.video}", file=sys.stderr)
        return 1

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval = max(1, int(fps * args.every))
    print(f"{args.video.name}: {total} frames @ {fps:.1f} fps "
          f"-> saving every {interval} frames")

    saved = index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % interval == 0:
            cv2.imwrite(str(out_dir / f"frame_{index:06d}.jpg"), frame)
            saved += 1
        index += 1

    cap.release()
    print(f"Saved {saved} frames to {out_dir}")
    print("Next: label them (Roboflow / labelImg), then run split_dataset.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
