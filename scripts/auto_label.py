#!/usr/bin/env python3
"""Pre-label new frames with an existing model, for human review.

Labelling every frame by hand does not scale. Running the current model over
new footage and correcting its output is far faster than starting from an
empty canvas. The output is a *draft* — it inherits the model's blind spots
and must be reviewed before it is used for training.

    python scripts/auto_label.py --images dataset/images/raw

Frames where the model finds nothing produce an empty label file. Those are
valid negatives in YOLO, but a large number of them usually means the
confidence threshold is too high for the new footage.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lpgdetect.paths import (DEFAULT_WEIGHTS, IMAGES_DIR,  # noqa: E402
                             LABELS_DIR, ensure_dir)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def read_image(path: Path):
    """Decode via raw bytes so odd encodings and non-ASCII paths still work.

    cv2.imread returns None silently in those cases, which is easy to mistake
    for an empty frame.
    """
    raw = np.frombuffer(path.read_bytes(), np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--images", type=Path, default=IMAGES_DIR / "raw")
    p.add_argument("--labels", type=Path, default=LABELS_DIR / "raw")
    p.add_argument("--conf", type=float, default=0.40,
                   help="Confidence threshold for a box to be written")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.weights.exists():
        print(f"Weights not found: {args.weights}", file=sys.stderr)
        return 1
    if not args.images.exists():
        print(f"Image directory not found: {args.images}", file=sys.stderr)
        return 1

    from ultralytics import YOLO

    label_dir = ensure_dir(args.labels)
    model = YOLO(str(args.weights))
    files = sorted(p for p in args.images.iterdir()
                   if p.suffix.lower() in IMAGE_SUFFIXES)
    print(f"Auto-labelling {len(files)} frames at conf>={args.conf}")

    labelled = skipped = empty = boxes_written = 0
    for i, path in enumerate(files, 1):
        try:
            image = read_image(path)
            if image is None:
                raise ValueError("could not decode image")
        except Exception as exc:
            print(f"  skipped {path.name}: {exc}", file=sys.stderr)
            skipped += 1
            continue

        try:
            results = model(image, conf=args.conf, verbose=False)[0]
        except Exception as exc:
            print(f"  inference failed on {path.name}: {exc}",
                  file=sys.stderr)
            skipped += 1
            continue

        lines = []
        if results.boxes is not None:
            for box in results.boxes:
                cls = int(box.cls[0])
                # xywhn is centre-x, centre-y, width, height normalised to
                # 0-1: exactly the YOLO label format, so predictions can be
                # written straight back out as training labels.
                cx, cy, bw, bh = (float(v) for v in box.xywhn[0])
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        (label_dir / f"{path.stem}.txt").write_text("\n".join(lines))
        boxes_written += len(lines)
        labelled += 1
        if not lines:
            empty += 1

        if i % 100 == 0:
            print(f"  {i}/{len(files)} processed")

    print(f"\nLabelled {labelled} frames ({boxes_written} boxes), "
          f"{empty} with no detections, {skipped} skipped")
    print(f"Drafts written to {label_dir} — review before training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
