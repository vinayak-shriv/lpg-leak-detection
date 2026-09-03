#!/usr/bin/env python3
"""Normalise mixed-format YOLO labels to plain bounding boxes.

The Roboflow export in this project is a mixed bag: most lines are ordinary
detection boxes (``cls cx cy w h``) but some are segmentation polygons
(``cls x1 y1 x2 y2 ...``). A detection model cannot consume polygons, and
Ultralytics does not reliably reject them, so they quietly distort training.

This script rewrites every polygon as its axis-aligned bounding box and
leaves genuine box lines untouched.

    python scripts/normalize_labels.py --labels dataset/labels/all
    python scripts/normalize_labels.py --labels dataset/labels/all --dry-run
"""
import argparse
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lpgdetect.paths import LABELS_DIR  # noqa: E402


def polygon_to_box(cls: int, coords: List[float]) -> Optional[str]:
    """Convert a normalised polygon to a normalised ``cls cx cy w h`` line."""
    xs = coords[0::2]
    ys = coords[1::2]
    if not xs or not ys:
        return None

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min
    if w <= 0 or h <= 0:
        return None  # degenerate polygon, nothing to train on

    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def convert_file(path: Path) -> tuple[List[str], int, int]:
    """Return normalised lines plus counts of converted and dropped lines."""
    out: List[str] = []
    converted = dropped = 0

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()

        try:
            cls = int(parts[0])
            values = [float(v) for v in parts[1:]]
        except (ValueError, IndexError):
            dropped += 1
            continue

        if len(values) == 4:
            out.append(line)          # already a bounding box
        elif len(values) >= 6 and len(values) % 2 == 0:
            box = polygon_to_box(cls, values)
            if box is None:
                dropped += 1
            else:
                out.append(box)
                converted += 1
        else:
            dropped += 1              # not a shape we can interpret

    return out, converted, dropped


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", type=Path, default=LABELS_DIR / "all",
                   help="Directory of label .txt files to normalise in place")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would change without writing")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.labels.exists():
        print(f"Label directory not found: {args.labels}", file=sys.stderr)
        return 1

    files = sorted(args.labels.glob("*.txt"))
    if not files:
        print(f"No label files in {args.labels}", file=sys.stderr)
        return 1

    files_changed = total_converted = total_dropped = 0
    for path in files:
        lines, converted, dropped = convert_file(path)
        total_converted += converted
        total_dropped += dropped
        if converted or dropped:
            files_changed += 1
            if not args.dry_run:
                path.write_text("\n".join(lines) + ("\n" if lines else ""))

    verb = "would convert" if args.dry_run else "converted"
    print(f"Scanned {len(files)} label files")
    print(f"  {verb} {total_converted} polygon(s) to bounding boxes")
    print(f"  dropped {total_dropped} unparseable line(s)")
    print(f"  {files_changed} file(s) affected")
    if args.dry_run:
        print("\nDry run: nothing was written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
