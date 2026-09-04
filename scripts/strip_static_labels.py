#!/usr/bin/env python3
"""Remove mislabeled static-object boxes from the vid_* label set.

Investigation (2026-09-04) found that retrained models consistently fired on
a small number of fixed screen positions (glare, fixed rig components)
instead of real rising bubbles. Checking the ground-truth labels showed why:
99.5% of all boxes across the 513 vid_* frames sit within ~20px of just a
handful of static positions. auto_label.py's drafts were apparently never
actually corrected for this during review (see docs/DATASET.md).

This script finds those static clusters directly from the label data (a
position hit in a large fraction of frames, within a small radius, cannot be
a moving bubble) and drops boxes near them, leaving genuine, varied
detections in place. Frames left with zero boxes become background (no
`bubble` instance) rather than being deleted -- valid, useful negatives for
training.

    python scripts/strip_static_labels.py --check-only   # report only
    python scripts/strip_static_labels.py                # rewrite labels
"""
import argparse
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lpgdetect.paths import LABELS_DIR  # noqa: E402

FRAME_W, FRAME_H = 1280, 720


def load_boxes(path: Path) -> List[Tuple[str, float, float, float, float]]:
    boxes = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        boxes.append(tuple(parts))
    return boxes


def pixel_center(box) -> Tuple[float, float]:
    _, cx, cy, _, _ = box
    return float(cx) * FRAME_W, float(cy) * FRAME_H


def find_static_clusters(all_centers: List[Tuple[float, float]], n_frames: int,
                         radius: float, min_frame_fraction: float) -> List[Tuple[float, float]]:
    """Greedily find dense position clusters hit in many separate frames.

    A cluster is "static" when a large fraction of all frames have a box
    within `radius` px of it -- real bubbles rise, so no single pixel
    position should recur across a large share of frames.
    """
    remaining = list(all_centers)
    clusters = []
    min_hits = int(n_frames * min_frame_fraction)

    while remaining:
        best_center, best_count = None, 0
        for cx, cy in remaining:
            count = sum(1 for px, py in remaining
                       if (px - cx) ** 2 + (py - cy) ** 2 <= radius ** 2)
            if count > best_count:
                best_center, best_count = (cx, cy), count

        if best_count < min_hits:
            break

        clusters.append(best_center)
        remaining = [(px, py) for px, py in remaining
                    if (px - best_center[0]) ** 2 + (py - best_center[1]) ** 2 > radius ** 2]

    return clusters


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", type=Path, default=LABELS_DIR / "all")
    p.add_argument("--radius", type=float, default=20.0,
                   help="Px radius around a static cluster to strip")
    p.add_argument("--min-frame-fraction", type=float, default=0.10,
                   help="A position hit in at least this share of frames is static")
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args()

    vid_files = sorted(args.labels.glob("vid_*.txt"))
    if not vid_files:
        print(f"No vid_*.txt files found under {args.labels}", file=sys.stderr)
        return 1

    file_boxes = {f: load_boxes(f) for f in vid_files}
    all_centers = [pixel_center(b) for boxes in file_boxes.values() for b in boxes]
    total_boxes = len(all_centers)

    clusters = find_static_clusters(all_centers, len(vid_files),
                                    args.radius, args.min_frame_fraction)

    print(f"{len(vid_files)} vid_ label files, {total_boxes} total boxes")
    print(f"Found {len(clusters)} static cluster(s) "
         f"(radius={args.radius}px, min_frame_fraction={args.min_frame_fraction}):")
    for cx, cy in clusters:
        print(f"  ({cx:.0f}, {cy:.0f})")

    kept_total = 0
    dropped_total = 0
    now_empty = 0
    for f, boxes in file_boxes.items():
        kept = []
        for box in boxes:
            cx, cy = pixel_center(box)
            is_static = any((cx - sx) ** 2 + (cy - sy) ** 2 <= args.radius ** 2
                           for sx, sy in clusters)
            if is_static:
                dropped_total += 1
            else:
                kept.append(box)
        kept_total += len(kept)
        if not kept:
            now_empty += 1
        if not args.check_only:
            f.write_text("\n".join(" ".join(b) for b in kept) +
                        ("\n" if kept else ""))

    print(f"\nKept {kept_total} boxes, dropped {dropped_total} "
         f"({100 * dropped_total / total_boxes:.1f}%)")
    print(f"{now_empty} of {len(vid_files)} frames now have zero boxes "
         f"(kept as background negatives)")

    if args.check_only:
        print("\n--check-only: no files were modified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
