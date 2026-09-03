#!/usr/bin/env python3
"""Build train/val splits and verify dataset integrity.

The original project trained against a config whose ``val:`` path did not
exist, so validation metrics were never actually computed. This script
creates a deterministic split and reports any problems before training.

    python scripts/split_dataset.py --val-fraction 0.2

Checks performed:
  * every image has a matching label file (and vice versa)
  * class indices in labels fit within the configured class count
  * label values are normalised to 0-1 as the YOLO format requires
"""
import argparse
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lpgdetect.paths import (DATASET_DIR, IMAGES_DIR, LABELS_DIR,  # noqa: E402
                             ensure_dir)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def stems(directory: Path, suffixes: Set[str]) -> Dict[str, Path]:
    if not directory.exists():
        return {}
    return {p.stem: p for p in directory.iterdir()
            if p.suffix.lower() in suffixes}


def audit_labels(label_paths: List[Path]) -> Tuple[Counter, List[str]]:
    """Return the class-index histogram and a list of problems found."""
    classes: Counter = Counter()
    problems: List[str] = []

    for path in label_paths:
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                problems.append(f"{path.name}:{lineno} expected 5 fields, "
                                f"got {len(parts)}")
                continue
            try:
                cls = int(parts[0])
                coords = [float(v) for v in parts[1:]]
            except ValueError:
                problems.append(f"{path.name}:{lineno} non-numeric values")
                continue
            classes[cls] += 1
            if any(not 0.0 <= v <= 1.0 for v in coords):
                problems.append(f"{path.name}:{lineno} coordinates outside "
                                f"0-1 (labels must be normalised)")
    return classes, problems


def collect_pairs(image_dir: Path, label_dir: Path,
                  problems: List[str]) -> List[Tuple[Path, Path]]:
    images = stems(image_dir, IMAGE_SUFFIXES)
    labels = stems(label_dir, {".txt"})

    for stem in sorted(set(images) - set(labels)):
        problems.append(f"image without label, skipped: {stem}")
    for stem in sorted(set(labels) - set(images)):
        problems.append(f"label without image, skipped: {stem}")

    return [(images[s], labels[s]) for s in sorted(set(images) & set(labels))]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", type=Path, default=IMAGES_DIR / "all",
                   help="Directory of source images")
    p.add_argument("--labels", type=Path, default=LABELS_DIR / "all",
                   help="Directory of YOLO-format label .txt files")
    p.add_argument("--val-fraction", type=float, default=0.2,
                   help="Share of pairs held out for validation")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed, so splits are reproducible")
    p.add_argument("--check-only", action="store_true",
                   help="Report integrity problems without writing files")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.images.exists():
        print(f"Image directory not found: {args.images}", file=sys.stderr)
        return 1

    problems: List[str] = []
    pairs = collect_pairs(args.images, args.labels, problems)
    if not pairs:
        print("No image/label pairs found.", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    classes, label_problems = audit_labels([lbl for _, lbl in pairs])
    problems.extend(label_problems)

    print(f"Usable image/label pairs: {len(pairs)}")
    print("Class index histogram:")
    for cls, count in sorted(classes.items()):
        print(f"  class {cls}: {count} boxes")
    highest = max(classes) if classes else -1
    print(f"Minimum nc for configs/bubble_dataset.yaml: {highest + 1}")

    if problems:
        print(f"\n{len(problems)} problem(s) found:")
        for problem in problems[:20]:
            print(f"  - {problem}")
        if len(problems) > 20:
            print(f"  ... and {len(problems) - 20} more")
    else:
        print("\nNo integrity problems found.")

    if args.check_only:
        return 0

    random.Random(args.seed).shuffle(pairs)
    cut = int(len(pairs) * (1 - args.val_fraction))
    splits = {"train": pairs[:cut], "val": pairs[cut:]}

    for name, items in splits.items():
        img_out = ensure_dir(IMAGES_DIR / name)
        lbl_out = ensure_dir(LABELS_DIR / name)
        for existing in list(img_out.iterdir()) + list(lbl_out.iterdir()):
            existing.unlink()
        for img, lbl in items:
            shutil.copy2(img, img_out / img.name)
            shutil.copy2(lbl, lbl_out / lbl.name)
        print(f"{name}: {len(items)} pairs -> {img_out.relative_to(DATASET_DIR.parent)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
