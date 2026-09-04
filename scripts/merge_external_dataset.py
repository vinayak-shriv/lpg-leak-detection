#!/usr/bin/env python3
"""Fold Roboflow downloads in dataset/external_raw/ into the main dataset.

Copies every train/valid/test image and label from each source under
dataset/external_raw/<prefix>/, remapping every class index to 0 (this
project uses a single `bubble` class -- see docs/DATASET.md) and renaming
files with the source prefix to avoid collisions, matching the existing
vid_*/rf_* convention.

    python scripts/merge_external_dataset.py
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lpgdetect.paths import DATASET_DIR, IMAGES_DIR, LABELS_DIR, ensure_dir  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SPLITS = ["train", "valid", "test"]


def remap_label(src: Path, dst: Path) -> None:
    """Copy a YOLO label file, forcing every class index to 0."""
    lines = []
    for line in src.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        lines.append(" ".join(["0"] + parts[1:]))
    dst.write_text("\n".join(lines) + ("\n" if lines else ""))


def merge_source(source_dir: Path, prefix: str, img_out: Path,
                 lbl_out: Path) -> int:
    count = 0
    for split in SPLITS:
        img_dir = source_dir / split / "images"
        lbl_dir = source_dir / split / "labels"
        if not img_dir.exists():
            continue
        for img in img_dir.iterdir():
            if img.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            lbl = lbl_dir / (img.stem + ".txt")
            if not lbl.exists():
                continue
            out_name = f"{prefix}_{split}_{img.stem}"
            shutil.copy2(img, img_out / f"{out_name}{img.suffix}")
            remap_label(lbl, lbl_out / f"{out_name}.txt")
            count += 1
    return count


def main() -> int:
    external_raw = DATASET_DIR / "external_raw"
    if not external_raw.exists():
        print(f"No {external_raw} found -- run "
             "scripts/fetch_supplemental_bubble_data.py first.",
             file=sys.stderr)
        return 1

    img_out = ensure_dir(IMAGES_DIR / "all")
    lbl_out = ensure_dir(LABELS_DIR / "all")

    total = 0
    for source_dir in sorted(external_raw.iterdir()):
        if not source_dir.is_dir():
            continue
        prefix = source_dir.name
        n = merge_source(source_dir, prefix, img_out, lbl_out)
        print(f"{prefix}: merged {n} image/label pairs")
        total += n

    print(f"\nMerged {total} pairs total. Run "
         "scripts/split_dataset.py --check-only next to audit, then "
         "scripts/split_dataset.py --val-fraction 0.2 to rebuild the split.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
