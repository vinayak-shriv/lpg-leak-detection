#!/usr/bin/env python3
"""Fine-tune YOLOv8 on the bubble dataset.

    python scripts/train.py --epochs 50 --model yolov8n.pt

Augmentation is deliberately constrained by the physics of the problem:
bubbles only ever rise, so rotation and vertical flips would teach the model
motion that cannot occur in a water bath. Horizontal flips and brightness
and scale jitter are kept, since those reflect real variation in camera
placement and tank lighting.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lpgdetect.paths import DATASET_YAML, REPO_ROOT  # noqa: E402


def resolve_device(requested: str) -> str:
    """Pick a device, defaulting to GPU when one is actually present."""
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        return "0"
    print("No CUDA device found, training on CPU (this will be slow).")
    return "cpu"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DATASET_YAML)
    p.add_argument("--model", default="yolov8n.pt",
                   help="Starting checkpoint; yolov8s.pt trades speed for "
                        "accuracy once the pipeline works")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--batch", type=int, default=8,
                   help="Reduce on out-of-memory errors")
    p.add_argument("--device", default="auto",
                   help="'auto', 'cpu', or a CUDA index such as '0'")
    p.add_argument("--name", default="bubble_run",
                   help="Run name under runs/detect/")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.data.exists():
        print(f"Dataset config not found: {args.data}", file=sys.stderr)
        return 1

    from ultralytics import YOLO

    device = resolve_device(args.device)
    model = YOLO(args.model)

    model.train(
        data=str(args.data),
        project=str(REPO_ROOT / "runs" / "detect"),
        name=args.name,
        epochs=args.epochs,
        patience=15,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=4,
        device=device,
        optimizer="AdamW",   # generalises better than SGD on small datasets
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        augment=True,
        hsv_h=0.015,         # slight hue shift for water colour variation
        hsv_s=0.5,
        hsv_v=0.4,           # brightness: tank lighting varies noticeably
        degrees=0.0,         # no rotation: bubbles rise vertically
        translate=0.1,
        scale=0.5,           # zoom: bubbles appear at different distances
        fliplr=0.5,          # horizontal flip is physically plausible
        flipud=0.0,          # vertical flip is not: bubbles never fall
        mosaic=1.0,
        mixup=0.1,
        box=7.5,
        cls=0.5,
        save=True,
        save_period=10,
        plots=True,
        val=True,
        iou=0.5,
        conf=0.25,
    )

    metrics = model.val()
    print("\nValidation results")
    print(f"  mAP50:     {metrics.box.map50:.3f}")
    print(f"  mAP50-95:  {metrics.box.map:.3f}")
    print(f"  Precision: {metrics.box.mp:.3f}")
    print(f"  Recall:    {metrics.box.mr:.3f}")
    print(f"\nWeights: runs/detect/{args.name}/weights/best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
