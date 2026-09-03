#!/usr/bin/env python3
"""Upload frames to a Roboflow project for labelling.

The API key is read from the environment, never stored in the file:

    export ROBOFLOW_API_KEY="..."      # Windows: setx ROBOFLOW_API_KEY "..."
    python scripts/upload_to_roboflow.py --images dataset/images/raw
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lpgdetect.paths import IMAGES_DIR  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", type=Path, default=IMAGES_DIR / "raw")
    p.add_argument("--workspace", default=os.environ.get("ROBOFLOW_WORKSPACE"))
    p.add_argument("--project", default=os.environ.get("ROBOFLOW_PROJECT"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("Set ROBOFLOW_API_KEY in your environment first.",
              file=sys.stderr)
        return 1
    if not (args.workspace and args.project):
        print("Provide --workspace and --project (or set ROBOFLOW_WORKSPACE "
              "and ROBOFLOW_PROJECT).", file=sys.stderr)
        return 1
    if not args.images.exists():
        print(f"Image directory not found: {args.images}", file=sys.stderr)
        return 1

    from roboflow import Roboflow

    project = (Roboflow(api_key=api_key)
               .workspace(args.workspace)
               .project(args.project))
    project.upload(str(args.images))
    print(f"Uploaded {args.images} to {args.workspace}/{args.project}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
