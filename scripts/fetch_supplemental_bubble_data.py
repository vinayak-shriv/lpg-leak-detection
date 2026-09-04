#!/usr/bin/env python3
"""Download supplemental public bubble-detection datasets from Roboflow Universe.

The project's own vid_* labels turned out to be almost entirely mislabeled
static objects, not real bubbles (see docs/DATASET.md and the 2026-09-04
investigation notes) -- there's effectively no genuine positive signal from
the bundled media/test.mp4. This pulls in additional real, licensed
underwater/leak-test bubble datasets to widen what the model actually learns
"bubble" to look like, beyond the single existing rf_ (Air Leak Bubble
Detection) source.

The API key is read from the environment, never stored in the file:

    export ROBOFLOW_API_KEY="..."      # Windows: setx ROBOFLOW_API_KEY "..."
    python scripts/fetch_supplemental_bubble_data.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lpgdetect.paths import DATASET_DIR  # noqa: E402

# (workspace, project, version, prefix, license)
SOURCES = [
    ("techit-sritrakul", "bubble-leak-test-in-oil-cooler", 4, "oil_",
     "Public Domain"),
    ("deakin-hbdei", "bubble-dataset-kkrbu", 1, "deakin_", "CC BY 4.0"),
]


def main() -> int:
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("Set ROBOFLOW_API_KEY in your environment first.",
             file=sys.stderr)
        return 1

    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)

    staging = DATASET_DIR / "external_raw"
    staging.mkdir(parents=True, exist_ok=True)

    for workspace, project_name, version, prefix, license_name in SOURCES:
        print(f"\n=== {workspace}/{project_name} v{version} ({license_name}) ===")
        project = rf.workspace(workspace).project(project_name)
        dataset = project.version(version).download(
            "yolov8", location=str(staging / prefix.rstrip("_")))
        print(f"Downloaded to {dataset.location}")

    print("\nDone. Run scripts/merge_external_dataset.py next to fold these "
         "into dataset/images/all and dataset/labels/all with prefixed "
         "filenames.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
