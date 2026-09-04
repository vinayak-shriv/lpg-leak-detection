# LPG Water Bath Leak Detection

Detects leaking LPG cylinders from water bath test footage by finding rising
bubbles with a YOLOv8 detector and confirming them with motion analysis
across frames.

Built during an internship in Process Engineering & Quality Assurance at an
HPCL LPG bottling plant. **Prototype — not a validated safety system.** See
[Limitations](#limitations).

## The problem

Filled cylinders are submerged in a water bath and an operator watches for
bubbles escaping from the valve or weld seam. It is a monotonous visual task
on a moving conveyor, and attention drifts.

Single-frame detection alone does not solve it. A water bath is full of
things that look like bubbles to a detector: surface glare, reflections of
overhead lighting, condensation, and specks on the tank glass. What
distinguishes a real leak is not appearance but **behaviour** — leak bubbles
rise, and they rise in a near-straight vertical line.

So the system works in two stages: YOLO proposes candidates in each frame,
then a tracking layer confirms or rejects them based on how they move.

## How detection works

1. **Region of interest** — the frame is cropped to the water bath, so the
   model never sees the tank rim or the floor.
2. **Per-frame detection** — YOLOv8 proposes bubble boxes above a confidence
   threshold. Boxes outside a plausible radius range are discarded.
3. **Tracking** — each detection is matched to the nearest track from the
   previous frame, building a trail of positions per bubble.
4. **Motion gating** — a track that is not rising is rejected outright,
   whatever the model's confidence. This is what removes glare and static
   specks.
5. **Confidence scoring** — surviving tracks are scored on a blend of
   detector confidence (45%), how many frames the track has persisted (30%),
   and how straight its trail is (25%).
6. **Aggregation** — confirmed bubbles feed a decaying per-cylinder score and
   a heatmap of trail origins, which approximates the leak point.

The frame width is split into equal zones, one per cylinder, so the output
identifies *which* cylinder is leaking rather than just that something is.

## Quick start

```bash
git clone <your-repo-url>
cd lpg-leak-detection
pip install -r requirements.txt

# Run on the included sample footage
python scripts/detect.py

# Headless (server, container, SSH — no display available)
python scripts/detect.py --no-display --save-video outputs/annotated.mp4
```

Detection needs only `weights/best.pt`, which is committed. Before
*training*, build the splits once — they are generated from
`dataset/{images,labels}/all/` rather than committed, to keep the repository
from carrying every image twice:

```bash
python scripts/split_dataset.py
```

## Repository layout

```
src/lpgdetect/        Library code
  paths.py            Repo-root-relative paths
  tracking.py         BubbleTracker, confidence scoring
  overlays.py         Heatmap and per-cylinder status bars
  pipeline.py         Frame loop, yields per-frame results
scripts/              Command-line entry points
tests/                Unit tests for tracking and scoring
configs/              Dataset config for training
weights/best.pt       Trained detector
dataset/              Labelled images and YOLO labels
media/test.mp4        Sample water bath footage
notebooks/            Original Colab exploration
```

Everything in `src/` is importable and testable on its own; `scripts/` only
handles argument parsing and I/O.

## Scripts

| Script | Purpose |
|---|---|
| `export_frames.py` | Sample frames from footage at a fixed time interval |
| `auto_label.py` | Draft labels for new frames using existing weights |
| `find_bubble_candidates.py` | Find candidate bubble frames via motion (background subtraction), not a YOLO model — avoids bootstrapping label errors from `auto_label.py` |
| `strip_static_labels.py` | Detect and remove labels clustered at fixed screen positions (mislabeled static objects, not bubbles) |
| `fetch_supplemental_bubble_data.py` | Download additional licensed bubble datasets from Roboflow Universe (needs `ROBOFLOW_API_KEY`) |
| `merge_external_dataset.py` | Fold a downloaded external dataset into `dataset/{images,labels}/all`, remapping classes to `0` |
| `normalize_labels.py` | Convert segmentation polygons to bounding boxes |
| `split_dataset.py` | Build train/val splits, audit dataset integrity |
| `train.py` | Fine-tune YOLOv8 on the bubble dataset |
| `validate.py` | Detection metrics and/or annotated prediction video |
| `detect.py` | Full pipeline on a video, with CSV logging |
| `upload_to_roboflow.py` | Push frames to Roboflow for labelling |

Every script supports `--help`.

## Retraining

```bash
python scripts/export_frames.py --video media/new_footage.mp4
python scripts/auto_label.py --images dataset/images/raw   # draft labels
# review and correct the drafts before continuing
python scripts/normalize_labels.py --labels dataset/labels/all
python scripts/split_dataset.py --check-only               # audit first
python scripts/split_dataset.py --val-fraction 0.2
python scripts/train.py --epochs 50
```

`split_dataset.py --check-only` reports orphaned images, malformed lines,
un-normalised coordinates, and the class-index histogram. Run it before
training rather than after a wasted run.

## Design notes

**Why a detector rather than classical CV?** Hough circles and blob
detection need thresholds retuned for every lighting condition. Bubbles vary
in size, overlap each other, and sit against a moving reflective background.
A learned detector handles that variation better.

**Why not trust YOLO confidence alone?** A single frame cannot distinguish a
bubble from a reflection — they can look nearly identical. Only motion
separates them, and motion needs multiple frames.

**Why a hand-written tracker?** Bubbles are small, numerous, and move with
roughly constant upward velocity. Nearest-centroid matching is sufficient
and adds no dependency. A Kalman-filter tracker would handle occlusion and
crossing paths better; see Limitations.

**Augmentation follows the physics.** Rotation and vertical flips are
disabled during training because bubbles only rise — augmenting with falling
bubbles would teach motion that cannot occur. Horizontal flips, brightness
and scale jitter are kept, since those reflect real camera and lighting
variation.

## Limitations

- **Prototype, not validated.** No measured false-negative rate against
  known-leaking cylinders. It has not been trialled against the manual
  inspection it would supplement, and must not replace it.
- **Fixed camera assumptions.** ROI fractions and the five cylinder zones
  assume one camera placement and tank layout. They are CLI arguments rather
  than constants, but a different setup still needs retuning.
- **Tracking is simple.** Nearest-centroid matching with a fixed 50 px
  threshold degrades when bubbles cross paths, when a bubble is briefly
  occluded, or at a frame rate very different from the development footage.
  SORT or DeepSORT would be the next step.
- **Single class.** The model detects "bubble" only. Leak severity is
  inferred from bubble rate and per-cylinder score, not learned. Telling a
  slow weep from a fast leak reliably would need severity-labelled data.
- **Dataset is small and narrow.** Training data comes from one video plus a
  public bubble dataset. Generalisation to other tanks, water clarity, and
  lighting is untested.
- **Confidence weights are hand-tuned.** The 45/30/25 blend was chosen by
  inspection on the sample footage, not fitted or validated.

## Dataset

Four sources are merged in `dataset/` — see `docs/DATASET.md` for the full
breakdown, licenses, and history:

- `vid_*` — frames extracted from `media/test.mp4`. **Not a reliable source
  of real bubble examples** — investigation found 99.5% of its original
  labels were mislabeled static glare, not bubbles, and the footage itself
  may not show visible leak activity. Kept as background-only data pending
  manual relabeling of confirmed real footage.
- `rf_*`, `oil_*`, `deakin_*` — real, licensed bubble/leak-test datasets
  from Roboflow Universe (CC BY 4.0 / Public Domain). Attribution is
  required for the CC BY sources if you redistribute this repository.

All labels use a single class (`0`, bubble); source datasets with multiple
bubble-related classes are remapped to it by `merge_external_dataset.py`.

## Tests

```bash
pytest
```

Tests cover the tracking and scoring logic — that a static object is never
classified as rising, that nearby bubbles do not collapse onto one track,
that a non-rising track is rejected regardless of detector confidence.

## License

Code is MIT licensed (see `LICENSE`). The `rf_*` portion of the dataset is
CC BY 4.0 and retains its own terms.
