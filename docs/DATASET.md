# Dataset

## Sources

| Prefix | Origin | Images | License |
|---|---|---|---|
| `vid_` | Frames from `media/test.mp4`, drafted with `auto_label.py` and reviewed | 513 | Project-owned |
| `rf_` | [Air Leak Bubble Detection](https://universe.roboflow.com/vision-test-ic1cb/air-leak-bubble-detection), Roboflow Universe | 966 | CC BY 4.0 |

Filename prefixes keep the two sources distinguishable after merging, so a
split can be inspected for source imbalance and either source can be removed
without guesswork.

## Format

Standard YOLO detection labels, one `.txt` per image, one box per line:

```
<class> <centre_x> <centre_y> <width> <height>
```

All four coordinates are normalised to 0–1 relative to image dimensions.
A single class is used: `0` = bubble.

## Known history

- **Mixed formats.** The Roboflow export contained 1,540 segmentation
  polygon lines alongside detection boxes. `normalize_labels.py` converts
  polygons to their axis-aligned bounding boxes.
- **Class count.** An earlier config declared `nc: 2` with a `large_leak`
  class, but no label in the dataset ever used class index 1. The config now
  declares `nc: 1`.
- **Orphan image.** `vid_frame_003416.jpg` has no label file and is skipped
  by `split_dataset.py`. Either label it or delete it.

## Rebuilding the split

```bash
python scripts/split_dataset.py --check-only      # audit
python scripts/split_dataset.py --val-fraction 0.2
```

The split is seeded (`--seed`, default 42), so it is reproducible. Images
and labels are copied into `dataset/{images,labels}/{train,val}`, which are
cleared first.

## Caveats

Both sources are narrow. The video frames come from a single camera position
in one bath, and consecutive frames — even sampled at 0.5 s — remain
correlated. A random train/val split therefore puts visually similar frames
on both sides, which flatters validation metrics.

A stricter evaluation would hold out an entire separate video. Treat the
current validation numbers as a sanity check, not an estimate of field
performance.
