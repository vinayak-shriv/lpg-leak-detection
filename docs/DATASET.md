# Dataset

## Sources

| Prefix | Origin | Images | License |
|---|---|---|---|
| `vid_` | Frames from `media/test.mp4`, drafted with `auto_label.py` | 513 (20 real, see Known history) | Project-owned |
| `rf_` | [Air Leak Bubble Detection](https://universe.roboflow.com/vision-test-ic1cb/air-leak-bubble-detection), Roboflow Universe | 966 | CC BY 4.0 |
| `oil_` | [Bubble Leak Test in Oil Cooler](https://universe.roboflow.com/techit-sritrakul/bubble-leak-test-in-oil-cooler), Roboflow Universe | 2,252 (incl. 3x in-export augmentation, see Known history) | Public Domain |
| `deakin_` | [Bubble Dataset](https://universe.roboflow.com/deakin-hbdei/bubble-dataset-kkrbu), Roboflow Universe | 1,639 | CC BY 4.0 |

Filename prefixes keep sources distinguishable after merging, so a split can
be inspected for source imbalance and any source can be removed without
guesswork. `oil_` and `deakin_` are fetched with
`scripts/fetch_supplemental_bubble_data.py` (needs `ROBOFLOW_API_KEY`) and
folded in with `scripts/merge_external_dataset.py`, which also remaps their
label class indices to `0` (`deakin_`'s original `Bubble`/`Water_Bubble`
classes both become `bubble`).

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
- **`vid_*` labels were almost entirely wrong (2026-09-04).** Retraining
  produced models that detected zero bubbles on `media/test.mp4` despite
  a reasonable held-out mAP. Investigation found 4,633 of 4,654 `vid_*`
  boxes (99.5%) sat within 20px of just 10 fixed screen positions ---
  static glare and rig components, not bubbles, apparently mislabeled by
  `auto_label.py`'s drafts and never corrected during review.
  `scripts/strip_static_labels.py` removed them; the 493 now-empty `vid_*`
  frames are kept as background negatives.
  Worse: the ~20 boxes that *did* survive that cleanup turned out to also be
  near-static (two more loose clusters revisited at near-identical positions
  across dozens of non-consecutive frames), and visually inspecting frames
  spread across the full 4-minute video found no obviously visible rising
  bubbles anywhere. `media/test.mp4` may not contain genuine, visible leak
  activity at all -- it may be closer to a demo/stock clip (identical
  background throughout, a scrolling data-ticker and logo baked into frame)
  than a live water-bath recording. Practically: **treat `vid_*` as
  background-only data for now**, not a source of positive bubble examples,
  until someone confirms real leak footage and relabels it by hand.
  `scripts/find_bubble_candidates.py` (motion-based, background-subtraction
  candidate finder) can help surface real candidates from new footage
  without bootstrapping from a YOLO model's own biases.
- **`oil_` augmentation.** The oil-cooler export applies 3x in-export
  augmentation (including 50% vertical flip) per source image. Fine for a
  single-frame detector's appearance learning, but it means the random
  train/val split can put near-duplicate augmented siblings on both sides,
  flattering validation the same way correlated `vid_*` frames do (see
  Caveats below).

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
