"""Centroid-based multi-object tracking for rising bubbles.

A single YOLO frame cannot tell a real leak bubble apart from glare, a
reflection or a static speck on the tank glass. Bubbles are distinguished by
their *motion over time*: they rise, and they rise in a fairly straight line.
This module supplies that temporal layer on top of per-frame detections.
"""
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class Detection:
    """One YOLO detection in a single frame, in full-frame pixel coordinates."""

    cx: int
    cy: int
    r: int
    conf: float
    box: Tuple[int, int, int, int] = field(default=(0, 0, 0, 0))


class BubbleTracker:
    """Greedy nearest-centroid tracker.

    Each detection in a new frame is matched to the closest track from the
    previous frame within ``max_match_dist`` pixels. Unmatched detections
    start new tracks; tracks that go unmatched are dropped.

    This is a deliberately minimal stand-in for a SORT-style tracker. Bubbles
    are small, numerous, and move with roughly constant upward velocity, so
    centroid proximity is sufficient and avoids a heavier dependency. The
    trade-off is documented in the README under Limitations.
    """

    def __init__(self, max_trail: int = 30, max_match_dist: float = 50.0,
                 min_trail_for_motion: int = 4, rise_threshold_px: float = 2.0,
                 straightness_scale: float = 25.0):
        self.max_trail = max_trail
        self.max_match_dist = max_match_dist
        self.min_trail_for_motion = min_trail_for_motion
        self.rise_threshold_px = rise_threshold_px
        self.straightness_scale = straightness_scale

        self.trails: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=max_trail)
        )
        self.ages: Dict[int, int] = defaultdict(int)
        self.prev_centers: Dict[int, Tuple[int, int]] = {}
        self.next_id = 0

    def update(self, detections: Sequence[Detection]) -> Dict[int, Detection]:
        """Advance the tracker by one frame.

        Returns a mapping of track id -> detection for tracks alive this frame.
        """
        new_centers: Dict[int, Tuple[int, int]] = {}
        matched: Dict[int, Detection] = {}
        claimed: set = set()

        for det in detections:
            best_id = None
            best_dist = self.max_match_dist

            for pid, (px, py) in self.prev_centers.items():
                # A previous track may only absorb one detection per frame,
                # otherwise two nearby bubbles collapse onto a single id.
                if pid in claimed:
                    continue
                dist = float(np.hypot(det.cx - px, det.cy - py))
                if dist < best_dist:
                    best_dist = dist
                    best_id = pid

            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
            else:
                claimed.add(best_id)

            new_centers[best_id] = (det.cx, det.cy)
            self.trails[best_id].append((det.cx, det.cy))
            self.ages[best_id] += 1
            matched[best_id] = det

        self.prev_centers = new_centers
        self._prune(matched)
        return matched

    def _prune(self, matched: Dict[int, Detection]) -> None:
        """Drop tracks that were not seen this frame."""
        for tid in list(self.ages.keys()):
            if tid not in matched:
                self.ages.pop(tid, None)
                self.trails.pop(tid, None)

    def is_rising(self, tid: int) -> bool:
        """True when the track's centroid moves consistently upward.

        Image coordinates put the origin at the top-left, so a *decreasing*
        y means upward motion.
        """
        trail = list(self.trails[tid])
        if len(trail) < self.min_trail_for_motion:
            return False
        mid = len(trail) // 2
        y_old = float(np.mean([p[1] for p in trail[:mid]]))
        y_new = float(np.mean([p[1] for p in trail[mid:]]))
        return y_old > y_new + self.rise_threshold_px

    def trail_straightness(self, tid: int) -> float:
        """Score a track's path from 1.0 (straight vertical) to 0.0 (erratic).

        Real bubbles rise near-vertically; glare and reflections jitter
        horizontally. Horizontal spread is therefore a useful discriminator.
        """
        trail = list(self.trails[tid])
        if len(trail) < self.min_trail_for_motion:
            return 0.5  # unknown — neither rewarded nor punished
        x_std = float(np.std([p[0] for p in trail]))
        return max(0.0, 1.0 - x_std / self.straightness_scale)

    def active_ids(self) -> List[int]:
        return list(self.ages.keys())


def compute_confidence(det: Detection, tracker: BubbleTracker, tid: int,
                       w_yolo: float = 0.45, w_age: float = 0.30,
                       w_straight: float = 0.25,
                       age_saturation: int = 12) -> float:
    """Blend per-frame and temporal evidence into one confidence value.

    A detection that is not rising is rejected outright: no amount of YOLO
    confidence should promote a stationary object to a leak bubble.
    """
    if not tracker.is_rising(tid):
        return 0.0

    age_score = min(tracker.ages[tid] / age_saturation, 1.0)
    straightness = tracker.trail_straightness(tid)

    confidence = (
        det.conf * w_yolo
        + age_score * w_age
        + straightness * w_straight
    )
    return min(confidence, 0.99)
