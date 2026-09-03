"""Tests for the tracking and confidence logic.

These cover the part of the system that is genuinely ours rather than
Ultralytics': deciding which detections count as rising bubbles.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lpgdetect.overlays import CylinderMonitor, status_for  # noqa: E402
from lpgdetect.tracking import (BubbleTracker, Detection,  # noqa: E402
                                compute_confidence)


def det(cx, cy, conf=0.8, r=8):
    return Detection(cx=cx, cy=cy, r=r, conf=conf)


def feed(tracker, points, conf=0.8):
    """Push a sequence of positions through the tracker, one per frame."""
    tid = None
    for cx, cy in points:
        matched = tracker.update([det(cx, cy, conf)])
        tid = next(iter(matched))
    return tid


class TestTracking:
    def test_same_bubble_keeps_one_id(self):
        tracker = BubbleTracker()
        for cy in range(200, 150, -5):
            matched = tracker.update([det(100, cy)])
        assert list(matched) == [0]

    def test_distant_detection_starts_new_track(self):
        tracker = BubbleTracker(max_match_dist=50)
        tracker.update([det(100, 200)])
        matched = tracker.update([det(400, 200)])  # far beyond the threshold
        assert list(matched) == [1]

    def test_two_bubbles_do_not_collapse_onto_one_id(self):
        """Nearby bubbles must not be merged into a single track."""
        tracker = BubbleTracker()
        tracker.update([det(100, 200), det(130, 200)])
        matched = tracker.update([det(100, 195), det(130, 195)])
        assert len(matched) == 2

    def test_lost_track_is_pruned(self):
        tracker = BubbleTracker()
        tracker.update([det(100, 200)])
        tracker.update([])
        assert tracker.active_ids() == []


class TestRisingDetection:
    def test_rising_bubble_is_detected(self):
        tracker = BubbleTracker()
        tid = feed(tracker, [(100, y) for y in (240, 230, 220, 210, 200)])
        assert tracker.is_rising(tid)

    def test_sinking_object_is_not_rising(self):
        tracker = BubbleTracker()
        tid = feed(tracker, [(100, y) for y in (200, 210, 220, 230, 240)])
        assert not tracker.is_rising(tid)

    def test_static_object_is_not_rising(self):
        """Glare and specks sit still; they must never count as bubbles."""
        tracker = BubbleTracker()
        tid = feed(tracker, [(100, 200)] * 6)
        assert not tracker.is_rising(tid)

    def test_short_track_is_not_yet_rising(self):
        tracker = BubbleTracker()
        tid = feed(tracker, [(100, 210), (100, 205)])
        assert not tracker.is_rising(tid)


class TestStraightness:
    def test_vertical_trail_scores_high(self):
        tracker = BubbleTracker()
        tid = feed(tracker, [(100, y) for y in range(240, 190, -10)])
        assert tracker.trail_straightness(tid) > 0.9

    def test_erratic_trail_scores_low(self):
        # Jitter stays inside the tracker's match distance, so this is one
        # wobbling track rather than a series of new ones.
        tracker = BubbleTracker()
        tid = feed(tracker, [(100, 240), (130, 230), (100, 220),
                             (130, 210), (100, 200), (130, 190)])
        assert tracker.trail_straightness(tid) < 0.5


class TestConfidence:
    def test_non_rising_track_is_rejected_outright(self):
        tracker = BubbleTracker()
        tid = feed(tracker, [(100, 200)] * 6, conf=0.99)
        detection = det(100, 200, conf=0.99)
        assert compute_confidence(detection, tracker, tid) == 0.0

    def test_rising_straight_persistent_track_scores_high(self):
        tracker = BubbleTracker()
        points = [(100, y) for y in range(300, 150, -10)]
        tid = feed(tracker, points, conf=0.9)
        score = compute_confidence(det(100, 150, conf=0.9), tracker, tid)
        assert score > 0.8

    def test_confidence_never_reaches_certainty(self):
        tracker = BubbleTracker()
        points = [(100, y) for y in range(400, 100, -10)]
        tid = feed(tracker, points, conf=1.0)
        assert compute_confidence(det(100, 100, conf=1.0),
                                  tracker, tid) <= 0.99


class TestStatus:
    @pytest.mark.parametrize("score,expected", [
        (0.0, "CLEAR"), (0.30, "CLEAR"), (0.50, "SUSPECT"), (0.90, "LEAK"),
    ])
    def test_status_thresholds(self, score, expected):
        assert status_for(score)[0] == expected


class TestCylinderMonitor:
    def test_bubble_scores_the_zone_it_sits_in(self):
        monitor = CylinderMonitor(500, ["A", "B", "C", "D", "E"])
        monitor.update(cx=450, confidence=0.9)  # rightmost zone
        assert monitor.scores[4] > 0
        assert sum(monitor.scores[:4]) == 0

    def test_scores_decay_towards_clear(self):
        monitor = CylinderMonitor(500, ["A", "B"])
        monitor.update(cx=100, confidence=0.9)
        before = monitor.scores[0]
        monitor.decay()
        assert monitor.scores[0] < before

    def test_edge_position_does_not_overflow(self):
        monitor = CylinderMonitor(500, ["A", "B"])
        monitor.update(cx=499, confidence=0.5)  # last pixel
        assert len(monitor.scores) == 2
