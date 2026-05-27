"""Tests for the per-video identity tracker (item #8)."""
from __future__ import annotations

import numpy as np

from dfdetect.preprocess.tracker import assign_tracks
from dfdetect.types import FaceCrop


def _fc(frame: int, bbox: tuple[int, int, int, int]) -> FaceCrop:
    return FaceCrop(image=np.zeros((1, 1, 3), dtype=np.uint8), bbox=bbox, frame_index=frame)


def test_single_face_gets_one_track():
    faces = [_fc(i, (10, 10, 60, 60)) for i in range(5)]
    assign_tracks(faces, iou_threshold=0.3)
    assert {f.track_id for f in faces} == {0}


def test_two_persistent_faces_get_two_tracks():
    # Two stable boxes, one on the left, one on the right.
    faces = []
    for i in range(5):
        faces.append(_fc(i, (10, 10, 60, 60)))
        faces.append(_fc(i, (200, 10, 250, 60)))
    assign_tracks(faces, iou_threshold=0.3)
    assert len({f.track_id for f in faces}) == 2


def test_drift_within_iou_threshold_keeps_one_track():
    # Box shifts a few pixels each frame; IoU remains high.
    faces = [_fc(i, (10 + i, 10 + i, 60 + i, 60 + i)) for i in range(10)]
    assign_tracks(faces, iou_threshold=0.3)
    assert len({f.track_id for f in faces}) == 1


def test_disjoint_boxes_get_separate_tracks():
    a = _fc(0, (10, 10, 60, 60))
    b = _fc(1, (300, 300, 360, 360))
    assign_tracks([a, b], iou_threshold=0.3)
    assert a.track_id != b.track_id


def test_gap_exceeds_max_gap_starts_new_track():
    a = _fc(0, (10, 10, 60, 60))
    b = _fc(100, (10, 10, 60, 60))   # same place but very far in time
    assign_tracks([a, b], iou_threshold=0.3, max_gap_frames=30)
    assert a.track_id != b.track_id
