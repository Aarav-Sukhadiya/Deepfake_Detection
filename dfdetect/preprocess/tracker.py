"""Cheap per-video identity tracker.

When ``largest_face_only=False`` and the source is a video, we get multiple
face crops per frame. Aggregating them all together blurs a fake speaker with
a real co-presenter; aggregating by *track* gives each identity its own score
so the per-file verdict can be derived from the worst (or any chosen)
identity.

The tracker is intentionally simple: greedy IoU + centroid matching across
consecutive frames, no Kalman filter, no re-id. It's enough to keep two or
three stable speakers separated for a few minutes of news/interview footage,
which is the case IMPROVEMENTS.md item #8 calls out. Detection failures on a
single frame are tolerated via a small ``max_gap`` lookback.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from ..types import FaceCrop


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class _Track:
    track_id: int
    last_bbox: tuple[int, int, int, int]
    last_frame: int
    faces: list[FaceCrop] = field(default_factory=list)


def assign_tracks(
    faces: Iterable[FaceCrop],
    iou_threshold: float = 0.3,
    max_gap_frames: int = 30,
) -> list[FaceCrop]:
    """Assign a ``track_id`` to each face in-place and return them in order.

    Faces are matched against open tracks by IoU on the bbox; the best match
    above ``iou_threshold`` claims the face. Tracks with no update for more
    than ``max_gap_frames`` are closed and won't match new faces. Frames need
    not arrive contiguously — we use the ``frame_index`` on each ``FaceCrop``.
    """
    open_tracks: list[_Track] = []
    out: list[FaceCrop] = []
    next_id = 0

    # Group faces by frame index so all detections in a frame can compete
    # against open tracks together (avoids two faces claiming the same track).
    by_frame: dict[int, list[FaceCrop]] = {}
    for f in faces:
        by_frame.setdefault(f.frame_index, []).append(f)

    for frame_idx in sorted(by_frame):
        # Retire stale tracks.
        open_tracks = [t for t in open_tracks if frame_idx - t.last_frame <= max_gap_frames]
        frame_faces = by_frame[frame_idx]

        # Greedy match: for each face, pick the best-IoU open track that's
        # still unclaimed this frame.
        claimed: set[int] = set()
        for f in frame_faces:
            best_track: _Track | None = None
            best_iou = iou_threshold
            for t in open_tracks:
                if t.track_id in claimed:
                    continue
                score = _iou(f.bbox, t.last_bbox)
                if score > best_iou:
                    best_iou = score
                    best_track = t
            if best_track is None:
                best_track = _Track(track_id=next_id, last_bbox=f.bbox, last_frame=frame_idx)
                next_id += 1
                open_tracks.append(best_track)
            best_track.last_bbox = f.bbox
            best_track.last_frame = frame_idx
            best_track.faces.append(f)
            claimed.add(best_track.track_id)
            f.track_id = best_track.track_id
            out.append(f)
    return out
