"""Unit tests for scene-aware sampling. Synthetic frames, no real video file."""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from dfdetect.config import VideoConfig  # noqa: E402
from dfdetect.preprocess.frame_sampler import (  # noqa: E402
    _hsv_histogram,
    _plan_indices,
    detect_scene_boundaries,
)


def _solid(rgb: tuple[int, int, int], size: int = 64) -> np.ndarray:
    f = np.zeros((size, size, 3), dtype=np.uint8)
    f[:] = rgb
    return f


def test_detect_scene_boundaries_flags_color_change():
    """A run of identical frames followed by a sharp colour swap fires a boundary."""
    frames = [_solid((255, 0, 0))] * 5 + [_solid((0, 255, 0))] * 5
    hists = [_hsv_histogram(f, thumb_size=32) for f in frames]
    boundaries = detect_scene_boundaries(hists, threshold=0.6)
    # Index 0 is always included; the second boundary is at the swap (idx 5).
    assert boundaries[0] == 0
    assert 5 in boundaries


def test_detect_scene_boundaries_uniform_clip():
    """Identical frames should produce only the opening boundary."""
    frames = [_solid((128, 128, 128))] * 8
    hists = [_hsv_histogram(f, thumb_size=32) for f in frames]
    assert detect_scene_boundaries(hists, threshold=0.6) == [0]


def test_plan_indices_uniform_matches_step():
    cfg = VideoConfig(frames_per_second=1.0, max_frames=10, sampling_strategy="uniform")
    indices = _plan_indices(total=60, fps=30.0, cfg=cfg, fetch_frame=lambda i: _solid((0, 0, 0)))
    # 30 fps / 1 fps = step 30 → indices [0, 30].
    assert indices == [0, 30]


def test_plan_indices_scene_returns_only_boundaries():
    """Build a synthetic 'video' with one cut and verify the scene strategy
    returns the boundary indices, not the uniform schedule."""
    # 30 source frames: first 15 red, last 15 green. fps=30 → cut at frame 15.
    red = _solid((255, 0, 0))
    green = _solid((0, 255, 0))

    def fetch(i: int) -> np.ndarray:
        return red if i < 15 else green

    cfg = VideoConfig(
        frames_per_second=1.0,
        max_frames=10,
        sampling_strategy="scene",
        scene_threshold=0.6,
        scene_stride=1,   # probe every source frame so the cut is detected
    )
    indices = _plan_indices(total=30, fps=30.0, cfg=cfg, fetch_frame=fetch)
    # Opening boundary plus the cut at idx 15.
    assert indices[0] == 0
    assert 15 in indices
    # No spurious mid-scene picks.
    assert all(i in (0, 15) for i in indices)


def test_plan_indices_hybrid_unions_uniform_and_scene():
    red = _solid((255, 0, 0))
    green = _solid((0, 255, 0))

    def fetch(i: int) -> np.ndarray:
        return red if i < 15 else green

    cfg = VideoConfig(
        frames_per_second=1.0,
        max_frames=10,
        sampling_strategy="hybrid",
        scene_threshold=0.6,
        scene_stride=1,
    )
    indices = _plan_indices(total=60, fps=30.0, cfg=cfg, fetch_frame=fetch)
    # Uniform alone would be [0, 30]; scene alone would be [0, 15]; hybrid = [0, 15, 30].
    assert indices == [0, 15, 30]


def test_plan_indices_respects_max_frames_cap():
    cfg = VideoConfig(frames_per_second=10.0, max_frames=3, sampling_strategy="uniform")
    indices = _plan_indices(total=300, fps=30.0, cfg=cfg, fetch_frame=lambda i: _solid((0, 0, 0)))
    assert len(indices) == 3


def test_video_config_rejects_out_of_range_threshold():
    with pytest.raises(ValueError):
        VideoConfig(scene_threshold=1.5)
