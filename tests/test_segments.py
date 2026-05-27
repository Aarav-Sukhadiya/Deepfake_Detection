"""Tests for segment derivation (item #26)."""
from __future__ import annotations

from dfdetect.inference.aggregator import derive_segments
from dfdetect.types import FrameScore


def _fs(t: float, score: float) -> FrameScore:
    return FrameScore(frame_index=int(t * 30), timestamp_s=t, score=score)


def test_empty_input_returns_no_segments():
    assert derive_segments([], threshold=0.5) == []


def test_all_low_scores_returns_no_segments():
    frames = [_fs(t, 0.1) for t in range(0, 10)]
    assert derive_segments(frames, threshold=0.5) == []


def test_single_high_run_one_segment():
    frames = [_fs(0, 0.1), _fs(1, 0.9), _fs(2, 0.95), _fs(3, 0.8), _fs(4, 0.1)]
    segs = derive_segments(frames, threshold=0.5)
    assert len(segs) == 1
    assert segs[0].t_start == 1
    assert segs[0].t_end == 3
    assert segs[0].n_frames == 3
    assert segs[0].score_peak == 0.95
    assert abs(segs[0].score_mean - (0.9 + 0.95 + 0.8) / 3) < 1e-9


def test_merge_gap_combines_close_runs():
    # Two high runs separated by a single low frame at t=2; merge_gap_s=2.0
    # should glue them into one segment so a momentary dip doesn't split it.
    frames = [_fs(0, 0.9), _fs(1, 0.85), _fs(2, 0.1), _fs(3, 0.92), _fs(4, 0.88)]
    segs = derive_segments(frames, threshold=0.5, merge_gap_s=2.0)
    assert len(segs) == 1
    assert segs[0].t_start == 0
    assert segs[0].t_end == 4
    assert segs[0].n_frames == 4   # the 0.1 frame is NOT counted as part of the merge


def test_gap_larger_than_merge_threshold_keeps_runs_separate():
    # Three high-score islands separated by long stretches of below-threshold
    # frames; merge_gap_s=2.0 is much smaller than the gaps so they stay split.
    frames = [
        _fs(0, 0.9),
        _fs(1, 0.1),
        _fs(2, 0.1),
        _fs(3, 0.1),
        _fs(4, 0.1),
        _fs(5, 0.1),
        _fs(6, 0.9),
        _fs(7, 0.1),
        _fs(8, 0.1),
        _fs(9, 0.1),
        _fs(10, 0.1),
        _fs(11, 0.9),
    ]
    segs = derive_segments(frames, threshold=0.5, merge_gap_s=2.0)
    assert len(segs) == 3


def test_segments_sorted_even_if_input_unordered():
    frames = [_fs(5, 0.9), _fs(1, 0.9), _fs(3, 0.1)]
    segs = derive_segments(frames, threshold=0.5, merge_gap_s=0.5)
    assert [s.t_start for s in segs] == sorted(s.t_start for s in segs)
