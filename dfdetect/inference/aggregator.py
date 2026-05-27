"""Combine per-frame scores into a single video-level score, then map to a verdict."""
from __future__ import annotations

from collections.abc import Sequence

from ..config import InferenceConfig
from ..types import FrameScore, Segment, Verdict


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def _median(xs: Sequence[float]) -> float:
    # ``statistics.median`` has ~30x overhead vs an inline sort+index for the
    # 8-64 element lists this aggregator typically sees.
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) * 0.5


def aggregate(
    scores: list[float],
    cfg: InferenceConfig,
    weights: Sequence[float] | None = None,
) -> float:
    """Collapse per-face scores into a single per-file score.

    ``weights`` (one per score, same length) is required for the
    ``quality_weighted`` strategy and ignored for the others. Pass per-face
    quality scores (see :meth:`dfdetect.types.FaceCrop.quality_weight`) so
    tiny / low-confidence detections don't dominate.
    """
    if not scores:
        return 0.0
    if cfg.aggregation == "mean":
        return _mean(scores)
    if cfg.aggregation == "median":
        return _median(scores)
    if cfg.aggregation == "max":
        return float(max(scores))
    if cfg.aggregation == "weighted":
        # Compute each stat once — ``weighted`` would otherwise pay for two
        # sorts (median twice) and two summations (mean twice).
        w = cfg.weighted
        mean = _mean(scores)
        median = _median(scores)
        peak = max(scores)
        return w.mean * mean + w.median * median + w.max * peak
    if cfg.aggregation == "quality_weighted":
        if not weights or len(weights) != len(scores):
            return _mean(scores)
        total_w = 0.0
        weighted_sum = 0.0
        for s, wt in zip(scores, weights):
            weighted_sum += s * wt
            total_w += wt
        if total_w <= 0:
            return _mean(scores)
        return weighted_sum / total_w
    raise ValueError(f"Unknown aggregation: {cfg.aggregation}")


def verdict_from_score(
    score: float,
    threshold: float,
    uncertain_band: tuple[float, float],
    faces_analyzed: int,
) -> Verdict:
    if faces_analyzed == 0:
        return Verdict.NO_FACE
    lo, hi = uncertain_band
    if lo <= score <= hi:
        return Verdict.UNCERTAIN
    return Verdict.FAKE if score >= threshold else Verdict.REAL


def derive_segments(
    frame_scores: list[FrameScore],
    threshold: float,
    merge_gap_s: float = 2.0,
) -> list[Segment]:
    """Run-length-encode high-scoring frame runs into contiguous segments.

    Frames with ``score >= threshold`` start (or extend) a segment. Two
    segments separated by less than ``merge_gap_s`` seconds are merged so a
    single dip mid-fake doesn't split the report. Returns segments sorted by
    start time. Frames are assumed to be in timestamp order.
    """
    if not frame_scores:
        return []
    # Sort defensively — the pipeline already submits frames in order, but a
    # caller pulling from storage might not.
    ordered = sorted(frame_scores, key=lambda fs: fs.timestamp_s)

    raw: list[tuple[float, float, list[float]]] = []
    current_start: float | None = None
    current_scores: list[float] = []
    current_end: float = 0.0
    for fs in ordered:
        if fs.score >= threshold:
            if current_start is None:
                current_start = fs.timestamp_s
            current_end = fs.timestamp_s
            current_scores.append(fs.score)
        else:
            if current_start is not None:
                raw.append((current_start, current_end, current_scores))
                current_start = None
                current_scores = []
    if current_start is not None:
        raw.append((current_start, current_end, current_scores))

    if not raw:
        return []

    merged: list[tuple[float, float, list[float]]] = [raw[0]]
    for start, end, scores in raw[1:]:
        prev_start, prev_end, prev_scores = merged[-1]
        if start - prev_end <= merge_gap_s:
            merged[-1] = (prev_start, end, prev_scores + scores)
        else:
            merged.append((start, end, scores))

    return [
        Segment(
            t_start=start,
            t_end=end,
            score_peak=max(scores),
            score_mean=_mean(scores),
            n_frames=len(scores),
        )
        for start, end, scores in merged
    ]
