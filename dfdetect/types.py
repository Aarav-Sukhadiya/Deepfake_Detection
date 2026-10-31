"""Shared dataclasses passed between pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class MediaKind(StrEnum):
    IMAGE = "image"
    VIDEO = "video"


class Verdict(StrEnum):
    REAL = "REAL"
    FAKE = "FAKE"
    UNCERTAIN = "UNCERTAIN"
    NO_FACE = "NO_FACE"
    ERROR = "ERROR"


@dataclass
class MediaItem:
    path: Path
    kind: MediaKind | None             # None when the file couldn't be classified
    size_bytes: int
    error: str | None = None


@dataclass
class FaceCrop:
    """One face crop ready for the model."""
    image: Any                 # numpy.ndarray (H, W, 3) RGB uint8
    bbox: tuple[int, int, int, int]
    frame_index: int = 0       # 0 for images, frame number for videos
    timestamp_s: float = 0.0   # 0 for images
    detector_confidence: float = 1.0  # 0..1, detector-provided when available
    track_id: int | None = None        # cross-frame identity (multi-face videos)

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    def quality_weight(self) -> float:
        """Heuristic quality score in (0, 1].

        Larger, higher-confidence detections get more weight when aggregating
        per-face scores into a per-file verdict, so a tiny background face
        doesn't drag a video's score the same way a centred talking head does.
        Floors at 0.05 so a genuine detection is never fully ignored.
        """
        area_w = min(1.0, self.area / (256 * 256))
        conf = max(0.0, min(1.0, self.detector_confidence))
        return max(0.05, 0.5 * area_w + 0.5 * conf)


@dataclass
class FrameScore:
    frame_index: int
    timestamp_s: float
    score: float


@dataclass
class Segment:
    """A contiguous run of suspicious frames inside one video.

    ``score_peak`` / ``score_mean`` are computed over the frames covered. Two
    runs separated by a gap of one sampling step are merged (see
    :func:`dfdetect.inference.aggregator.derive_segments`) so a momentary dip
    doesn't split what is otherwise one continuous fake region.
    """
    t_start: float
    t_end: float
    score_peak: float
    score_mean: float
    n_frames: int


@dataclass
class FileResult:
    path: Path
    kind: MediaKind | None
    verdict: Verdict
    score: float                       # final aggregated score in [0, 1]
    faces_analyzed: int = 0
    frame_scores: list[FrameScore] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    processing_time_s: float = 0.0
    model_name: str = ""
    model_hash: str = ""
    error: str | None = None
    heatmap_path: Path | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "kind": self.kind.value if self.kind else None,
            "verdict": self.verdict.value,
            "score": round(self.score, 4),
            "faces_analyzed": self.faces_analyzed,
            "frame_scores": [
                {"frame": fs.frame_index, "t": round(fs.timestamp_s, 3), "score": round(fs.score, 4)}
                for fs in self.frame_scores
            ],
            "segments": [
                {
                    "t_start": round(seg.t_start, 3),
                    "t_end": round(seg.t_end, 3),
                    "score_peak": round(seg.score_peak, 4),
                    "score_mean": round(seg.score_mean, 4),
                    "n_frames": seg.n_frames,
                }
                for seg in self.segments
            ],
            "processing_time_s": round(self.processing_time_s, 3),
            "model_name": self.model_name,
            "model_hash": self.model_hash,
            "error": self.error,
            "heatmap_path": str(self.heatmap_path) if self.heatmap_path else None,
            "created_at": self.created_at.isoformat(),
        }
