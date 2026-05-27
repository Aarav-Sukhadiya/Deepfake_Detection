"""Audio-detector interface scaffold (IMPROVEMENTS.md #22).

The visual branch is fully wired; the audio branch is intentionally a stub.
Drop a concrete detector under this package (e.g. ``rawnet2.py``) that
implements :class:`AudioDetector`, register it via a sibling registry, and
plug it into ``Pipeline._result_for_item`` once you're ready.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class AudioDetector(ABC):
    """A waveform/spectrogram → fake probability classifier."""

    name: str = "base-audio"

    @abstractmethod
    def score(self, media_path: Path) -> float:
        """Return the per-file fake probability for the audio track."""

    @abstractmethod
    def segments(self, media_path: Path) -> list[tuple[float, float, float]]:
        """Return ``[(t_start, t_end, peak_score), ...]`` for suspicious audio runs."""
