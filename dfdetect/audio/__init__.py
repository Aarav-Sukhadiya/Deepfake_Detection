"""Audio-deepfake branch (planned, IMPROVEMENTS.md #22).

Today this is a placeholder package. A full implementation will:

* Extract audio with ``ffmpeg`` (or ``decord.AudioReader``) at 16 kHz mono.
* Run a model such as RawNet2 or AASIST on the waveform/spectrogram.
* Produce a parallel ``audio_score`` and ``audio_segments`` per file, surfaced
  alongside the visual verdict.

See :class:`AudioDetector` for the contract a real implementation must satisfy.
"""
from .base import AudioDetector  # noqa: F401
