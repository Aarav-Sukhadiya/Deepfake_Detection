"""Active-learning loop scaffold (IMPROVEMENTS.md #23).

Sketch of the future flow:

1. Each ``analyze`` run already persists per-file scores. Surface
   ``Verdict.UNCERTAIN`` rows from the SQLite store in the Gradio UI.
2. Accept a user label (``REAL``/``FAKE``) plus an optional confidence.
3. Append the labelled face crops to ``~/.cache/dfdetect/active_learning/``
   keyed by ``(run_id, result_id)``, so the offline trainer (see
   ``dfdetect/training/``) can pick them up without re-running inference.
4. Optionally write a sidecar ``labels.csv`` for downstream tooling.

This module is a placeholder until #27 (training UI) lands — the labelling
surface and the trainer should be designed together so the label format is
the same on both sides.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class UserLabel:
    run_id: int
    result_id: int
    label: str          # "REAL" | "FAKE"
    confidence: float   # 0..1, user-reported certainty
    note: str = ""


def store_label(label: UserLabel, root: Path) -> Path:  # pragma: no cover - scaffold
    """Persist a user label. NOT YET WIRED — raises NotImplementedError."""
    raise NotImplementedError(
        "Active-learning persistence is a v2 feature. "
        "See IMPROVEMENTS.md #23 and #27 (training UI)."
    )
