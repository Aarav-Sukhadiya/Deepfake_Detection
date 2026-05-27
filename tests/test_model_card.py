"""Test the model-card renderer (item #25)."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from dfdetect.config import Config
from dfdetect.report.model_card import render_model_card
from dfdetect.types import FileResult, MediaKind, Verdict


def _result(verdict: Verdict, score: float) -> FileResult:
    return FileResult(
        path=Path("/tmp/x.png"),
        kind=MediaKind.IMAGE,
        verdict=verdict,
        score=score,
        faces_analyzed=1,
        processing_time_s=0.1,
        model_name="efficientnet_b4",
        model_hash="abcd1234",
        created_at=datetime.now(UTC),
    )


def test_model_card_renders_expected_sections(tmp_path: Path):
    cfg = Config()
    results = [_result(Verdict.FAKE, 0.9), _result(Verdict.REAL, 0.1)]
    out = render_model_card(cfg, results, tmp_path / "card.md")
    body = out.read_text()
    assert "# Model Card" in body
    assert "efficientnet_b4" in body
    assert "abcd1234" in body
    assert "Decision policy" in body
    assert "FAKE" in body
    assert "Known limitations" in body
