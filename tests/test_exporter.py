import csv
import json
from datetime import UTC, datetime
from pathlib import Path

from dfdetect.report.exporter import write_csv, write_html, write_json
from dfdetect.types import FileResult, FrameScore, MediaKind, Verdict


def _sample(kind=MediaKind.IMAGE, verdict=Verdict.REAL, score=0.12) -> FileResult:
    return FileResult(
        path=Path("/tmp/a.png"),
        kind=kind,
        verdict=verdict,
        score=score,
        faces_analyzed=1,
        frame_scores=[FrameScore(0, 0.0, score)],
        processing_time_s=0.5,
        model_name="efficientnet_b4",
        model_hash="abcd",
        created_at=datetime.now(UTC),
    )


def test_write_json_roundtrip(tmp_path: Path):
    out = write_json([_sample()], tmp_path / "r.json")
    data = json.loads(out.read_text())
    assert data[0]["verdict"] == "REAL"


def test_write_csv_has_header(tmp_path: Path):
    out = write_csv([_sample()], tmp_path / "r.csv")
    rows = list(csv.DictReader(out.open()))
    assert rows[0]["verdict"] == "REAL"
    assert rows[0]["model_name"] == "efficientnet_b4"


def test_write_html_renders(tmp_path: Path):
    out = write_html([_sample()], tmp_path / "r.html", title="t")
    body = out.read_text()
    assert "<table>" in body
    assert "REAL" in body


def test_write_html_includes_svg_detail_for_videos(tmp_path: Path):
    from dfdetect.types import Segment
    video = FileResult(
        path=Path("/tmp/clip.mp4"),
        kind=MediaKind.VIDEO,
        verdict=Verdict.FAKE,
        score=0.85,
        faces_analyzed=10,
        frame_scores=[
            FrameScore(0, 0.0, 0.4),
            FrameScore(30, 1.0, 0.9),
            FrameScore(60, 2.0, 0.95),
        ],
        segments=[Segment(t_start=1.0, t_end=2.0, score_peak=0.95, score_mean=0.925, n_frames=2)],
        processing_time_s=1.5,
        model_name="efficientnet_b4",
        model_hash="abcd",
        created_at=datetime.now(UTC),
    )
    out = write_html([video], tmp_path / "r.html", threshold=0.5, uncertain_band=(0.4, 0.6))
    body = out.read_text()
    assert "<svg" in body
    assert "polyline" in body
    assert "FAKE" in body
    # Segment time range should be rendered in the per-file detail block.
    assert "1.00s" in body and "2.00s" in body
