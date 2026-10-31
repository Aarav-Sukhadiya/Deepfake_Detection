"""Regression tests for bugs fixed during the cleanup pass. Each test names
the failure it guards against."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from dfdetect.config import Config, load_config
from dfdetect.inference.aggregator import reverdict
from dfdetect.preprocess.tracker import assign_tracks
from dfdetect.report.exporter import write_html
from dfdetect.storage.db import SCHEMA_VERSION, ResultsStore
from dfdetect.types import FaceCrop, FileResult, FrameScore, MediaKind, Segment, Verdict

# --- storage ----------------------------------------------------------------


def _make_v2_db(path: Path) -> None:
    """A results.db as written by schema v2 (JSON frame_scores/segments)."""
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE schema_version (id INTEGER PRIMARY KEY, version INTEGER, updated_at DATETIME);
        INSERT INTO schema_version (version, updated_at) VALUES (2, CURRENT_TIMESTAMP);
        CREATE TABLE runs (id INTEGER PRIMARY KEY, started_at DATETIME, model_name VARCHAR(64),
            model_hash VARCHAR(64), config_json TEXT, parent_run_id INTEGER, kind VARCHAR(16));
        CREATE TABLE results (id INTEGER PRIMARY KEY, run_id INTEGER, path VARCHAR(1024),
            kind VARCHAR(16), verdict VARCHAR(16), score FLOAT, faces_analyzed INTEGER,
            processing_time_s FLOAT, error TEXT, heatmap_path VARCHAR(1024),
            frame_scores JSON, segments JSON, created_at DATETIME);
        INSERT INTO runs VALUES (1, '2026-01-01 00:00:00', 'm', 'h', '{}', NULL, 'analyze');
        """
    )
    con.execute(
        "INSERT INTO results VALUES (7, 1, '/v.mp4', 'video', 'FAKE', 0.9, 2, 1.0, NULL, NULL, ?, ?, "
        "'2026-01-01 00:00:00')",
        (
            json.dumps([{"frame": 0, "t": 0.0, "score": 0.8}, {"frame": 30, "t": 1.0, "score": 0.95}]),
            json.dumps([{"t_start": 0.0, "t_end": 1.0, "score_peak": 0.95, "score_mean": 0.875, "n_frames": 2}]),
        ),
    )
    con.commit()
    con.close()


def test_v2_db_migrates_json_into_tables(tmp_path: Path):
    db = tmp_path / "results.db"
    _make_v2_db(db)
    store = ResultsStore(db)
    assert store.schema_version() == SCHEMA_VERSION
    [r] = store.get_run(1)
    assert [fs.frame_index for fs in r.frame_scores] == [0, 30]
    assert r.segments[0].n_frames == 2
    assert r.model_name == "m"  # read back from the parent run
    # Re-opening is a no-op, not a second migration.
    assert len(ResultsStore(db).get_run(1)[0].frame_scores) == 2
    tables = {t for (t,) in sqlite3.connect(db).execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "results_old" not in tables


def test_fresh_db_has_no_legacy_json_columns_and_has_indexes(tmp_path: Path):
    db = tmp_path / "results.db"
    ResultsStore(db)
    con = sqlite3.connect(db)
    cols = {r[1] for r in con.execute("PRAGMA table_info(results)")}
    assert "segments" not in cols and "frame_scores" not in cols
    idx = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='index'")}
    assert {"ix_frame_scores_result_id", "ix_segments_result_id"} <= idx


def test_save_run_accepts_unclassified_item(tmp_path: Path):
    """Files whose type couldn't be detected have kind=None; save_run used to
    crash on ``None.value`` after the whole analysis had finished."""
    store = ResultsStore(tmp_path / "results.db")
    bad = FileResult(path=Path("/x.bin"), kind=None, verdict=Verdict.ERROR, score=0.0, error="input: boom")
    run_id = store.save_run([bad], "m", "h", config_dict={})
    [back] = store.get_run(run_id)
    assert back.kind is None and back.verdict == Verdict.ERROR


def test_list_runs_counts_results(tmp_path: Path):
    store = ResultsStore(tmp_path / "results.db")
    r = FileResult(path=Path("/a.png"), kind=MediaKind.IMAGE, verdict=Verdict.REAL, score=0.1)
    rid = store.save_run([r, r, r], "m", "h", config_dict={})
    empty = store.save_run([], "m", "h", config_dict={})
    counts = {row["id"]: row["n_results"] for row in store.list_runs()}
    assert counts == {rid: 3, empty: 0}


# --- verdicts ---------------------------------------------------------------


def test_reverdict_keeps_error_results():
    err = FileResult(
        path=Path("/a.mp4"), kind=MediaKind.VIDEO, verdict=Verdict.ERROR,
        score=0.0, faces_analyzed=5, error="inference: OOM",
    )
    assert reverdict(err, 0.5, (0.4, 0.6)) == Verdict.ERROR
    ok = FileResult(path=Path("/b.png"), kind=MediaKind.IMAGE, verdict=Verdict.REAL, score=0.9, faces_analyzed=1)
    assert reverdict(ok, 0.5, (0.4, 0.6)) == Verdict.FAKE


# --- report -----------------------------------------------------------------


def test_html_report_escapes_filenames(tmp_path: Path):
    evil = FileResult(
        path=Path("/tmp/<script>alert(1)</script>.mp4"), kind=MediaKind.VIDEO, verdict=Verdict.FAKE,
        score=0.9, faces_analyzed=1, frame_scores=[FrameScore(0, 0.0, 0.9)],
        segments=[Segment(0.0, 0.0, 0.9, 0.9, 1)],
    )
    body = write_html([evil], tmp_path / "r.html").read_text()
    assert "<script>alert(1)</script>" not in body
    assert "&lt;script&gt;" in body


# --- tracker ----------------------------------------------------------------


def _fc(frame: int, bbox: tuple[int, int, int, int]) -> FaceCrop:
    return FaceCrop(image=np.zeros((1, 1, 3), dtype=np.uint8), bbox=bbox, frame_index=frame)


def test_tracker_matches_best_iou_regardless_of_detection_order():
    # Tracks A=(0..100) and B=(40..140). In frame 1, X=(15..115) prefers A
    # (IoU .74 vs .60) but Y=(0..100) *is* A (IoU 1.0). The old per-face loop
    # handled X first, gave it A, and pushed Y onto B.
    a0, b0 = _fc(0, (0, 0, 100, 100)), _fc(0, (40, 0, 140, 100))
    x1, y1 = _fc(1, (15, 0, 115, 100)), _fc(1, (0, 0, 100, 100))
    assign_tracks([a0, b0, x1, y1], iou_threshold=0.3)
    assert y1.track_id == a0.track_id
    assert x1.track_id == b0.track_id


def test_track_gap_scales_with_sampling_step():
    from dfdetect.pipeline import _track_gap

    # 1 fps sampling of a 60 fps clip: samples are 60 source frames apart.
    faces = [_fc(i * 60, (10, 10, 60, 60)) for i in range(5)]
    assign_tracks(faces, max_gap_frames=_track_gap(faces))
    assert {f.track_id for f in faces} == {0}


# --- config / CLI -----------------------------------------------------------


def test_explicit_missing_config_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_config_rejects_zero_fps():
    with pytest.raises(ValueError):
        Config(video={"frames_per_second": 0})


def test_cli_rejects_bad_output_format_before_analysis(tmp_path: Path):
    from dfdetect.cli import app

    res = CliRunner().invoke(app, ["analyze", str(tmp_path), "--output", "pdf"])
    assert res.exit_code != 0
    assert "pdf" in res.output


# --- training split ---------------------------------------------------------


def test_group_split_keeps_video_frames_together(tmp_path: Path):
    from dfdetect.training.trainer import _group_split, _Sample

    samples = []
    for v in range(10):
        p = tmp_path / f"v{v}.mp4"
        p.write_bytes(b"x")
        samples += [_Sample(path=p, label=v % 2, frame_index=i) for i in range(5)]
    # A hardlinked duplicate of v0 must land on the same side as v0.
    dup = tmp_path / "v0_dup.mp4"
    dup.hardlink_to(tmp_path / "v0.mp4")
    samples += [_Sample(path=dup, label=0, frame_index=i) for i in range(5)]

    train, val = _group_split(samples, 0.2, np.random.default_rng(0))
    assert len(train) + len(val) == len(samples) and val
    train_inodes = {s.path.stat().st_ino for s in train}
    val_inodes = {s.path.stat().st_ino for s in val}
    assert not train_inodes & val_inodes
