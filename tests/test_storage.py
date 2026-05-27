from datetime import UTC, datetime
from pathlib import Path

from dfdetect.storage.db import ResultsStore
from dfdetect.types import FileResult, FrameScore, MediaKind, Verdict


def _make_result(score: float = 0.81, verdict: Verdict = Verdict.FAKE) -> FileResult:
    return FileResult(
        path=Path("/tmp/x.mp4"),
        kind=MediaKind.VIDEO,
        verdict=verdict,
        score=score,
        faces_analyzed=4,
        frame_scores=[FrameScore(0, 0.0, score)],
        processing_time_s=1.23,
        model_name="efficientnet_b4",
        model_hash="abcd",
        created_at=datetime.now(UTC),
    )


def test_save_and_read_run(tmp_path: Path):
    store = ResultsStore(tmp_path / "results.db")
    r = FileResult(
        path=Path("/tmp/x.mp4"),
        kind=MediaKind.VIDEO,
        verdict=Verdict.FAKE,
        score=0.81,
        faces_analyzed=4,
        frame_scores=[
            FrameScore(0, 0.0, 0.7),
            FrameScore(30, 1.0, 0.9),
        ],
        processing_time_s=1.23,
        model_name="efficientnet_b4",
        model_hash="abcd",
        created_at=datetime.now(UTC),
    )
    run_id = store.save_run([r], "efficientnet_b4", "abcd", config_dict={"threshold": 0.5})
    assert run_id > 0

    runs = store.list_runs()
    assert any(row["id"] == run_id for row in runs)

    rows = store.get_run(run_id)
    assert len(rows) == 1
    assert rows[0].verdict == Verdict.FAKE
    assert rows[0].score == 0.81
    assert len(rows[0].frame_scores) == 2


def test_reverdict_persists_as_derived_run(tmp_path: Path):
    store = ResultsStore(tmp_path / "results.db")
    parent_id = store.save_run(
        [_make_result(score=0.55, verdict=Verdict.UNCERTAIN)],
        model_name="efficientnet_b4",
        model_hash="abcd",
        config_dict={"threshold": 0.5, "uncertain_band": [0.4, 0.6]},
    )

    # Re-threshold so the verdict flips to REAL.
    updated = store.get_run(parent_id)
    for r in updated:
        r.verdict = Verdict.REAL

    derived_id = store.save_reverdict_run(
        parent_run_id=parent_id,
        results=updated,
        threshold=0.7,
        uncertain_band=(0.65, 0.75),
    )
    assert derived_id != parent_id

    runs = {r["id"]: r for r in store.list_runs()}
    assert runs[derived_id]["kind"] == "reverdict"
    assert runs[derived_id]["parent_run_id"] == parent_id
    assert runs[parent_id]["kind"] == "analyze"

    derived_rows = store.get_run(derived_id)
    assert derived_rows[0].verdict == Verdict.REAL
    # Score is unchanged — only verdict is re-derived.
    assert derived_rows[0].score == 0.55


def test_query_filters_by_verdict_and_score(tmp_path: Path):
    store = ResultsStore(tmp_path / "results.db")
    a = _make_result(score=0.9, verdict=Verdict.FAKE)
    b = _make_result(score=0.2, verdict=Verdict.REAL)
    store.save_run([a, b], "m", "h", config_dict={})

    fakes = store.query(verdict="FAKE")
    assert len(fakes) == 1 and fakes[0].score == 0.9

    high = store.query(min_score=0.5)
    assert len(high) == 1 and high[0].score == 0.9


def test_schema_version_is_recorded(tmp_path: Path):
    from dfdetect.storage.db import SCHEMA_VERSION

    store = ResultsStore(tmp_path / "results.db")
    assert store.schema_version() == SCHEMA_VERSION
